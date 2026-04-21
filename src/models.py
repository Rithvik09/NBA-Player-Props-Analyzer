from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV
try:
    # sklearn >= 1.6; replaces CalibratedClassifierCV(cv='prefit') which was removed in 1.8
    from sklearn.frozen import FrozenEstimator
except ImportError:  # older sklearn — caller falls back to cv='prefit' path below
    FrozenEstimator = None
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from nba_api.stats.endpoints import TeamGameLog, CommonPlayerInfo, LeagueGameFinder
from nba_api.stats.endpoints import playergamelog, LeagueDashPtDefend
import scipy.stats
import numpy as np
import pandas as pd
import joblib
import os
import time
from datetime import datetime
from .injury_tracker import InjuryTracker

class EnhancedMLPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.injury_tracker = InjuryTracker()
        self.models_trained = False

        # adaptive confidence thresholds (can be overridden by calibrate_confidence_thresholds)
        self.conf_high_threshold = 0.08
        self.conf_med_threshold = 0.04

        # load saved confidence thresholds if present
        _thresh_path = os.path.join(model_dir, 'conf_thresholds.joblib')
        if os.path.exists(_thresh_path):
            try:
                _thresholds = joblib.load(_thresh_path)
                self.conf_high_threshold = _thresholds.get('high', 0.08)
                self.conf_med_threshold = _thresholds.get('med', 0.04)
            except Exception:
                pass

        # model versioning — load from file or start at "0"
        _version_path = os.path.join(model_dir, 'model_version.txt')
        if os.path.exists(_version_path):
            try:
                with open(_version_path, 'r') as _vf:
                    self._model_version = _vf.read().strip()
            except Exception:
                self._model_version = "0"
        else:
            self._model_version = "0"

        # load saved models if they exist, otherwise start fresh
        try:
            self.classification_model = joblib.load(f'{model_dir}/classification_model.joblib')
            self.regression_model = joblib.load(f'{model_dir}/regression_model.joblib')
            self.scaler = joblib.load(f'{model_dir}/scaler.joblib')
            self.models_trained = True
            print("models loaded from disk")
        except Exception:
            self.classification_model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.regression_model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.scaler = StandardScaler()

        # per-prop models: prop_type -> {classification_model, regression_model, scaler, calibrated_clf}
        self.prop_models = {}
        self._load_prop_models()

        self.position_matchup_cache = {}   # evicted when > 200 entries
        self.team_context_cache = {}       # evicted when > 100 entries
        self._pt_defend_cache = None       # league-wide defensive data, fetched once per season
        self._pt_defend_season = None      # track which season it was fetched for
        self._CACHE_MAX_POSITION = 200
        self._CACHE_MAX_TEAM = 100

    @property
    def model_version(self):
        return self._model_version

    def _load_prop_models(self):
        """Load any previously saved per-prop models from disk."""
        if not os.path.isdir(self.model_dir):
            return
        for fname in os.listdir(self.model_dir):
            if fname.startswith('clf_cal_') and fname.endswith('.joblib'):
                prop_type = fname[len('clf_cal_'):-len('.joblib')]
                reg_path = os.path.join(self.model_dir, f'reg_{prop_type}.joblib')
                scaler_path = os.path.join(self.model_dir, f'scaler_{prop_type}.joblib')
                try:
                    # Use a saved scaler if available; fall back to identity (XGBoost doesn't need scaling)
                    if os.path.exists(scaler_path):
                        _scaler = joblib.load(scaler_path)
                    else:
                        _scaler = FunctionTransformer()  # identity passthrough
                    bundle = {
                        'calibrated_clf': joblib.load(os.path.join(self.model_dir, fname)),
                        'scaler': _scaler,
                    }
                    # Regression model is optional for binary props (double_double, triple_double)
                    if os.path.exists(reg_path):
                        bundle['regression_model'] = joblib.load(reg_path)
                    else:
                        bundle['regression_model'] = None
                    self.prop_models[prop_type] = bundle
                except Exception as e:
                    print(f"failed to load per-prop model for {prop_type}: {e}")


    def _get_injury_history(self, player_id):
        """Pull injury history from game logs for the past two seasons."""
        try:
            # figure out which seasons to pull based on current month
            current_year = datetime.now().year
            current_month = datetime.now().month

            if 1 <= current_month <= 7:
                seasons = [f"{current_year-1}-{str(current_year)[2:]}",
                          f"{current_year-2}-{str(current_year-1)[2:]}"]
            else:
                seasons = [f"{current_year}-{str(current_year+1)[2:]}",
                          f"{current_year-1}-{str(current_year)[2:]}"]

            all_games = []
            for season in seasons:
                try:
                    games = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season
                    ).get_data_frames()[0]
                    time.sleep(0.6)
                    all_games.append(games)
                except Exception as e:
                    print(f"couldn't fetch game log for season {season}: {e}")
                    continue

            if not all_games:
                return self._get_default_injury_history()

            games_df = pd.concat(all_games, ignore_index=True)
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            games_df = games_df.sort_values('GAME_DATE')

            games_df['DAYS_BETWEEN'] = games_df['GAME_DATE'].diff().dt.days

            # gaps over 7 days are probably injuries
            injury_gaps = games_df[games_df['DAYS_BETWEEN'] > 7]
            recent_injuries = []

            for _, gap in injury_gaps.iterrows():
                recent_injuries.append({
                    'date': gap['GAME_DATE'],
                    'days_missed': gap['DAYS_BETWEEN'],
                    'is_recent': (datetime.now() - gap['GAME_DATE'].to_pydatetime()).days < 60
                })

            total_gaps = len(injury_gaps)
            recent_gaps = sum(1 for inj in recent_injuries if inj['is_recent'])
            total_days_missed = injury_gaps['DAYS_BETWEEN'].sum()
            recent_days_missed = sum(inj['days_missed'] for inj in recent_injuries if inj['is_recent'])

            if recent_gaps > 0 or recent_days_missed > 30:
                injury_risk = 'high'
            elif total_gaps > 2:
                injury_risk = 'medium'
            else:
                injury_risk = 'low'

            return {
                'recent_injuries': recent_injuries,
                'games_missed': total_gaps,
                'total_days_missed': total_days_missed,
                'injury_risk': injury_risk
            }

        except Exception as e:
            print(f"error getting injury history: {e}")
            return self._get_default_injury_history()

    def _get_default_injury_history(self):
        """Fallback when injury data isn't available."""
        return {
            'recent_injuries': [],
            'games_missed': 0,
            'total_days_missed': 0,
            'injury_risk': 'low'
        }

    def _get_matchup_history(self, player_id, opponent_team_id):
        """Get this player's historical stats against a specific team.

        Concatenates regular-season AND playoff matchups:
          - Regular season: covers recent meetings (good sample size, fresh data).
          - Playoffs: small sample but very high signal during a series, where
            the same teams meet 4–7 times. Playoff history may be stale (teams
            might not have met in the playoffs for years), so regular-season
            data anchors the average; playoff rows just add to it.
        """
        try:
            frames = []
            for stype in ('Regular Season', 'Playoffs'):
                try:
                    df = LeagueGameFinder(
                        player_id_nullable=player_id,
                        vs_team_id_nullable=opponent_team_id,
                        season_type_nullable=stype,
                    ).get_data_frames()[0]
                    time.sleep(0.6)
                    if df is not None and len(df) > 0:
                        frames.append(df)
                except Exception as _inner:
                    # One season type failing shouldn't kill the other
                    print(f"matchup history ({stype}) fetch failed: {_inner}")
                    continue

            if not frames:
                return None

            gamefinder = pd.concat(frames, ignore_index=True)

            if len(gamefinder) == 0:
                return None

            return {
                'avg_points': float(gamefinder['PTS'].mean()),
                'avg_assists': float(gamefinder['AST'].mean()),
                'avg_rebounds': float(gamefinder['REB'].mean()),
                'games_played': len(gamefinder),
                'success_rate': float((gamefinder['PLUS_MINUS'] > 0).mean())
            }
        except Exception as e:
            print(f"error getting matchup history: {e}")
            return None

    def _get_pt_defend_data(self):
        """Fetches LeagueDashPtDefend for the current season (cached per session)."""
        current_season = self._get_current_season()
        if self._pt_defend_cache is not None and self._pt_defend_season == current_season:
            return self._pt_defend_cache

        try:
            df = LeagueDashPtDefend(
                league_id='00',
                per_mode_simple='PerGame',
                season=current_season,
                season_type_all_star='Regular Season',
                defense_category='Overall'
            ).get_data_frames()[0]
            time.sleep(0.6)
            self._pt_defend_cache = df
            self._pt_defend_season = current_season
            return df
        except Exception as e:
            print(f"failed to fetch LeagueDashPtDefend: {e}")
            return None

    def _get_current_season(self):
        current_year = datetime.now().year
        current_month = datetime.now().month
        if 1 <= current_month <= 7:
            return f"{current_year-1}-{str(current_year)[2:]}"
        return f"{current_year}-{str(current_year+1)[2:]}"

    def get_position_matchup_stats(self, position, team_id):
        """How well the opponent team defends against a specific position (from LeagueDashPtDefend)."""
        cache_key = f"{position}_{team_id}"
        if cache_key in self.position_matchup_cache:
            return self.position_matchup_cache[cache_key]

        # map whatever position string we get to nba_api codes
        position_group_map = {
            'G': ['G', 'G-F', 'F-G'],
            'Guard': ['G', 'G-F', 'F-G'],
            'SG': ['G', 'G-F', 'F-G'],
            'PG': ['G', 'G-F', 'F-G'],
            'G-F': ['G', 'G-F', 'F-G'],
            'F-G': ['G', 'G-F', 'F-G'],
            'F': ['F', 'F-G', 'G-F', 'F-C', 'C-F'],
            'Forward': ['F', 'F-G', 'G-F', 'F-C', 'C-F'],
            'SF': ['F', 'F-G', 'G-F', 'F-C', 'C-F'],
            'PF': ['F', 'F-G', 'G-F', 'F-C', 'C-F'],
            'F-C': ['F', 'F-C', 'C-F'],
            'C-F': ['C', 'C-F', 'F-C'],
            'C': ['C', 'C-F', 'F-C'],
            'Center': ['C', 'C-F', 'F-C'],
            'Forward-Center': ['F', 'F-C', 'C-F'],
            'Center-Forward': ['C', 'C-F', 'F-C'],
            'Forward- Center': ['F', 'F-C', 'C-F'],
        }
        valid_positions = position_group_map.get(position, [position])

        try:
            defend_df = self._get_pt_defend_data()
            if defend_df is None or defend_df.empty:
                return self._get_default_position_matchup()

            # only care about this team's defenders at the right position
            team_defenders = defend_df[
                (defend_df['PLAYER_LAST_TEAM_ID'] == int(team_id)) &
                (defend_df['PLAYER_POSITION'].isin(valid_positions))
            ]

            if team_defenders.empty:
                return self._get_default_position_matchup()

            # weight by games played so bench guys don't skew it
            weights = team_defenders['G'].values
            if weights.sum() == 0:
                return self._get_default_position_matchup()

            avg_fg_pct   = float(np.average(team_defenders['D_FG_PCT'],     weights=weights))
            avg_pct_pm   = float(np.average(team_defenders['PCT_PLUSMINUS'], weights=weights))
            # pts_allowed: weighted avg FGM * 2 (field goals, not perfect but consistent proxy)
            pts_allowed  = float(np.average(team_defenders['D_FGM'] * 2,    weights=weights))
            # defensive_rating: 100 baseline shifted by normalised plus/minus (PCT_PLUSMINUS is a fraction)
            def_rating   = 100.0 + avg_pct_pm * 100.0

            matchup_stats = {
                'pts_allowed_per_game': pts_allowed if not np.isnan(pts_allowed) else 15.0,
                'defensive_rating':     def_rating  if not np.isnan(def_rating)  else 110.0,
                'effective_fg_pct':     avg_fg_pct  if not np.isnan(avg_fg_pct)  else 0.47,
            }

            if len(self.position_matchup_cache) >= self._CACHE_MAX_POSITION:
                self.position_matchup_cache.clear()
            self.position_matchup_cache[cache_key] = matchup_stats
            return matchup_stats

        except Exception as e:
            print(f"error in get_position_matchup_stats: {e}")
            return self._get_default_position_matchup()

    def _get_default_position_matchup(self):
        """Fallback matchup stats when we have nothing better."""
        return {
            'pts_allowed_per_game': 15.0,
            'defensive_rating': 110.0,
            'effective_fg_pct': 0.47
        }

    def get_player_context(self, player_id, opponent_team_id):
        """Pull together position, injury history, and matchup info for a player."""
        try:
            player_info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            time.sleep(0.6)
            position = player_info['POSITION'].iloc[0]

            injury_history = self._get_injury_history(player_id)

            matchup_history = self._get_matchup_history(player_id, opponent_team_id)

            position_matchup = self.get_position_matchup_stats(position, opponent_team_id)

            team_id = int(player_info['TEAM_ID'].iloc[0])

            return {
                'position': position,
                'team_id': team_id,
                'injury_history': injury_history,
                'matchup_history': matchup_history,
                'position_matchup': position_matchup
            }
        except Exception as e:
            print(f"error getting player context: {e}")
            return None

    def _calculate_team_form(self, games_df):
        """Win rate and scoring trend from recent games."""
        try:
            wins = float((games_df['WL'] == 'W').mean())
            avg_points = float(games_df['PTS'].mean())

            return {
                'win_pct': wins,
                'avg_points': avg_points,
                'trend': 'up' if wins > 0.5 else 'down' if wins < 0.5 else 'neutral'
            }
        except Exception as e:
            print(f"error calculating team form: {e}")
            return {
                'win_pct': 0.5,
                'avg_points': 100.0,
                'trend': 'neutral'
            }

    def _calculate_defensive_rating(self, games_df):
        """Estimates def rating from box score — proxied via pts_allowed / possessions * 100."""
        try:
            oreb_dreb_sum = games_df['OREB'].mean() + games_df['DREB'].mean()
            oreb_factor = (games_df['OREB'].mean() / oreb_dreb_sum) if oreb_dreb_sum > 0 else 0.33
            possessions = (
                games_df['FGA'].mean() +
                0.4 * games_df['FTA'].mean() -
                1.07 * oreb_factor * (games_df['FGA'].mean() - games_df['FGM'].mean()) +
                games_df['TOV'].mean()
            )
            if possessions <= 0:
                return 110.0

            # team's own points per 100 possessions
            off_rating = (float(games_df['PTS'].mean()) / possessions) * 100

            # PLUS_MINUS = pts_scored - pts_allowed per game
            # pts_allowed ≈ pts_scored - plus_minus
            # def_rating ≈ (pts_allowed / possessions) * 100
            avg_plus_minus = float(games_df['PLUS_MINUS'].mean()) if 'PLUS_MINUS' in games_df.columns else 0.0
            pts_allowed_est = float(games_df['PTS'].mean()) - avg_plus_minus
            def_rating = (pts_allowed_est / possessions) * 100

            # clamp to realistic NBA range
            return float(max(90.0, min(130.0, def_rating)))

        except Exception as e:
            return 110.0

    def get_team_context(self, team_id):
        """Recent form, pace, defense, and injury situation for a team."""
        if team_id in self.team_context_cache:
            return self.team_context_cache[team_id]

        try:
            # pull recent games
            team_games = TeamGameLog(
                team_id=team_id,
                season_type_all_star='Regular Season'
            ).get_data_frames()[0]
            time.sleep(0.6)

            if len(team_games) == 0:
                return self._get_default_context()

            # grab injury info
            injury_info = self.injury_tracker.get_team_injuries(team_id)

            recent_games = team_games.head(10)
            possessions_per_game = self._calculate_estimated_pace(recent_games)
            pts_per_game = float(recent_games['PTS'].mean())

            defensive_rating = self._calculate_defensive_rating(recent_games)

            injury_impact = self._calculate_injury_impact(team_id)
            adjusted_pace = possessions_per_game * (1 - injury_impact * 0.1)
            adjusted_pts = pts_per_game * (1 - injury_impact * 0.15)

            context = {
                'pace': float(adjusted_pace),
                'offensive_rating': float(adjusted_pts / (adjusted_pace / 100)),
                'defensive_rating': float(defensive_rating),
                'recent_form': self._calculate_team_form(recent_games),
                'rest_days': self._calculate_rest_days(recent_games),
                'injury_impact': injury_impact,
                'injuries': {
                    'total_players_out': injury_info['total_players_out'],
                    'key_players_out': injury_info['key_players_out'],
                    'active_injuries': injury_info['active_injuries']
                }
            }

            if len(self.team_context_cache) >= self._CACHE_MAX_TEAM:
                self.team_context_cache.clear()
            self.team_context_cache[team_id] = context
            return context

        except Exception as e:
            print(f"error getting team context: {e}")
            return self._get_default_context()

    def _calculate_estimated_pace(self, games_df):
        """Oliver possession formula to estimate pace from box score stats."""
        try:
            fga = float(games_df['FGA'].mean()) if 'FGA' in games_df.columns else 85.0
            fta = float(games_df['FTA'].mean()) if 'FTA' in games_df.columns else 22.0
            oreb = float(games_df['OREB'].mean()) if 'OREB' in games_df.columns else 10.0
            tov = float(games_df['TOV'].mean()) if 'TOV' in games_df.columns else 14.0

            # Oliver: FGA - OREB + TOV + 0.44*FTA
            estimated_pace = fga - oreb + tov + 0.44 * fta

            return float(max(estimated_pace, 90.0))
        except Exception as e:
            print(f"error calculating pace: {e}")
            return 100.0

    def _get_default_context(self):
        """Neutral defaults when we can't get real team data."""
        return {
            'pace': 100.0,
            'offensive_rating': 110.0,
            'defensive_rating': 110.0,
            'recent_form': {
                'win_pct': 0.5,
                'avg_points': 100.0,
                'trend': 'neutral'
            },
            'rest_days': 2,
            'injury_impact': 0.1,
            'injuries': {
                'total_players_out': 0,
                'key_players_out': 0,
                'active_injuries': []
            }
        }

    def _calculate_rest_days(self, games_df):
        """Days since last game."""
        try:
            if len(games_df) < 2:
                return 1

            last_game = pd.to_datetime(games_df['GAME_DATE'].iloc[0])
            today = pd.Timestamp.now()

            return int((today - last_game).days)
        except Exception as e:
            print(f"error calculating rest days: {e}")
            return 2

    def _calculate_injury_impact(self, team_id):
        """Weighted injury impact score based on who's out and how important they are."""
        try:
            injury_data = self.injury_tracker.get_team_injuries(team_id)

            if not injury_data:
                return 0.0

            total_impact = float(injury_data.get('total_impact', 0))
            key_players_out = int(injury_data.get('key_players_out', 0))
            total_players_out = int(injury_data.get('total_players_out', 0))

            weighted_impact = (
                0.6 * min(total_impact, 1.0) +
                0.3 * min(key_players_out / 3, 1.0) +
                0.1 * min(total_players_out / 5, 1.0)
            )

            return min(max(weighted_impact, 0.0), 1.0)

        except Exception as e:
            print(f"error calculating injury impact: {e}")
            return 0.1

    def prepare_features(self, player_stats, player_context, team_context, opponent_context):
        """Build the feature dict for the ML models from all available context.

        player_stats may contain extra efficiency/location keys populated by
        analyze_prop_bet() before calling this method:
          home_avg, away_avg, home_games, away_games,
          avg_minutes, recent_minutes, fg_pct, recent_fg_pct, ft_pct,
          usage_rate, trend_slope, b2b_flag
        All are optional — missing ones fall back to neutral defaults.
        """
        features = {}

        # ---- core rolling stats ----
        # values list is newest-first from _get_stat_dict; reverse for chronological order
        _all_vals = list(reversed(player_stats.get('values') or [0]))
        _last5 = _all_vals[-5:] if len(_all_vals) >= 5 else _all_vals
        features.update({
            'recent_avg':   float(player_stats.get('last5_avg', 0)),
            'season_avg':   float(player_stats.get('avg', 0)),
            # stddev/max/min over full history to match training (train uses full hist, not just last5)
            'max_recent':   float(max(_all_vals)),
            'min_recent':   float(min(_all_vals)),
            'stddev':       float(np.std(_all_vals)) if len(_all_vals) > 1 else 0.0,
            'games_played': len(_all_vals),
        })

        # ---- location splits (already in stat_data from _get_stat_dict) ----
        features.update({
            'home_avg':   float(player_stats.get('home_avg',   features['recent_avg'])),
            'away_avg':   float(player_stats.get('away_avg',   features['recent_avg'])),
            'home_games': int(player_stats.get('home_games',   0)),
            'away_games': int(player_stats.get('away_games',   0)),
        })

        # ---- efficiency / usage (injected by analyze_prop_bet) ----
        features.update({
            'avg_minutes':   float(player_stats.get('avg_minutes',   24.0)),
            'recent_minutes': float(player_stats.get('recent_minutes', 24.0)),
            'fg_pct':        float(player_stats.get('fg_pct',         0.45)),
            'recent_fg_pct': float(player_stats.get('recent_fg_pct',  0.45)),
            'ft_pct':        float(player_stats.get('ft_pct',         0.75)),
            'usage_rate':    float(player_stats.get('usage_rate',     18.0)),
            'trend_slope':   float(player_stats.get('trend_slope',     0.0)),
            'b2b_flag':      int(player_stats.get('b2b_flag',          0)),
        })

        # ---- extended game-log derived features ----
        # _all_vals is already chronological (oldest-first) from the fix above
        _vals = _all_vals  # chronological order, consistent with training
        _last5_ext  = _vals[-5:]  if len(_vals) >= 5  else _vals
        _last10_ext = _vals[-10:] if len(_vals) >= 10 else _vals
        _last3_ext  = _vals[-3:]  if len(_vals) >= 3  else _vals
        _seas_avg_ext = float(np.mean(_vals)) if _vals else 0.0
        _seas_std_ext = float(np.std(_vals)) if len(_vals) > 1 else 1.0

        def _slope_ext(arr):
            if len(arr) < 2:
                return 0.0
            try:
                return float(np.polyfit(range(len(arr)), arr, 1)[0])
            except Exception:
                return 0.0

        features.update({
            'fg3_pct_recent':    float(player_stats.get('fg3_pct_recent', 0.33)),
            'fga_per_game':      float(player_stats.get('fga_per_game', 15.0)),
            'fg3a_per_game':     float(player_stats.get('fg3a_per_game', 5.0)),
            'fta_per_game':      float(player_stats.get('fta_per_game', 4.0)),
            'oreb_per_game':     float(player_stats.get('oreb_per_game', 1.0)),
            'dreb_per_game':     float(player_stats.get('dreb_per_game', 3.0)),
            'plus_minus_avg':    float(player_stats.get('plus_minus_avg', 0.0)),
            'fouls_per_game':    float(player_stats.get('fouls_per_game', 2.0)),
            'win_rate_last10':   float(player_stats.get('win_rate_last10', 0.5)),
            'points_per_shot':   float(player_stats.get('points_per_shot',
                                      float(features.get('recent_avg', 0.0)) / max(float(player_stats.get('fga_per_game', 15.0)), 1.0))),
            'ast_to_tov_ratio':  float(player_stats.get('ast_to_tov_ratio', 1.5)),
            'reb_rate_per_36':   float(player_stats.get('reb_rate_per_36', 0.0)),
            'scoring_efficiency_trend': float(player_stats.get('scoring_efficiency_trend', 0.0)),
            'usage_trend':       float(player_stats.get('usage_trend', 0.0)),
            'minutes_volatility': float(player_stats.get('minutes_volatility', 3.0)),
            'blowout_game_pct':  float(player_stats.get('blowout_game_pct', 0.2)),
            'close_game_pct':    float(player_stats.get('close_game_pct', 0.3)),
            'consistency_score': float(player_stats.get('consistency_score',
                                      max(0.0, 1.0 - (_seas_std_ext / max(_seas_avg_ext, 0.1))))),
            'ceiling_game_frequency': float(player_stats.get('ceiling_game_frequency', 0.1)),
            'recent_variance_spike':  float(player_stats.get('recent_variance_spike', 0.0)),
            # Trend = mean(last-N) - season_avg, matching training definition in train_models.py
            'last_3_games_trend':  float(player_stats.get('last_3_games_trend',
                                         float(np.mean(_last3_ext)) - _seas_avg_ext if _last3_ext else 0.0)),
            'last_5_games_trend':  float(player_stats.get('last_5_games_trend',
                                         float(np.mean(_last5_ext)) - _seas_avg_ext if _last5_ext else 0.0)),
            'last_10_games_trend': float(player_stats.get('last_10_games_trend',
                                         float(np.mean(_last10_ext)) - _seas_avg_ext if _last10_ext else 0.0)),
            'games_above_season_avg_last5': float(player_stats.get('games_above_season_avg_last5',
                                                  float(sum(1 for v in _last5_ext if v > _seas_avg_ext)))),
            'days_since_last_game':  float(player_stats.get('days_since_last_game', 2.0)),
            'games_in_last_7_days':  float(player_stats.get('games_in_last_7_days', 3.0)),
        })

        # ---- team style + opponent baseline (from PrecomputedStore team_stats) ----
        features.update({
            'team_pts_fb':              float(player_stats.get('team_pts_fb', 12.0)),
            'opp_pts_fb_allowed':       float(player_stats.get('opp_pts_fb_allowed', 12.0)),
            'team_pts_off_tov':         float(player_stats.get('team_pts_off_tov', 16.0)),
            'opp_pts_off_tov_allowed':  float(player_stats.get('opp_pts_off_tov_allowed', 16.0)),
            'opp_pts_paint':            float(player_stats.get('opp_pts_paint', 44.0)),
            'opp_fga':                  float(player_stats.get('opp_fga', 86.0)),
            'opp_fg_pct':               float(player_stats.get('opp_fg_pct', 0.47)),
            'opp_fg3a':                 float(player_stats.get('opp_fg3a', 35.0)),
            'opp_fg3_pct':              float(player_stats.get('opp_fg3_pct', 0.36)),
            'opp_tov':                  float(player_stats.get('opp_tov', 14.0)),
            'opp_stl':                  float(player_stats.get('opp_stl', 7.0)),
            'opp_blk':                  float(player_stats.get('opp_blk', 5.0)),
            'opp_off_rating':           float(player_stats.get('opp_off_rating', 110.0)),
            'opp_def_rating_last5':     float(player_stats.get('opp_def_rating_last5', 110.0)),
            'opp_blocks_per_game_last5': float(player_stats.get('opp_blocks_per_game_last5', 5.0)),
            'opp_steals_per_game_last5': float(player_stats.get('opp_steals_per_game_last5', 7.0)),
            'lg_pts_fb':                float(player_stats.get('lg_pts_fb', 12.0)),
            'lg_opp_pts_fb':            float(player_stats.get('lg_opp_pts_fb', 12.0)),
            'lg_pts_off_tov':           float(player_stats.get('lg_pts_off_tov', 16.0)),
            'lg_opp_pts_off_tov':       float(player_stats.get('lg_opp_pts_off_tov', 16.0)),
            'lg_fga':                   float(player_stats.get('lg_fga', 86.0)),
            'lg_fg_pct':                float(player_stats.get('lg_fg_pct', 0.47)),
            'lg_fg3a':                  float(player_stats.get('lg_fg3a', 35.0)),
            'lg_tov':                   float(player_stats.get('lg_tov', 14.0)),
            'lg_stl':                   float(player_stats.get('lg_stl', 7.0)),
        })

        # ---- DVP (Defence vs Position) deltas + primary defender ----
        # Injected by analyze_prop_bet() from PrecomputedStore.
        # Zero defaults = league-average defence, consistent with training.
        features.update({
            'dvp_gp':                    int(player_stats.get('dvp_gp', 0)),
            'dvp_pts_delta':             float(player_stats.get('dvp_pts_delta',  0.0)),
            'dvp_reb_delta':             float(player_stats.get('dvp_reb_delta',  0.0)),
            'dvp_ast_delta':             float(player_stats.get('dvp_ast_delta',  0.0)),
            'dvp_fg3m_delta':            float(player_stats.get('dvp_fg3m_delta', 0.0)),
            'dvp_stl_delta':             float(player_stats.get('dvp_stl_delta',  0.0)),
            'dvp_blk_delta':             float(player_stats.get('dvp_blk_delta',  0.0)),
            'dvp_tov_delta':             float(player_stats.get('dvp_tov_delta',  0.0)),
            'primary_defender_score01':  float(player_stats.get('primary_defender_score01', 0.0)),
        })

        # ---- Referee tendencies ----
        features.update({
            'ref_foul_rate':     float(player_stats.get('ref_foul_rate', 0.0)),
            'ref_home_bias':     float(player_stats.get('ref_home_bias', 0.5)),
            'ref_pace_tendency': float(player_stats.get('ref_pace_tendency', 100.0)),
        })
        # ---- Rolling DVP deltas ----
        features.update({
            'dvp_pts_delta_last5':   float(player_stats.get('dvp_pts_delta_last5', 0.0)),
            'dvp_pts_delta_last10':  float(player_stats.get('dvp_pts_delta_last10', 0.0)),
            'dvp_reb_delta_last5':   float(player_stats.get('dvp_reb_delta_last5', 0.0)),
            'dvp_ast_delta_last5':   float(player_stats.get('dvp_ast_delta_last5', 0.0)),
            'dvp_fg3m_delta_last5':  float(player_stats.get('dvp_fg3m_delta_last5', 0.0)),
        })
        # ---- Opponent foul rates ----
        features.update({
            'opp_foul_rate_per48': float(player_stats.get('opp_foul_rate_per48', 20.0)),
            'opp_foul_rate_last5': float(player_stats.get('opp_foul_rate_last5', 20.0)),
        })
        # ---- Injury trajectory ----
        features.update({
            'games_since_return':         float(player_stats.get('games_since_return', 0.0)),
            'missed_games_before_return': float(player_stats.get('missed_games_before_return', 0.0)),
        })
        # ---- Calendar position ----
        features.update({
            'days_into_season':       float(player_stats.get('days_into_season', 90.0)),
            'season_phase_numeric':   float(player_stats.get('season_phase_numeric', 0.5)),
            'games_remaining_approx': float(player_stats.get('games_remaining_approx', 40.0)),
        })
        # ---- Defender health and lineup ----
        features.update({
            'primary_defender_active':  float(player_stats.get('primary_defender_active', 1.0)),
            'opp_lineup_changes_last5': float(player_stats.get('opp_lineup_changes_last5', 0.0)),
        })
        # ---- Implied game total ----
        features.update({
            'implied_game_total': float(player_stats.get('implied_game_total', 220.0)),
        })

        # ---- advanced player stats (official NBA advanced measure) ----
        features.update({
            'usg_pct_official':   float(player_stats.get('usg_pct_official', 0.18)),
            'ts_pct_official':    float(player_stats.get('ts_pct_official', 0.55)),
            'efg_pct_official':   float(player_stats.get('efg_pct_official', 0.50)),
            'ast_pct_official':   float(player_stats.get('ast_pct_official', 0.15)),
            'oreb_pct_official':  float(player_stats.get('oreb_pct_official', 0.05)),
            'dreb_pct_official':  float(player_stats.get('dreb_pct_official', 0.15)),
            'reb_pct_official':   float(player_stats.get('reb_pct_official', 0.10)),
            'pie':                float(player_stats.get('pie', 0.10)),
            'player_off_rating':  float(player_stats.get('player_off_rating', 110.0)),
            'player_def_rating':  float(player_stats.get('player_def_rating', 110.0)),
            'player_pace':        float(player_stats.get('player_pace', 100.0)),
            'net_rating_player':  float(player_stats.get('net_rating_player', 0.0)),
        })
        # ---- bio ----
        features.update({
            'player_age':           float(player_stats.get('player_age', 26.0)),
            'player_height_inches': float(player_stats.get('player_height_inches', 78.0)),
            'player_weight':        float(player_stats.get('player_weight', 220.0)),
            'years_experience':     float(player_stats.get('years_experience', 5.0)),
        })
        # ---- clutch ----
        features.update({
            'clutch_pts_per_game':  float(player_stats.get('clutch_pts_per_game', 0.0)),
            'clutch_fg_pct':        float(player_stats.get('clutch_fg_pct', 0.45)),
            'clutch_fg3_pct':       float(player_stats.get('clutch_fg3_pct', 0.33)),
            'clutch_fta_per_game':  float(player_stats.get('clutch_fta_per_game', 0.0)),
            'clutch_plus_minus':    float(player_stats.get('clutch_plus_minus', 0.0)),
            'clutch_min_per_game':  float(player_stats.get('clutch_min_per_game', 0.0)),
            'clutch_games':         float(player_stats.get('clutch_games', 0)),
        })
        # ---- hustle ----
        features.update({
            'contested_shots_per_game': float(player_stats.get('contested_shots_per_game', 3.0)),
            'deflections_per_game':     float(player_stats.get('deflections_per_game', 1.0)),
            'charges_drawn_per_game':   float(player_stats.get('charges_drawn_per_game', 0.1)),
            'screen_assists_per_game':  float(player_stats.get('screen_assists_per_game', 0.5)),
        })
        # ---- shot profile ----
        features.update({
            'open_shot_fg_pct':    float(player_stats.get('open_shot_fg_pct', 0.50)),
            'open_shot_frequency': float(player_stats.get('open_shot_frequency', 0.30)),
            'tight_shot_fg_pct':   float(player_stats.get('tight_shot_fg_pct', 0.38)),
            'tight_shot_frequency': float(player_stats.get('tight_shot_frequency', 0.15)),
            'catch_shoot_fg_pct':  float(player_stats.get('catch_shoot_fg_pct', 0.40)),
            'catch_shoot_frequency': float(player_stats.get('catch_shoot_frequency', 0.25)),
            'pullup_fg_pct':       float(player_stats.get('pullup_fg_pct', 0.40)),
            'pullup_frequency':    float(player_stats.get('pullup_frequency', 0.20)),
        })
        # ---- synergy play types ----
        features.update({
            'iso_poss_pct':      float(player_stats.get('iso_poss_pct', 0.0)),
            'iso_ppp':           float(player_stats.get('iso_ppp', 0.9)),
            'pnr_bh_poss_pct':   float(player_stats.get('pnr_bh_poss_pct', 0.0)),
            'pnr_bh_ppp':        float(player_stats.get('pnr_bh_ppp', 0.9)),
            'pnr_roll_poss_pct': float(player_stats.get('pnr_roll_poss_pct', 0.0)),
            'pnr_roll_ppp':      float(player_stats.get('pnr_roll_ppp', 0.9)),
            'spotup_poss_pct':   float(player_stats.get('spotup_poss_pct', 0.0)),
            'spotup_ppp':        float(player_stats.get('spotup_ppp', 1.0)),
            'transition_poss_pct': float(player_stats.get('transition_poss_pct', 0.0)),
            'transition_ppp':    float(player_stats.get('transition_ppp', 1.1)),
            'postup_poss_pct':   float(player_stats.get('postup_poss_pct', 0.0)),
            'cut_poss_pct':      float(player_stats.get('cut_poss_pct', 0.0)),
        })
        # ---- on/off ----
        features.update({
            'on_court_net_rating':  float(player_stats.get('on_court_net_rating', 0.0)),
            'off_court_net_rating': float(player_stats.get('off_court_net_rating', 0.0)),
            'on_off_differential':  float(player_stats.get('on_off_differential', 0.0)),
        })
        # ---- shot zones ----
        features.update({
            'rim_fga_pct':         float(player_stats.get('rim_fga_pct', 0.25)),
            'rim_fg_pct':          float(player_stats.get('rim_fg_pct', 0.62)),
            'paint_fga_pct':       float(player_stats.get('paint_fga_pct', 0.30)),
            'paint_fg_pct':        float(player_stats.get('paint_fg_pct', 0.55)),
            'midrange_fga_pct':    float(player_stats.get('midrange_fga_pct', 0.20)),
            'midrange_fg_pct':     float(player_stats.get('midrange_fg_pct', 0.42)),
            'corner3_fga_pct':     float(player_stats.get('corner3_fga_pct', 0.10)),
            'corner3_fg_pct':      float(player_stats.get('corner3_fg_pct', 0.38)),
            'above_break3_fga_pct': float(player_stats.get('above_break3_fga_pct', 0.25)),
            'above_break3_fg_pct': float(player_stats.get('above_break3_fg_pct', 0.35)),
        })
        # ---- quarter splits ----
        features.update({
            'q1_avg':       float(player_stats.get('q1_avg', 0.0)),
            'q2_avg':       float(player_stats.get('q2_avg', 0.0)),
            'q3_avg':       float(player_stats.get('q3_avg', 0.0)),
            'q4_avg':       float(player_stats.get('q4_avg', 0.0)),
            'q4_min_per_game': float(player_stats.get('q4_min_per_game', 0.0)),
        })
        # ---- opponent shot zone defense ----
        features.update({
            'opp_rim_fg_pct_allowed':         float(player_stats.get('opp_rim_fg_pct_allowed', 0.62)),
            'opp_paint_fg_pct_allowed':       float(player_stats.get('opp_paint_fg_pct_allowed', 0.55)),
            'opp_midrange_fg_pct_allowed':    float(player_stats.get('opp_midrange_fg_pct_allowed', 0.42)),
            'opp_corner3_fg_pct_allowed':     float(player_stats.get('opp_corner3_fg_pct_allowed', 0.38)),
            'opp_above_break3_fg_pct_allowed': float(player_stats.get('opp_above_break3_fg_pct_allowed', 0.35)),
        })
        # ---- shot quality matchup (player zone FG% minus opponent zone FG% allowed) ----
        features.update({
            'rim_shot_quality_matchup':         features['rim_fg_pct'] - features['opp_rim_fg_pct_allowed'],
            'midrange_shot_quality_matchup':    features['midrange_fg_pct'] - features['opp_midrange_fg_pct_allowed'],
            'corner3_shot_quality_matchup':     features['corner3_fg_pct'] - features['opp_corner3_fg_pct_allowed'],
            'above_break3_shot_quality_matchup': features['above_break3_fg_pct'] - features['opp_above_break3_fg_pct_allowed'],
        })
        # ---- synergy team defense ----
        features.update({
            'opp_pnr_ppp_allowed':        float(player_stats.get('opp_pnr_ppp_allowed', 0.9)),
            'opp_iso_ppp_allowed':        float(player_stats.get('opp_iso_ppp_allowed', 0.9)),
            'opp_spotup_ppp_allowed':     float(player_stats.get('opp_spotup_ppp_allowed', 1.0)),
            'opp_transition_ppp_allowed': float(player_stats.get('opp_transition_ppp_allowed', 1.1)),
            'opp_postup_ppp_allowed':     float(player_stats.get('opp_postup_ppp_allowed', 0.9)),
        })
        # ---- synergy matchup advantage (player PPP - opponent PPP allowed, positive = player has edge) ----
        features.update({
            'pnr_matchup_advantage':        features['pnr_bh_ppp'] - features['opp_pnr_ppp_allowed'],
            'iso_matchup_advantage':        features['iso_ppp'] - features['opp_iso_ppp_allowed'],
            'spotup_matchup_advantage':     features['spotup_ppp'] - features['opp_spotup_ppp_allowed'],
            'transition_matchup_advantage': features['transition_ppp'] - features['opp_transition_ppp_allowed'],
        })

        # ---- Group 1: Player Tracking Stats ----
        features.update({
            'tracking_avg_speed':          float(player_stats.get('tracking_avg_speed', 4.5)),
            'tracking_avg_speed_off':      float(player_stats.get('tracking_avg_speed_off', 4.8)),
            'tracking_avg_speed_def':      float(player_stats.get('tracking_avg_speed_def', 4.2)),
            'tracking_dist_miles':         float(player_stats.get('tracking_dist_miles', 2.5)),
            'tracking_dist_miles_off':     float(player_stats.get('tracking_dist_miles_off', 1.3)),
            'tracking_dist_miles_def':     float(player_stats.get('tracking_dist_miles_def', 1.2)),
            'tracking_touches_pg':         float(player_stats.get('tracking_touches_pg', 50.0)),
            'tracking_time_of_poss_pg':    float(player_stats.get('tracking_time_of_poss_pg', 2.5)),
            'tracking_avg_drib_per_touch': float(player_stats.get('tracking_avg_drib_per_touch', 1.5)),
            'tracking_passes_made_pg':     float(player_stats.get('tracking_passes_made_pg', 30.0)),
            'tracking_potential_ast_pg':   float(player_stats.get('tracking_potential_ast_pg', 5.0)),
            'tracking_secondary_ast_pg':   float(player_stats.get('tracking_secondary_ast_pg', 1.0)),
        })
        # ---- Group 2: Team Standings / Game Importance ----
        features.update({
            'team_win_pct':                    float(player_stats.get('team_win_pct', 0.5)),
            'team_conf_rank':                  float(player_stats.get('team_conf_rank', 8.0)),
            'team_games_back':                 float(player_stats.get('team_games_back', 5.0)),
            'team_current_streak':             float(player_stats.get('team_current_streak', 0.0)),
            'team_l10_wins':                   float(player_stats.get('team_l10_wins', 5.0)),
            'team_home_win_pct':               float(player_stats.get('team_home_win_pct', 0.5)),
            'opp_win_pct':                     float(player_stats.get('opp_win_pct', 0.5)),
            'opp_conf_rank':                   float(player_stats.get('opp_conf_rank', 8.0)),
            'opp_games_back':                  float(player_stats.get('opp_games_back', 5.0)),
            'opp_current_streak':              float(player_stats.get('opp_current_streak', 0.0)),
            'opp_l10_wins':                    float(player_stats.get('opp_l10_wins', 5.0)),
            'opp_road_win_pct':                float(player_stats.get('opp_road_win_pct', 0.5)),
            'win_pct_diff':                    float(player_stats.get('win_pct_diff', 0.0)),
            'is_playoff_race_game':            float(player_stats.get('is_playoff_race_game', 0.0)),
        })
        # ---- Group 3: Scoring Breakdown by Method ----
        features.update({
            'pct_pts_3pt':      float(player_stats.get('pct_pts_3pt', 0.25)),
            'pct_pts_paint':    float(player_stats.get('pct_pts_paint', 0.30)),
            'pct_pts_ft':       float(player_stats.get('pct_pts_ft', 0.15)),
            'pct_pts_midrange': float(player_stats.get('pct_pts_midrange', 0.20)),
            'pct_uast_fgm':     float(player_stats.get('pct_uast_fgm', 0.40)),
            'pct_ast_fgm':      float(player_stats.get('pct_ast_fgm', 0.60)),
        })
        # ---- Group 4: Win/Loss Performance Splits ----
        features.update({
            'stat_in_wins':               float(player_stats.get('stat_in_wins', 0.0)),
            'stat_in_losses':             float(player_stats.get('stat_in_losses', 0.0)),
            'win_loss_performance_split': float(player_stats.get('win_loss_performance_split', 0.0)),
            'over_rate_in_wins':          float(player_stats.get('over_rate_in_wins', 0.5)),
        })
        # ---- Group 5: Opponent Rest & Schedule Context ----
        features.update({
            'opp_days_rest':   float(player_stats.get('opp_days_rest', 2.0)),
            'opp_b2b':         float(player_stats.get('opp_b2b', 0.0)),
            'rest_advantage':  float(player_stats.get('rest_advantage', 0.0)),
        })
        # ---- Group 6: Additional Derived Features ----
        features.update({
            'ast_pct_to_usg_ratio':      float(player_stats.get('ast_pct_to_usg_ratio', 0.83)),
            'defensive_burden':          float(player_stats.get('defensive_burden', 58.0)),
            'shot_profile_fit':          float(player_stats.get('shot_profile_fit', 0.09)),
            'pace_adjusted_projection':  float(player_stats.get('pace_adjusted_projection', 0.0)),
            'form_momentum':             float(player_stats.get('form_momentum', 0.0)),
        })
        # ---- Group 7: Player vs Opponent Historical Splits ----
        features.update({
            'vs_opp_avg_pts': float(player_stats.get('vs_opp_avg_pts', 0.0)),
            'vs_opp_fg_pct':  float(player_stats.get('vs_opp_fg_pct', 0.45)),
            'vs_opp_ts_pct':  float(player_stats.get('vs_opp_ts_pct', 0.55)),
            'vs_opp_gp':      float(player_stats.get('vs_opp_gp', 0.0)),
            'vs_opp_avg_min': float(player_stats.get('vs_opp_avg_min', 30.0)),
        })
        # ---- Group 8: Opponent trend features ----
        features.update({
            'opp_def_rating_trend':          float(player_stats.get('opp_def_rating_trend', 0.0)),
            'opp_def_rating_home_away_split': float(player_stats.get('opp_def_rating_home_away_split', 0.0)),
            'pts_vs_top10_defenses':         float(player_stats.get('pts_vs_top10_defenses', float(player_stats.get('season_avg', 0.0)))),
            'pts_vs_bottom10_defenses':      float(player_stats.get('pts_vs_bottom10_defenses', float(player_stats.get('season_avg', 0.0)))),
        })

        # ---- Group A: Derived Efficiency Features (computed from existing player_stats keys) ----
        features.update({
            'touch_efficiency': min(float(player_stats.get('tracking_passes_made_pg', 0.0)) / max(float(player_stats.get('tracking_touches_pg', 1.0)), 1.0), 1.0),
            'pace_adjusted_variance': float(player_stats.get('stddev', 2.0)) * (100.0 / max(float(player_stats.get('team_pace', 100.0)), 80.0)),
            'clutch_efficiency_delta': float(player_stats.get('clutch_pts_per_game', 0.0)) / max(float(player_stats.get('season_avg', 1.0)), 0.1) - 1.0,
            'rim_volume_quality': float(player_stats.get('rim_fga_pct', 0.25)) * max(float(player_stats.get('rim_shot_quality_matchup', 0.0)) + 1.0, 0.1),
            'play_specialization_score': max(float(player_stats.get('iso_poss_pct', 0.0)), float(player_stats.get('pnr_bh_poss_pct', 0.0)), float(player_stats.get('spotup_poss_pct', 0.0)), float(player_stats.get('transition_poss_pct', 0.0)), float(player_stats.get('postup_poss_pct', 0.0))),
            'best_play_type_ppp': max(float(player_stats.get('iso_ppp', 0.9)), float(player_stats.get('pnr_bh_ppp', 0.9)), float(player_stats.get('spotup_ppp', 0.9)), float(player_stats.get('transition_ppp', 0.9))),
            'ts_vs_zone_expected': float(player_stats.get('ts_pct_official', 0.55)) - (
                float(player_stats.get('rim_fga_pct', 0.25)) * 0.67 +
                float(player_stats.get('paint_fga_pct', 0.30)) * 0.55 +
                float(player_stats.get('midrange_fga_pct', 0.20)) * 0.42 +
                float(player_stats.get('corner3_fga_pct', 0.10)) * 0.38 +
                float(player_stats.get('above_break3_fga_pct', 0.25)) * 0.36
            ),
            'usage_stability': max(0.0, 1.0 - min(abs(float(player_stats.get('usage_trend', 0.0))), 0.05) / 0.05),
            'recent_form_confidence': min(float(player_stats.get('games_played', 5)), 20.0) / 20.0,
            'def_toughness_composite': (float(player_stats.get('opp_def_rating_last5', 110.0)) / 100.0) * (1.0 - float(player_stats.get('opp_fg_pct', 0.47))),
            'load_efficiency_ratio': float(player_stats.get('points_per_shot', 1.0)) / max(float(player_stats.get('fga_per_game', 10.0)) / max(float(player_stats.get('avg_minutes', 30.0)), 1.0), 0.1),
            'tracking_dist_per_touch': float(player_stats.get('tracking_dist_miles', 2.5)) / max(float(player_stats.get('tracking_touches_pg', 50.0)), 1.0),
        })

        # ---- Groups B/C/D: Historical vs Opponent, Team Rest Splits, YoY Stats ----
        features.update({
            'opp_b2b_def_rating':          float(player_stats.get('opp_b2b_def_rating', 112.0)),
            'opp_b2b_pace':                float(player_stats.get('opp_b2b_pace', 100.0)),
            'opp_b2b_pts_allowed':         float(player_stats.get('opp_b2b_pts_allowed', 115.0)),
            'opp_rested_def_rating':       float(player_stats.get('opp_rested_def_rating', 110.0)),
            'opp_rested_pace':             float(player_stats.get('opp_rested_pace', 100.0)),
            'opp_rest_def_rating_delta':   float(player_stats.get('opp_rest_def_rating_delta', 2.0)),
            'yoy_pts_change':              float(player_stats.get('yoy_pts_change', 0.0)),
            'yoy_ts_change':               float(player_stats.get('yoy_ts_change', 0.0)),
            'yoy_usage_change':            float(player_stats.get('yoy_usage_change', 0.0)),
            'seasons_in_league':           float(player_stats.get('seasons_in_league', 5)),
        })

        # ---- player context (matchup history + position defence) ----
        if player_context:
            matchup_history = player_context.get('matchup_history') or {}
            position_matchup = player_context.get('position_matchup') or {}
            injury_risk_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
            injury_risk_str = player_context.get('injury_history', {}).get('injury_risk', 'low')

            features.update({
                'vs_team_avg':          float(matchup_history.get('avg_points', 0)),
                'matchup_games':        int(matchup_history.get('games_played', 0)),
                'matchup_success_rate': float(matchup_history.get('success_rate', 0)),
                'pos_pts_allowed':      float(position_matchup.get('pts_allowed_per_game', 0)),
                'pos_def_rating':       float(position_matchup.get('defensive_rating', 0)),
                'effective_fg_pct':     float(position_matchup.get('effective_fg_pct', 0.47)),
                'injury_risk':          injury_risk_map.get(injury_risk_str, 0.0),
            })

        # ---- team context ----
        if team_context:
            features.update({
                'team_pace':       float(team_context.get('pace', 0)),
                'team_off_rating': float(team_context.get('offensive_rating', 0)),
                'team_def_rating': float(team_context.get('defensive_rating', 0)),
                'team_form':       float(team_context.get('recent_form', {}).get('win_pct', 0)),
                'rest_days':       int(team_context.get('rest_days', 1)),
                'team_injuries':   float(team_context.get('injury_impact', 0)),
            })
        # Ensure these keys always exist at inference with sensible defaults
        # matching the training zero-fill values in data_collector._group_defaults
        features.setdefault('team_pace', 100.0)
        features.setdefault('team_off_rating', 110.0)
        features.setdefault('team_def_rating', 110.0)
        features.setdefault('team_form', 0.5)
        features.setdefault('rest_days', 2)
        features.setdefault('team_injuries', 0.0)

        # ---- opponent context ----
        if opponent_context:
            features.update({
                'opp_pace':       float(opponent_context.get('pace', 0)),
                'opp_def_rating': float(opponent_context.get('defensive_rating', 0)),
                'opp_form':       float(opponent_context.get('recent_form', {}).get('win_pct', 0)),
                'opp_injuries':   float(opponent_context.get('injury_impact', 0)),
            })
        # Same guard for opponent context keys
        features.setdefault('opp_pace', 100.0)
        features.setdefault('opp_def_rating', 110.0)
        features.setdefault('opp_form', 0.5)
        features.setdefault('opp_injuries', 0.0)

        # ---- extended injury detail ----
        if team_context and 'injuries' in team_context:
            features.update({
                'team_injury_impact':     float(team_context['injury_impact']),
                'team_key_players_out':   int(team_context['injuries']['key_players_out']),
                'team_total_players_out': int(team_context['injuries']['total_players_out']),
            })

        if opponent_context and 'injuries' in opponent_context:
            features.update({
                'opp_injury_impact':     float(opponent_context['injury_impact']),
                'opp_key_players_out':   int(opponent_context['injuries']['key_players_out']),
                'opp_total_players_out': int(opponent_context['injuries']['total_players_out']),
            })

        # ---- Fill remaining model features from available data ----
        # Naming aliases (training used different key names)
        _vals = list(player_stats.get('values') or [0])
        _arr = np.array(_vals[::-1], dtype=float)  # chronological order
        _seas_avg = float(np.mean(_arr)) if len(_arr) > 0 else 0.0
        _seas_std = float(np.std(_arr)) if len(_arr) > 1 else 1.0
        _fga = features.get('fga_per_game', 15.0)
        _fta = features.get('fta_per_game', 4.0)
        _pts = features.get('recent_avg', 0.0)

        features.setdefault('is_back_to_back', float(player_stats.get('b2b_flag', 0)))
        features.setdefault('is_home_game_num', float(player_stats.get('is_home', 0.5)))
        features.setdefault('mins_last5', features.get('recent_minutes', 24.0))
        features.setdefault('mins_season', features.get('avg_minutes', 24.0))
        features.setdefault('ft_pct_recent', features.get('ft_pct', 0.75))
        features.setdefault('fg_pct_recent', features.get('recent_fg_pct', 0.45))
        features.setdefault('first_quarter_avg', features.get('q1_avg', 0.0))
        features.setdefault('fourth_quarter_avg', features.get('q4_avg', 0.0))
        features.setdefault('clutch_minutes_per_game', features.get('clutch_min_per_game', 0.0))
        features.setdefault('isolation_pct', features.get('iso_poss_pct', 0.0))
        features.setdefault('spot_up_pct', features.get('spotup_poss_pct', 0.0))
        features.setdefault('post_up_pct', features.get('postup_poss_pct', 0.0))
        features.setdefault('corner_3_pct', features.get('corner3_fg_pct', 0.38))
        features.setdefault('above_break_3_pct', features.get('above_break3_fg_pct', 0.35))
        features.setdefault('corner_three_pct', features.get('corner3_fg_pct', 0.38))
        features.setdefault('above_break_three_pct', features.get('above_break3_fg_pct', 0.35))
        features.setdefault('catch_and_shoot_pct', features.get('catch_shoot_fg_pct', 0.40))
        features.setdefault('pull_up_shot_pct', features.get('pullup_fg_pct', 0.40))
        features.setdefault('wide_open_shot_pct', features.get('open_shot_fg_pct', 0.50))
        features.setdefault('days_rest_opponent', features.get('opp_days_rest', 2.0))
        features.setdefault('opponent_back_to_back', features.get('opp_b2b', 0.0))

        # Derived from shot zones + per-game rates
        features.setdefault('rim_fga_per_game', features.get('rim_fga_pct', 0.25) * _fga)
        features.setdefault('paint_fga_per_game', features.get('paint_fga_pct', 0.30) * _fga)
        features.setdefault('mid_range_fga_per_game', features.get('midrange_fga_pct', 0.20) * _fga)
        features.setdefault('restricted_area_fg_pct', features.get('rim_fg_pct', 0.62))
        features.setdefault('mid_range_frequency', features.get('midrange_fga_pct', 0.20))
        features.setdefault('contested_shot_pct', 0.30)
        features.setdefault('open_shot_pct', features.get('open_shot_frequency', 0.30))
        features.setdefault('paint_touch_frequency', features.get('paint_fga_pct', 0.30))
        features.setdefault('paint_pts_per_game', features.get('pct_pts_paint', 0.30) * _seas_avg)
        features.setdefault('paint_attempts_per_game', features.get('paint_fga_pct', 0.30) * _fga)
        features.setdefault('paint_fg_pct', features.get('paint_fg_pct', 0.55))
        features.setdefault('restricted_area_attempts', features.get('rim_fga_pct', 0.25) * _fga)
        features.setdefault('paint_touch_to_points', features.get('pct_pts_paint', 0.30))

        # Efficiency metrics derivable from existing features
        _ts_denom = 2.0 * (_fga + 0.44 * _fta)
        features.setdefault('true_shooting_pct', features.get('ts_pct_official', (_pts / _ts_denom) if _ts_denom > 0 else 0.55))
        features.setdefault('assist_percentage', features.get('ast_pct_official', 0.15))
        features.setdefault('rebound_percentage', features.get('reb_pct_official', 0.10))
        features.setdefault('dreb_rate', features.get('dreb_pct_official', 0.15))
        features.setdefault('oreb_rate', features.get('oreb_pct_official', 0.05))
        features.setdefault('total_reb_rate', features.get('reb_pct_official', 0.10))
        features.setdefault('ft_rate', _fta / max(_fga, 1.0))
        features.setdefault('fta_rate_player', _fta / max(_fga, 1.0))
        features.setdefault('ft_attempts_per_game', _fta)
        features.setdefault('fouls_drawn_per_game', _fta * 0.6)
        features.setdefault('and_one_frequency', _fta * 0.05)
        features.setdefault('foul_drawing_ability', _fta / max(_fga, 1.0))

        # Per-100 possessions (approximate from per-game assuming ~100 poss/game)
        _min_ratio = float(player_stats.get('avg_minutes', 30.0)) / 48.0
        _poss_per_game = features.get('team_pace', 100.0) * _min_ratio
        _poss = max(_poss_per_game, 1.0)
        # Per-100 possessions: stat_per_game / poss_per_game * 100
        _ast_pg  = float(player_stats.get('ast_per_game',  features.get('ast_to_tov_ratio', 1.5) * 1.5))
        _reb_pg  = features.get('dreb_per_game', 3.0) + features.get('oreb_per_game', 1.0)
        _stl_pg  = float(player_stats.get('stl_per_game',  1.0))
        _blk_pg  = float(player_stats.get('blk_per_game',  0.5))
        _tov_pg  = float(player_stats.get('tov_per_game',
                         (_ast_pg / max(float(player_stats.get('ast_to_tov_ratio', 2.0)), 0.1))))
        features.setdefault('pts_per_100', _seas_avg / _poss * 100.0)
        features.setdefault('ast_per_100', _ast_pg / _poss * 100.0)
        features.setdefault('reb_per_100', _reb_pg  / _poss * 100.0)
        features.setdefault('stl_per_100', _stl_pg  / _poss * 100.0)
        features.setdefault('blk_per_100', _blk_pg  / _poss * 100.0)
        features.setdefault('tov_per_100', _tov_pg  / _poss * 100.0)

        # Rolling window stats from values array
        _ewm03 = float(pd.Series(_arr).ewm(alpha=0.3).mean().iloc[-1]) if len(_arr) > 0 else _seas_avg
        _ewm05 = float(pd.Series(_arr).ewm(alpha=0.5).mean().iloc[-1]) if len(_arr) > 0 else _seas_avg
        _last7  = _arr[-7:]  if len(_arr) >= 7  else _arr
        _last14 = _arr[-14:] if len(_arr) >= 14 else _arr
        _last30 = _arr[-30:] if len(_arr) >= 30 else _arr
        features.setdefault('ewm_alpha_0.3', _ewm03)
        features.setdefault('ewm_alpha_0.5', _ewm05)
        features.setdefault('rolling_7day_avg', float(np.mean(_last7)) if len(_last7) > 0 else _seas_avg)
        features.setdefault('rolling_14day_avg', float(np.mean(_last14)) if len(_last14) > 0 else _seas_avg)
        features.setdefault('rolling_30day_avg', float(np.mean(_last30)) if len(_last30) > 0 else _seas_avg)
        features.setdefault('games_above_season_avg_7day', float(np.sum(_last7 > _seas_avg)) if len(_last7) > 0 else 3.5)
        features.setdefault('games_above_season_avg_14day', float(np.sum(_last14 > _seas_avg)) if len(_last14) > 0 else 7.0)

        def _slope(a):
            if len(a) < 2: return 0.0
            try: return float(np.polyfit(range(len(a)), a, 1)[0])
            except Exception: return 0.0

        _last5 = _arr[-5:] if len(_arr) >= 5 else _arr
        _last10 = _arr[-10:] if len(_arr) >= 10 else _arr
        features.setdefault('trend_slope_5games', _slope(_last5))
        features.setdefault('trend_slope_10games', _slope(_last10))
        features.setdefault('volatility_ratio', _seas_std / max(_seas_avg, 0.1))
        features.setdefault('momentum_score', (_ewm05 - _seas_avg) / max(_seas_std, 0.1))
        features.setdefault('hot_hand_indicator', float(np.mean(_last5) > _seas_avg * 1.1) if len(_last5) > 0 else 0.0)

        # Minutes-based features
        _avg_min = float(player_stats.get('avg_minutes', 30.0))
        _rec_min = float(player_stats.get('recent_minutes', _avg_min))
        _min_vals = [_avg_min] * len(_vals)  # approximate; real per-game minutes not stored separately
        features.setdefault('avg_minutes_last_3', _rec_min)
        features.setdefault('minutes_last_3_games', _rec_min * 3)
        features.setdefault('minutes_last_5_games', _rec_min * 5)
        features.setdefault('minutes_last_7_games', _rec_min * 7)
        features.setdefault('minutes_fatigue_score', max(0.0, (_rec_min - _avg_min) / max(_avg_min, 1.0)))

        # Consecutive streaks (over/under vs season avg — no line available here)
        _consec_over = 0
        for v in reversed(list(_arr)):
            if v > _seas_avg: _consec_over += 1
            else: break
        _consec_under = 0
        for v in reversed(list(_arr)):
            if v <= _seas_avg: _consec_under += 1
            else: break
        features.setdefault('consecutive_over_games', float(_consec_over))
        features.setdefault('consecutive_under_games', float(_consec_under))

        # Rest advantage
        _own_rest = features.get('rest_days', 2.0)
        _opp_rest = features.get('days_rest_opponent', features.get('opp_days_rest', 2.0))
        features.setdefault('rest_advantage', _own_rest - _opp_rest)
        features.setdefault('rest_advantage_abs', abs(_own_rest - _opp_rest))
        features.setdefault('both_teams_rested', float(_own_rest >= 2 and _opp_rest >= 2))

        # Player age/experience flags
        _age = features.get('player_age', 26.0)
        _exp = features.get('years_experience', 5.0)
        features.setdefault('is_rookie', float(_exp <= 1))
        features.setdefault('is_veteran', float(_exp >= 10))

        # Opponent def context
        features.setdefault('opp_win_rate_last10', float(player_stats.get('opp_win_rate_last10', 0.5)))
        features.setdefault('opp_def_rating_last10', float(player_stats.get('opp_def_rating_last10', 110.0)))
        features.setdefault('opp_pace_last5', float(player_stats.get('opp_pace_last5', 100.0)))

        # Tracking derived
        _touches = features.get('tracking_touches_pg', 50.0)
        _time_poss = features.get('tracking_time_of_poss_pg', 2.5)
        _dist = features.get('tracking_dist_miles', 2.5)
        features.setdefault('avg_dribbles_per_touch', features.get('tracking_avg_drib_per_touch', 1.5))
        features.setdefault('avg_seconds_per_touch', (_time_poss * 60.0) / max(_touches, 1.0))
        features.setdefault('avg_points_per_touch', _seas_avg / max(_touches, 1.0))
        features.setdefault('touches_per_game', _touches)
        features.setdefault('touches_per_possession', _touches / max(_poss, 1.0))
        features.setdefault('time_of_possession_per_game', _time_poss)
        features.setdefault('elbow_touches_per_game', _touches * 0.05)
        features.setdefault('post_touches_per_game', features.get('postup_poss_pct', 0.0) * _touches)
        features.setdefault('paint_touches_per_game', features.get('paint_fga_pct', 0.30) * _touches)
        features.setdefault('front_court_touches_per_game', _touches * 0.7)

        # Shot clock / shot timing
        features.setdefault('avg_shot_clock_time', 14.0)
        features.setdefault('avg_shot_distance', 14.0)
        features.setdefault('late_clock_shot_frequency', 0.15)
        features.setdefault('early_clock_shot_frequency', 0.20)
        features.setdefault('shot_quality_vs_expected', features.get('ts_vs_zone_expected', 0.0))

        # Lineup / role features
        features.setdefault('net_rating_with_starters', features.get('on_court_net_rating', 0.0))
        features.setdefault('usage_rate_with_star_out', features.get('usage_rate', 18.0) * 1.05)
        features.setdefault('minutes_with_starting_lineup_pct', 0.6)
        features.setdefault('five_man_unit_net_rating', features.get('on_court_net_rating', 0.0))
        features.setdefault('lineups_played_count', 50.0)
        features.setdefault('off_court_plus_minus', features.get('off_court_net_rating', 0.0))
        features.setdefault('on_court_plus_minus', features.get('on_court_net_rating', 0.0))
        features.setdefault('top_lineup_minutes_pct', 0.4)
        features.setdefault('net_rating', features.get('net_rating_player', 0.0))
        features.setdefault('lineup_continuity', 0.7)
        features.setdefault('lineup_stability_score', float(player_stats.get('lineup_stability_score', 0.7)))
        features.setdefault('teammate_chemistry_score', 0.5)
        features.setdefault('bench_strength', 0.5)
        features.setdefault('new_teammate_games', 0.0)
        features.setdefault('primary_teammate_out', 0.0)
        features.setdefault('secondary_teammate_out', 0.0)

        # Game situation features
        features.setdefault('blowout_probability', features.get('blowout_game_pct', 0.2))
        features.setdefault('close_game_probability', features.get('close_game_pct', 0.3))
        features.setdefault('expected_game_script', 0.5)
        features.setdefault('historical_game_script_avg', 0.5)
        features.setdefault('head_to_head_avg', features.get('vs_team_avg', 0.0))
        features.setdefault('head_to_head_games', features.get('matchup_games', 0))
        features.setdefault('position_vs_position_dvp', features.get('dvp_pts_delta', 0.0))
        features.setdefault('matchup_pace', (features.get('team_pace', 100.0) + features.get('opp_pace', 100.0)) / 2.0)

        # Clutch / performance situational
        features.setdefault('clutch_performance_score', features.get('clutch_pts_per_game', 0.0) / max(_seas_avg, 0.1))
        features.setdefault('fourth_quarter_usage_rate', features.get('usage_rate', 18.0))
        features.setdefault('crunch_time_usage', features.get('clutch_min_per_game', 0.0))
        features.setdefault('garbage_time_minutes_pct', 0.05)
        features.setdefault('typical_substitution_minute', 20.0)

        # Performance by game state (defaults to season avg)
        features.setdefault('performance_when_leading', _seas_avg * 0.95)
        features.setdefault('performance_when_trailing', _seas_avg * 1.05)
        features.setdefault('performance_when_tied', _seas_avg)
        features.setdefault('performance_in_overtime', _seas_avg)
        features.setdefault('performance_by_score_differential', 0.0)
        features.setdefault('stat_in_wins', features.get('stat_in_wins', _seas_avg))
        features.setdefault('stat_in_losses', features.get('stat_in_losses', _seas_avg))

        # Defender features
        features.setdefault('career_vs_defender', features.get('vs_team_avg', 0.0))
        features.setdefault('recent_vs_defender', features.get('vs_team_avg', 0.0))
        features.setdefault('primary_defender_rating', features.get('opp_def_rating', 110.0))
        features.setdefault('primary_defender_age', 26.0)
        features.setdefault('defender_size_mismatch', 0.0)
        features.setdefault('defender_recent_form', 0.5)
        features.setdefault('defender_switching_frequency', 0.3)
        features.setdefault('def_fg_pct_allowed', features.get('opp_fg_pct', 0.47))
        features.setdefault('def_rating_individual', features.get('opp_def_rating', 110.0))
        features.setdefault('deflections_per_game', float(player_stats.get('deflections_per_game', 1.0)))

        # Arena / travel
        features.setdefault('arena_altitude', 0.0)
        features.setdefault('arena_capacity', 19000.0)
        features.setdefault('home_court_advantage_rating', 3.0)
        features.setdefault('player_vs_arena', 0.0)
        features.setdefault('travel_distance', 1000.0)
        features.setdefault('time_zone_change', 0.0)
        features.setdefault('coast_to_coast', 0.0)

        # Schedule context
        features.setdefault('recent_away_streak', float(player_stats.get('recent_away_streak', 0.0)))
        features.setdefault('team_win_streak', float(player_stats.get('team_win_streak', 0.0)))
        features.setdefault('team_loss_streak', float(player_stats.get('team_loss_streak', 0.0)))

        # Season context
        features.setdefault('playoff_implications', 0.5)
        features.setdefault('rivalry_game', 0.0)
        features.setdefault('national_tv_game', 0.0)
        features.setdefault('season_phase', features.get('season_phase_numeric', 0.5))
        features.setdefault('playoff_seeding_impact', 0.5)
        features.setdefault('tanking_indicator', 0.0)
        features.setdefault('must_win_situation', 0.0)
        features.setdefault('games_back_from_playoff', features.get('team_games_back', 5.0))

        # Model performance tracking (neutral defaults until graded predictions accumulate)
        features.setdefault('model_accuracy_player', 0.55)
        features.setdefault('avg_prediction_error_player', _seas_std)
        features.setdefault('calibration_score_player', 0.5)

        # Hit rate and edge (used by classifier)
        features.setdefault('hit_rate', float(player_stats.get('hit_rate', 0.5)))
        features.setdefault('edge', float(player_stats.get('edge', 0.0)))
        features.setdefault('is_home', float(player_stats.get('is_home', 0.5)))
        features.setdefault('location_avg', float(player_stats.get('location_avg', _seas_avg)))

        # Additional derived
        features.setdefault('vs_team_last_season_avg', features.get('vs_team_avg', 0.0))
        features.setdefault('vs_team_home_away_split', 0.0)
        features.setdefault('vs_team_win_pct', 0.5)
        features.setdefault('shot_selection_rating', features.get('ts_pct_official', 0.55))
        features.setdefault('bad_shot_frequency', 1.0 - features.get('ts_pct_official', 0.55))
        features.setdefault('shot_clock_management', 0.5)
        features.setdefault('pnr_ball_handler_pct', features.get('pnr_bh_poss_pct', 0.0))
        features.setdefault('pnr_roll_man_pct', features.get('pnr_roll_poss_pct', 0.0))
        features.setdefault('transition_pct', features.get('transition_poss_pct', 0.0))

        # pct_fga splits
        _rim_pct = features.get('rim_fga_pct', 0.25)
        _paint_pct = features.get('paint_fga_pct', 0.30)
        _mid_pct = features.get('midrange_fga_pct', 0.20)
        _c3_pct = features.get('corner3_fga_pct', 0.10)
        _ab3_pct = features.get('above_break3_fga_pct', 0.25)
        _total3 = _c3_pct + _ab3_pct
        features.setdefault('pct_fga_2pt', 1.0 - _total3)
        features.setdefault('pct_fga_3pt', _total3)
        features.setdefault('pct_pts_in_paint', features.get('pct_pts_paint', 0.30))
        features.setdefault('pct_pts_off_tov', 0.10)
        features.setdefault('pct_pts_fb', 0.12)

        # Additional missing features from NUMERIC_FEATURE_KEYS
        features.setdefault('opp_def_rating_last10', features.get('opp_def_rating_last5', 110.0))
        features.setdefault('opp_pace_last5', features.get('opp_pace', 100.0))
        features.setdefault('opp_win_rate_last10', 0.5)
        features.setdefault('is_rookie', float(features.get('years_experience', 5.0) <= 1))
        features.setdefault('is_veteran', float(features.get('years_experience', 5.0) >= 10))
        features.setdefault('first_quarter_avg', features.get('q1_avg', 0.0))
        features.setdefault('fourth_quarter_avg', features.get('q4_avg', 0.0))
        features.setdefault('clutch_performance_score', features.get('clutch_pts_per_game', 0.0) / max(_seas_avg, 0.1))
        features.setdefault('shot_selection_rating', features.get('ts_pct_official', 0.55))
        features.setdefault('bad_shot_frequency', 1.0 - features.get('ts_pct_official', 0.55))
        features.setdefault('shot_clock_management', 12.0)
        features.setdefault('player_age', 26.0)
        features.setdefault('years_experience', 5.0)
        features.setdefault('net_rating_player', 0.0)
        features.setdefault('rebound_contested_pct', 0.4)
        features.setdefault('rebound_positioning_score', 0.5)
        features.setdefault('paint_touch_to_points', features.get('pct_pts_paint', 0.30))
        features.setdefault('performance_in_overtime', _seas_avg)
        features.setdefault('performance_by_score_differential', 0.0)
        features.setdefault('both_teams_rested', float(features.get('rest_days', 2) >= 2 and features.get('days_rest_opponent', 2) >= 2))
        features.setdefault('head_to_head_avg', features.get('vs_team_avg', 0.0))
        features.setdefault('head_to_head_games', features.get('matchup_games', 0))
        features.setdefault('position_vs_position_dvp', features.get('dvp_pts_delta', 0.0))
        features.setdefault('matchup_pace', (features.get('team_pace', 100.0) + features.get('opp_pace', 100.0)) / 2.0)
        features.setdefault('historical_game_script_avg', features.get('blowout_game_pct', 0.2) - features.get('close_game_pct', 0.3))
        features.setdefault('defender_switching_frequency', 0.3)
        features.setdefault('blowout_probability', features.get('blowout_game_pct', 0.2))
        features.setdefault('close_game_probability', features.get('close_game_pct', 0.3))
        features.setdefault('expected_game_script', features.get('blowout_game_pct', 0.2) - features.get('close_game_pct', 0.3))

        # ---- Market / odds line movement features ----
        # These are injected by the caller (analyze_prop_bet) when odds data is available.
        # Default to neutral values when no odds tracking is active.
        features.setdefault('opening_line', 0.0)
        features.setdefault('current_line', 0.0)
        features.setdefault('line_movement', 0.0)
        features.setdefault('line_movement_pct', 0.0)
        features.setdefault('implied_over_prob', 0.5)
        features.setdefault('implied_under_prob', 0.5)
        features.setdefault('market_consensus_std', 0.0)
        features.setdefault('sharp_action_score', 0.0)
        features.setdefault('line_velocity', 0.0)
        features.setdefault('stale_line_flag', 0.0)
        features.setdefault('bookmaker_count', 0.0)

        # Intensity / playoff context — sourced from player_stats (populated upstream)
        features['is_playoff'] = float(player_stats.get('is_playoff', 0.0) or 0.0)
        features['is_play_in'] = float(player_stats.get('is_play_in', 0.0) or 0.0)
        features['series_game_num'] = float(player_stats.get('series_game_num', 0.0) or 0.0)
        features['team_series_wins_in'] = float(player_stats.get('team_series_wins_in', 0.0) or 0.0)
        features['opp_series_wins_in'] = float(player_stats.get('opp_series_wins_in', 0.0) or 0.0)
        features['is_elimination_game'] = float(player_stats.get('is_elimination_game', 0.0) or 0.0)
        features['playoff_home'] = float(player_stats.get('playoff_home', 0.0) or 0.0)
        features['playoff_away'] = float(player_stats.get('playoff_away', 0.0) or 0.0)

        return features

    def predict(self, features, line, prop_type=None):
        """Run prediction — blends statistical baseline with ML models if trained.

        When prop_type is provided and a per-prop model exists, that model is used
        instead of the global fallback.  The interface is fully backward-compatible:
        callers that omit prop_type get the original behaviour unchanged.
        """
        try:
            recent_avg = features.get('recent_avg', 0)
            season_avg = features.get('season_avg', 0)
            std_dev = features.get('stddev', 0)

            # stat-only baseline
            stat_predicted_value = (0.7 * recent_avg + 0.3 * season_avg)
            stat_z_score = (line - stat_predicted_value) / (std_dev + 1e-6)
            stat_over_prob = 1 - scipy.stats.norm.cdf(stat_z_score)

            predicted_value = stat_predicted_value
            over_prob = stat_over_prob

            # choose which set of ML models to use
            use_prop_model = (
                prop_type is not None
                and prop_type in self.prop_models
            )

            if use_prop_model:
                try:
                    prop_bundle = self.prop_models[prop_type]
                    _clf = prop_bundle['calibrated_clf']
                    _reg = prop_bundle['regression_model']
                    _scaler = prop_bundle['scaler']
                    _models_ready = True
                except KeyError as _ke:
                    print(f"per-prop bundle for '{prop_type}' is incomplete ({_ke}), falling back to global models")
                    use_prop_model = False

            if not use_prop_model:
                _clf = self.classification_model
                _reg = self.regression_model
                _scaler = self.scaler
                _models_ready = self.models_trained

            # blend with trained ML models when available
            if _models_ready:
                try:
                    features_df = pd.DataFrame([features])

                    def _align_to_model(df, estimator, extra=None):
                        """Zero-fill missing cols and reorder to match estimator's expected features."""
                        # unwrap custom wrappers like IsotonicCalibratedModel (.base_estimator)
                        # and sklearn 1.6+ CalibratedClassifierCV (.estimator)
                        actual = getattr(estimator, 'estimator', getattr(estimator, 'base_estimator', estimator))
                        feat_names = None
                        if hasattr(actual, 'feature_names_in_'):
                            feat_names = [str(f) for f in actual.feature_names_in_]
                        elif hasattr(actual, 'get_booster'):
                            feat_names = actual.get_booster().feature_names
                        elif hasattr(estimator, 'feature_names_in_'):
                            feat_names = [str(f) for f in estimator.feature_names_in_]
                        if feat_names is None:
                            return df
                        row = {col: df[col].iloc[0] if col in df.columns else 0.0 for col in feat_names}
                        if extra:
                            for k, v in extra.items():
                                if k in feat_names:
                                    row[k] = v
                        return pd.DataFrame([row], columns=feat_names)

                    # align for scaler / regressor (scaler takes precedence if it has feature names)
                    if _reg is not None:
                        scaler_estimator = _scaler if hasattr(_scaler, 'feature_names_in_') else _reg
                        features_aligned = _align_to_model(features_df, scaler_estimator)
                        features_aligned = features_aligned.fillna(0.0)
                        features_scaled = _scaler.transform(features_aligned)
                        ml_pred = float(_reg.predict(features_scaled)[0])
                    else:
                        # Binary props (double_double, triple_double) — no regression model
                        ml_pred = stat_predicted_value

                    # classifier may expect additional features (e.g. 'line')
                    clf_features = _align_to_model(features_df, _clf, extra={'line': line})
                    clf_features = clf_features.fillna(0.0)
                    ml_prob = float(_clf.predict_proba(clf_features)[0, 1])

                    # 25/75 blend: small stats anchor + ML dominant
                    predicted_value = 0.25 * stat_predicted_value + 0.75 * ml_pred
                    blended_z = (line - predicted_value) / (std_dev + 1e-6)
                    blended_stat_prob = 1 - scipy.stats.norm.cdf(blended_z)
                    over_prob = 0.25 * blended_stat_prob + 0.75 * ml_prob
                except Exception as e:
                    print(f"ML inference failed, falling back to stats: {e}")
            # when models aren't trained yet, the stat-only baseline is already set above — nothing more to do

            edge = ((predicted_value - line) / line) if line > 0 else 0

            prob_strength = abs(over_prob - 0.5)
            edge_strength = abs(edge)
            confidence = self._calculate_confidence(prob_strength, edge_strength)

            over_prob = max(0.0, min(1.0, over_prob))  # clamp to valid probability range
            recommendation = self._generate_recommendation(over_prob, predicted_value, line, edge, confidence)

            return {
                'over_probability': float(over_prob),
                'predicted_value': float(predicted_value),
                'recommendation': recommendation,
                'confidence': confidence,
                'edge': float(edge)
            }

        except Exception as e:
            print(f"prediction error: {e}")
            return {
                'over_probability': 0.5,
                'predicted_value': features.get('season_avg', line),
                'recommendation': 'PASS',
                'confidence': 'LOW',
                'edge': 0.0
            }

    def _calculate_confidence(self, prob_strength, edge_strength):
        """HIGH/MEDIUM/LOW based on how far prob and edge are from neutral.

        Thresholds are adaptive: they default to 0.08/0.04 but can be updated
        by calibrate_confidence_thresholds() based on historical graded logs.
        """
        confidence_score = (0.7 * prob_strength + 0.3 * edge_strength)

        if confidence_score > self.conf_high_threshold:
            return 'HIGH'
        elif confidence_score > self.conf_med_threshold:
            return 'MEDIUM'
        return 'LOW'

    def calibrate_confidence_thresholds(self, graded_logs):
        """Derive HIGH/MEDIUM thresholds from graded prediction logs.

        Parameters
        ----------
        graded_logs : list of dict
            Each dict must contain:
              - 'confidence_score': float  (the raw score, i.e. 0.7*prob_strength + 0.3*edge_strength)
              - 'correct': int             (1 if the bet was correct, 0 otherwise)

        The method searches for the lowest threshold such that predictions
        above it hit >60% (HIGH) and the band below that hits >52% (MEDIUM).
        Results are persisted to conf_thresholds.joblib.
        """
        if not graded_logs:
            return

        if len(graded_logs) < 10:
            print(f"calibrate_confidence_thresholds: only {len(graded_logs)} graded samples — keeping current thresholds")
            return

        scores = np.array([float(l['confidence_score']) for l in graded_logs])
        correct = np.array([int(l['correct']) for l in graded_logs])

        # candidate thresholds between the 50th and 99th percentile
        candidates = np.percentile(scores, np.arange(50, 100, 1))
        candidates = sorted(set(candidates))

        best_high = self.conf_high_threshold
        best_med = self.conf_med_threshold

        # find the lowest threshold where hit-rate above it exceeds 60%
        for thresh in candidates:
            mask = scores >= thresh
            if mask.sum() < 10:
                continue
            hit_rate = correct[mask].mean()
            if hit_rate > 0.60:
                best_high = float(thresh)
                break

        # find the lowest threshold where hit-rate in (best_med, best_high) exceeds 52%
        for thresh in candidates:
            if thresh >= best_high:
                break
            mask = (scores >= thresh) & (scores < best_high)
            if mask.sum() < 10:
                continue
            hit_rate = correct[mask].mean()
            if hit_rate > 0.52:
                best_med = float(thresh)
                break

        self.conf_high_threshold = best_high
        self.conf_med_threshold = best_med

        joblib.dump(
            {'high': self.conf_high_threshold, 'med': self.conf_med_threshold},
            os.path.join(self.model_dir, 'conf_thresholds.joblib')
        )
        print(f"confidence thresholds updated — HIGH>{self.conf_high_threshold:.4f}, "
              f"MEDIUM>{self.conf_med_threshold:.4f}")

    def _generate_recommendation(self, prob, predicted_value, line, edge, confidence):
        """Turn probability + confidence into a betting recommendation."""
        if confidence == 'LOW':
            return 'PASS'

        if prob > 0.6 and edge > 0.05:
            return 'STRONG OVER'
        elif prob < 0.4 and edge < -0.05:
            return 'STRONG UNDER'
        elif prob > 0.55 and edge > 0.03:
            return 'LEAN OVER'
        elif prob < 0.45 and edge < -0.03:
            return 'LEAN UNDER'

        return 'PASS'

    def _make_calibrated_clf(self, base_clf, n_samples):
        """Wrap a fitted GradientBoostingClassifier in CalibratedClassifierCV."""
        method = 'isotonic' if n_samples >= 100 else 'sigmoid'
        # sklearn 1.8 removed cv='prefit'; wrap already-fitted estimator in FrozenEstimator instead
        if FrozenEstimator is not None:
            cal = CalibratedClassifierCV(FrozenEstimator(base_clf), method=method, cv=None)
        else:
            cal = CalibratedClassifierCV(base_clf, method=method, cv='prefit')
        return cal

    def train(self, training_data):
        """Train both models and save them to disk."""
        if not training_data:
            raise ValueError("No training data provided")

        # build feature matrix and both target arrays
        X = pd.DataFrame([data['features'] for data in training_data])
        X = X.fillna(0)  # guard against NaN from mismatched feature sets across sample sources
        y_class = [1 if data['result'] > data['line'] else 0 for data in training_data]
        y_reg = [data['result'] for data in training_data]

        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42
        )

        # split train further so calibration uses held-out data
        X_fit, X_cal, y_class_fit, y_class_cal = train_test_split(
            X_train, y_class_train, test_size=0.2, random_state=42
        )

        X_fit_scaled = self.scaler.fit_transform(X_fit)
        X_cal_scaled = self.scaler.transform(X_cal)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_scaled = self.scaler.transform(X_train)  # for regressor below

        # classifier — fit on 80% of train, calibrate on held-out 20%
        self.classification_model.fit(X_fit_scaled, y_class_fit)
        cal_clf = self._make_calibrated_clf(self.classification_model, len(X_cal))
        cal_clf.fit(X_cal_scaled, y_class_cal)
        class_auc = roc_auc_score(y_class_test,
            cal_clf.predict_proba(X_test_scaled)[:, 1])

        # regressor
        self.regression_model.fit(X_train_scaled, y_reg_train)
        reg_rmse = np.sqrt(mean_squared_error(y_reg_test,
            self.regression_model.predict(X_test_scaled)))

        print(f"Classification AUC: {class_auc:.3f}")
        print(f"Regression RMSE: {reg_rmse:.3f}")

        # store calibrated classifier as the active global classifier
        self.classification_model = cal_clf

        joblib.dump(self.classification_model, f'{self.model_dir}/classification_model.joblib')
        joblib.dump(self.regression_model, f'{self.model_dir}/regression_model.joblib')
        joblib.dump(self.scaler, f'{self.model_dir}/scaler.joblib')
        self.models_trained = True

        # ------------------------------------------------------------------ #
        # per-prop models                                                      #
        # ------------------------------------------------------------------ #
        _MIN_PROP_SAMPLES = 50
        prop_results = {}

        # group training samples by prop_type (key may be absent — skip those)
        from collections import defaultdict
        prop_buckets = defaultdict(list)
        for sample in training_data:
            pt = sample.get('prop_type')
            if pt:
                prop_buckets[pt].append(sample)

        for prop_type, samples in prop_buckets.items():
            if len(samples) < _MIN_PROP_SAMPLES:
                print(f"skipping per-prop model for '{prop_type}': only {len(samples)} samples")
                continue

            try:
                Xp = pd.DataFrame([s['features'] for s in samples])
                Xp = Xp.fillna(0)  # same NaN guard as global training
                yp_class = [1 if s['result'] > s['line'] else 0 for s in samples]
                yp_reg = [s['result'] for s in samples]

                Xp_train, Xp_test, ypc_train, ypc_test, ypr_train, ypr_test = train_test_split(
                    Xp, yp_class, yp_reg, test_size=0.2, random_state=42
                )

                prop_scaler = StandardScaler()

                # split train further for held-out calibration
                Xp_fit, Xp_cal, ypc_fit, ypc_cal = train_test_split(
                    Xp_train, ypc_train, test_size=0.2, random_state=42
                )
                Xp_fit_s = prop_scaler.fit_transform(Xp_fit)
                Xp_cal_s = prop_scaler.transform(Xp_cal)
                Xp_test_s = prop_scaler.transform(Xp_test)

                prop_clf = GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
                )
                prop_clf.fit(Xp_fit_s, ypc_fit)
                prop_cal = self._make_calibrated_clf(prop_clf, len(Xp_cal))
                prop_cal.fit(Xp_cal_s, ypc_cal)

                # regressor uses the full train set (no calibration needed)
                Xp_train_s_full = prop_scaler.transform(Xp_train)
                prop_reg = GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
                )
                prop_reg.fit(Xp_train_s_full, ypr_train)

                prop_auc = roc_auc_score(ypc_test, prop_cal.predict_proba(Xp_test_s)[:, 1])
                prop_rmse = np.sqrt(mean_squared_error(ypr_test, prop_reg.predict(Xp_test_s)))
                print(f"[{prop_type}] AUC: {prop_auc:.3f}, RMSE: {prop_rmse:.3f}")

                joblib.dump(prop_cal,   os.path.join(self.model_dir, f'clf_cal_{prop_type}.joblib'))
                joblib.dump(prop_reg,   os.path.join(self.model_dir, f'reg_{prop_type}.joblib'))
                joblib.dump(prop_scaler, os.path.join(self.model_dir, f'scaler_{prop_type}.joblib'))

                self.prop_models[prop_type] = {
                    'calibrated_clf': prop_cal,
                    'regression_model': prop_reg,
                    'scaler': prop_scaler,
                }
                prop_results[prop_type] = {'auc': float(prop_auc), 'rmse': float(prop_rmse)}

            except Exception as e:
                print(f"error training per-prop model for '{prop_type}': {e}")

        # ------------------------------------------------------------------ #
        # model versioning                                                     #
        # ------------------------------------------------------------------ #
        try:
            new_version = str(int(self._model_version) + 1)
        except ValueError:
            new_version = "1"
        self._model_version = new_version
        with open(os.path.join(self.model_dir, 'model_version.txt'), 'w') as _vf:
            _vf.write(self._model_version)
        print(f"model version incremented to {self._model_version}")

        return {
            'auc': float(class_auc),
            'rmse': float(reg_rmse),
            'prop_results': prop_results,
            'model_version': self._model_version,
        }
