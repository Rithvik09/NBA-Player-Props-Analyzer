from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
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
                if not os.path.exists(reg_path) or not os.path.exists(scaler_path):
                    print(f"skipping incomplete per-prop bundle for '{prop_type}': missing reg or scaler file")
                    continue
                try:
                    self.prop_models[prop_type] = {
                        'calibrated_clf': joblib.load(os.path.join(self.model_dir, fname)),
                        'regression_model': joblib.load(reg_path),
                        'scaler': joblib.load(scaler_path),
                    }
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
        """Get this player's historical stats against a specific team."""
        try:
            gamefinder = LeagueGameFinder(
                player_id_nullable=player_id,
                vs_team_id_nullable=opponent_team_id,
                season_type_nullable='Regular Season'
            ).get_data_frames()[0]

            time.sleep(0.6)

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
        """Build the feature dict for the ML models from all available context."""
        features = {}

        # use only the last 5 games for max/min/stddev — matching the training data computation
        _all_vals = player_stats.get('values') or [0]
        _last5 = _all_vals[:5] if len(_all_vals) >= 5 else _all_vals
        features.update({
            'recent_avg':  float(player_stats.get('last5_avg', 0)),
            'season_avg':  float(player_stats.get('avg', 0)),
            'max_recent':  float(max(_last5)),
            'min_recent':  float(min(_last5)),
            'stddev':      float(np.std(_last5)),
            'games_played': len(_all_vals),
        })

        if player_context:
            matchup_history = player_context.get('matchup_history') or {}
            position_matchup = player_context.get('position_matchup') or {}
            injury_risk_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
            injury_risk_str = player_context.get('injury_history', {}).get('injury_risk', 'low')

            features.update({
                'vs_team_avg': float(matchup_history.get('avg_points', 0)),
                'matchup_games': int(matchup_history.get('games_played', 0)),
                'matchup_success_rate': float(matchup_history.get('success_rate', 0)),
                'pos_pts_allowed': float(position_matchup.get('pts_allowed_per_game', 0)),
                'pos_def_rating': float(position_matchup.get('defensive_rating', 0)),
                'injury_risk': injury_risk_map.get(injury_risk_str, 0.0)
            })

        if team_context:
            features.update({
                'team_pace': float(team_context.get('pace', 0)),
                'team_off_rating': float(team_context.get('offensive_rating', 0)),
                'team_def_rating': float(team_context.get('defensive_rating', 0)),
                'team_form': float(team_context.get('recent_form', {}).get('win_pct', 0)),
                'rest_days': int(team_context.get('rest_days', 1)),
                'team_injuries': float(team_context.get('injury_impact', 0))
            })

        if opponent_context:
            features.update({
                'opp_pace': float(opponent_context.get('pace', 0)),
                'opp_def_rating': float(opponent_context.get('defensive_rating', 0)),
                'opp_form': float(opponent_context.get('recent_form', {}).get('win_pct', 0)),
                'opp_injuries': float(opponent_context.get('injury_impact', 0))
            })

        if team_context and 'injuries' in team_context:
            features.update({
                'team_injury_impact': float(team_context['injury_impact']),
                'team_key_players_out': int(team_context['injuries']['key_players_out']),
                'team_total_players_out': int(team_context['injuries']['total_players_out'])
            })

        if opponent_context and 'injuries' in opponent_context:
            features.update({
                'opp_injury_impact': float(opponent_context['injury_impact']),
                'opp_key_players_out': int(opponent_context['injuries']['key_players_out']),
                'opp_total_players_out': int(opponent_context['injuries']['total_players_out'])
            })

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
                    # align columns to what scaler was trained on
                    if hasattr(_scaler, 'feature_names_in_'):
                        for col in _scaler.feature_names_in_:
                            if col not in features_df.columns:
                                features_df[col] = 0.0
                        features_df = features_df[_scaler.feature_names_in_]
                    features_df = features_df.fillna(0.0)  # prevent NaN from propagating through the scaler
                    features_scaled = _scaler.transform(features_df)

                    ml_pred = float(_reg.predict(features_scaled)[0])
                    ml_prob = float(_clf.predict_proba(features_scaled)[0, 1])

                    # 50/50 blend: statistical baseline + ML models
                    predicted_value = 0.5 * stat_predicted_value + 0.5 * ml_pred
                    blended_z = (line - predicted_value) / (std_dev + 1e-6)
                    blended_stat_prob = 1 - scipy.stats.norm.cdf(blended_z)
                    over_prob = 0.5 * blended_stat_prob + 0.5 * ml_prob
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
