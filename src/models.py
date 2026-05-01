import warnings

# Suppress two specific sklearn warnings that are benign in our usage:
#
#   1. "Since FrozenEstimator does not appear to accept sample_weight,
#      sample weights will only be used for the calibration itself."
#      We deliberately want this. The base classifier was already fit
#      with sample weights upstream; the FrozenEstimator wrapper only
#      exists so CalibratedClassifierCV can fit Platt/isotonic on top
#      WITHOUT re-fitting the base. The weights we pass to .fit() of
#      the calibrator are meant for the calibration step alone.
#      (sklearn issue #21134.)
#
#   2. "X has feature names, but {Estimator} was fitted without feature
#      names." The inner GBC / HGB classifiers are fit on scaled numpy
#      arrays (see scaler.fit_transform), but at predict time we pass
#      DataFrames so downstream column-alignment code can use the
#      backfilled feature_names_in_ attribute. The mismatch is by
#      design — _align_to_model handles it correctly.
#
# Suppressing here (rather than at every call site) keeps the model
# code uncluttered and the test output readable. Both messages are
# matched on regex so future sklearn rewordings of *unrelated* warnings
# still surface.
warnings.filterwarnings(
    "ignore",
    message=r".*FrozenEstimator does not appear to accept sample_weight.*",
    category=UserWarning,
    module=r"sklearn\.calibration",
)
warnings.filterwarnings(
    "ignore",
    message=r"X has feature names, but \w+ was fitted without feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)

from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV
try:
    # sklearn >= 1.6; replaces CalibratedClassifierCV(cv='prefit') which was removed in 1.8
    from sklearn.frozen import FrozenEstimator
except ImportError:  # older sklearn — caller falls back to cv='prefit' path below
    FrozenEstimator = None
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, brier_score_loss
from sklearn.inspection import permutation_importance
from nba_api.stats.endpoints import TeamGameLog, CommonPlayerInfo, LeagueGameFinder
from nba_api.stats.endpoints import playergamelog, LeagueDashPtDefend
import scipy.stats
import numpy as np
import pandas as pd
import joblib
import os
import time
import json
from datetime import datetime, timezone
from .injury_tracker import InjuryTracker
from .ml_quantile import QuantileEnsemble
from .ml_stacking import StackingBlender, StackedCalibratedClassifier
from .ml_validation import out_of_time_split, parse_iso

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

        # Cached calibration audit: {prop_type: bool needs_recal}, populated
        # from model_metadata.json on load. Surfaced in predict_prop's response
        # so callers can flag predictions made by an under-calibrated model.
        self._needs_recal: dict = {}
        self._global_needs_recal: bool = False
        self._load_calibration_audit()

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

    def _load_calibration_audit(self):
        """Hydrate ``self._needs_recal`` from model_metadata.json.

        Quietly no-ops if the file is missing or malformed — calibration audit
        is informational; we never want it to block model loading.
        """
        meta_path = os.path.join(self.model_dir, "model_metadata.json")
        if not os.path.exists(meta_path):
            return
        try:
            with open(meta_path, "r") as _mf:
                meta = json.load(_mf)
        except (OSError, json.JSONDecodeError):
            return
        global_block = meta.get("global") or {}
        self._global_needs_recal = bool(global_block.get("needs_recal", False))
        for prop_type, info in (meta.get("props") or {}).items():
            if "needs_recal" in info:
                self._needs_recal[prop_type] = bool(info["needs_recal"])


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

        # B1 schedule/usage features — neutral defaults at serve time so the
        # column alignment in _align_to_model still finds them. Training-time
        # values come from data_collector.py.
        features.setdefault('three_in_four_flag', 0.0)
        features.setdefault(
            'minutes_last3_avg',
            float(features.get('recent_minutes', 24.0) or 24.0),
        )
        features.setdefault('minutes_trend_5', 0.0)
        features.setdefault('team_script_volatility_10', 0.0)
        features.setdefault('garbage_time_pct_5', 0.0)

        # B2 lite — rotation-disruption proxies. Serve-time defaults assume
        # "no recent role change": jumps == 1.0 (recent==baseline), zero
        # volatility, zero outlier share. These are the values an
        # equilibrium-state player would show. Training-time values come
        # from src/data_collector.py.
        features.setdefault('minutes_jump_3v10', 1.0)
        features.setdefault('usage_jump_3v10', 1.0)
        features.setdefault('minutes_volatility_10', 0.0)
        features.setdefault('outlier_minutes_share_10', 0.0)

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

            # blend with trained ML models when available. Defaults so
            # PI fields are always present in the response (None when the
            # regressor isn't a QuantileEnsemble or ML inference failed).
            pi80_lower = None
            pi80_upper = None
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
                    pi80_lower = None
                    pi80_upper = None
                    quantile_prob_over = None
                    if _reg is not None:
                        scaler_estimator = _scaler if hasattr(_scaler, 'feature_names_in_') else _reg
                        features_aligned = _align_to_model(features_df, scaler_estimator)
                        features_aligned = features_aligned.fillna(0.0)
                        features_scaled = _scaler.transform(features_aligned)
                        ml_pred = float(_reg.predict(features_scaled)[0])
                        # If the regressor is a QuantileEnsemble, surface the
                        # conformal-calibrated 80% PI and the quantile-derived
                        # P(over). The latter is calibrated by construction
                        # (interpolates the empirical CDF at three knots) and
                        # is a strictly better signal than 1 - Φ(z) on heavy-
                        # tailed distributions like 3PM and blocks.
                        if hasattr(_reg, "predict_intervals") and hasattr(_reg, "prob_over"):
                            try:
                                _intv = _reg.predict_intervals(features_scaled)
                                pi80_lower = float(_intv["lower"][0])
                                pi80_upper = float(_intv["upper"][0])
                                quantile_prob_over = float(
                                    _reg.prob_over(features_scaled, line)[0]
                                )
                            except Exception:
                                pi80_lower = pi80_upper = quantile_prob_over = None
                    else:
                        # Binary props (double_double, triple_double) — no regression model
                        ml_pred = stat_predicted_value

                    # classifier may expect additional features (e.g. 'line')
                    clf_features = _align_to_model(features_df, _clf, extra={'line': line})
                    clf_features = clf_features.fillna(0.0)
                    ml_prob = float(_clf.predict_proba(clf_features)[0, 1])

                    # 25/75 blend: small stats anchor + ML dominant.
                    # If we have a quantile-based prob_over, swap it in for
                    # the gaussian-CDF approximation — it accounts for
                    # asymmetric and heavy-tailed result distributions.
                    predicted_value = 0.25 * stat_predicted_value + 0.75 * ml_pred
                    if quantile_prob_over is not None:
                        regressor_prob = quantile_prob_over
                    else:
                        blended_z = (line - predicted_value) / (std_dev + 1e-6)
                        regressor_prob = 1 - scipy.stats.norm.cdf(blended_z)
                    over_prob = 0.25 * regressor_prob + 0.75 * ml_prob
                except Exception as e:
                    print(f"ML inference failed, falling back to stats: {e}")
                    pi80_lower = pi80_upper = None
            # when models aren't trained yet, the stat-only baseline is already set above — nothing more to do

            edge = ((predicted_value - line) / line) if line > 0 else 0

            prob_strength = abs(over_prob - 0.5)
            edge_strength = abs(edge)
            confidence = self._calculate_confidence(prob_strength, edge_strength)

            over_prob = max(0.0, min(1.0, over_prob))  # clamp to valid probability range
            recommendation = self._generate_recommendation(over_prob, predicted_value, line, edge, confidence)

            # Calibration warning: surface when the model that produced this
            # prediction was flagged as poorly calibrated at training time
            # (ECE > 0.07). Per-prop flag wins if we used a per-prop model;
            # otherwise we fall back to the global flag.
            if use_prop_model:
                _calibration_warning = bool(self._needs_recal.get(prop_type, False))
            else:
                _calibration_warning = bool(self._global_needs_recal)

            return {
                'over_probability': float(over_prob),
                'predicted_value': float(predicted_value),
                'recommendation': recommendation,
                'confidence': confidence,
                'edge': float(edge),
                'model_calibration_warning': _calibration_warning,
                # Conformal-calibrated 80% prediction interval. ``None``
                # when the regressor isn't a QuantileEnsemble (e.g. legacy
                # GradientBoostingRegressor was loaded from disk).
                'pi80_lower': pi80_lower,
                'pi80_upper': pi80_upper,
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

    # ────────────────────────────────────────────────────────────────────
    # Calibration audit helper
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _reliability_bins(probs, y_true, n_bins=10):
        """Per-bin (mean_pred, mean_actual, n) for a reliability diagram.

        A perfectly calibrated model has mean_pred ≈ mean_actual in every
        non-empty bin. Large gaps in any bin → that probability range is
        miscalibrated and is a candidate for isotonic re-fit (item C2).

        Returns a list of dicts so it serialises cleanly to JSON.
        """
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y_true, dtype=float)
        if probs.size == 0:
            return []
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        out = []
        for k in range(n_bins):
            lo, hi = edges[k], edges[k + 1]
            mask = (probs >= lo) & (probs < hi if k < n_bins - 1 else probs <= hi)
            n = int(mask.sum())
            if n == 0:
                continue
            out.append({
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": n,
                "mean_pred": float(probs[mask].mean()),
                "mean_actual": float(y[mask].mean()),
            })
        return out

    @staticmethod
    def _expected_calibration_error(reliability_bins):
        """Sample-weighted L1 gap between predicted and observed rates (ECE).

        ECE = Σ_b (n_b / N) · |mean_pred_b - mean_actual_b|

        Lower is better. <0.03 is well-calibrated for prop markets; >0.07 is
        a red flag that suggests an isotonic refit on the calibration split
        would help (item C2). The flag is advisory — it's written into
        model_metadata.json so the training pipeline can be re-run with
        ``method='isotonic'`` next time around without changing the train()
        signature.
        """
        if not reliability_bins:
            return None
        total = sum(b["n"] for b in reliability_bins)
        if total <= 0:
            return None
        return float(
            sum(
                (b["n"] / total) * abs(b["mean_pred"] - b["mean_actual"])
                for b in reliability_bins
            )
        )

    # ────────────────────────────────────────────────────────────────────
    # Training helpers — sample weighting + temporal split
    # ────────────────────────────────────────────────────────────────────
    def _compute_recency_weights(self, timestamps, halflife_days=None):
        """Exponential-decay sample weights based on game age.

        Weight at age=0 is 1.0; at age=halflife_days it's 0.5; at 2× halflife it's
        0.25; etc. Why decay rather than uniform: the league is non-stationary
        (rule changes, role shifts, pace drift), so a 2021 game is *less*
        informative than a 2025 game about how the player will perform tomorrow.
        Forcing the model to ignore recency wastes signal in the recent tail.

        Falls back to uniform weights if any timestamp is missing — the OOT
        split is the more important guarantee, weighting is the boost on top.
        """
        if halflife_days is None:
            halflife_days = float(os.environ.get("RECENCY_HALFLIFE_DAYS", 365.0))
        if halflife_days <= 0:
            return np.ones(len(timestamps))
        parsed = []
        for t in timestamps:
            if t is None:
                return np.ones(len(timestamps))
            try:
                parsed.append(parse_iso(t))
            except Exception:
                return np.ones(len(timestamps))
        if not parsed:
            return np.ones(0)
        ref = max(parsed)
        ages_days = np.array([
            max(0.0, (ref - t).total_seconds() / 86400.0) for t in parsed
        ])
        return np.power(0.5, ages_days / float(halflife_days))

    def _temporal_split_indices(self, timestamps, n, holdout_days=None):
        """Return (train_idx, test_idx, mode_str). OOT if all timestamps are
        present and produce both non-empty sides; random fallback otherwise.

        We ALWAYS need both halves to be non-empty. If the OOT holdout would
        produce an empty train set (rare — happens with synthetic test data
        where every row has the same timestamp), we fall back to random.
        """
        if holdout_days is None:
            holdout_days = int(os.environ.get("OOT_HOLDOUT_DAYS", 30))
        if all(t is not None for t in timestamps):
            try:
                train_mask, oot_mask = out_of_time_split(
                    timestamps, holdout_days=holdout_days
                )
                if train_mask.any() and oot_mask.any():
                    idx = np.arange(n)
                    return idx[train_mask], idx[oot_mask], "oot"
            except Exception:
                pass
        # Fallback: random 80/20 with a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        cut = int(0.8 * n)
        return perm[:cut], perm[cut:], "random"

    @staticmethod
    def _conformal_split(idx, weights, rng_seed=43, cal_frac=0.2):
        """Independent 80/20 split for regressor fit + conformal-calibration.

        Why a separate split from the classifier's fit/cal/blend: conformal
        calibration is a property of the *regressor*, not the classifier.
        Sharing the classifier's cal or blend split would mean the regressor
        sees its own conformal-calibration set during fit, which destroys
        the held-out guarantee. A different RNG seed (43, not 42) keeps it
        decorrelated from the classifier splits but still deterministic.

        Returns ``(fit_idx, cal_idx, w_fit, w_cal)``.
        """
        rng = np.random.default_rng(rng_seed)
        shuffled = rng.permutation(np.asarray(idx))
        n = len(shuffled)
        cut = max(1, int((1.0 - cal_frac) * n))
        fit_i, cal_i = shuffled[:cut], shuffled[cut:]
        return fit_i, cal_i, weights[fit_i], weights[cal_i]

    @staticmethod
    def _three_way_split(idx, weights, rng_seed=42):
        """Split indices 60/20/20 into (fit, cal, blend) with stable shuffling.

        Why three slices: we need disjoint data for (a) fitting base models,
        (b) calibrating their probability outputs, and (c) training the
        stacking blender on the calibrated outputs. Any overlap leaks
        optimism into downstream metrics.
        """
        rng = np.random.default_rng(rng_seed)
        shuffled = rng.permutation(idx)
        n = len(shuffled)
        c1 = int(0.6 * n)
        c2 = int(0.8 * n)
        i_fit, i_cal, i_blend = shuffled[:c1], shuffled[c1:c2], shuffled[c2:]
        return i_fit, i_cal, i_blend, weights[i_fit], weights[i_cal], weights[i_blend]

    # Sign of expected influence on the target. ``+1`` means "more of this
    # feature should not decrease the prediction"; ``-1`` is the inverse.
    # Anything not listed is unconstrained (0). Why we bother: HGB will
    # happily fit a non-monotonic spline on a noisy small-sample feature
    # (e.g. blocks with 4k rows) and produce a learned "more recent_avg →
    # lower prediction" curve over a tiny range. Constraining the sign
    # closes that failure mode at near-zero training cost.
    #
    # ``_REG_MONOTONIC`` applies to the regressor (predicts the stat value).
    # ``_CLF_MONOTONIC`` applies to the classifier (predicts P(stat > line))
    # and includes ``line`` itself, which the classifier sees as a feature.
    _REG_MONOTONIC = {
        "recent_avg": +1, "season_avg": +1,
        "minutes_last3_avg": +1, "minutes_trend_5": +1,
        "recent_minutes": +1, "season_minutes": +1, "avg_minutes": +1,
        "pace_team": +1, "pace_opp": +1,
        "is_back_to_back": -1, "three_in_four_flag": -1,
    }
    _CLF_MONOTONIC = {
        **_REG_MONOTONIC,
        "line": -1,  # higher line => lower P(over) all else equal
    }

    @staticmethod
    def _build_monotonic_cst(feature_names, rules):
        """Map a feature-name → sign dict to the integer array HGB expects.

        Returns ``None`` if no constraints apply (avoids HGB doing extra
        bookkeeping for an all-zeros vector). Features unknown to ``rules``
        get 0 (unconstrained).
        """
        if feature_names is None:
            return None
        arr = np.array(
            [int(rules.get(str(f), 0)) for f in feature_names], dtype=int
        )
        if not np.any(arr):
            return None
        return arr

    @staticmethod
    def _adaptive_depth(n_samples: int) -> int:
        """Heuristic max_depth that scales with training set size.

        With shared depth=5 across props, blocks/steals (≈ 5k rows) overfit
        and points/rebounds (≈ 50k rows) underfit. The bins below produced
        the cleanest Brier curves on synthetic-then-real ablations:

          n < 1000  → depth=3   (thin, must regularize hard)
          n < 5000  → depth=4
          n < 20000 → depth=5
          n ≥ 20000 → depth=7

        Combined with early-stopping ``max_iter``, this self-tunes both
        depth and iteration count without an outer grid search.
        """
        if n_samples < 1000:
            return 3
        if n_samples < 5000:
            return 4
        if n_samples < 20000:
            return 5
        return 7

    def _make_alt_classifier(self, monotonic_cst=None, n_samples: int | None = None):
        """Second base classifier for the stacking ensemble.

        Different algorithm family (HGBC vs GBC) → different bias-variance
        profile → blender has something to actually choose between. Same
        family with different hyperparameters tends to produce highly
        correlated probabilities; the blender then trivially picks one.

        ``n_samples`` controls the sample-size-adaptive ``max_depth``;
        defaults to a conservative depth=5 if unknown.
        """
        depth = self._adaptive_depth(n_samples) if n_samples is not None else 5
        kwargs = dict(
            learning_rate=0.05,
            # Upper bound; early_stopping picks the real number of trees
            max_iter=500,
            max_depth=depth,
            random_state=43,  # deliberately != 42 so it sees a different fit
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
        )
        if monotonic_cst is not None:
            kwargs["monotonic_cst"] = monotonic_cst
        return HistGradientBoostingClassifier(**kwargs)

    def train(self, training_data):
        """Train classifier + regressor + stacking blender, save artefacts.

        Pipeline (per global and per-prop loops):
          1. Build X, y_class, y_reg, timestamps from training_data.
          2. Temporal split → train_idx / test_idx (OOT preferred).
          3. Recency-weighted sample weights from timestamps.
          4. Three-way split inside train: fit / cal / blend (60/20/20).
          5. Fit base A (GBC) + base B (HGBC) on fit split.
          6. Calibrate each on cal split.
          7. Fit StackingBlender on blend split using calibrated probs.
          8. Wrap (base_a_cal, base_b_cal, blender) into StackedCalibratedClassifier.
          9. Fit QuantileEnsemble regressor on the FULL train set.
         10. Compute test-set Brier + reliability + feature importances.
         11. Save models + write metadata for drift monitoring.
        """
        if not training_data:
            raise ValueError("No training data provided")

        # build feature matrix and both target arrays
        X = pd.DataFrame([data['features'] for data in training_data])
        X = X.fillna(0)  # guard against NaN from mismatched feature sets across sample sources
        y_class = np.array([1 if d['result'] > d['line'] else 0 for d in training_data])
        y_reg = np.array([d['result'] for d in training_data], dtype=float)
        timestamps = [d.get('timestamp') for d in training_data]

        # ── A2: temporal split (OOT preferred, random fallback) ──────────
        train_idx, test_idx, split_mode = self._temporal_split_indices(
            timestamps, len(X)
        )
        print(f"split mode: {split_mode}  train={len(train_idx)}  test={len(test_idx)}")

        # ── A3: recency-weighted sample weights ──────────────────────────
        weights_all = self._compute_recency_weights(timestamps)

        # Apply both index splits
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_class_train, y_class_test = y_class[train_idx], y_class[test_idx]
        y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]
        w_train, w_test = weights_all[train_idx], weights_all[test_idx]

        # Three-way split inside train: fit / cal / blend
        (i_fit, i_cal, i_blend,
         w_fit, w_cal, w_blend) = self._three_way_split(
            np.arange(len(X_train)), w_train,
        )
        X_fit = X_train.iloc[i_fit]
        X_cal = X_train.iloc[i_cal]
        X_blend = X_train.iloc[i_blend]
        y_fit = y_class_train[i_fit]
        y_cal = y_class_train[i_cal]
        y_blend = y_class_train[i_blend]

        # ── Scaler (fit on the fit split only, never on test) ────────────
        X_fit_scaled = self.scaler.fit_transform(X_fit)
        X_cal_scaled = self.scaler.transform(X_cal)
        X_blend_scaled = self.scaler.transform(X_blend)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_scaled = self.scaler.transform(X_train)

        # ── A5: two base classifiers + calibration + blender ────────────
        # Base A: existing GradientBoostingClassifier
        self.classification_model.fit(X_fit_scaled, y_fit, sample_weight=w_fit)
        cal_a = self._make_calibrated_clf(self.classification_model, len(X_cal))
        cal_a.fit(X_cal_scaled, y_cal, sample_weight=w_cal)

        # Base B: HistGradientBoostingClassifier (different family).
        # Apply monotonic constraints (line ↓, recent_avg ↑, etc.) so the
        # tree splits can't fit the wrong sign on noisy small-sample features.
        clf_feat_names = (
            list(self.scaler.feature_names_in_)
            if hasattr(self.scaler, "feature_names_in_") else None
        )
        clf_mono = self._build_monotonic_cst(clf_feat_names, self._CLF_MONOTONIC)
        base_b = self._make_alt_classifier(
            monotonic_cst=clf_mono, n_samples=len(X_fit),
        )
        base_b.fit(X_fit_scaled, y_fit, sample_weight=w_fit)
        cal_b = self._make_calibrated_clf(base_b, len(X_cal))
        cal_b.fit(X_cal_scaled, y_cal, sample_weight=w_cal)

        # Train the blender on disjoint blend split using calibrated probs
        try:
            p_a_blend = cal_a.predict_proba(X_blend_scaled)[:, 1]
            p_b_blend = cal_b.predict_proba(X_blend_scaled)[:, 1]
            base_probs_blend = np.column_stack([p_a_blend, p_b_blend])
            if len(np.unique(y_blend)) >= 2:
                blender = StackingBlender(
                    base_names=["gbc_cal", "hgbc_cal"]
                ).fit(base_probs_blend, y_blend, sample_weight=w_blend)
                stacked_clf = StackedCalibratedClassifier(cal_a, cal_b, blender)
            else:
                # Blend split is single-class — degenerate case; fall back to base A
                print("blender fit skipped: blend split has only one class")
                stacked_clf = cal_a
        except Exception as e:
            print(f"blender training failed, falling back to single classifier: {e}")
            stacked_clf = cal_a

        # Inner classifiers were fit on scaled numpy arrays so they don't carry
        # feature_names_in_. Backfill from the scaler so downstream column-
        # alignment code in inference still works (see _align_to_model).
        if hasattr(self.scaler, "feature_names_in_"):
            stacked_clf.feature_names_in_ = np.array(list(self.scaler.feature_names_in_))

        # Test metrics on the held-out test split
        test_probs = stacked_clf.predict_proba(X_test_scaled)[:, 1]
        class_auc = roc_auc_score(y_class_test, test_probs) if len(np.unique(y_class_test)) >= 2 else float('nan')
        # Brier with sample weights so recent test points dominate the measure
        try:
            class_brier = float(brier_score_loss(y_class_test, test_probs, sample_weight=w_test))
        except Exception:
            class_brier = float('nan')

        # ── A4: QuantileEnsemble regressor ───────────────────────────────
        # Same idea as the HGB classifier: enforce that recent_avg etc. push
        # the predicted stat *up*, not down. Note ``line`` isn't a feature
        # for the regressor (it predicts the stat itself), so use the
        # _REG_MONOTONIC table.
        reg_mono = self._build_monotonic_cst(clf_feat_names, self._REG_MONOTONIC)
        # Independent 80/20 split for fit + conformal calibration. CQR
        # turns "soft" quantile-regression bands into bands with a
        # provable marginal coverage guarantee on the calibration split.
        rfi, rci, w_rfi, w_rci = self._conformal_split(
            np.arange(len(X_train_scaled)), w_train,
        )
        quantile_reg = QuantileEnsemble(
            monotonic_cst=reg_mono,
            max_depth=self._adaptive_depth(len(rfi)),
        ).fit(
            X_train_scaled[rfi], y_reg_train[rfi], sample_weight=w_rfi,
        )
        try:
            quantile_reg.calibrate(X_train_scaled[rci], y_reg_train[rci])
        except Exception as e:  # noqa: BLE001
            print(f"conformal calibration skipped (global): {e}")
        # Wrap into a thin estimator that exposes feature_names_in_ matching
        # the scaler so the inference-side _align_to_model works unchanged
        if hasattr(self.scaler, "feature_names_in_"):
            quantile_reg.feature_names_in_ = np.array(list(self.scaler.feature_names_in_))
        reg_predictions = quantile_reg.predict(X_test_scaled)
        reg_rmse = float(np.sqrt(mean_squared_error(y_reg_test, reg_predictions)))
        # Interval coverage at the configured (lo, hi) — sanity-check on calibration
        intervals = quantile_reg.predict_intervals(X_test_scaled)
        coverage_80 = float(np.mean(
            (y_reg_test >= intervals["lower"]) & (y_reg_test <= intervals["upper"])
        ))

        # ── C2: global reliability bins + Expected Calibration Error audit ──
        # An ECE above ~0.07 is our advisory threshold to refit calibration with
        # ``method='isotonic'`` next round. We just record the flag in metadata;
        # the operator (or CI) can read it and re-train accordingly.
        global_reliability = self._reliability_bins(test_probs, y_class_test)
        global_ece = self._expected_calibration_error(global_reliability)
        global_needs_recal = bool(global_ece is not None and global_ece > 0.07)

        print(f"Classification AUC: {class_auc:.3f}  Brier: {class_brier:.4f}")
        print(f"Regression RMSE (median): {reg_rmse:.3f}  PI80 coverage: {coverage_80:.3f}")
        if global_ece is not None:
            print(
                f"Calibration ECE: {global_ece:.4f}"
                + ("  ⚠ recommend isotonic refit" if global_needs_recal else "")
            )

        # Replace globals with the new artefacts (downstream code keeps working
        # because StackedCalibratedClassifier mimics the sklearn classifier API)
        self.classification_model = stacked_clf
        self.regression_model = quantile_reg

        joblib.dump(self.classification_model, f'{self.model_dir}/classification_model.joblib')
        joblib.dump(self.regression_model, f'{self.model_dir}/regression_model.joblib')
        joblib.dump(self.scaler, f'{self.model_dir}/scaler.joblib')
        self.models_trained = True

        # ── A7: feature importances (HGB-native + permutation on test) ──
        # HGB's split-gain importance is fast but biased toward high-cardinality
        # features; permutation importance is slow but is the right answer for
        # "what does the model actually rely on at predict time." We dump both
        # and let the operator compare.
        feat_importance_dump = {}
        try:
            hgb_importance = getattr(base_b, "feature_importances_", None)
            if hgb_importance is None and hasattr(base_b, "_predictors"):
                # HistGradientBoostingClassifier doesn't expose feature_importances_
                # directly — skip the native one.
                hgb_importance = None
            if hgb_importance is not None:
                feat_importance_dump["hgbc_native"] = {
                    str(c): float(v) for c, v in zip(X.columns, hgb_importance)
                }
        except Exception:
            pass
        try:
            # Permutation importance on a sample of test rows for speed
            sample_n = min(500, len(X_test_scaled))
            if sample_n >= 30:
                perm = permutation_importance(
                    base_b, X_test_scaled[:sample_n], y_class_test[:sample_n],
                    n_repeats=3, random_state=42, n_jobs=1,
                )
                feat_importance_dump["permutation"] = {
                    str(c): float(v) for c, v in zip(X.columns, perm.importances_mean)
                }
        except Exception as e:
            print(f"permutation importance skipped: {e}")

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
                yp_class = np.array([1 if s['result'] > s['line'] else 0 for s in samples])
                yp_reg = np.array([s['result'] for s in samples], dtype=float)
                p_timestamps = [s.get('timestamp') for s in samples]

                # Apply the same OOT split + recency weighting + 3-way fit/cal/blend
                # treatment as the global model. Per-prop sample counts are smaller,
                # so the OOT holdout window must clip down for sparse props.
                p_train_idx, p_test_idx, p_split_mode = self._temporal_split_indices(
                    p_timestamps, len(Xp),
                )
                p_weights_all = self._compute_recency_weights(p_timestamps)

                Xp_train = Xp.iloc[p_train_idx].reset_index(drop=True)
                Xp_test = Xp.iloc[p_test_idx].reset_index(drop=True)
                ypc_train, ypc_test = yp_class[p_train_idx], yp_class[p_test_idx]
                ypr_train, ypr_test = yp_reg[p_train_idx], yp_reg[p_test_idx]
                wp_train, wp_test = p_weights_all[p_train_idx], p_weights_all[p_test_idx]

                # If a prop's blend split would be tiny (< 10 rows), skip stacking
                # for that prop and fall back to a single calibrated classifier.
                use_stacking = len(Xp_train) >= 80

                prop_scaler = StandardScaler()
                if use_stacking:
                    (pi_fit, pi_cal, pi_blend,
                     wp_fit, wp_cal, wp_blend) = self._three_way_split(
                        np.arange(len(Xp_train)), wp_train,
                    )
                else:
                    # 80/20 fit/cal — no blender
                    cut = int(0.8 * len(Xp_train))
                    rng = np.random.default_rng(42)
                    perm = rng.permutation(len(Xp_train))
                    pi_fit, pi_cal = perm[:cut], perm[cut:]
                    wp_fit, wp_cal = wp_train[pi_fit], wp_train[pi_cal]

                Xp_fit_s = prop_scaler.fit_transform(Xp_train.iloc[pi_fit])
                Xp_cal_s = prop_scaler.transform(Xp_train.iloc[pi_cal])
                Xp_test_s = prop_scaler.transform(Xp_test)
                Xp_train_s_full = prop_scaler.transform(Xp_train)
                ypc_fit, ypc_cal = ypc_train[pi_fit], ypc_train[pi_cal]

                # Per-prop monotonic constraint vector (uses the prop scaler's
                # feature names, which may differ from the global scaler's set
                # if some features dropped out for this prop)
                prop_feat_names = (
                    list(prop_scaler.feature_names_in_)
                    if hasattr(prop_scaler, "feature_names_in_") else None
                )
                prop_clf_mono = self._build_monotonic_cst(
                    prop_feat_names, self._CLF_MONOTONIC
                )
                prop_reg_mono = self._build_monotonic_cst(
                    prop_feat_names, self._REG_MONOTONIC
                )

                # Base A
                prop_clf_a = GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
                )
                prop_clf_a.fit(Xp_fit_s, ypc_fit, sample_weight=wp_fit)
                prop_cal_a = self._make_calibrated_clf(prop_clf_a, len(pi_cal))
                prop_cal_a.fit(Xp_cal_s, ypc_cal, sample_weight=wp_cal)

                if use_stacking and len(np.unique(ypc_train[pi_blend])) >= 2:
                    Xp_blend_s = prop_scaler.transform(Xp_train.iloc[pi_blend])
                    ypc_blend = ypc_train[pi_blend]
                    # Base B (monotonic-constrained, depth scales with prop size)
                    prop_clf_b = self._make_alt_classifier(
                        monotonic_cst=prop_clf_mono, n_samples=len(pi_fit),
                    )
                    prop_clf_b.fit(Xp_fit_s, ypc_fit, sample_weight=wp_fit)
                    prop_cal_b = self._make_calibrated_clf(prop_clf_b, len(pi_cal))
                    prop_cal_b.fit(Xp_cal_s, ypc_cal, sample_weight=wp_cal)
                    # Blender
                    bp_blend = np.column_stack([
                        prop_cal_a.predict_proba(Xp_blend_s)[:, 1],
                        prop_cal_b.predict_proba(Xp_blend_s)[:, 1],
                    ])
                    prop_blender = StackingBlender(
                        base_names=["gbc_cal", "hgbc_cal"]
                    ).fit(bp_blend, ypc_blend, sample_weight=wp_blend)
                    prop_stacked = StackedCalibratedClassifier(
                        prop_cal_a, prop_cal_b, prop_blender,
                    )
                else:
                    prop_stacked = prop_cal_a

                # Backfill feature names from scaler (inner classifiers were fit
                # on scaled numpy arrays and don't carry them otherwise)
                if hasattr(prop_scaler, "feature_names_in_"):
                    prop_stacked.feature_names_in_ = np.array(
                        list(prop_scaler.feature_names_in_)
                    )

                # Quantile regressor: 80/20 fit + conformal split, monotonic-constrained,
                # depth scales with this prop's sample size.
                p_rfi, p_rci, w_p_rfi, w_p_rci = self._conformal_split(
                    np.arange(len(Xp_train_s_full)), wp_train,
                )
                prop_reg = QuantileEnsemble(
                    monotonic_cst=prop_reg_mono,
                    max_depth=self._adaptive_depth(len(p_rfi)),
                ).fit(
                    Xp_train_s_full[p_rfi], ypr_train[p_rfi], sample_weight=w_p_rfi,
                )
                try:
                    prop_reg.calibrate(
                        Xp_train_s_full[p_rci], ypr_train[p_rci],
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"conformal calibration skipped ({prop_type}): {e}")
                if hasattr(prop_scaler, "feature_names_in_"):
                    prop_reg.feature_names_in_ = np.array(list(prop_scaler.feature_names_in_))

                # Test metrics
                test_probs_p = prop_stacked.predict_proba(Xp_test_s)[:, 1]
                prop_auc = float(
                    roc_auc_score(ypc_test, test_probs_p)
                    if len(np.unique(ypc_test)) >= 2 else float('nan')
                )
                try:
                    prop_brier = float(brier_score_loss(ypc_test, test_probs_p, sample_weight=wp_test))
                except Exception:
                    prop_brier = float('nan')
                prop_rmse = float(np.sqrt(
                    mean_squared_error(ypr_test, prop_reg.predict(Xp_test_s))
                ))
                # ── C1/C2: reliability bins + ECE audit per prop ──
                prop_reliability = self._reliability_bins(test_probs_p, ypc_test)
                prop_ece = self._expected_calibration_error(prop_reliability)
                prop_needs_recal = bool(prop_ece is not None and prop_ece > 0.07)
                # Residual std for anomaly detection in monitoring
                resid_std = float(np.std(ypr_test - prop_reg.predict(Xp_test_s)))

                print(f"[{prop_type}] AUC: {prop_auc:.3f}, Brier: {prop_brier:.4f}, "
                      f"RMSE: {prop_rmse:.3f}  ({p_split_mode} split, "
                      f"stacked={use_stacking})")

                joblib.dump(prop_stacked, os.path.join(self.model_dir, f'clf_cal_{prop_type}.joblib'))
                joblib.dump(prop_reg, os.path.join(self.model_dir, f'reg_{prop_type}.joblib'))
                joblib.dump(prop_scaler, os.path.join(self.model_dir, f'scaler_{prop_type}.joblib'))

                self.prop_models[prop_type] = {
                    'calibrated_clf': prop_stacked,
                    'regression_model': prop_reg,
                    'scaler': prop_scaler,
                }
                prop_results[prop_type] = {
                    'auc': prop_auc,
                    'rmse': prop_rmse,
                    'training_brier': prop_brier,
                    'resid_std': resid_std,
                    'split_mode': p_split_mode,
                    'n_train': int(len(Xp_train)),
                    'n_test': int(len(Xp_test)),
                    'stacked': bool(use_stacking),
                    'reliability': prop_reliability,
                    'ece': prop_ece,
                    'needs_recal': prop_needs_recal,
                }

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

        # ── A6: write per-prop training Brier + reliability + importances ──
        # The drift endpoint (/healthz/drift) reads training_brier from this
        # file to detect decay. Without it, the endpoint silently no-ops.
        # We also extend each prop dict with the reliability bins so the
        # operator can spot mis-calibration without re-running anything.
        try:
            metadata_path = os.path.join(self.model_dir, "model_metadata.json")
            try:
                with open(metadata_path, "r") as _mf:
                    metadata = json.load(_mf)
            except (FileNotFoundError, json.JSONDecodeError):
                metadata = {"schema": 1, "props": {}}
            metadata.setdefault("props", {})
            for prop_type, info in prop_results.items():
                # Merge into existing per-prop metadata; don't blow away
                # historic fields like feature_count
                metadata["props"].setdefault(prop_type, {})
                for k, v in info.items():
                    metadata["props"][prop_type][k] = v
            metadata["global"] = {
                "auc": float(class_auc) if not np.isnan(class_auc) else None,
                "training_brier": float(class_brier) if not np.isnan(class_brier) else None,
                "rmse": float(reg_rmse),
                "pi80_coverage": float(coverage_80),
                "split_mode": split_mode,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "halflife_days": float(os.environ.get("RECENCY_HALFLIFE_DAYS", 365.0)),
                "model_version": self._model_version,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "reliability": global_reliability,
                "ece": global_ece,
                "needs_recal": global_needs_recal,
            }
            if feat_importance_dump:
                metadata["feature_importance"] = feat_importance_dump
            with open(metadata_path, "w") as _mf:
                json.dump(metadata, _mf, indent=2, default=float)
            print(f"wrote training metadata to {metadata_path}")
        except Exception as e:  # never let metadata I/O fail the training run
            print(f"failed to write model_metadata.json: {e}")

        return {
            'auc': float(class_auc) if not np.isnan(class_auc) else None,
            'rmse': float(reg_rmse),
            'training_brier': float(class_brier) if not np.isnan(class_brier) else None,
            'pi80_coverage': float(coverage_80),
            'split_mode': split_mode,
            'prop_results': prop_results,
            'model_version': self._model_version,
        }

    # ────────────────────────────────────────────────────────────────────
    # Walk-forward backtest — honest performance distribution across time
    # ────────────────────────────────────────────────────────────────────
    def walk_forward_evaluate(
        self,
        training_data,
        n_folds: int = 5,
        fold_days: int = 30,
        min_train_samples: int = 200,
        write_metadata: bool = True,
    ) -> dict:
        """K-fold rolling-origin evaluation.

        A single OOT split is one sample of generalization performance and
        can mislead — a lucky test window can hide overfitting, an unlucky
        one can panic you off a good model. Walk-forward gives a small
        distribution: train through time t, test on (t, t+fold_days],
        then slide t forward and repeat.

        Parameters
        ----------
        training_data : list[dict]
            Same shape as ``train()`` input. Must include ``timestamp``
            on each row — without timestamps, walk-forward is meaningless.
        n_folds : int
            How many rolling test windows to evaluate. The most recent
            ``fold_days`` is fold 0, the next-most-recent is fold 1, etc.
        fold_days : int
            Length of each test window in days.
        min_train_samples : int
            Skip a fold if it has fewer than this many train samples
            (the model would be too thin to compare meaningfully).
        write_metadata : bool
            If True, persist per-prop walk-forward aggregates into
            ``model_metadata.json`` under ``props.<prop>.walk_forward``
            so ``/healthz/drift`` can read ``brier_cal_mean`` from there.

        Returns
        -------
        dict with keys:
          - ``folds``: per-fold metrics (auc, brier, rmse, pi80_coverage,
            n_train, n_test, test_start, test_end)
          - ``aggregates``: mean + std across folds
          - ``per_prop``: same shape, per prop_type
        """
        import tempfile
        from datetime import timedelta

        # Filter & sort by timestamp
        with_ts = []
        for d in training_data:
            t = d.get("timestamp")
            if t is None:
                continue
            try:
                with_ts.append((parse_iso(t), d))
            except Exception:
                continue
        if len(with_ts) < n_folds * min_train_samples:
            raise ValueError(
                f"walk_forward needs ≥ {n_folds * min_train_samples} timestamped "
                f"samples, got {len(with_ts)}"
            )
        with_ts.sort(key=lambda r: r[0])
        sorted_ts = [r[0] for r in with_ts]
        sorted_data = [r[1] for r in with_ts]
        max_t = sorted_ts[-1]

        fold_results = []
        per_prop: dict[str, list] = {}

        for i in range(n_folds):
            test_end = max_t - timedelta(days=i * fold_days)
            test_start = test_end - timedelta(days=fold_days)
            train_data = [
                d for ts, d in zip(sorted_ts, sorted_data) if ts < test_start
            ]
            test_data = [
                d for ts, d in zip(sorted_ts, sorted_data)
                if test_start <= ts < test_end
            ]
            if len(train_data) < min_train_samples or len(test_data) < 20:
                # Skip degenerate folds rather than emit nonsense metrics
                continue

            # Train a *fresh* predictor in a tmp dir so we don't pollute the
            # caller's saved artefacts. The new predictor inherits the
            # current class config (env vars for halflife etc.).
            with tempfile.TemporaryDirectory(prefix="wf_") as tmpd:
                fold_predictor = self.__class__(model_dir=tmpd)
                metrics = fold_predictor.train(train_data)
                # Score on the held-out test window using the same predict path
                test_brier_num, test_brier_den = 0.0, 0
                test_correct = []
                test_probs_list = []
                test_labels = []
                test_resids = []
                test_inside_pi80 = []
                per_prop_local: dict[str, dict] = {}
                for d in test_data:
                    feat = d.get("features") or {}
                    line = float(d.get("line") or 0.0)
                    result = float(d.get("result") or 0.0)
                    pt = d.get("prop_type")
                    pred = fold_predictor.predict(feat, line=line, prop_type=pt)
                    p = float(pred.get("over_probability", 0.5))
                    yhat = float(pred.get("predicted_value", line))
                    y = 1 if result > line else 0
                    test_probs_list.append(p)
                    test_labels.append(y)
                    test_brier_num += (p - y) ** 2
                    test_brier_den += 1
                    test_resids.append(result - yhat)
                    pp = per_prop_local.setdefault(
                        pt or "_global",
                        {"y": [], "p": [], "resid": []},
                    )
                    pp["y"].append(y)
                    pp["p"].append(p)
                    pp["resid"].append(result - yhat)

                if test_brier_den > 0:
                    fold_brier = test_brier_num / test_brier_den
                else:
                    fold_brier = float("nan")
                fold_auc = (
                    float(roc_auc_score(test_labels, test_probs_list))
                    if len(set(test_labels)) >= 2 else float("nan")
                )
                fold_rmse = (
                    float(np.sqrt(np.mean(np.square(test_resids))))
                    if test_resids else float("nan")
                )
                fold_results.append({
                    "fold": i,
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "n_train": len(train_data),
                    "n_test": len(test_data),
                    "auc": fold_auc,
                    "brier": float(fold_brier),
                    "rmse": fold_rmse,
                    "global_train_auc": metrics.get("auc"),
                    "global_train_brier": metrics.get("training_brier"),
                })
                # Per-prop aggregation
                for pt, pp in per_prop_local.items():
                    if len(pp["y"]) < 5:
                        continue
                    bins = self._reliability_bins(
                        np.array(pp["p"]), np.array(pp["y"])
                    )
                    pp_brier = float(np.mean(
                        (np.array(pp["p"]) - np.array(pp["y"])) ** 2
                    ))
                    per_prop.setdefault(pt, []).append({
                        "fold": i,
                        "n_test": len(pp["y"]),
                        "brier": pp_brier,
                        "ece": self._expected_calibration_error(bins),
                    })

        # Aggregates
        def _agg(values):
            arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))])
            if arr.size == 0:
                return {"mean": None, "std": None, "n": 0}
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "n": int(arr.size),
            }

        aggregates = {
            "auc": _agg([f["auc"] for f in fold_results]),
            "brier": _agg([f["brier"] for f in fold_results]),
            "rmse": _agg([f["rmse"] for f in fold_results]),
            "n_folds_completed": len(fold_results),
        }
        per_prop_agg = {
            pt: {
                "brier_cal_mean": _agg([f["brier"] for f in folds])["mean"],
                "ece_mean": _agg([f["ece"] for f in folds])["mean"],
                "n_folds": len(folds),
            }
            for pt, folds in per_prop.items() if pt != "_global"
        }

        # Optionally persist per-prop walk-forward to model_metadata.json so
        # /healthz/drift can read brier_cal_mean from the established path.
        if write_metadata and per_prop_agg:
            try:
                meta_path = os.path.join(self.model_dir, "model_metadata.json")
                try:
                    with open(meta_path, "r") as _mf:
                        meta = json.load(_mf)
                except (FileNotFoundError, json.JSONDecodeError):
                    meta = {"schema": 1, "props": {}}
                meta.setdefault("props", {})
                for pt, agg in per_prop_agg.items():
                    meta["props"].setdefault(pt, {})
                    meta["props"][pt]["walk_forward"] = agg
                with open(meta_path, "w") as _mf:
                    json.dump(meta, _mf, indent=2, default=float)
            except Exception as e:  # noqa: BLE001
                print(f"walk_forward metadata write failed: {e}")

        return {
            "folds": fold_results,
            "aggregates": aggregates,
            "per_prop": per_prop_agg,
        }
