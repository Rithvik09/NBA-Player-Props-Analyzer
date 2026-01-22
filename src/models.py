from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from nba_api.stats.endpoints import TeamGameLog, CommonPlayerInfo, LeagueGameFinder, CommonTeamRoster
from nba_api.stats.endpoints import playergamelog, TeamDashboardByGeneralSplits
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams
import scipy.stats
import numpy as np
import pandas as pd
import joblib
import os
import time
import json
from typing import Any
from datetime import datetime, timedelta
from .injury_tracker import InjuryTracker
from .ml_features import build_feature_vector, build_classifier_vector, NUMERIC_FEATURE_KEYS, CLASSIFIER_EXTRA_KEYS
import os as _os
from .incremental_models import IncrementalModelManager

class EnhancedMLPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.injury_tracker = InjuryTracker()

        current_year = datetime.now().year
        current_month = datetime.now().month
        if 1 <= current_month <= 7:
            self.current_season = f"{current_year-1}-{str(current_year)[2:]}"
        else:
            self.current_season = f"{current_year}-{str(current_year+1)[2:]}"
        
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
        self.position_matchup_cache = {}
        self.team_context_cache = {}
        self.team_context_ttl_seconds = 60 * 60 * 2  # 2h

        self._league_team_dash_cache = {}
        self._league_team_dash_cached_at = 0
        self._league_team_dash_ttl_seconds = 60 * 60 * 6  # 6h

        self.model_mode = _os.environ.get("MODEL_MODE", "batch").strip().lower()
        self.incremental = IncrementalModelManager(self.model_dir)

        self.trained_regressors: dict[str, Any] = {}
        self.trained_classifiers_raw: dict[str, Any] = {}
        self.trained_classifiers_cal: dict[str, Any] = {}
        self.model_metadata: dict[str, Any] = {}
        self._load_trained_models()

    def _load_trained_models(self):
        try:
            meta_path = os.path.join(self.model_dir, "model_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.model_metadata = json.load(f)

            props = (self.model_metadata.get("props") or {}) if self.model_metadata else {}
            for prop, info in props.items():
                if info.get("type") == "regression":
                    path = os.path.join(self.model_dir, f"reg_{prop}.joblib")
                    if os.path.exists(path):
                        self.trained_regressors[prop] = joblib.load(path)

                clf_raw = os.path.join(self.model_dir, f"clf_raw_{prop}.joblib")
                clf_cal = os.path.join(self.model_dir, f"clf_cal_{prop}.joblib")
                if os.path.exists(clf_raw):
                    self.trained_classifiers_raw[prop] = joblib.load(clf_raw)
                if os.path.exists(clf_cal):
                    self.trained_classifiers_cal[prop] = joblib.load(clf_cal)
        except Exception as e:
            self.trained_regressors = {}
            self.trained_classifiers_raw = {}
            self.trained_classifiers_cal = {}
            self.model_metadata = {}

    def get_model_artifact_info(self, prop_type: str | None, model_source: str | None = None) -> dict[str, Any] | None:
        try:
            if not prop_type:
                return None
            src = str(model_source or "").strip().lower()

            paths: list[str] = []
            if src in ("trained_classifier_calibrated", "trained_classifier_raw", "trained_regressor"):
                if src == "trained_regressor":
                    paths = [os.path.join(self.model_dir, f"reg_{prop_type}.joblib")]
                elif src == "trained_classifier_raw":
                    paths = [os.path.join(self.model_dir, f"clf_raw_{prop_type}.joblib")]
                else:
                    paths = [os.path.join(self.model_dir, f"clf_cal_{prop_type}.joblib")]
            elif src == "incremental_sgd":
                p = self.incremental._paths(prop_type)  # type: ignore[attr-defined]
                paths = [p["reg"], p["clf"], p["scaler_reg"], p["scaler_clf"]]
            else:
                return None

            existing = [p for p in paths if os.path.exists(p)]
            if not existing:
                return None

            mt = max(os.path.getmtime(p) for p in existing)
            updated_at = int(mt)
            return {
                "source": model_source,
                "paths": existing,
                "updated_at": updated_at,
                "updated_at_iso": datetime.fromtimestamp(updated_at).isoformat(),
            }
        except Exception:
            return None

    def _trained_resid_std(self, prop_type: str) -> float | None:
        try:
            props = self.model_metadata.get("props", {})
            v = props.get(prop_type, {}).get("resid_std", None)
            return float(v) if v is not None else None
        except Exception:
            return None

    def _predict_with_trained_regressor(self, features: dict, line: float, prop_type: str):
        model = self.trained_regressors.get(prop_type)
        if not model:
            return None

        fv = build_feature_vector(features)
        pred = float(model.predict(fv.X)[0])
        pred = float(max(pred, 0.0))

        resid_std = self._trained_resid_std(prop_type) or 1.0
        player_std = float(features.get("stddev", 0.0) or 0.0)
        sigma = float(max(0.5, 0.75 * resid_std + 0.25 * player_std))

        z = (line - pred) / (sigma + 1e-6)
        over_prob = float(1 - scipy.stats.norm.cdf(z))

        edge = ((pred - line) / line) if line > 0 else 0.0
        prob_strength = abs(over_prob - 0.5)
        edge_strength = abs(edge)
        confidence = self._calculate_confidence(prob_strength, edge_strength)
        recommendation = self._generate_recommendation(over_prob, pred, line, edge, confidence)

        return {
            "over_probability": over_prob,
            "predicted_value": pred,
            "recommendation": recommendation,
            "confidence": confidence,
            "edge": float(edge),
            "factors": {
                "model_source": "trained_regressor",
                "trained_resid_std": float(resid_std),
                "sigma": float(sigma),
            },
        }

    def _predict_with_trained_classifier(self, features: dict, line: float, prop_type: str):
        cal = self.trained_classifiers_cal.get(prop_type)
        raw = self.trained_classifiers_raw.get(prop_type)
        if not cal and not raw:
            return None

        Xc = build_classifier_vector(features, line=float(line)).X
        model = cal or raw
        proba = float(model.predict_proba(Xc)[:, 1][0])

        if prop_type in ("double_double", "triple_double"):
            pred = float(proba)
        else:
            reg = self.trained_regressors.get(prop_type)
            if reg:
                Xr = build_feature_vector(features).X
                pred = float(max(0.0, float(reg.predict(Xr)[0])))
            else:
                pred = float(features.get("season_avg", line))

        edge = ((pred - line) / line) if line > 0 else 0.0
        prob_strength = abs(proba - 0.5)
        edge_strength = abs(edge)
        confidence = self._calculate_confidence(prob_strength, edge_strength)
        recommendation = self._generate_recommendation(proba, pred, line, edge, confidence)

        return {
            "over_probability": float(proba),
            "predicted_value": float(pred),
            "recommendation": recommendation,
            "confidence": confidence,
            "edge": float(edge),
            "factors": {
                "model_source": "trained_classifier_calibrated" if cal else "trained_classifier_raw",
            },
        }

    def _predict_with_trained_combo(self, features: dict, line: float, combo_prop: str):
        combo_map = {
            "pts_reb": ["points", "rebounds"],
            "pts_ast": ["points", "assists"],
            "ast_reb": ["assists", "rebounds"],
            "pts_ast_reb": ["points", "assists", "rebounds"],
            "stl_blk": ["steals", "blocks"],
        }
        parts = combo_map.get(combo_prop)
        if not parts:
            return None
        if any(p not in self.trained_regressors for p in parts):
            return None

        preds = []
        sigmas = []
        for p in parts:
            out = self._predict_with_trained_regressor(features, line=0.0, prop_type=p)
            if not out:
                return None
            preds.append(float(out["predicted_value"]))
            sigmas.append(float((out.get("factors") or {}).get("sigma", 1.0)))

        pred = float(sum(preds))
        sigma = float(np.sqrt(sum(s * s for s in sigmas))) if sigmas else 1.0
        z = (line - pred) / (sigma + 1e-6)
        over_prob = float(1 - scipy.stats.norm.cdf(z))

        edge = ((pred - line) / line) if line > 0 else 0.0
        prob_strength = abs(over_prob - 0.5)
        edge_strength = abs(edge)
        confidence = self._calculate_confidence(prob_strength, edge_strength)
        recommendation = self._generate_recommendation(over_prob, pred, line, edge, confidence)

        return {
            "over_probability": over_prob,
            "predicted_value": pred,
            "recommendation": recommendation,
            "confidence": confidence,
            "edge": float(edge),
            "factors": {
                "model_source": "trained_combo",
                "combo_parts": parts,
                "sigma": float(sigma),
            },
        }

    def _per_game(self, value, gp):
        try:
            v = float(value)
            g = float(gp) if gp else 0.0
            if g > 0 and v > 300:
                return v / g
            return v
        except Exception:
            return 0.0

    def _get_league_team_dash(self, season=None):
        season = season or self.current_season
        now = int(time.time())
        if self._league_team_dash_cache and (now - int(self._league_team_dash_cached_at) < self._league_team_dash_ttl_seconds):
            return self._league_team_dash_cache

        try:
            base_df = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense='Base'
            ).get_data_frames()[0]
            time.sleep(0.3)

            adv_df = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense='Advanced'
            ).get_data_frames()[0]
            time.sleep(0.3)

            misc_df = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense='Misc'
            ).get_data_frames()[0]

            def row_map(df):
                return {int(r['TEAM_ID']): r for _, r in df.iterrows()}

            base = row_map(base_df)
            adv = row_map(adv_df)
            misc = row_map(misc_df)

            team_stats = {}
            for team_id in base.keys():
                b = base.get(team_id, {})
                a = adv.get(team_id, {})
                m = misc.get(team_id, {})
                gp = float(b.get('GP', 0) or a.get('GP', 0) or m.get('GP', 0) or 0)

                team_stats[team_id] = {
                    # Advanced
                    'off_rating': float(a.get('OFF_RATING', a.get('E_OFF_RATING', 110.0)) or 110.0),
                    'def_rating': float(a.get('DEF_RATING', a.get('E_DEF_RATING', 110.0)) or 110.0),
                    'pace': float(a.get('PACE', a.get('E_PACE', 100.0)) or 100.0),

                    # Base box rates (per game)
                    'pts': self._per_game(b.get('PTS', 0), gp),
                    'fga': self._per_game(b.get('FGA', 0), gp),
                    'fg_pct': float(b.get('FG_PCT', 0.47) or 0.47),
                    'fg3a': self._per_game(b.get('FG3A', 0), gp),
                    'fg3_pct': float(b.get('FG3_PCT', 0.36) or 0.36),
                    'tov': self._per_game(b.get('TOV', 0), gp),
                    'stl': self._per_game(b.get('STL', 0), gp),
                    'blk': self._per_game(b.get('BLK', 0), gp),

                    # Misc style / transition (per game)
                    'pts_fb': self._per_game(m.get('PTS_FB', 0), gp),
                    'opp_pts_fb': self._per_game(m.get('OPP_PTS_FB', 0), gp),
                    'pts_off_tov': self._per_game(m.get('PTS_OFF_TOV', 0), gp),
                    'opp_pts_off_tov': self._per_game(m.get('OPP_PTS_OFF_TOV', 0), gp),
                    'pts_paint': self._per_game(m.get('PTS_PAINT', 0), gp),
                    'opp_pts_paint': self._per_game(m.get('OPP_PTS_PAINT', 0), gp),
                }

            # League averages for normalization / shrinkage
            def avg(field, default=0.0):
                vals = [v.get(field) for v in team_stats.values() if v.get(field) is not None]
                return float(np.mean(vals)) if vals else float(default)

            league_avgs = {
                'pts_fb': avg('pts_fb', 12.0),
                'opp_pts_fb': avg('opp_pts_fb', 12.0),
                'pts_off_tov': avg('pts_off_tov', 16.0),
                'opp_pts_off_tov': avg('opp_pts_off_tov', 16.0),
                'tov': avg('tov', 14.0),
                'stl': avg('stl', 7.0),
                'fga': avg('fga', 86.0),
                'fg_pct': avg('fg_pct', 0.47),
                'fg3a': avg('fg3a', 35.0),
                'pace': avg('pace', 100.0),
            }

            payload = {'teams': team_stats, 'league_avgs': league_avgs, 'season': season}
            self._league_team_dash_cache = payload
            self._league_team_dash_cached_at = now
            return payload

        except Exception as e:
            # If this fails, fall back to existing methods
            return {'teams': {}, 'league_avgs': {}, 'season': season}
        

    def _get_injury_history(self, player_id):
        """Analyze player's injury history from game logs"""
        try:
            # Get player's game logs for current and previous season
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
                games = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season
                ).get_data_frames()[0]
                time.sleep(0.6)
                all_games.append(games)

            if not all_games:
                return self._get_default_injury_history()

            games_df = pd.concat(all_games, ignore_index=True)
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            games_df = games_df.sort_values('GAME_DATE')

            games_df['DAYS_BETWEEN'] = games_df['GAME_DATE'].diff().dt.days

            # Identify likely injuries (gaps > 7 days)
            injury_gaps = games_df[games_df['DAYS_BETWEEN'] > 7]
            recent_injuries = []

            for _, gap in injury_gaps.iterrows():
                recent_injuries.append({
                    'date': gap['GAME_DATE'],
                    'days_missed': gap['DAYS_BETWEEN'],
                    'is_recent': (datetime.now() - gap['GAME_DATE'].to_pydatetime()).days < 60
                })

            # Calculate injury risk
            total_gaps = len(injury_gaps)
            recent_gaps = sum(1 for inj in recent_injuries if inj['is_recent'])
            total_days_missed = injury_gaps['DAYS_BETWEEN'].sum()

            if recent_gaps > 0 or total_days_missed > 30:
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
            print(f"Error getting injury history: {e}")
            return self._get_default_injury_history()

    def _get_default_injury_history(self):
        """Return default injury history when data unavailable"""
        return {
            'recent_injuries': [],
            'games_missed': 0,
            'total_days_missed': 0,
            'injury_risk': 'low'
        }

    def _get_matchup_history(self, player_id, opponent_team_id):
        """Get detailed matchup history against specific team"""
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
            print(f"Error getting matchup history: {e}")
            return None
        
    def get_position_matchup_stats(self, position, team_id, opponent_context=None):
        """
        Fast estimate of opponent defense vs a player's position.

        The previous implementation attempted to infer this by iterating across many teams/players and
        calling `LeagueGameFinder` repeatedly, which can take minutes. This version uses the opponent's
        overall context (defensive_rating + pace) and small position heuristics.
        """
        cache_key = f"{position}_{team_id}"
        if cache_key in self.position_matchup_cache:
            return self.position_matchup_cache[cache_key]
        
        try:
            ctx = opponent_context or self.get_team_context(team_id)
            def_rating = float((ctx or {}).get('defensive_rating', 110.0))
            pace = float((ctx or {}).get('pace', 100.0))

            # Bucket positions and apply a small bias
            pos = (position or '').upper()
            if 'G' in pos and 'F' not in pos:
                pos_bias = 1.0   # guards tend to score slightly more
            elif 'C' in pos and 'F' not in pos:
                pos_bias = -1.0  # centers slightly less (more role-dependent)
            else:
                pos_bias = 0.0

            # Convert defensive rating into an approximate "points allowed to position" signal.
            # Baseline roughly in the low-20s; worse defenses push this up.
            pts_allowed = 22.0 + pos_bias + (def_rating - 110.0) * 0.12
            pts_allowed = float(min(max(pts_allowed, 15.0), 30.0))

            # Effective FG% proxy: slightly lower for better defenses.
            effective_fg_pct = 0.47 - (110.0 - def_rating) * 0.001
            effective_fg_pct = float(min(max(effective_fg_pct, 0.40), 0.54))
            
                matchup_stats = {
                'pts_allowed_per_game': pts_allowed,
                'defensive_rating': float(def_rating),
                'effective_fg_pct': effective_fg_pct,
                'pace': float(pace)
            }

                self.position_matchup_cache[cache_key] = matchup_stats
                return matchup_stats
                
        except Exception:
            return self._get_default_position_matchup()
        
    def _get_default_position_matchup(self):
        """Return default position matchup stats"""
        return {
            'pts_allowed_per_game': 15.0,
            'defensive_rating': 110.0,
            'effective_fg_pct': 0.47,
            'pace': 100.0
        }

    def get_player_context(self, player_id, opponent_team_id, opponent_context=None):
        """Get comprehensive player context including injuries and matchups"""
        try:
            player_info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            position = player_info['POSITION'].iloc[0]
            team_id = int(player_info['TEAM_ID'].iloc[0]) if 'TEAM_ID' in player_info.columns else None
            
            injury_history = self._get_injury_history(player_id)
            
            matchup_history = self._get_matchup_history(player_id, opponent_team_id)
            
            position_matchup = self.get_position_matchup_stats(
                position,
                opponent_team_id,
                opponent_context=opponent_context
            )
            
            return {
                'position': position,
                'team_id': team_id,
                'injury_history': injury_history,
                'matchup_history': matchup_history,
                'position_matchup': position_matchup
            }
        except Exception as e:
            print(f"Error getting player context: {e}")
            return None

    def _calculate_team_form(self, games_df):
        """Calculate team's form using only available stats"""
        try:
            wins = float((games_df['WL'] == 'W').mean())
            avg_points = float(games_df['PTS'].mean())

            return {
                'win_pct': wins,
                'avg_points': avg_points,
                'trend': 'up' if wins > 0.5 else 'down' if wins < 0.5 else 'neutral'
            }
        except Exception as e:
            print(f"Error calculating team form: {e}")
            return {
                'win_pct': 0.5,
                'avg_points': 100.0,
                'trend': 'neutral'
            }

    def _calculate_defensive_rating(self, games_df):
        """Calculate team's defensive rating from game data"""
        try:
            away_games = games_df[games_df['MATCHUP'].str.contains('@')]
            home_games = games_df[~games_df['MATCHUP'].str.contains('@')]
            opp_pts_away = float(away_games['PTS'].mean()) if not away_games.empty else 0
            opp_pts_home = float(home_games['PTS'].mean()) if not home_games.empty else 0

            num_away = len(away_games)
            num_home = len(home_games)
            total_games = num_away + num_home
        
            if total_games == 0:
                return 110.0
            
            opp_pts = ((opp_pts_away * num_away) + (opp_pts_home * num_home)) / total_games

            possessions = (
                games_df['FGA'].mean() +  # Field goal attempts
                0.4 * games_df['FTA'].mean() -  # Free throw factor
                1.07 * (games_df['OREB'].mean() /
                        (games_df['OREB'].mean() + games_df['DREB'].mean())) *
                (games_df['FGA'].mean() - games_df['FGM'].mean()) +  
                games_df['TOV'].mean()  # Turnovers
            )

            # Calculate defensive rating (points allowed per 100 possessions)
            def_rating = (opp_pts / possessions) * 100 if possessions > 0 else 110.0

            return float(def_rating)

        except Exception as e:
            return 110.0

    def get_team_context(self, team_id, include_injuries: bool = True, season: str | None = None):
        """Get comprehensive team context including injuries"""
        season = season or self.current_season
        cache_key = (int(team_id), bool(include_injuries), str(season))
        cached = self.team_context_cache.get(cache_key)
        if cached and (int(time.time()) - int(cached.get('_cached_at', 0)) < self.team_context_ttl_seconds):
            return cached

        try:
            dash = self._get_league_team_dash(season=season)
            team_dash = (dash.get('teams') or {}).get(int(team_id), {})
            league_avgs = dash.get('league_avgs') or {}

            # Get injury information
            if include_injuries:
            injury_info = self.injury_tracker.get_team_injuries(team_id)
            else:
                injury_info = {
                    'active_injuries': [],
                    'total_impact': 0.0,
                    'key_players_out': 0,
                    'total_players_out': 0
                }
            injury_impact = float(injury_info.get('total_impact', 0.0) or 0.0)

            # Prefer league dash rates for stability and speed
            possessions_per_game = float(team_dash.get('pace', 100.0) or 100.0)
            off_rating = float(team_dash.get('off_rating', 110.0) or 110.0)
            defensive_rating = float(team_dash.get('def_rating', 110.0) or 110.0)

            adjusted_pace = possessions_per_game * (1 - injury_impact * 0.08)
            adjusted_off_rating = off_rating * (1 - injury_impact * 0.10)

            context = {
                '_cached_at': int(time.time()),
                'pace': float(adjusted_pace),
                'offensive_rating': float(adjusted_off_rating),
                'defensive_rating': float(defensive_rating),
                'recent_form': {'win_pct': 0.5, 'avg_points': float(team_dash.get('pts', 100.0) or 100.0), 'trend': 'neutral'},
                'rest_days': 2,
                'injury_impact': injury_impact,
                'injuries': {
                    'total_players_out': injury_info['total_players_out'],
                    'key_players_out': injury_info['key_players_out'],
                    'active_injuries': injury_info['active_injuries']
                },

                # Team style factors (per game, normalized fields)
                'style': {
                    'pts_fb': float(team_dash.get('pts_fb', 0.0) or 0.0),
                    'opp_pts_fb': float(team_dash.get('opp_pts_fb', 0.0) or 0.0),
                    'pts_off_tov': float(team_dash.get('pts_off_tov', 0.0) or 0.0),
                    'opp_pts_off_tov': float(team_dash.get('opp_pts_off_tov', 0.0) or 0.0),
                    'pts_paint': float(team_dash.get('pts_paint', 0.0) or 0.0),
                    'opp_pts_paint': float(team_dash.get('opp_pts_paint', 0.0) or 0.0),
                },
                'base': {
                    'fga': float(team_dash.get('fga', 0.0) or 0.0),
                    'fg_pct': float(team_dash.get('fg_pct', 0.47) or 0.47),
                    'fg3a': float(team_dash.get('fg3a', 0.0) or 0.0),
                    'fg3_pct': float(team_dash.get('fg3_pct', 0.36) or 0.36),
                    'tov': float(team_dash.get('tov', 0.0) or 0.0),
                    'stl': float(team_dash.get('stl', 0.0) or 0.0),
                    'blk': float(team_dash.get('blk', 0.0) or 0.0),
                },
                'league_avgs': league_avgs
            }

            self.team_context_cache[cache_key] = context
            return context

        except Exception as e:
            print(f"Error getting team context: {e}")
            return self._get_default_context()

    def _calculate_estimated_pace(self, games_df):
        """Calculate estimated pace from available stats"""
        try:
            fga = float(games_df['FGA'].mean()) if 'FGA' in games_df.columns else 85.0
            fta = float(games_df['FTA'].mean()) if 'FTA' in games_df.columns else 22.0
            pts = float(games_df['PTS'].mean())

            # pace formula
            estimated_pace = (fga + 0.4 * fta) or (pts / 1.1)

            return float(max(estimated_pace, 90.0))
        except Exception as e:
            print(f"Error calculating pace: {e}")
            return 100.0

    def _get_default_context(self):
        """Return default context when data is unavailable"""
        return {
            '_cached_at': int(time.time()),
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
            },
            'style': {
                'pts_fb': 12.0,
                'opp_pts_fb': 12.0,
                'pts_off_tov': 16.0,
                'opp_pts_off_tov': 16.0,
                'pts_paint': 48.0,
                'opp_pts_paint': 48.0
            },
            'base': {
                'fga': 86.0,
                'fg_pct': 0.47,
                'fg3a': 35.0,
                'fg3_pct': 0.36,
                'tov': 14.0,
                'stl': 7.0,
                'blk': 5.0
            },
            'league_avgs': {
                'pts_fb': 12.0,
                'opp_pts_fb': 12.0,
                'tov': 14.0,
                'stl': 7.0,
                'fga': 86.0,
                'fg_pct': 0.47,
                'pace': 100.0
            }
        }

    def _calculate_rest_days(self, games_df):
        """Calculate days of rest before next game"""
        try:
            if len(games_df) < 2:
                return 1

            last_game = pd.to_datetime(games_df['GAME_DATE'].iloc[0])
            today = pd.Timestamp.now()

            return int((today - last_game).days)
        except Exception as e:
            print(f"Error calculating rest days: {e}")
            return 2

    def _calculate_injury_impact(self, team_id):
        """Calculate impact of current injuries on team based on InjuryTracker data"""
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
            print(f"Error calculating injury impact: {e}")
            return 0.1

    def prepare_features(self, features, player_stats, player_context, team_context, opponent_context):
        """Prepare features for ML models including all context"""
        features = {}
        
        features.update({
            'recent_avg': float(player_stats.get('last5_avg', 0)),
            'season_avg': float(player_stats.get('avg', 0)),
            'max_recent': float(max(player_stats.get('values', [0]))),
            'min_recent': float(min(player_stats.get('values', [0]))),
            'stddev': float(np.std(player_stats.get('values', [0]))),
            'games_played': len(player_stats.get('values', [])),
        })
        
        if player_context:
            matchup_history = player_context.get('matchup_history', {})
            position_matchup = player_context.get('position_matchup', {})
            
            features.update({
                'vs_team_avg': float(matchup_history.get('avg_points', 0)),
                'matchup_games': int(matchup_history.get('games_played', 0)),
                'matchup_success_rate': float(matchup_history.get('success_rate', 0)),
                'pos_pts_allowed': float(position_matchup.get('pts_allowed_per_game', 0)),
                'pos_def_rating': float(position_matchup.get('defensive_rating', 0)),
                'injury_risk': float(player_context.get('injury_history', {}).get('injury_risk', 0))
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
        """Fast prediction using contextual adjustments (no training required)."""
        try:
            mode = (self.model_mode or "batch").strip().lower()
            print(f"[predict] prop_type={prop_type}, line={line}, mode={mode}")

            def _try_incremental():
                if not prop_type:
                    print(f"[incremental] skipped: no prop_type")
                    return None
                try:
                    Xr = build_feature_vector(features).X.values.astype(float)
                    Xc = build_classifier_vector(features, line=float(line)).X.values.astype(float)
                    print(f"[incremental] Xr.shape={Xr.shape}, Xc.shape={Xc.shape}")
                    out = self.incremental.predict(prop_type, X_reg=Xr, X_clf=Xc)
                    print(f"[incremental] raw output: {out}")
                except Exception as e:
                    print(f"[incremental] prediction error for {prop_type}: {e}")
                    import traceback
                    traceback.print_exc()
                    return None
                if out and out.get("over_probability") is not None:
                    proba = float(out["over_probability"])
                    pred = out.get("predicted_value")
                    if pred is None:
                        # For DD/TD, predicted value is the probability.
                        if prop_type in ("double_double", "triple_double"):
                            pred = proba
                        else:
                            pred = float(features.get("season_avg", line))
                    pred = float(pred)
                    edge = ((pred - line) / line) if line > 0 else 0.0
                    prob_strength = abs(proba - 0.5)
                    edge_strength = abs(edge)
                    confidence = self._calculate_confidence(prob_strength, edge_strength)
                    recommendation = self._generate_recommendation(proba, pred, line, edge, confidence)
                    print(f"[incremental] SUCCESS: proba={proba:.3f}, pred={pred:.1f}, rec={recommendation}")
                    return {
                        "over_probability": proba,
                        "predicted_value": pred,
                        "recommendation": recommendation,
                        "confidence": confidence,
                        "edge": float(edge),
                        "factors": {
                            "model_source": "incremental_sgd",
                        },
                    }
                print(f"[incremental] returned None or missing over_probability")
                return None

            # Hybrid preference:
            # - mode=incremental => incremental first, then trained, then heuristic
            # - otherwise => trained first, then incremental fallback (if available), then heuristic
            if mode == "incremental":
                print(f"[predict] trying incremental first (mode=incremental)")
                inc = _try_incremental()
                if inc:
                    return inc
                print(f"[predict] incremental failed, checking trained models...")

            # Prefer direct probability classifiers when available (best for sportsbook lines).
            if prop_type and (prop_type in self.trained_classifiers_cal or prop_type in self.trained_classifiers_raw):
                clf_out = self._predict_with_trained_classifier(features, line, prop_type=prop_type)
                if clf_out:
                    clf_out["factors"] = {
                        **(clf_out.get("factors", {}) or {}),
                        "minutes_ratio": float(features.get("mins_last5", 0) / (features.get("mins_season", 1) or 1)) if (features.get("mins_last5") and features.get("mins_season")) else 1.0,
                        "opp_def_rating": float(features.get("opp_def_rating", 110.0) or 110.0),
                        "dvp_gp": int(features.get("dvp_gp", 0) or 0),
                        "primary_defender_score01": float(features.get("primary_defender_score01", 0.0) or 0.0),
                    }
                    # Add comprehensive factor analysis
                    factor_analysis = self._analyze_all_factors(
                        features, clf_out["factors"], prop_type, line,
                        clf_out["predicted_value"], clf_out["over_probability"]
                    )
                    clf_out["factor_analysis"] = factor_analysis
                    # Recalculate confidence with factor strength
                    prob_strength = abs(clf_out["over_probability"] - 0.5)
                    edge_strength = abs(clf_out["edge"])
                    clf_out["confidence"] = self._calculate_confidence(
                        prob_strength, edge_strength, factor_analysis["factor_strength"]
                    )
                    # Regenerate recommendation with updated confidence
                    clf_out["recommendation"] = self._generate_recommendation(
                        clf_out["over_probability"], clf_out["predicted_value"],
                        line, clf_out["edge"], clf_out["confidence"]
                    )
                    return clf_out

            # If we have a trained regressor for this prop, use it (still fast).
            if prop_type and prop_type in ("pts_reb", "pts_ast", "ast_reb", "pts_ast_reb", "stl_blk"):
                combo = self._predict_with_trained_combo(features, line, combo_prop=prop_type)
                if combo:
                    combo["factors"] = {
                        **(combo.get("factors", {}) or {}),
                        "minutes_ratio": float(features.get("mins_last5", 0) / (features.get("mins_season", 1) or 1)) if (features.get("mins_last5") and features.get("mins_season")) else 1.0,
                        "opp_def_rating": float(features.get("opp_def_rating", 110.0) or 110.0),
                        "dvp_gp": int(features.get("dvp_gp", 0) or 0),
                        "primary_defender_score01": float(features.get("primary_defender_score01", 0.0) or 0.0),
                    }
                    return combo

            if prop_type and prop_type in self.trained_regressors:
                trained = self._predict_with_trained_regressor(features, line, prop_type=prop_type)
                if trained:
                    # Merge in a subset of the heuristic factor breakdown for explainability
                    trained["factors"] = {
                        **(trained.get("factors", {}) or {}),
                        "minutes_ratio": float(features.get("mins_last5", 0) / (features.get("mins_season", 1) or 1)) if (features.get("mins_last5") and features.get("mins_season")) else 1.0,
                        "opp_def_rating": float(features.get("opp_def_rating", 110.0) or 110.0),
                        "dvp_gp": int(features.get("dvp_gp", 0) or 0),
                        "primary_defender_score01": float(features.get("primary_defender_score01", 0.0) or 0.0),
                    }
                    # Add comprehensive factor analysis
                    factor_analysis = self._analyze_all_factors(
                        features, trained["factors"], prop_type, line,
                        trained["predicted_value"], trained["over_probability"]
                    )
                    trained["factor_analysis"] = factor_analysis
                    # Recalculate confidence with factor strength
                    prob_strength = abs(trained["over_probability"] - 0.5)
                    edge_strength = abs(trained["edge"])
                    trained["confidence"] = self._calculate_confidence(
                        prob_strength, edge_strength, factor_analysis["factor_strength"]
                    )
                    # Regenerate recommendation with updated confidence
                    trained["recommendation"] = self._generate_recommendation(
                        trained["over_probability"], trained["predicted_value"],
                        line, trained["edge"], trained["confidence"]
                    )
                    return trained

            # If we didn't have trained batch models for this prop, fall back to incremental (hybrid).
            if mode in ("batch", "hybrid", "auto", ""):
                print(f"[predict] no trained models found, trying incremental fallback...")
                inc = _try_incremental()
                if inc:
                    return inc
                print(f"[predict] incremental also failed, falling back to heuristics")

            # NOTE: This predictor is currently rule-based and does not use the scaler/models.
            # Do NOT attempt to scale the raw feature dict (it can contain non-numeric values like
            # `dvp_position` or `primary_defender_name`).
            recent_avg = float(features.get('recent_avg', 0))
            season_avg = float(features.get('season_avg', 0))
            std_dev = float(features.get('stddev', 0))

            # Minutes / role signal (REDUCED CAP: was 1.25, now 1.15 to avoid over-amplification)
            mins_last5 = float(features.get('mins_last5', 0)) or 0.0
            mins_season = float(features.get('mins_season', 0)) or 0.0
            minutes_ratio = (mins_last5 / mins_season) if mins_last5 > 0 and mins_season > 0 else 1.0
            minutes_ratio = float(min(max(minutes_ratio, 0.80), 1.15))

            # Context
            vs_team_avg = float(features.get('vs_team_avg', 0))
            matchup_games = int(features.get('matchup_games', 0) or 0)
            opp_def_rating = float(features.get('opp_def_rating', 110.0))
            team_pace = float(features.get('team_pace', 100.0))
            opp_pace = float(features.get('opp_pace', 100.0))
            team_off_rating = float(features.get('team_off_rating', 110.0))
            team_def_rating = float(features.get('team_def_rating', 110.0))
            opp_off_rating = float(features.get('opp_off_rating', 110.0))
            rest_days = int(features.get('rest_days', 2) or 2)
            team_key_out = int(features.get('team_key_players_out', 0) or 0)
            opp_injury_impact = float(features.get('opp_injury_impact', 0.0) or 0.0)
            opp_key_out = int(features.get('opp_key_players_out', 0) or 0)
            is_home_game = features.get('is_home_game', None)
            home_split_avg = float(features.get('home_split_avg', season_avg) or season_avg)
            away_split_avg = float(features.get('away_split_avg', season_avg) or season_avg)
            home_split_n = int(features.get('home_split_n', 0) or 0)
            away_split_n = int(features.get('away_split_n', 0) or 0)
            recent_away_streak = int(features.get('recent_away_streak', 0) or 0)

            # Team style / opportunity features (league normalized)
            team_pts_fb = float(features.get('team_pts_fb', 0.0) or 0.0)
            opp_pts_fb_allowed = float(features.get('opp_pts_fb_allowed', 0.0) or 0.0)
            lg_pts_fb = float(features.get('lg_pts_fb', 12.0) or 12.0)
            lg_opp_pts_fb = float(features.get('lg_opp_pts_fb', 12.0) or 12.0)

            team_pts_off_tov = float(features.get('team_pts_off_tov', 0.0) or 0.0)
            opp_pts_off_tov_allowed = float(features.get('opp_pts_off_tov_allowed', 0.0) or 0.0)
            lg_pts_off_tov = float(features.get('lg_pts_off_tov', 16.0) or 16.0)
            lg_opp_pts_off_tov = float(features.get('lg_opp_pts_off_tov', 16.0) or 16.0)

            opp_fga = float(features.get('opp_fga', 0.0) or 0.0)
            opp_fg_pct = float(features.get('opp_fg_pct', 0.47) or 0.47)
            opp_fg3a = float(features.get('opp_fg3a', 0.0) or 0.0)
            lg_fga = float(features.get('lg_fga', 86.0) or 86.0)
            lg_fg_pct = float(features.get('lg_fg_pct', 0.47) or 0.47)
            lg_fg3a = float(features.get('lg_fg3a', 35.0) or 35.0)

            opp_tov = float(features.get('opp_tov', 0.0) or 0.0)
            opp_stl = float(features.get('opp_stl', 0.0) or 0.0)
            lg_tov = float(features.get('lg_tov', 14.0) or 14.0)
            lg_stl = float(features.get('lg_stl', 7.0) or 7.0)

            opp_pts_paint = float(features.get('opp_pts_paint', 0.0) or 0.0)

            # Precomputed DVP (defense vs position) deltas + elite defender
            dvp_gp = int(features.get('dvp_gp', 0) or 0)
            dvp_position = features.get('dvp_position', None)
            primary_defender_score01 = float(features.get('primary_defender_score01', 0.0) or 0.0)

            # Pace factor (bounded)
            expected_pace = (0.55 * team_pace + 0.45 * opp_pace)
            pace_factor = float(min(max(expected_pace / 100.0, 0.93), 1.07))

            # Opponent minutes trend proxy: competitiveness/blowout risk affects whether starters play full minutes.
            team_net = team_off_rating - team_def_rating
            opp_net = opp_off_rating - opp_def_rating
            net_diff = team_net - opp_net
            blowout_minutes_factor = 1.0
            # Start applying a reduction after ~6 net rating points of mismatch.
            mismatch = max(0.0, abs(net_diff) - 6.0)
            if mismatch > 0:
                blowout_minutes_factor = float(min(max(1.0 - 0.01 * mismatch, 0.90), 1.0))
                # Away games are slightly more volatile
                if is_home_game is False:
                    blowout_minutes_factor = float(min(max(blowout_minutes_factor - 0.01, 0.88), 1.0))
            # REDUCED CAP: was 0.70-1.28, now 0.80-1.15
            minutes_ratio = float(min(max(minutes_ratio * blowout_minutes_factor, 0.80), 1.15))

            # Matchup factor: only trust if sample exists
            matchup_blend = 0.0
            if matchup_games >= 2 and vs_team_avg > 0:
                matchup_blend = min(0.18, 0.06 + 0.02 * min(matchup_games, 6))

            # Tier 1: Extract new features
            plus_minus_avg = float(features.get('plus_minus_avg', 0.0) or 0.0)
            win_rate_last10 = float(features.get('win_rate_last10', 0.5) or 0.5)
            fg_pct_recent = float(features.get('fg_pct_recent', 0.0) or 0.0)
            fouls_per_game = float(features.get('fouls_per_game', 0.0) or 0.0)
            
            # Tier 2 Quick Wins: momentum/trend features
            last_3_trend = float(features.get('last_3_games_trend', 0.0) or 0.0)
            last_5_trend = float(features.get('last_5_games_trend', 0.0) or 0.0)
            last_10_trend = float(features.get('last_10_games_trend', 0.0) or 0.0)
            is_back_to_back = bool(features.get('is_back_to_back', 0.0))
            games_in_last_7 = int(features.get('games_in_last_7_days', 0) or 0)

            # Momentum adjustment: hot teams/players perform better (REDUCED from ±0.35/0.25)
            momentum_adj = 0.0
            if win_rate_last10 > 0.7:
                momentum_adj = 0.25  # winning team, player likely getting more opportunities
            elif win_rate_last10 < 0.3:
                momentum_adj = -0.20  # losing team, may have reduced role/minutes

            # Plus/minus indicates overall impact beyond box score
            impact_adj = 0.0
            if plus_minus_avg > 5.0:
                impact_adj = 0.25  # high-impact player
            elif plus_minus_avg < -5.0:
                impact_adj = -0.20  # negative impact (may lose minutes)

            # Shooting efficiency for scoring props (REMOVED - redundant with ts_adj and pps_adj)
            # We now use True Shooting % as the primary efficiency metric
            efficiency_adj = 0.0

            # Foul trouble risk (reduces minutes)
            foul_adj = 0.0
            if fouls_per_game > 4.0:
                foul_adj = -0.30  # high foul rate may limit minutes

            # Tier 2: Momentum/trend adjustments (REDUCED to avoid overlap with momentum_adj)
            trend_adj = 0.0
            if last_3_trend > 1.5:  # strong upward 3-game trend
                trend_adj = 0.35
            elif last_3_trend < -1.5:  # strong downward trend
                trend_adj = -0.30
            elif last_5_trend > 0.8:  # moderate upward trend
                trend_adj = 0.20
            elif last_5_trend < -0.8:  # moderate downward trend
                trend_adj = -0.18

            # Tier 2: Fatigue adjustments (REDUCED - overlaps with rest_adj)
            fatigue_adj = 0.0
            if is_back_to_back:
                fatigue_adj = -0.35  # back-to-back significantly impacts performance
            elif games_in_last_7 >= 5:
                fatigue_adj = -0.20  # heavy schedule
            elif games_in_last_7 <= 2:
                fatigue_adj = 0.15  # well-rested

            # Tier 2+: Advanced metrics adjustments
            usage_rate = float(features.get('usage_rate', 0.0) or 0.0)
            true_shooting = float(features.get('true_shooting_pct', 0.0) or 0.0)
            pie = float(features.get('pie', 0.0) or 0.0)
            clutch_fg_pct = float(features.get('clutch_fg_pct', 0.0) or 0.0)
            altitude = float(features.get('arena_altitude', 0) or 0)
            coast_to_coast = bool(features.get('coast_to_coast', 0.0))
            
            # Usage rate: high usage = more opportunities (REDUCED weights)
            usage_rate_adj = 0.0
            if usage_rate > 0.30:  # Very high usage (30%+)
                usage_rate_adj = 0.30
            elif usage_rate > 0.25:  # High usage
                usage_rate_adj = 0.18
            elif usage_rate < 0.15:  # Low usage role player
                usage_rate_adj = -0.15
            
            # True shooting efficiency (PRIMARY efficiency metric, REDUCED weights)
            ts_adj = 0.0
            if prop_type in ('points', 'three_pointers', 'pts_ast', 'pts_reb', 'pts_ast_reb'):
                if true_shooting > 0.60:  # Elite efficiency
                    ts_adj = 0.28
                elif true_shooting < 0.50:  # Poor efficiency
                    ts_adj = -0.25
            
            # PIE (Player Impact Estimate) - overall contribution
            pie_adj = 0.0
            if pie > 0.15:  # High impact player (top tier)
                pie_adj = 0.30
            elif pie < 0.05:  # Low impact
                pie_adj = -0.15
            
            # Altitude effect (Denver, Utah)
            altitude_adj = 0.0
            if altitude > 4000:  # High altitude arenas
                # Visiting teams struggle, home teams benefit
                if features.get('is_home_game', True):
                    altitude_adj = 0.25  # Home team advantage
                else:
                    altitude_adj = -0.35  # Visiting team disadvantage
            
            # Coast-to-coast travel fatigue
            travel_adj = 0.0
            if coast_to_coast:
                travel_adj = -0.30  # 3+ hour timezone change impacts performance
            
            # Tier 4: Extract new efficiency and consistency features
            points_per_shot = float(features.get('points_per_shot', 0.0) or 0.0)
            ast_to_tov_ratio = float(features.get('ast_to_tov_ratio', 0.0) or 0.0)
            usage_trend = float(features.get('usage_trend', 0.0) or 0.0)
            consistency_score = float(features.get('consistency_score', 0.0) or 0.0)
            ceiling_game_frequency = float(features.get('ceiling_game_frequency', 0.0) or 0.0)
            blowout_game_pct = float(features.get('blowout_game_pct', 0.0) or 0.0)
            close_game_pct = float(features.get('close_game_pct', 0.0) or 0.0)
            minutes_volatility = float(features.get('minutes_volatility', 0.0) or 0.0)
            
            # Tier 4: Efficiency adjustments (REMOVED - redundant with ts_adj)
            # Points per shot is essentially the same as True Shooting %
            pps_adj = 0.0
            
            # Tier 4: Assist to turnover ratio (for playmaking props) - REDUCED
            ast_tov_adj = 0.0
            if prop_type in ('assists', 'pts_ast', 'ast_reb', 'pts_ast_reb'):
                if ast_to_tov_ratio > 3.0:  # Elite ball security
                    ast_tov_adj = 0.18
                elif ast_to_tov_ratio < 1.5:  # Poor ball security
                    ast_tov_adj = -0.15
            
            # Tier 4: Usage trend (increasing role) - REDUCED, overlaps with usage_rate_adj
            usage_trend_adj = 0.0
            if usage_trend > 2.0:  # Significantly increased role
                usage_trend_adj = 0.22
            elif usage_trend < -2.0:  # Significantly decreased role
                usage_trend_adj = -0.20
            
            # Tier 4: Consistency boost (consistent players are safer bets) - REDUCED
            consistency_adj = 0.0
            if consistency_score > 0.75:  # Very consistent
                consistency_adj = 0.12  # Reduces uncertainty, nudge prediction up slightly
            elif consistency_score < 0.50:  # Very volatile
                consistency_adj = -0.10  # High variance, slight negative adjustment
            
            # Tier 4: Ceiling game frequency (explosive upside) - REDUCED
            ceiling_adj = 0.0
            if ceiling_game_frequency > 0.25:  # Frequently hits ceiling
                ceiling_adj = 0.15
            
            # Tier 4: Game script risk (blowouts reduce minutes) - REDUCED
            game_script_adj = 0.0
            if blowout_game_pct > 0.35:  # Frequent blowouts
                game_script_adj = -0.18  # Minutes at risk
            elif close_game_pct > 0.50:  # Frequently in close games
                game_script_adj = 0.15  # More clutch minutes
            
            # Tier 4: Minutes volatility risk - REDUCED
            mins_volatility_adj = 0.0
            if minutes_volatility > 8.0:  # Very inconsistent minutes
                mins_volatility_adj = -0.18
            elif minutes_volatility < 3.0:  # Very stable minutes
                mins_volatility_adj = 0.12
            
            # ============================================================================
            # TIER 5: Player Streaks, Game Importance, Rotation, Matchup (FREE FEATURES)
            # ============================================================================
            
            # Tier 5: Player streaks vs betting lines
            consecutive_over = int(features.get('consecutive_over_games', 0) or 0)
            consecutive_under = int(features.get('consecutive_under_games', 0) or 0)
            hot_hand = float(features.get('hot_hand_indicator', 0.0) or 0.0)
            variance_spike = float(features.get('recent_variance_spike', 0.0) or 0.0)
            
            streak_adj = 0.0
            if consecutive_over >= 3:
                # On a hot streak vs the line - moderate positive signal
                streak_adj = 0.20
            elif consecutive_under >= 3:
                # On a cold streak vs the line - moderate negative signal
                streak_adj = -0.20
            
            # Hot hand indicator (independent of streaks)
            hothand_adj = 0.0
            if hot_hand > 0.20:  # Significantly above season hit rate
                hothand_adj = 0.15
            elif hot_hand < -0.20:  # Significantly below season hit rate
                hothand_adj = -0.15
            
            # Variance spike adjustment
            variance_adj = 0.0
            if variance_spike > 0.5:  # Variance suddenly increased
                variance_adj = -0.10  # More unpredictable, slight negative
            
            # Tier 5: Game importance
            playoff_impact = float(features.get('playoff_seeding_impact', 0.5) or 0.5)
            tanking = float(features.get('tanking_indicator', 0.0) or 0.0)
            must_win = float(features.get('must_win_situation', 0.0) or 0.0)
            
            importance_adj = 0.0
            if must_win > 0.5:  # Must-win game
                importance_adj = 0.25  # Players tend to step up
            elif tanking > 0.5:  # Tanking team
                importance_adj = -0.20  # Players may rest/lower effort
            elif playoff_impact > 0.7:  # High stakes
                importance_adj = 0.12
            
            # Tier 5: Rotation patterns
            fourth_q_usage = float(features.get('fourth_quarter_usage_rate', 0.2) or 0.2)
            garbage_time_pct = float(features.get('garbage_time_minutes_pct', 0.05) or 0.05)
            crunch_usage = float(features.get('crunch_time_usage', 0.2) or 0.2)
            
            rotation_adj = 0.0
            if fourth_q_usage > 0.30:  # High 4th Q usage
                if prop_type in ('points', 'pts_ast', 'pts_reb', 'pts_ast_reb'):
                    rotation_adj = 0.18  # Closers score more
            if garbage_time_pct > 0.15:  # Lots of garbage time
                rotation_adj -= 0.12  # Lower quality minutes
            
            # Tier 5: Specific matchup history
            career_vs = float(features.get('career_vs_defender', 0.0) or 0.0)
            recent_vs = float(features.get('recent_vs_defender', 0.0) or 0.0)
            
            matchup_hist_adj = 0.0
            if recent_vs > 0 and career_vs > 0:
                # If recent performance vs defender is significantly different from career
                matchup_diff = recent_vs - career_vs
                if abs(matchup_diff) > 3.0:  # Meaningful difference
                    matchup_hist_adj = matchup_diff * 0.10  # Moderate weight
            
            # ============================================================================
            # TIER 6: Shot Quality, Touch/Usage, Extended Lineup Context
            # ============================================================================
            
            # Tier 6: Shot quality impact
            open_shot_pct = float(features.get('open_shot_pct', 0.3) or 0.3)
            contested_pct = float(features.get('contested_shot_pct', 0.5) or 0.5)
            shot_quality_vs_exp = float(features.get('shot_quality_vs_expected', 0.0) or 0.0)
            restricted_area_pct = float(features.get('restricted_area_fg_pct', 0.0) or 0.0)
            
            shot_quality_adj = 0.0
            if prop_type in ('points', 'pts_ast', 'pts_reb', 'pts_ast_reb'):
                # High quality shots = positive adjustment
                if open_shot_pct > 0.4:  # Lots of open looks
                    shot_quality_adj = 0.20
                elif contested_pct > 0.6:  # Heavily contested
                    shot_quality_adj = -0.18
                
                # Over/under-performing expected FG%
                if shot_quality_vs_exp > 0.03:  # Hot shooting
                    shot_quality_adj += 0.15
                elif shot_quality_vs_exp < -0.03:  # Cold shooting
                    shot_quality_adj -= 0.15
            
            # Tier 6: Touch & ball dominance
            touches_pg = float(features.get('touches_per_game', 0.0) or 0.0)
            time_of_poss = float(features.get('time_of_possession_per_game', 0.0) or 0.0)
            avg_points_per_touch = float(features.get('avg_points_per_touch', 0.0) or 0.0)
            
            touch_efficiency_adj = 0.0
            if prop_type in ('points', 'assists', 'pts_ast', 'pts_ast_reb'):
                # High touch players get more opportunities
                if touches_pg > 80:  # Very high usage player
                    touch_efficiency_adj = 0.18
                elif touches_pg < 40:  # Low usage
                    touch_efficiency_adj = -0.12
                
                # Efficiency per touch matters
                if avg_points_per_touch > 0.5:  # Efficient scorer
                    touch_efficiency_adj += 0.10
            
            # Tier 6: Lineup synergy effects
            on_off_diff = float(features.get('on_off_differential', 0.0) or 0.0)
            net_rating_starters = float(features.get('net_rating_with_starters', 0.0) or 0.0)
            
            lineup_synergy_adj = 0.0
            if on_off_diff > 5.0:  # Team much better with player
                lineup_synergy_adj = 0.15
            elif on_off_diff < -5.0:  # Team worse with player
                lineup_synergy_adj = -0.12
            
            if net_rating_starters > 8.0:  # Great lineup performance
                lineup_synergy_adj += 0.10

            base = (0.62 * recent_avg + 0.28 * season_avg + matchup_blend * vs_team_avg)

            # Defense adjustment: worse defenses inflate, better defenses deflate
            defense_adj = -0.085 * (opp_def_rating - 110.0)

            # Rest adjustment (small, bounded)
            if rest_days <= 0:
                rest_adj = -0.6
            elif rest_days == 1:
                rest_adj = -0.3
            elif rest_days >= 3:
                rest_adj = 0.25
            else:
                rest_adj = 0.0

            # Injury usage bump (applies mostly to scoring/playmaking props) - REDUCED and RENAMED
            teammate_out_adj = 0.0
            if prop_type in ('points', 'assists', 'pts_ast', 'pts_ast_reb'):
                teammate_out_adj = 0.35 * min(team_key_out, 2)  # Max +0.70

            # Opponent injuries: missing key defenders/rotation usually softens defense and can increase pace/efficiency.
            # REDUCED and CAPPED to avoid stacking with teammate_out_adj
            opp_injury_adj = 0.0
            if opp_injury_impact > 0:
                # weight slightly more if multiple key players are out
                scale = min(max(opp_injury_impact + 0.05 * min(opp_key_out, 3), 0.0), 0.5)
                if prop_type in ('points', 'three_pointers', 'pts_ast', 'pts_reb', 'pts_ast_reb'):
                    opp_injury_adj = 0.55 * scale  # Max +0.275
                elif prop_type in ('assists',):
                    opp_injury_adj = 0.35 * scale
                elif prop_type in ('rebounds', 'ast_reb'):
                    opp_injury_adj = 0.15 * scale
                elif prop_type in ('steals', 'stl_blk'):
                    opp_injury_adj = 0.20 * scale
                elif prop_type in ('turnovers',):
                    # weaker opponent defense => slightly fewer pressured turnovers
                    opp_injury_adj = -0.10 * scale
                opp_injury_adj = float(min(max(opp_injury_adj, -0.3), 0.5))

            # Home/away advantage (small, bounded)
            home_adj = 0.0
            if is_home_game is True:
                home_adj = 0.35
            elif is_home_game is False:
                home_adj = -0.35

            # Travel fatigue proxy: away + short rest + ongoing road streak
            travel_adj = 0.0
            if is_home_game is False and rest_days <= 1:
                travel_adj -= 0.35
            if is_home_game is False and recent_away_streak >= 3:
                travel_adj -= 0.20

            # Transition adjustment (points/assists/3s benefit from fast-break environment)
            transition_delta = 0.0
            if lg_pts_fb > 0 and lg_opp_pts_fb > 0:
                transition_delta = 0.5 * ((team_pts_fb - lg_pts_fb) + (opp_pts_fb_allowed - lg_opp_pts_fb))

            transition_adj = 0.0
            if prop_type in ('points', 'assists', 'three_pointers', 'pts_ast', 'pts_ast_reb'):
                transition_adj = 0.08 * transition_delta

            # Points-off-turnovers environment (steals/turnovers correlate with how often teams create/cough up live-ball miscues).
            off_tov_delta = 0.0
            if lg_pts_off_tov > 0 and lg_opp_pts_off_tov > 0:
                off_tov_delta = 0.5 * ((team_pts_off_tov - lg_pts_off_tov) + (opp_pts_off_tov_allowed - lg_opp_pts_off_tov))

            off_tov_adj = 0.0
            if prop_type in ('steals', 'stl_blk'):
                off_tov_adj = 0.03 * off_tov_delta
            elif prop_type in ('turnovers',):
                # more opponent points off turnovers allowed can imply pressure; small bump
                off_tov_adj = 0.02 * (opp_pts_off_tov_allowed - lg_opp_pts_off_tov)
            off_tov_adj = float(min(max(off_tov_adj, -0.6), 0.6))

            # Rebound opportunities proxy: opponent misses (more misses => more rebound chances)
            misses = max(0.0, opp_fga * (1.0 - opp_fg_pct))
            lg_misses = max(0.0, lg_fga * (1.0 - lg_fg_pct))
            misses_delta = misses - lg_misses
            reb_adj = 0.0
            if prop_type in ('rebounds', 'pts_reb', 'ast_reb', 'pts_ast_reb'):
                reb_adj = 0.04 * misses_delta

            # 3PA environment (small): teams that take lots of 3s can increase volatility and slightly lift 3PM props.
            three_env_adj = 0.0
            if prop_type in ('three_pointers',) and lg_fg3a > 0:
                three_env_adj = 0.02 * ((opp_fg3a - lg_fg3a) / 5.0)
                three_env_adj = float(min(max(three_env_adj, -0.3), 0.3))

            # Turnovers / steals opportunity proxies
            tov_adj = 0.0
            stl_adj = 0.0
            if prop_type in ('turnovers',):
                # more opponent steals => more pressure => slightly higher turnover expectation
                tov_adj = 0.10 * (opp_stl - lg_stl)
            if prop_type in ('steals', 'stl_blk'):
                # more opponent turnovers => more steal opportunities
                stl_adj = 0.06 * (opp_tov - lg_tov)

            # Blocks proxy: teams that score more in paint tend to generate more rim events
            blk_adj = 0.0
            if prop_type in ('blocks', 'stl_blk'):
                blk_adj = 0.015 * ((opp_pts_paint - 48.0) / 2.0)

            # DVP adjustment (shrinkage by GP, REDUCED CAP from 0.22 to 0.15)
            dvp_w = 0.0
            if dvp_gp >= 5:
                dvp_w = min(0.15, 0.04 + 0.003 * min(dvp_gp, 40))

            dvp_adj = 0.0
            if dvp_w > 0:
                # REDUCED multipliers (was 0.35-0.45, now 0.25-0.32) and CAP (was ±1.2, now ±0.60)
                if prop_type in ('points',):
                    dvp_adj = dvp_w * float(features.get('dvp_pts_delta', 0.0) or 0.0) * 0.25
                elif prop_type in ('assists',):
                    dvp_adj = dvp_w * float(features.get('dvp_ast_delta', 0.0) or 0.0) * 0.25
                elif prop_type in ('rebounds',):
                    dvp_adj = dvp_w * float(features.get('dvp_reb_delta', 0.0) or 0.0) * 0.22
                elif prop_type in ('three_pointers',):
                    dvp_adj = dvp_w * float(features.get('dvp_fg3m_delta', 0.0) or 0.0) * 0.32
                elif prop_type in ('steals',):
                    dvp_adj = dvp_w * float(features.get('dvp_stl_delta', 0.0) or 0.0) * 0.18
                elif prop_type in ('blocks',):
                    dvp_adj = dvp_w * float(features.get('dvp_blk_delta', 0.0) or 0.0) * 0.18
                elif prop_type in ('turnovers',):
                    dvp_adj = dvp_w * float(features.get('dvp_tov_delta', 0.0) or 0.0) * 0.18
                elif prop_type == 'pts_reb':
                    dvp_adj = dvp_w * (float(features.get('dvp_pts_delta', 0.0) or 0.0) * 0.18 +
                                       float(features.get('dvp_reb_delta', 0.0) or 0.0) * 0.15)
                elif prop_type == 'pts_ast':
                    dvp_adj = dvp_w * (float(features.get('dvp_pts_delta', 0.0) or 0.0) * 0.18 +
                                       float(features.get('dvp_ast_delta', 0.0) or 0.0) * 0.18)
                elif prop_type == 'ast_reb':
                    dvp_adj = dvp_w * (float(features.get('dvp_ast_delta', 0.0) or 0.0) * 0.16 +
                                       float(features.get('dvp_reb_delta', 0.0) or 0.0) * 0.16)
                elif prop_type == 'pts_ast_reb':
                    dvp_adj = dvp_w * (float(features.get('dvp_pts_delta', 0.0) or 0.0) * 0.16 +
                                       float(features.get('dvp_ast_delta', 0.0) or 0.0) * 0.16 +
                                       float(features.get('dvp_reb_delta', 0.0) or 0.0) * 0.13)
                elif prop_type == 'stl_blk':
                    dvp_adj = dvp_w * (float(features.get('dvp_stl_delta', 0.0) or 0.0) * 0.13 +
                                       float(features.get('dvp_blk_delta', 0.0) or 0.0) * 0.13)
                dvp_adj = float(min(max(dvp_adj, -0.60), 0.60))

            # Elite defender adjustment (REDUCED from -0.65 to -0.45, cap from -0.9 to -0.60)
            defender_adj = 0.0
            if primary_defender_score01 > 0:
                if prop_type in ('points', 'pts_ast', 'pts_reb', 'pts_ast_reb'):
                    defender_adj = -0.45 * primary_defender_score01
                elif prop_type in ('assists',):
                    defender_adj = -0.18 * primary_defender_score01
                elif prop_type in ('three_pointers',):
                    defender_adj = -0.13 * primary_defender_score01
                elif prop_type in ('turnovers',):
                    defender_adj = 0.15 * primary_defender_score01
                defender_adj = float(min(max(defender_adj, -0.60), 0.25))

            # Calculate total adjustments (with REBALANCED weights + Tier 5 + Tier 6)
            total_adjustments = (defense_adj + rest_adj + teammate_out_adj + opp_injury_adj + 
                               home_adj + travel_adj + transition_adj + off_tov_adj + reb_adj + 
                               three_env_adj + tov_adj + stl_adj + blk_adj + dvp_adj + defender_adj + 
                               momentum_adj + impact_adj + efficiency_adj + foul_adj + trend_adj + 
                               fatigue_adj + usage_rate_adj + ts_adj + pie_adj + altitude_adj + 
                               pps_adj + ast_tov_adj + usage_trend_adj + consistency_adj + 
                               ceiling_adj + game_script_adj + mins_volatility_adj + 
                               # Tier 5 additions
                               streak_adj + hothand_adj + variance_adj + importance_adj + 
                               rotation_adj + matchup_hist_adj +
                               # Tier 6 additions
                               shot_quality_adj + touch_efficiency_adj + lineup_synergy_adj)
            
            # CAP total adjustments to prevent extreme predictions (±5.0 points max, increased for Tier 6)
            total_adjustments = float(min(max(total_adjustments, -5.0), 5.0))
            
            predicted_value = (base + total_adjustments) * pace_factor * minutes_ratio
            predicted_value = float(max(predicted_value, 0.0))

            # Location split shrinkage: blend toward home/away-specific average when sample exists.
            loc_avg = None
            loc_n = 0
            if is_home_game is True:
                loc_avg = home_split_avg
                loc_n = home_split_n
            elif is_home_game is False:
                loc_avg = away_split_avg
                loc_n = away_split_n

            if loc_avg is not None and loc_n >= 3:
                loc_w = min(0.22, 0.06 + 0.02 * min(loc_n, 7))
                predicted_value = float((1 - loc_w) * predicted_value + loc_w * float(loc_avg))

            edge = ((predicted_value - line) / line) if line > 0 else 0
        
            # Prob estimate: t-distribution handles small samples better than normal.
            n = int(features.get('games_played', 0) or 0)
            df = max(n - 1, 1)
            scale = (std_dev + 1e-6)
            z_score = (line - predicted_value) / scale
            over_prob = 1 - scipy.stats.t.cdf(z_score, df=df)
        
            # Calculate confidence
            prob_strength = abs(over_prob - 0.5)
            edge_strength = abs(edge)
            
            # Build factors dict for analysis
            factors_dict = {
                    'minutes_ratio': minutes_ratio,
                    'pace_factor': pace_factor,
                    'team_net_rating': float(team_net),
                    'opp_net_rating': float(opp_net),
                    'net_diff': float(net_diff),
                    'blowout_minutes_factor': float(blowout_minutes_factor),
                    'opp_def_rating': float(opp_def_rating),
                    'rest_days': int(rest_days),
                    'matchup_blend': float(matchup_blend),
                    'defense_adj': float(defense_adj),
                    'rest_adj': float(rest_adj),
                    'teammate_out_adj': float(teammate_out_adj),
                    'opp_injury_impact': float(opp_injury_impact),
                    'opp_key_players_out': int(opp_key_out),
                    'opp_injury_adj': float(opp_injury_adj),
                    'total_adjustments': float(total_adjustments),
                    'home_adj': float(home_adj),
                    'travel_adj': float(travel_adj),
                    'transition_adj': float(transition_adj),
                    'off_tov_adj': float(off_tov_adj),
                    'reb_adj': float(reb_adj),
                    'three_env_adj': float(three_env_adj),
                    'tov_adj': float(tov_adj),
                    'stl_adj': float(stl_adj),
                    'blk_adj': float(blk_adj),
                    'dvp_position': dvp_position,
                    'dvp_gp': int(dvp_gp),
                    'dvp_w': float(dvp_w),
                    'dvp_adj': float(dvp_adj),
                    'primary_defender_score01': float(primary_defender_score01),
                    'defender_adj': float(defender_adj),
                    # Tier 1 additions
                    'momentum_adj': float(momentum_adj),
                    'impact_adj': float(impact_adj),
                    'efficiency_adj': float(efficiency_adj),
                    'foul_adj': float(foul_adj),
                    'win_rate_last10': float(win_rate_last10),
                    'plus_minus_avg': float(plus_minus_avg),
                    'fg_pct_recent': float(fg_pct_recent),
                    'fouls_per_game': float(fouls_per_game),
                    # Tier 2 Quick Wins
                    'trend_adj': float(trend_adj),
                    'fatigue_adj': float(fatigue_adj),
                    'last_3_trend': float(last_3_trend),
                    'last_5_trend': float(last_5_trend),
                    'is_back_to_back': int(is_back_to_back),
                    'games_in_last_7': int(games_in_last_7),
                    # Tier 2+: Advanced metrics
                    'usage_rate': float(usage_rate),
                    'usage_rate_adj': float(usage_rate_adj),
                    'true_shooting_pct': float(true_shooting),
                    'ts_adj': float(ts_adj),
                    'pie': float(pie),
                    'pie_adj': float(pie_adj),
                    'clutch_fg_pct': float(clutch_fg_pct),
                    'altitude': float(altitude),
                    'altitude_adj': float(altitude_adj),
                    'coast_to_coast': int(coast_to_coast),
                    # Tier 4: New efficiency and consistency adjustments
                    'pps_adj': float(pps_adj),
                    'ast_tov_adj': float(ast_tov_adj),
                    'usage_trend_adj': float(usage_trend_adj),
                    'consistency_adj': float(consistency_adj),
                    'ceiling_adj': float(ceiling_adj),
                    'game_script_adj': float(game_script_adj),
                    'mins_volatility_adj': float(mins_volatility_adj),
                    'points_per_shot': float(points_per_shot),
                    'ast_to_tov_ratio': float(ast_to_tov_ratio),
                    'usage_trend': float(usage_trend),
                    'consistency_score': float(consistency_score),
                    'ceiling_game_frequency': float(ceiling_game_frequency),
                    'blowout_game_pct': float(blowout_game_pct),
                    'close_game_pct': float(close_game_pct),
                    'minutes_volatility': float(minutes_volatility),
                }
            
            # Add comprehensive factor analysis
            factor_analysis = self._analyze_all_factors(
                features, factors_dict, prop_type, line,
                predicted_value, over_prob
            )
            
            # Recalculate confidence with factor strength
            confidence = self._calculate_confidence(
                prob_strength, edge_strength, factor_analysis["factor_strength"]
            )
            
            # Regenerate recommendation with updated confidence
            recommendation = self._generate_recommendation(over_prob, predicted_value, line, edge, confidence)
        
            return {
                'over_probability': float(over_prob),
                'predicted_value': float(predicted_value),
                'recommendation': recommendation,
                'confidence': confidence,
                'edge': float(edge),
                'factor_analysis': factor_analysis,
                'factors': factors_dict
            }
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'over_probability': 0.5,
                'predicted_value': features.get('season_avg', line),
                'recommendation': 'PASS',
                'confidence': 'LOW',
                'edge': 0.0,
                'factors': {}
            }

    def _calculate_confidence(self, prob_strength, edge_strength, factor_strength=0.0):
        """Calculate prediction confidence based on probability, edge strength, and factor alignment"""
        # Factor strength (0-1) measures how many factors align with the prediction
        base_score = (0.6 * prob_strength + 0.3 * edge_strength + 0.1 * factor_strength)
        confidence_score = min(1.0, base_score)
    
        if confidence_score > 0.15:
            return 'HIGH'
        elif confidence_score > 0.1:
            return 'MEDIUM'
        return 'LOW'

    def _analyze_all_factors(self, features, factors_dict, prop_type, line, predicted_value, over_prob):
        """
        Comprehensive factor analysis - evaluates all factors and generates detailed insights.
        Returns: {
            'positive_factors': [{'name': str, 'impact': float, 'value': any, 'explanation': str}],
            'negative_factors': [...],
            'key_drivers': [...],  # Top 5-10 most impactful factors
            'factor_strength': float,  # 0-1, how aligned factors are
            'detailed_explanation': str
        }
        """
        positive_factors = []
        negative_factors = []
        factor_impacts = []
        
        # Helper to add factor with impact calculation
        def add_factor(name, value, impact, explanation, threshold=0.0):
            if abs(impact) > threshold:
                factor_info = {
                    'name': name,
                    'value': value,
                    'impact': float(impact),
                    'explanation': explanation
                }
                if impact > 0:
                    positive_factors.append(factor_info)
                else:
                    negative_factors.append(factor_info)
                factor_impacts.append(abs(impact))
        
        # Extract key values from factors_dict
        f = factors_dict
        
        # 1. DEFENSIVE MATCHUP FACTORS (High Impact)
        opp_def_rating = float(f.get('opp_def_rating', 110.0) or 110.0)
        def_impact = (110.0 - opp_def_rating) * 0.15  # Lower D-rating = better for offense
        if abs(def_impact) > 0.5:
            add_factor(
                'Opponent Defense Rating',
                f"{opp_def_rating:.1f}",
                def_impact,
                f"Opponent allows {opp_def_rating:.1f} points per 100 possessions. {'Favorable matchup' if def_impact > 0 else 'Tough defensive matchup'}."
            )
        
        dvp_adj = float(f.get('dvp_adj', 0.0) or 0.0)
        if abs(dvp_adj) > 0.3:
            dvp_pos = f.get('dvp_position', 'N/A')
            add_factor(
                f'DVP vs {dvp_pos}',
                f"{dvp_adj:+.1f}",
                dvp_adj,
                f"Opponent ranks {'weak' if dvp_adj > 0 else 'strong'} against {dvp_pos} position. Historical data shows {abs(dvp_adj):.1f} point {'advantage' if dvp_adj > 0 else 'disadvantage'}."
            )
        
        defender_score = float(f.get('primary_defender_score01', 0.0) or 0.0)
        defender_adj = float(f.get('defender_adj', 0.0) or 0.0)
        if abs(defender_adj) > 0.3:
            defender_name = f.get('primary_defender_name', 'Unknown')
            add_factor(
                f'Primary Defender: {defender_name}',
                f"{defender_score:.2f}",
                defender_adj,
                f"Facing {'elite' if defender_score > 0.7 else 'strong' if defender_score > 0.5 else 'average'} defender. Expected {abs(defender_adj):.1f} point {'reduction' if defender_adj < 0 else 'boost'}."
            )
        
        # 2. RECENT FORM & MOMENTUM (High Impact)
        momentum_adj = float(f.get('momentum_adj', 0.0) or 0.0)
        if abs(momentum_adj) > 0.3:
            add_factor(
                'Recent Momentum',
                f"{momentum_adj:+.1f}",
                momentum_adj,
                f"Player showing {'strong upward' if momentum_adj > 0 else 'declining'} trend in recent games."
            )
        
        trend_adj = float(f.get('trend_adj', 0.0) or 0.0)
        if abs(trend_adj) > 0.3:
            add_factor(
                'Performance Trend',
                f"{trend_adj:+.1f}",
                trend_adj,
                f"{'Improving' if trend_adj > 0 else 'Declining'} performance trajectory over last 5-10 games."
            )
        
        hot_hand = float(f.get('hot_hand_indicator', 0.0) or 0.0)
        if hot_hand > 0.6:
            add_factor(
                'Hot Hand Indicator',
                f"{hot_hand:.2f}",
                hot_hand * 1.5,
                f"Player in hot streak - {hot_hand*100:.0f}% confidence. Recent games significantly above average."
            )
        elif hot_hand < 0.3:
            add_factor(
                'Cold Streak',
                f"{hot_hand:.2f}",
                -(1.0 - hot_hand) * 1.2,
                f"Player in cold streak - recent performance {hot_hand*100:.0f}% of normal. May be due for regression."
            )
        
        # 3. INJURY & ROSTER IMPACT (Very High Impact)
        opp_key_out = int(f.get('opp_key_players_out', 0) or 0)
        opp_injury_adj = float(f.get('opp_injury_adj', 0.0) or 0.0)
        if opp_key_out > 0 or abs(opp_injury_adj) > 0.5:
            add_factor(
                'Opponent Injuries',
                f"{opp_key_out} key players out",
                opp_injury_adj,
                f"Opponent missing {opp_key_out} key player(s). {'Weaker defense expected' if opp_injury_adj > 0 else 'Still strong despite injuries'}."
            )
        
        teammate_out_adj = float(f.get('teammate_out_adj', 0.0) or 0.0)
        if abs(teammate_out_adj) > 0.5:
            add_factor(
                'Teammate Availability',
                f"{teammate_out_adj:+.1f}",
                teammate_out_adj,
                f"Key teammate(s) {'out' if teammate_out_adj > 0 else 'returning'}. {'Increased usage expected' if teammate_out_adj > 0 else 'Usage may normalize'}."
            )
        
        # 4. GAME CONTEXT (Medium-High Impact)
        rest_adj = float(f.get('rest_adj', 0.0) or 0.0)
        rest_days = int(f.get('rest_days', 2) or 2)
        if abs(rest_adj) > 0.3:
            rest_status = 'Well-rested' if rest_days >= 2 else 'Short rest' if rest_days == 1 else 'Back-to-back'
            add_factor(
                'Rest Days',
                f"{rest_days} days",
                rest_adj,
                f"{rest_status}. {'Optimal recovery' if rest_adj > 0 else 'Fatigue may impact performance'}."
            )
        
        is_back_to_back = int(f.get('is_back_to_back', 0) or 0)
        if is_back_to_back:
            add_factor(
                'Back-to-Back Game',
                "Yes",
                -1.2,
                "Playing second game in two nights. Typically see 5-10% reduction in performance."
            )
        
        home_adj = float(f.get('home_adj', 0.0) or 0.0)
        if abs(home_adj) > 0.3:
            add_factor(
                'Home Court Advantage',
                "Home" if home_adj > 0 else "Away",
                home_adj,
                f"{'Home court' if home_adj > 0 else 'Road game'} typically provides {abs(home_adj):.1f} point {'boost' if home_adj > 0 else 'reduction'}."
            )
        
        travel_adj = float(f.get('travel_adj', 0.0) or 0.0)
        if abs(travel_adj) > 0.5:
            travel_dist = f.get('travel_distance', 0)
            add_factor(
                'Travel Impact',
                f"{travel_dist:.0f} miles",
                travel_adj,
                f"{'Long travel' if travel_adj < 0 else 'Minimal travel'}. {'Fatigue factor' if travel_adj < 0 else 'Well-rested'}."
            )
        
        # 5. EFFICIENCY & USAGE (Medium Impact)
        usage_rate = float(f.get('usage_rate', 0.0) or 0.0)
        usage_rate_adj = float(f.get('usage_rate_adj', 0.0) or 0.0)
        if abs(usage_rate_adj) > 0.3:
            add_factor(
                'Usage Rate',
                f"{usage_rate:.1f}%",
                usage_rate_adj,
                f"{'High' if usage_rate > 25 else 'Moderate' if usage_rate > 20 else 'Low'} usage rate ({usage_rate:.1f}%). {'More opportunities' if usage_rate_adj > 0 else 'Fewer touches expected'}."
            )
        
        true_shooting = float(f.get('true_shooting_pct', 0.5) or 0.5)
        ts_adj = float(f.get('ts_adj', 0.0) or 0.0)
        if abs(ts_adj) > 0.3:
            add_factor(
                'Shooting Efficiency',
                f"{true_shooting:.1%}",
                ts_adj,
                f"{'Elite' if true_shooting > 0.6 else 'Good' if true_shooting > 0.55 else 'Below average'} true shooting. {'Efficient scorer' if ts_adj > 0 else 'Inefficiency concerns'}."
            )
        
        # 6. PACE & GAME SCRIPT (Medium Impact)
        pace_factor = float(f.get('pace_factor', 1.0) or 1.0)
        if abs(pace_factor - 1.0) > 0.05:
            add_factor(
                'Game Pace',
                f"{pace_factor:.2f}x",
                (pace_factor - 1.0) * 2.0,
                f"{'Fast-paced' if pace_factor > 1.0 else 'Slow-paced'} game expected. {'More possessions' if pace_factor > 1.0 else 'Fewer opportunities'}."
            )
        
        blowout_factor = float(f.get('blowout_minutes_factor', 1.0) or 1.0)
        if blowout_factor < 0.9:
            add_factor(
                'Blowout Risk',
                f"{blowout_factor:.2f}x",
                (blowout_factor - 1.0) * 1.5,
                "Potential blowout scenario. May see reduced minutes in 4th quarter."
            )
        
        # 7. MATCHUP HISTORY (Medium Impact)
        matchup_blend = float(f.get('matchup_blend', 0.0) or 0.0)
        if abs(matchup_blend) > 0.5:
            add_factor(
                'Historical Matchup',
                f"{matchup_blend:+.1f}",
                matchup_blend,
                f"Past performance vs this opponent shows {abs(matchup_blend):.1f} point {'advantage' if matchup_blend > 0 else 'disadvantage'}."
            )
        
        career_vs_defender = float(f.get('career_vs_defender', 0.0) or 0.0)
        if abs(career_vs_defender) > 1.0:
            add_factor(
                'Career vs Defender',
                f"{career_vs_defender:+.1f}",
                career_vs_defender * 0.3,
                f"Historical performance against this defender: {career_vs_defender:+.1f} vs average."
            )
        
        # 8. ADVANCED METRICS (Lower-Medium Impact)
        consistency = float(f.get('consistency_score', 0.5) or 0.5)
        if consistency < 0.4:
            add_factor(
                'Consistency',
                f"{consistency:.2f}",
                -(0.5 - consistency) * 1.0,
                "High variance player. Less predictable performance."
            )
        
        ceiling_freq = float(f.get('ceiling_game_frequency', 0.0) or 0.0)
        if ceiling_freq > 0.3:
            add_factor(
                'Ceiling Games',
                f"{ceiling_freq:.1%}",
                ceiling_freq * 0.8,
                f"Frequently exceeds expectations ({ceiling_freq:.1%} of games). Upside potential."
            )
        
        # 9. ROTATION & MINUTES (Medium Impact)
        minutes_ratio = float(f.get('minutes_ratio', 1.0) or 1.0)
        if abs(minutes_ratio - 1.0) > 0.1:
            add_factor(
                'Recent Minutes Trend',
                f"{minutes_ratio:.2f}x",
                (minutes_ratio - 1.0) * 1.5,
                f"{'Increased' if minutes_ratio > 1.0 else 'Reduced'} playing time recently. {'More opportunities' if minutes_ratio > 1.0 else 'Limited role'}."
            )
        
        fourth_q_usage = float(f.get('fourth_quarter_usage_rate', 0.2) or 0.2)
        if fourth_q_usage > 0.3:
            add_factor(
                'Crunch Time Usage',
                f"{fourth_q_usage:.1%}",
                (fourth_q_usage - 0.2) * 1.0,
                "High usage in clutch situations. More opportunities in close games."
            )
        
        # 10. GAME IMPORTANCE (Lower Impact)
        playoff_impact = float(f.get('playoff_seeding_impact', 0.5) or 0.5)
        must_win = float(f.get('must_win_situation', 0.0) or 0.0)
        if must_win > 0.7:
            add_factor(
                'Must-Win Game',
                "Yes",
                0.8,
                "High-stakes game. Players typically elevate performance."
            )
        
        # Calculate factor strength (alignment score)
        total_positive_impact = sum(f['impact'] for f in positive_factors)
        total_negative_impact = abs(sum(f['impact'] for f in negative_factors))
        total_impact = total_positive_impact + total_negative_impact
        
        if total_impact > 0:
            factor_strength = min(1.0, (total_positive_impact / total_impact) if over_prob > 0.5 else (total_negative_impact / total_impact))
        else:
            factor_strength = 0.5
        
        # Sort factors by absolute impact
        all_factors = positive_factors + negative_factors
        all_factors.sort(key=lambda x: abs(x['impact']), reverse=True)
        key_drivers = all_factors[:10]  # Top 10 most impactful
        
        # Generate detailed explanation
        explanation_parts = []
        if key_drivers:
            top_driver = key_drivers[0]
            explanation_parts.append(f"Primary factor: {top_driver['name']} ({top_driver['explanation']})")
        
        if len(positive_factors) > len(negative_factors):
            explanation_parts.append(f"{len(positive_factors)} positive factors vs {len(negative_factors)} negative factors favor the OVER.")
        elif len(negative_factors) > len(positive_factors):
            explanation_parts.append(f"{len(negative_factors)} negative factors vs {len(positive_factors)} positive factors favor the UNDER.")
        else:
            explanation_parts.append("Mixed signals from factors - recommendation based on edge and probability.")
        
        detailed_explanation = " ".join(explanation_parts)
        
        return {
            'positive_factors': sorted(positive_factors, key=lambda x: abs(x['impact']), reverse=True),
            'negative_factors': sorted(negative_factors, key=lambda x: abs(x['impact']), reverse=True),
            'key_drivers': key_drivers,
            'factor_strength': float(factor_strength),
            'total_positive_impact': float(total_positive_impact),
            'total_negative_impact': float(total_negative_impact),
            'detailed_explanation': detailed_explanation
        }

    def _generate_recommendation(self, prob, predicted_value, line, edge, confidence):
        """Generate betting recommendation based on probability and confidence"""
        # Convert edge percentage to absolute points for clearer thresholds
        edge_points = abs(predicted_value - line)
        
        # Determine direction: is predicted value above or below the line?
        is_over = predicted_value > line
        is_under = predicted_value < line
        
        # Large edge threshold: >8 points difference (significant value)
        large_edge_over = is_over and edge_points > 8.0
        large_edge_under = is_under and edge_points > 8.0
        # Medium edge threshold: >5 points difference
        medium_edge_over = is_over and edge_points > 5.0
        medium_edge_under = is_under and edge_points > 5.0
        
        # STRONG recommendations: high probability OR large edge in correct direction
        if prob > 0.65 and (edge > 0.05 or large_edge_over):
            return 'STRONG OVER'
        elif prob < 0.35 and (edge < -0.05 or large_edge_under):
            return 'STRONG UNDER'
        
        # LEAN recommendations: moderate probability OR medium edge in correct direction
        if prob > 0.58 and (edge > 0.03 or medium_edge_over):
            return 'LEAN OVER'
        elif prob < 0.42 and (edge < -0.03 or medium_edge_under):
            return 'LEAN UNDER'
        
        # Even with LOW confidence, if edge is very large in correct direction, still recommend
        if large_edge_over and prob > 0.52:
            return 'LEAN OVER'
        elif large_edge_under and prob < 0.48:
            return 'LEAN UNDER'
        
        # Only PASS if confidence is LOW AND edge is small
        if confidence == 'LOW' and not (medium_edge_over or medium_edge_under):
            return 'PASS'
    
        return 'PASS'

    def train(self, training_data):
        """Train both classification and regression models"""
        if not training_data:
            raise ValueError("No training data provided")
            
        # Prepare features and targets
        X = pd.DataFrame([data['features'] for data in training_data])
        y_class = [1 if data['result'] > data['line'] else 0 for data in training_data]
        y_reg = [data['result'] for data in training_data]
        
        X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2)
        _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classification model
        self.classification_model.fit(X_train_scaled, y_class_train)
        class_auc = roc_auc_score(y_class_test, 
            self.classification_model.predict_proba(X_test_scaled)[:, 1])
        
        # Train regression model
        self.regression_model.fit(X_train_scaled, y_reg_train)
        reg_rmse = np.sqrt(mean_squared_error(y_reg_test, 
            self.regression_model.predict(X_test_scaled)))
        
        print(f"Classification AUC: {class_auc:.3f}")
        print(f"Regression RMSE: {reg_rmse:.3f}")
        
        joblib.dump(self.classification_model, f'{self.model_dir}/classification_model.joblib')
        joblib.dump(self.regression_model, f'{self.model_dir}/regression_model.joblib')
        joblib.dump(self.scaler, f'{self.model_dir}/scaler.joblib')