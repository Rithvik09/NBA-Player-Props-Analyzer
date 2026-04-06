from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
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

        # Try to load pre-trained models from disk
        try:
            self.classification_model = joblib.load(f'{model_dir}/classification_model.joblib')
            self.regression_model = joblib.load(f'{model_dir}/regression_model.joblib')
            self.scaler = joblib.load(f'{model_dir}/scaler.joblib')
            self.models_trained = True
            print("Loaded pre-trained models from disk")
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

        self.position_matchup_cache = {}   # evicted when > 200 entries
        self.team_context_cache = {}       # evicted when > 100 entries
        self._pt_defend_cache = None       # league-wide defensive data, fetched once per season
        self._pt_defend_season = None      # track which season it was fetched for
        self._CACHE_MAX_POSITION = 200
        self._CACHE_MAX_TEAM = 100
        

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
                try:
                    games = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season
                    ).get_data_frames()[0]
                    time.sleep(0.6)
                    all_games.append(games)
                except Exception as e:
                    print(f"Error fetching injury history for season {season}: {e}")
                    continue

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
        
    def _get_pt_defend_data(self):
        """
        Fetch LeagueDashPtDefend for the current season — once per session.
        Returns the full DataFrame with all defenders across the league.
        """
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
            print(f"Error fetching LeagueDashPtDefend: {e}")
            return None

    def _get_current_season(self):
        current_year = datetime.now().year
        current_month = datetime.now().month
        if 1 <= current_month <= 7:
            return f"{current_year-1}-{str(current_year)[2:]}"
        return f"{current_year}-{str(current_year+1)[2:]}"

    def get_position_matchup_stats(self, position, team_id):
        """
        Get opponent team's defensive stats against a specific position.

        Uses LeagueDashPtDefend (1 API call per session, cached) instead of
        100+ individual LeagueGameFinder calls. Filters to the opponent team's
        defenders at the relevant position and computes weighted averages by
        games played, preserving the same accuracy as the original approach.
        """
        cache_key = f"{position}_{team_id}"
        if cache_key in self.position_matchup_cache:
            return self.position_matchup_cache[cache_key]

        # Map any position string → canonical nba_api position codes
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

            # Filter to the opponent team's defenders at this position group
            team_defenders = defend_df[
                (defend_df['PLAYER_LAST_TEAM_ID'] == int(team_id)) &
                (defend_df['PLAYER_POSITION'].isin(valid_positions))
            ]

            if team_defenders.empty:
                return self._get_default_position_matchup()

            # Weighted average by games played for accuracy
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
            print(f"Error in get_position_matchup_stats: {e}")
            return self._get_default_position_matchup()
        
    def _get_default_position_matchup(self):
        """Return default position matchup stats"""
        return {
            'pts_allowed_per_game': 15.0,
            'defensive_rating': 110.0,
            'effective_fg_pct': 0.47
        }

    def get_player_context(self, player_id, opponent_team_id):
        """Get comprehensive player context including injuries and matchups"""
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
        """
        Estimate defensive rating from TeamGameLog data.
        TeamGameLog only has the team's own stats (PTS, FGA, etc.), not the opponent's,
        so we derive a proxy: start from offensive rating and subtract the average
        PLUS_MINUS per 100 possessions.
          defensive_rating ≈ offensive_rating - avg_plus_minus_per_100_poss
        A team that outscores opponents by +5/game gives up ~5 fewer pts/game → better defense.
        """
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

            # Offensive rating: team's own points per 100 possessions
            off_rating = (float(games_df['PTS'].mean()) / possessions) * 100

            # Approximate defensive rating using PLUS_MINUS
            # PLUS_MINUS = pts_scored - pts_allowed per game
            # pts_allowed ≈ pts_scored - plus_minus
            # def_rating ≈ (pts_allowed / possessions) * 100
            avg_plus_minus = float(games_df['PLUS_MINUS'].mean()) if 'PLUS_MINUS' in games_df.columns else 0.0
            pts_allowed_est = float(games_df['PTS'].mean()) - avg_plus_minus
            def_rating = (pts_allowed_est / possessions) * 100

            # Sanity clamp: keep within realistic NBA range
            return float(max(90.0, min(130.0, def_rating)))

        except Exception as e:
            return 110.0

    def get_team_context(self, team_id):
        """Get comprehensive team context including injuries"""
        if team_id in self.team_context_cache:
            return self.team_context_cache[team_id]

        try:
            # Get team's recent games
            team_games = TeamGameLog(
                team_id=team_id,
                season_type_all_star='Regular Season'
            ).get_data_frames()[0]
            time.sleep(0.6)

            if len(team_games) == 0:
                return self._get_default_context()

            # Get injury information
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
            print(f"Error getting team context: {e}")
            return self._get_default_context()

    def _calculate_estimated_pace(self, games_df):
        """Calculate estimated pace from available stats using Oliver formula"""
        try:
            fga = float(games_df['FGA'].mean()) if 'FGA' in games_df.columns else 85.0
            fta = float(games_df['FTA'].mean()) if 'FTA' in games_df.columns else 22.0
            oreb = float(games_df['OREB'].mean()) if 'OREB' in games_df.columns else 10.0
            tov = float(games_df['TOV'].mean()) if 'TOV' in games_df.columns else 14.0

            # Oliver possession formula: FGA - OREB + TOV + 0.44*FTA
            estimated_pace = fga - oreb + tov + 0.44 * fta

            return float(max(estimated_pace, 90.0))
        except Exception as e:
            print(f"Error calculating pace: {e}")
            return 100.0

    def _get_default_context(self):
        """Return default context when data is unavailable"""
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

    def prepare_features(self, player_stats, player_context, team_context, opponent_context):
        """Prepare features for ML models including all context"""
        features = {}
        
        features.update({
            'recent_avg': float(player_stats.get('last5_avg', 0)),
            'season_avg': float(player_stats.get('avg', 0)),
            'max_recent': float(max(player_stats.get('values', [0]))),
            'min_recent': float(min(player_stats.get('values', [0]))),
            'stddev': float(np.std(player_stats.get('values') or [0])),
            'games_played': len(player_stats.get('values', [])),
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

    def predict(self, features, line):
        """Make predictions using both classification and regression models"""
        try:
            recent_avg = features.get('recent_avg', 0)
            season_avg = features.get('season_avg', 0)
            std_dev = features.get('stddev', 0)

            # Statistical baseline prediction
            stat_predicted_value = (0.7 * recent_avg + 0.3 * season_avg)
            stat_z_score = (line - stat_predicted_value) / (std_dev + 1e-6)
            stat_over_prob = 1 - scipy.stats.norm.cdf(stat_z_score)

            predicted_value = stat_predicted_value
            over_prob = stat_over_prob

            # Blend with trained ML models when available
            if self.models_trained:
                try:
                    features_df = pd.DataFrame([features])
                    # Align columns to what scaler was trained on
                    if hasattr(self.scaler, 'feature_names_in_'):
                        for col in self.scaler.feature_names_in_:
                            if col not in features_df.columns:
                                features_df[col] = 0.0
                        features_df = features_df[self.scaler.feature_names_in_]
                    features_scaled = self.scaler.transform(features_df)

                    ml_pred = float(self.regression_model.predict(features_scaled)[0])
                    ml_prob = float(self.classification_model.predict_proba(features_scaled)[0, 1])

                    # 50/50 blend: statistical baseline + ML models
                    predicted_value = 0.5 * stat_predicted_value + 0.5 * ml_pred
                    blended_z = (line - predicted_value) / (std_dev + 1e-6)
                    blended_stat_prob = 1 - scipy.stats.norm.cdf(blended_z)
                    over_prob = 0.5 * blended_stat_prob + 0.5 * ml_prob
                except Exception as e:
                    print(f"ML model inference failed, using statistical fallback: {e}")
            else:
                if not hasattr(self.scaler, 'mean_'):
                    self.scaler.mean_ = np.zeros(len(features))
                    self.scaler.scale_ = np.ones(len(features))
                    self.scaler.var_ = np.ones(len(features))
                    self.scaler.n_features_in_ = len(features)

            edge = ((predicted_value - line) / line) if line > 0 else 0

            # Calculate confidence
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
            print(f"Prediction error: {e}")
            return {
                'over_probability': 0.5,
                'predicted_value': features.get('season_avg', line),
                'recommendation': 'PASS',
                'confidence': 'LOW',
                'edge': 0.0
            }

    def _calculate_confidence(self, prob_strength, edge_strength):
        """Calculate prediction confidence based on probability and edge strength"""
        confidence_score = (0.7 * prob_strength + 0.3 * edge_strength)

        if confidence_score > 0.08:
            return 'HIGH'
        elif confidence_score > 0.04:
            return 'MEDIUM'
        return 'LOW'

    def _generate_recommendation(self, prob, predicted_value, line, edge, confidence):
        """Generate betting recommendation based on probability and confidence"""
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

    def train(self, training_data):
        """Train both classification and regression models"""
        if not training_data:
            raise ValueError("No training data provided")
            
        # Prepare features and targets
        X = pd.DataFrame([data['features'] for data in training_data])
        y_class = [1 if data['result'] > data['line'] else 0 for data in training_data]
        y_reg = [data['result'] for data in training_data]
        
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42
        )
        
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
        self.models_trained = True
        return {'auc': float(class_auc), 'rmse': float(reg_rmse)}