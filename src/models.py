from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from nba_api.stats.endpoints import TeamGameLog, CommonPlayerInfo, LeagueGameFinder
from nba_api.stats.endpoints import PlayerVsPlayer, TeamDashboardByGeneralSplits
import scipy.stats
import numpy as np
import pandas as pd
import joblib
import os
import time
from datetime import datetime, timedelta
from .injury_tracker import InjuryTracker

class EnhancedMLPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.injury_tracker = InjuryTracker()
        
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
        

    def _get_injury_history(self, player_id):
        """Analyze player's injury history and recent injuries"""
        # This would connect to an injury database/API
        # For now returning placeholder data
        return {
            'recent_injuries': [],
            'games_missed': 0,
            'injury_risk': 'low'
        }

    def _get_matchup_history(self, player_id, opponent_team_id):
        """Get detailed matchup history against specific team"""
        try:
            # Use LeagueGameFinder instead of PlayerVsPlayer
            gamefinder = LeagueGameFinder(
                player_id_nullable=player_id,
                vs_team_id_nullable=opponent_team_id,
                season_type_nullable='Regular Season'
            ).get_data_frames()[0]
        
            time.sleep(0.6)  # Rate limiting
        
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

    def _get_position_matchup_stats(self, position, team_id):
        """Get team's defensive stats against specific position"""
        cache_key = f"{position}_{team_id}"
        
        if cache_key in self.position_matchup_cache:
            return self.position_matchup_cache[cache_key]
            
        try:
            # Get team's defensive stats against position
            # This would use advanced stats APIs/databases
            # Returning placeholder data for now
            matchup_stats = {
                'pts_allowed_per_game': 20.5,
                'defensive_rating': 105.2,
                'effective_fg_pct': 0.52
            }
            
            self.position_matchup_cache[cache_key] = matchup_stats
            return matchup_stats
            
        except Exception as e:
            print(f"Error getting position matchup stats: {e}")
            return None

    def get_player_context(self, player_id, opponent_team_id):
        """Get comprehensive player context including injuries and matchups"""
        try:
            # Get player info including position
            player_info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            position = player_info['POSITION'].iloc[0]
            
            # Get recent injuries
            injury_history = self._get_injury_history(player_id)
            
            # Get matchup history against opponent
            matchup_history = self._get_matchup_history(player_id, opponent_team_id)
            
            # Get position-specific matchup data
            position_matchup = self._get_position_matchup_stats(position, opponent_team_id)
            
            return {
                'position': position,
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
            # Use only W/L and PTS which are guaranteed to be available
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

            if len(team_games) == 0:
                return self._get_default_context()

            # Get injury information
            injury_info = self.injury_tracker.get_team_injuries(team_id)

            # Calculate stats from recent games
            recent_games = team_games.head(10)
            possessions_per_game = self._calculate_estimated_pace(recent_games)
            pts_per_game = float(recent_games['PTS'].mean())

            # Adjust stats based on injuries
            injury_impact = injury_info['total_impact']
            adjusted_pace = possessions_per_game * (1 - injury_impact * 0.1)  # Slight pace reduction with injuries
            adjusted_pts = pts_per_game * (1 - injury_impact * 0.15)  # Larger scoring impact from injuries

            context = {
                'pace': float(adjusted_pace),
                'offensive_rating': float(adjusted_pts / (adjusted_pace / 100)),
                'defensive_rating': float(110.0),  # Default value
                'recent_form': self._calculate_team_form(recent_games),
                'rest_days': self._calculate_rest_days(recent_games),
                'injury_impact': injury_impact,
                'injuries': {
                    'total_players_out': injury_info['total_players_out'],
                    'key_players_out': injury_info['key_players_out'],
                    'active_injuries': injury_info['active_injuries']
                }
            }

            self.team_context_cache[team_id] = context
            return context

        except Exception as e:
            print(f"Error getting team context: {e}")
            return self._get_default_context()

    def _calculate_estimated_pace(self, games_df):
        """Calculate estimated pace from available stats"""
        try:
            # Use only guaranteed available columns
            fga = float(games_df['FGA'].mean()) if 'FGA' in games_df.columns else 85.0
            fta = float(games_df['FTA'].mean()) if 'FTA' in games_df.columns else 22.0
            pts = float(games_df['PTS'].mean())

            # Simplified pace estimation
            estimated_pace = (fga + 0.4 * fta) or (pts / 1.1)

            return float(max(estimated_pace, 90.0))  # Ensure reasonable minimum
        except Exception as e:
            print(f"Error calculating pace: {e}")
            return 100.0  # Default value

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
            'injury_impact': 0.1
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
            return 2  # Default value

    def _calculate_injury_impact(self, team_id):
        """Calculate impact of current injuries on team
        Would connect to an injury database/API in production
        Returns placeholder impact score (0-1)"""
        return 0.1  # Placeholder

    def prepare_features(self, features, player_stats, player_context, team_context, opponent_context):
        """Prepare features for ML models including all context"""
        features = {}
        
        # Basic player statistics
        features.update({
            'recent_avg': float(player_stats.get('last5_avg', 0)),
            'season_avg': float(player_stats.get('avg', 0)),
            'max_recent': float(max(player_stats.get('values', [0]))),
            'min_recent': float(min(player_stats.get('values', [0]))),
            'stddev': float(np.std(player_stats.get('values', [0]))),
            'games_played': len(player_stats.get('values', [])),
        })
        
        # Player context features
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
        
        # Team context features
        if team_context:
            features.update({
                'team_pace': float(team_context.get('pace', 0)),
                'team_off_rating': float(team_context.get('offensive_rating', 0)),
                'team_def_rating': float(team_context.get('defensive_rating', 0)),
                'team_form': float(team_context.get('recent_form', {}).get('win_pct', 0)),
                'rest_days': int(team_context.get('rest_days', 1)),
                'team_injuries': float(team_context.get('injury_impact', 0))
            })
        
        # Opponent context features
        if opponent_context:
            features.update({
                'opp_pace': float(opponent_context.get('pace', 0)),
                'opp_def_rating': float(opponent_context.get('defensive_rating', 0)),
                'opp_form': float(opponent_context.get('recent_form', {}).get('win_pct', 0)),
                'opp_injuries': float(opponent_context.get('injury_impact', 0))
            })

        # Injury-related features
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
            # Initialize scaler with some reasonable defaults if not fitted
            if not hasattr(self.scaler, 'mean_'):
                self.scaler.mean_ = np.zeros(len(features))
                self.scaler.scale_ = np.ones(len(features))
                self.scaler.var_ = np.ones(len(features))
                self.scaler.n_features_in_ = len(features)
        
            features_df = pd.DataFrame([features])
            features_scaled = self.scaler.transform(features_df)
        
            recent_avg = features.get('recent_avg', 0)
            season_avg = features.get('season_avg', 0)
            std_dev = features.get('stddev', 0)
        
            predicted_value = (0.7 * recent_avg + 0.3 * season_avg)
            edge = ((predicted_value - line) / line) if line > 0 else 0
        
            z_score = (line - predicted_value) / (std_dev + 1e-6)
            over_prob = 1 - scipy.stats.norm.cdf(z_score)
        
            # Calculate confidence
            prob_strength = abs(over_prob - 0.5)
            edge_strength = abs(edge)
            confidence = self._calculate_confidence(prob_strength, edge_strength)
        
            # Generate recommendation
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
    
        if confidence_score > 0.15:
            return 'HIGH'
        elif confidence_score > 0.1:
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
        
        # Split data
        X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2)
        _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2)
        
        # Scale features
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
        
        # Save models
        joblib.dump(self.classification_model, f'{self.model_dir}/classification_model.joblib')
        joblib.dump(self.regression_model, f'{self.model_dir}/regression_model.joblib')
        joblib.dump(self.scaler, f'{self.model_dir}/scaler.joblib')