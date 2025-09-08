import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, CommonPlayerInfo, PlayerVsPlayer, TeamDashboardByGeneralSplits
from nba_api.stats.static import players
from nba_api.stats.endpoints import TeamGameLog, CommonTeamRoster, LeagueGameFinder
import sqlite3
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import asyncio
from .models import EnhancedMLPredictor
from .situational_analyzer import SituationalAnalyzer
from .bankroll_manager import BankrollManager
from .parlay_optimizer import ParlayOptimizer
# Enterprise AI Models
try:
    from .ai_models.tensorflow_predictor import TensorFlowPredictor
    from .ai_models.pytorch_predictor import PyTorchPredictor
    ADVANCED_AI_AVAILABLE = True
except ImportError:
    print("Warning: Advanced AI models not available. Using standard models.")
    ADVANCED_AI_AVAILABLE = False
# Cloud Infrastructure
try:
    from .cloud.aws_integration import AWSIntegration
    CLOUD_AVAILABLE = True
except ImportError:
    print("Warning: Cloud integration not available. Using local storage.")
    CLOUD_AVAILABLE = False
# Live Odds Integration
try:
    from .live_data.odds_integration import OddsAggregator
    LIVE_ODDS_AVAILABLE = True
except ImportError:
    print("Warning: Live odds integration not available.")
    LIVE_ODDS_AVAILABLE = False
# Authentication System
try:
    from .auth.user_management import UserManager
    AUTH_AVAILABLE = True
except ImportError:
    print("Warning: User authentication not available.")
    AUTH_AVAILABLE = False 


class BasketballBettingHelper:
    def __init__(self, db_name='basketball_data.db', user_id=None):
        self.db_name = db_name
        self.user_id = user_id
        
        # Initialize standard ML predictor
        self.ml_predictor = EnhancedMLPredictor()
        
        # Initialize enterprise AI models if available
        if ADVANCED_AI_AVAILABLE:
            self.tensorflow_predictor = TensorFlowPredictor()
            self.pytorch_predictor = PyTorchPredictor()
            print("🤖 Advanced AI models (TensorFlow & PyTorch) loaded successfully!")
        else:
            self.tensorflow_predictor = None
            self.pytorch_predictor = None
        
        # Initialize cloud infrastructure
        if CLOUD_AVAILABLE:
            self.aws_integration = AWSIntegration()
            print("☁️ AWS cloud integration enabled!")
        else:
            self.aws_integration = None
        
        # Initialize live odds aggregator
        if LIVE_ODDS_AVAILABLE:
            self.odds_aggregator = OddsAggregator()
            print("📈 Live odds integration enabled!")
        else:
            self.odds_aggregator = None
        
        # Initialize user management
        if AUTH_AVAILABLE:
            self.user_manager = UserManager()
            print("👤 User authentication system enabled!")
        else:
            self.user_manager = None
        
        self.situational_analyzer = SituationalAnalyzer()
        self.bankroll_manager = BankrollManager(db_name)
        self.parlay_optimizer = ParlayOptimizer()
        
        current_year = datetime.now().year
        current_month = datetime.now().month
        

        if 1 <= current_month <= 7:
            self.current_season = f"{current_year-1}-{str(current_year)[2:]}"
        else:
            self.current_season = f"{current_year}-{str(current_year+1)[2:]}"
            
        print(f"Current season set to: {self.current_season}")
        self.create_tables()
        
    def get_db(self):
        return sqlite3.connect(self.db_name)
        
    def create_tables(self):
        conn = self.get_db()
        cursor = conn.cursor()
        
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY,
                full_name TEXT,
                first_name TEXT,
                last_name TEXT,
                is_active INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                game_date TEXT,
                matchup TEXT,
                wl TEXT,
                min INTEGER,
                pts INTEGER,
                ast INTEGER,
                reb INTEGER,
                stl INTEGER,
                blk INTEGER,
                turnover INTEGER,
                fg3m INTEGER,
                fg_pct REAL,
                fg3_pct REAL,
                ft_pct REAL,
                FOREIGN KEY (player_id) REFERENCES players (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def get_player_suggestions(self, partial_name):
        if len(partial_name) < 2:
            return []
            
        try:
            all_players = players.get_players()
            suggestions = [
                {
                    'id': player['id'],
                    'full_name': player['full_name'],
                    'is_active': player['is_active']
                }
                for player in all_players 
                if player['is_active'] and partial_name.lower() in player['full_name'].lower()
            ][:10]
            
            conn = self.get_db()
            cursor = conn.cursor()
            
            for player in suggestions:
                cursor.execute('''
                    INSERT OR REPLACE INTO players (id, full_name, is_active)
                    VALUES (?, ?, ?)
                ''', (
                    player['id'],
                    player['full_name'],
                    1
                ))
            
            conn.commit()
            conn.close()
            
            return suggestions
            
        except Exception as e:
            print(f"Error getting player suggestions: {e}")
            return []

    def get_player_stats(self, player_id):
        #Get player statistics
        try:
            current_year = datetime.now().year
            current_month = datetime.now().month
            
            if 1 <= current_month <= 7:
                current_season = f"{current_year-1}-{str(current_year)[2:]}"
                previous_season = f"{current_year-2}-{str(current_year-1)[2:]}"
            else:
                current_season = f"{current_year}-{str(current_year+1)[2:]}"
                previous_season = f"{current_year-1}-{str(current_year)[2:]}"
            
            seasons = [current_season, previous_season]
            print(f"Fetching seasons: {seasons}")
            
            all_games = []
            
            for season in seasons:
                try:
                    gamelog = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season
                    )
                    time.sleep(0.5)
                    games = gamelog.get_data_frames()[0]
                    print(f"Found {len(games)} games for {season}")
                    all_games.append(games)
                except Exception as e:
                    print(f"Error fetching {season} data: {e}")
                    continue
            
            if not all_games:
                raise Exception("Could not fetch any game data")
                
            games_df = pd.concat(all_games, ignore_index=True)
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            games_df = games_df.sort_values('GAME_DATE', ascending=False)
            
            games_df = games_df.head(20)
            print(f"Using {len(games_df)} most recent games")
            
            stats = {
                'games_played': len(games_df),
                'points': self._get_stat_dict(games_df, 'PTS'),
                'assists': self._get_stat_dict(games_df, 'AST'),
                'rebounds': self._get_stat_dict(games_df, 'REB'),
                'steals': self._get_stat_dict(games_df, 'STL'),
                'blocks': self._get_stat_dict(games_df, 'BLK'),
                'turnovers': self._get_stat_dict(games_df, 'TOV'),
                'three_pointers': self._get_stat_dict(games_df, 'FG3M'),
                'double_double': self._get_double_double_stats(games_df),
                'triple_double': self._get_triple_double_stats(games_df)
            }
            
            stats['combined_stats'] = {
                'pts_reb': self._get_combined_stat_dict(games_df, ['PTS', 'REB']),
                'pts_ast': self._get_combined_stat_dict(games_df, ['PTS', 'AST']),
                'ast_reb': self._get_combined_stat_dict(games_df, ['AST', 'REB']),
                'pts_ast_reb': self._get_combined_stat_dict(games_df, ['PTS', 'AST', 'REB']),
                'stl_blk': self._get_combined_stat_dict(games_df, ['STL', 'BLK'])
            }
            
            stats['dates'] = games_df['GAME_DATE'].dt.strftime('%Y-%m-%d').tolist()
            stats['matchups'] = games_df['MATCHUP'].tolist()
            stats['minutes'] = games_df['MIN'].tolist()
            stats['last_game_date'] = games_df['GAME_DATE'].max().strftime('%Y-%m-%d')
            
            stats['trends'] = self._calculate_trends(games_df)
            
            return stats
            
        except Exception as e:
            print(f"Error getting player stats: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_stat_dict(self, df, column):
        values = df[column].tolist()
        return {
            'values': values,
            'avg': float(df[column].mean()),
            'last5_avg': float(df[column].head().mean()),
            'max': float(df[column].max()),
            'min': float(df[column].min())
        }

    def _get_combined_stat_dict(self, df, columns):
        #Helper method
        combined = df[columns].sum(axis=1)
        return {
            'values': combined.tolist(),
            'avg': float(combined.mean()),
            'last5_avg': float(combined.head().mean())
        }

    def _get_double_double_stats(self, df):
        #Calculate double-double stats
        stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        double_doubles = df[stats].apply(lambda x: sum(x >= 10) >= 2, axis=1)
        return {
            'values': double_doubles.astype(int).tolist(),
            'avg': float(double_doubles.mean()),
            'last5_avg': float(double_doubles.head().mean())
        }

    def _get_triple_double_stats(self, df):
        #Calculate triple-double stats
        stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        triple_doubles = df[stats].apply(lambda x: sum(x >= 10) >= 3, axis=1)
        return {
            'values': triple_doubles.astype(int).tolist(),
            'avg': float(triple_doubles.mean()),
            'last5_avg': float(triple_doubles.head().mean())
        }

    def _calculate_trends(self, df):
        #Calculate performance trends
        trends = {}
        stats = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FG3M']
        
        for stat in stats:
            values = df[stat].values
            if len(values) >= 5:
                recent_values = values[:5]
                z = np.polyfit(range(len(recent_values)), recent_values, 1)
                slope = z[0]
                
                trends[stat.lower()] = {
                    'slope': float(slope),
                    'direction': 'Increasing' if slope > 0.1 else 'Decreasing' if slope < -0.1 else 'Stable',
                    'strength': abs(float(slope))
                }
        
        return trends

    async def analyze_prop_bet(self, player_id, prop_type, line, opponent_team_id):
        """Analyze prop bet for given player and line"""
        try:
            stats = self.get_player_stats(player_id)
            if not stats:
                return {
                    'success': False,
                    'error': 'Unable to retrieve player stats'
                }

            # Get more context
            player_context = self.ml_predictor.get_player_context(player_id, opponent_team_id)
            team_id = self._get_player_team_id(player_id)
            team_context = self.ml_predictor.get_team_context(team_id) if team_id else None
            opponent_context = self.ml_predictor.get_team_context(opponent_team_id)

            if prop_type == 'three_pointers':
                stat_data = stats.get('three_pointers', {})
                trend = stats.get('trends', {}).get('fg3m', {})
            elif prop_type in ['double_double', 'triple_double']:
                stat_data = stats.get(prop_type, {})
                trend = {}
            elif prop_type in ['points', 'assists', 'rebounds', 'steals', 'blocks', 'turnovers']:
                stat_data = stats.get(prop_type, {})
                trend = stats.get('trends', {}).get(prop_type, {})
            else:
                stat_data = stats.get('combined_stats', {}).get(prop_type, {})
                trend = {}

            values = stat_data.get('values', [])
            if not values:
                return {
                    'success': False,
                    'error': 'No historical data available'
                }

            hits = sum(1 for x in values if x > line)
            hit_rate = hits / len(values) if values else 0
            edge = ((stat_data.get('avg', 0) - line) / line) if line > 0 else 0

            features = {
                'recent_avg': float(stat_data.get('last5_avg', 0)),
                'season_avg': float(stat_data.get('avg', 0)), 
                'max_recent': float(max(values) if values else 0),
                'min_recent': float(min(values) if values else 0),
                'stddev': float(np.std(values) if values else 0),
                'games_played': len(values),
                'hit_rate': hit_rate,
                'edge': edge
            }

            # Advanced ML prediction ensemble
            ml_predictions = []
            
            # Standard ML prediction
            standard_prediction = self.ml_predictor.predict(features, line)
            if standard_prediction:
                ml_predictions.append({
                    'model': 'standard',
                    'weight': 0.3,
                    **standard_prediction
                })
            
            # TensorFlow prediction
            if self.tensorflow_predictor:
                tf_prediction = await self._get_tensorflow_prediction(player_id, prop_type, line, stats)
                if tf_prediction:
                    ml_predictions.append({
                        'model': 'tensorflow',
                        'weight': 0.4,
                        **tf_prediction
                    })
            
            # PyTorch prediction
            if self.pytorch_predictor:
                pytorch_prediction = await self._get_pytorch_prediction(player_id, prop_type, line, stats)
                if pytorch_prediction:
                    ml_predictions.append({
                        'model': 'pytorch', 
                        'weight': 0.3,
                        **pytorch_prediction
                    })
            
            # Ensemble prediction
            if ml_predictions:
                ml_prediction = self._ensemble_predictions(ml_predictions)
            else:
                ml_prediction = {
                    'over_probability': hit_rate,
                    'predicted_value': stat_data.get('avg', line),
                    'recommendation': 'PASS',
                    'confidence': 'LOW',
                    'edge': edge,
                    'model_consensus': 'NONE'
                }

            # Get live odds if available
            live_odds_data = None
            if self.odds_aggregator:
                live_odds_data = await self._get_live_odds_for_prop(player_id, prop_type, line)
            
            # Get situational analysis
            situation_analysis = self.situational_analyzer.analyze_game_situation(
                team_id, opponent_team_id, datetime.now().strftime('%Y-%m-%d')
            )
            
            return {
                'success': True,
                'hit_rate': hit_rate,
                'average': stat_data.get('avg', 0),
                'last5_average': stat_data.get('last5_avg', 0),
                'times_hit': hits,
                'total_games': len(values),
                'edge': edge,
                'trend': trend,
                'values': values,
                'predicted_value': ml_prediction.get('predicted_value', stat_data.get('avg', 0)),
                'over_probability': ml_prediction.get('over_probability', hit_rate),
                'recommendation': ml_prediction.get('recommendation', 'PASS'),
                'confidence': ml_prediction.get('confidence', 'LOW'),
                'model_consensus': ml_prediction.get('model_consensus', 'STANDARD_ONLY'),
                'ai_insights': ml_prediction.get('ai_insights', {}),
                'context': {
                    'player': player_context,
                    'team': team_context,
                    'opponent': opponent_context,
                    'situation': situation_analysis
                },
                'live_odds': live_odds_data,
                'enterprise_features': {
                    'tensorflow_enabled': self.tensorflow_predictor is not None,
                    'pytorch_enabled': self.pytorch_predictor is not None,
                    'cloud_enabled': self.aws_integration is not None,
                    'live_odds_enabled': self.odds_aggregator is not None
                }
            }

        except Exception as e:
            print(f"Error analyzing prop bet: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_player_team_id(self, player_id):
        """Helper method to get player's current team ID"""
        try:
            player_info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            time.sleep(0.6)  # Rate limiting
            return int(player_info['TEAM_ID'].iloc[0])
        except Exception as e:
            print(f"Error getting player team ID: {e}")
            return None
    
    def _get_final_recommendation(self, base_recommendation, confidence_score, situation_boost):
        """Generate final recommendation considering all factors"""
        if situation_boost > 0.15 and confidence_score > 80:
            if base_recommendation in ['OVER', 'UNDER']:
                return f'STRONG {base_recommendation}'
            elif base_recommendation in ['LEAN OVER', 'LEAN UNDER']:
                return base_recommendation.replace('LEAN ', '')
        
        if situation_boost > 0.1 and confidence_score > 70:
            if base_recommendation == 'PASS':
                return 'LEAN SITUATIONAL'
        
        return base_recommendation
    
    def analyze_multiple_props(self, props_list):
        """Analyze multiple props for parlay opportunities"""
        try:
            analyzed_props = []
            
            for prop in props_list:
                analysis = self.analyze_prop_bet(
                    prop['player_id'],
                    prop['prop_type'],
                    prop['line'],
                    prop['opponent_team_id']
                )
                
                if analysis.get('success'):
                    prop_data = {
                        'player_id': prop['player_id'],
                        'player_name': prop.get('player_name', 'Unknown'),
                        'prop_type': prop['prop_type'],
                        'line': prop['line'],
                        'over_probability': analysis.get('over_probability', 0.5),
                        'confidence_score': analysis.get('confidence_score', 50),
                        'edge': analysis.get('edge', 0),
                        'recommendation': analysis.get('recommendation', 'PASS'),
                        'predicted_value': analysis.get('predicted_value', prop['line'])
                    }
                    analyzed_props.append(prop_data)
            
            # Find parlay opportunities
            parlay_analysis = self.parlay_optimizer.analyze_single_game_parlay(analyzed_props)
            
            return {
                'success': True,
                'individual_props': analyzed_props,
                'parlay_opportunities': parlay_analysis,
                'summary': {
                    'total_props_analyzed': len(analyzed_props),
                    'positive_ev_individual': len([p for p in analyzed_props if p['edge'] > 0]),
                    'high_confidence_props': len([p for p in analyzed_props if p['confidence_score'] > 70]),
                    'parlay_recommendations': parlay_analysis.get('positive_ev_parlays', [])[:3]
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_bankroll_dashboard(self):
        """Get comprehensive bankroll management dashboard"""
        try:
            bankroll_info = self.bankroll_manager.get_bankroll_info()
            performance_metrics = self.bankroll_manager.calculate_performance_metrics()
            recent_bets = self.bankroll_manager.get_bet_history(20)
            
            return {
                'bankroll_info': bankroll_info,
                'performance_metrics': performance_metrics,
                'recent_bets': recent_bets,
                'recommendations': self._generate_bankroll_recommendations(bankroll_info, performance_metrics)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_bankroll_recommendations(self, bankroll_info, performance_metrics):
        """Generate bankroll management recommendations"""
        recommendations = []
        
        roi = performance_metrics.get('roi', 0)
        win_rate = performance_metrics.get('win_rate', 0)
        total_bets = performance_metrics.get('total_bets', 0)
        
        if total_bets < 10:
            recommendations.append("Build more betting history before adjusting unit sizes")
        
        if roi > 15:
            recommendations.append("Excellent ROI! Consider slightly increasing unit sizes")
        elif roi < -10:
            recommendations.append("Poor ROI. Reduce unit sizes and focus on higher confidence bets")
        
        if win_rate > 60:
            recommendations.append("High win rate suggests good selection. Maintain current strategy")
        elif win_rate < 45:
            recommendations.append("Low win rate. Focus on higher confidence predictions only")
        
        return recommendations
    
    def get_player_name(self, player_id):
        """Get player name from ID"""
        try:
            all_players = players.get_players()
            player = next((p for p in all_players if p['id'] == player_id), None)
            return player['full_name'] if player else 'Unknown Player'
        except:
            return 'Unknown Player'
    
    async def _get_tensorflow_prediction(self, player_id, prop_type, line, stats):
        """Get prediction from TensorFlow models"""
        try:
            if not self.tensorflow_predictor:
                return None
            
            # Prepare feature data for TensorFlow model
            feature_data = self._prepare_tensorflow_features(stats, prop_type)
            
            # Get ensemble prediction from multiple TF models
            prediction = await asyncio.to_thread(
                self.tensorflow_predictor.predict_ensemble, 
                feature_data, 
                line, 
                prop_type
            )
            
            return {
                'over_probability': prediction.get('probability', 0.5),
                'predicted_value': prediction.get('predicted_value', line),
                'confidence': prediction.get('confidence', 'MEDIUM'),
                'recommendation': prediction.get('recommendation', 'PASS'),
                'ai_insights': {
                    'lstm_confidence': prediction.get('lstm_confidence', 0),
                    'transformer_confidence': prediction.get('transformer_confidence', 0),
                    'cnn_confidence': prediction.get('cnn_confidence', 0),
                    'attention_weights': prediction.get('attention_weights', [])
                }
            }
        except Exception as e:
            print(f"TensorFlow prediction error: {e}")
            return None
    
    async def _get_pytorch_prediction(self, player_id, prop_type, line, stats):
        """Get prediction from PyTorch models"""
        try:
            if not self.pytorch_predictor:
                return None
            
            # Prepare feature data for PyTorch model
            feature_data = self._prepare_pytorch_features(stats, prop_type)
            
            # Get prediction from PyTorch ensemble
            prediction = await asyncio.to_thread(
                self.pytorch_predictor.predict_advanced,
                feature_data,
                line,
                prop_type
            )
            
            return {
                'over_probability': prediction.get('probability', 0.5),
                'predicted_value': prediction.get('predicted_value', line),
                'confidence': prediction.get('confidence', 'MEDIUM'),
                'recommendation': prediction.get('recommendation', 'PASS'),
                'ai_insights': {
                    'gnn_score': prediction.get('gnn_score', 0),
                    'vae_latent_score': prediction.get('vae_latent_score', 0),
                    'gan_generated_scenarios': prediction.get('scenarios', []),
                    'attention_map': prediction.get('attention_map', {})
                }
            }
        except Exception as e:
            print(f"PyTorch prediction error: {e}")
            return None
    
    def _prepare_tensorflow_features(self, stats, prop_type):
        """Prepare features for TensorFlow models"""
        # Create sequence data for LSTM/Transformer models
        if prop_type == 'points':
            values = stats.get('points', {}).get('values', [])
        elif prop_type == 'rebounds':
            values = stats.get('rebounds', {}).get('values', [])
        elif prop_type == 'assists':
            values = stats.get('assists', {}).get('values', [])
        else:
            values = stats.get(prop_type, {}).get('values', [])
        
        # Pad or truncate to sequence length
        sequence_length = 10
        if len(values) >= sequence_length:
            sequence = values[:sequence_length]
        else:
            sequence = values + [0] * (sequence_length - len(values))
        
        return {
            'sequence': sequence,
            'recent_avg': stats.get(prop_type, {}).get('last5_avg', 0),
            'season_avg': stats.get(prop_type, {}).get('avg', 0),
            'games_played': stats.get('games_played', 0),
            'trend_direction': stats.get('trends', {}).get(prop_type, {}).get('direction', 'Stable')
        }
    
    def _prepare_pytorch_features(self, stats, prop_type):
        """Prepare features for PyTorch models"""
        # Similar to TensorFlow but with additional graph features
        base_features = self._prepare_tensorflow_features(stats, prop_type)
        
        # Add graph neural network features (player connections, team dynamics)
        base_features.update({
            'player_network_centrality': 0.5,  # Placeholder - would be calculated from team data
            'team_chemistry_score': 0.7,       # Placeholder - would be calculated from team stats
            'opponent_defense_rating': 0.6     # Placeholder - would be fetched from opponent data
        })
        
        return base_features
    
    def _ensemble_predictions(self, predictions):
        """Combine predictions from multiple models"""
        if not predictions:
            return None
        
        total_weight = sum(p['weight'] for p in predictions)
        
        # Weighted average of probabilities
        weighted_prob = sum(p['over_probability'] * p['weight'] for p in predictions) / total_weight
        weighted_value = sum(p['predicted_value'] * p['weight'] for p in predictions) / total_weight
        
        # Consensus confidence
        confidences = [p['confidence'] for p in predictions]
        confidence_mapping = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'VERY_HIGH': 4}
        avg_confidence_score = sum(confidence_mapping.get(c, 2) for c in confidences) / len(confidences)
        
        if avg_confidence_score >= 3.5:
            consensus_confidence = 'VERY_HIGH'
        elif avg_confidence_score >= 2.5:
            consensus_confidence = 'HIGH'
        elif avg_confidence_score >= 1.5:
            consensus_confidence = 'MEDIUM'
        else:
            consensus_confidence = 'LOW'
        
        # Generate recommendation based on ensemble
        if weighted_prob > 0.65 and consensus_confidence in ['HIGH', 'VERY_HIGH']:
            recommendation = 'STRONG OVER'
        elif weighted_prob > 0.58:
            recommendation = 'OVER'
        elif weighted_prob > 0.52:
            recommendation = 'LEAN OVER'
        elif weighted_prob < 0.35 and consensus_confidence in ['HIGH', 'VERY_HIGH']:
            recommendation = 'STRONG UNDER'
        elif weighted_prob < 0.42:
            recommendation = 'UNDER'
        elif weighted_prob < 0.48:
            recommendation = 'LEAN UNDER'
        else:
            recommendation = 'PASS'
        
        # Aggregate AI insights
        ai_insights = {}
        for pred in predictions:
            if 'ai_insights' in pred:
                ai_insights[f\"{pred['model']}_insights\"] = pred['ai_insights']
        
        return {
            'over_probability': weighted_prob,
            'predicted_value': weighted_value,
            'confidence': consensus_confidence,
            'recommendation': recommendation,
            'model_consensus': f\"{len(predictions)} models agree\",
            'ai_insights': ai_insights,
            'edge': (weighted_value - weighted_prob) if weighted_prob > 0 else 0
        }
    
    async def _get_live_odds_for_prop(self, player_id, prop_type, line):
        """Get live odds data for a specific prop"""
        try:
            if not self.odds_aggregator:
                return None
            
            player_name = self.get_player_name(player_id)
            odds_data = await self.odds_aggregator.get_player_prop_odds(player_name, prop_type)
            
            if odds_data:
                return {
                    'best_over_odds': odds_data.get('best_over_odds'),
                    'best_under_odds': odds_data.get('best_under_odds'),
                    'line_movement': odds_data.get('line_movement', []),
                    'market_consensus': odds_data.get('market_consensus', {}),
                    'arbitrage_opportunities': odds_data.get('arbitrage_opportunities', []),
                    'sportsbooks': odds_data.get('sportsbooks', []),
                    'last_updated': odds_data.get('timestamp')
                }
            
            return None
        except Exception as e:
            print(f\"Live odds error: {e}\")
            return None
    
    async def get_enterprise_analytics_dashboard(self):
        \"\"\"Get comprehensive enterprise analytics dashboard\"\"\"
        try:
            dashboard_data = {}
            
            # Cloud storage analytics
            if self.aws_integration:
                cloud_metrics = await self.aws_integration.get_system_metrics()
                dashboard_data['cloud_metrics'] = cloud_metrics
            
            # Live market overview
            if self.odds_aggregator:
                market_overview = await self.odds_aggregator.get_market_overview()
                dashboard_data['market_overview'] = market_overview
            
            # AI model performance
            if self.tensorflow_predictor or self.pytorch_predictor:
                model_performance = await self._get_ai_model_performance()
                dashboard_data['ai_performance'] = model_performance
            
            # User analytics (if authenticated)
            if self.user_manager and self.user_id:
                user_analytics = await self.user_manager.get_user_analytics(self.user_id)
                dashboard_data['user_analytics'] = user_analytics
            
            return {
                'success': True,
                'dashboard_data': dashboard_data,
                'enterprise_status': {
                    'tensorflow_active': self.tensorflow_predictor is not None,
                    'pytorch_active': self.pytorch_predictor is not None,
                    'cloud_active': self.aws_integration is not None,
                    'live_odds_active': self.odds_aggregator is not None,
                    'auth_active': self.user_manager is not None
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _get_ai_model_performance(self):
        \"\"\"Get AI model performance metrics\"\"\"
        performance = {}
        
        if self.tensorflow_predictor:
            tf_metrics = await asyncio.to_thread(self.tensorflow_predictor.get_model_metrics)
            performance['tensorflow'] = tf_metrics
        
        if self.pytorch_predictor:
            pytorch_metrics = await asyncio.to_thread(self.pytorch_predictor.get_model_metrics)
            performance['pytorch'] = pytorch_metrics
        
        return performance
    
