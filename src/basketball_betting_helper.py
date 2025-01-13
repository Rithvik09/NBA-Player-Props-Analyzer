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
from .models import EnhancedMLPredictor 


class BasketballBettingHelper:
    def __init__(self, db_name='basketball_data.db'):
        self.db_name = db_name
        self.ml_predictor = EnhancedMLPredictor()
        
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

    def analyze_prop_bet(self, player_id, prop_type, line, opponent_team_id):
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

            # ML prediction
            ml_prediction = self.ml_predictor.predict(features, line)
            if not ml_prediction:
                ml_prediction = {
                    'over_probability': hit_rate,
                    'predicted_value': stat_data.get('avg', line),
                    'recommendation': 'PASS',
                    'confidence': 'LOW',
                    'edge': edge
                }

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
                'context': {
                    'player': player_context,
                    'team': team_context,
                    'opponent': opponent_context
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
    
