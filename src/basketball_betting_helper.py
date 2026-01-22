import pandas as pd
import numpy as np
from nba_api.stats.endpoints import (
    playergamelog, CommonPlayerInfo, PlayerVsPlayer, TeamDashboardByGeneralSplits,
    TeamGameLog, CommonTeamRoster, LeagueGameFinder,
    PlayerDashboardByGeneralSplits, PlayerDashboardByShootingSplits,
    PlayerDashboardByClutch
)
from nba_api.stats.static import players
import sqlite3
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from .models import EnhancedMLPredictor 
import json
from .precomputed_store import PrecomputedStore
import threading
from .precompute_jobs import update_precomputed
from .background_jobs import BackgroundJobRunner, load_config
from .arena_data import get_arena_info, get_home_court_advantage, calculate_travel_metrics


class BasketballBettingHelper:
    def __init__(self, db_name='basketball_data.db'):
        self.db_name = db_name
        self.ml_predictor = EnhancedMLPredictor()
        self.precomputed = PrecomputedStore(db_name)
        self._precompute_lock = threading.Lock()
        self._precompute_in_flight = False
        self._precompute_last_attempt = 0
        self._precompute_min_interval_seconds = 60 * 30  # 30 minutes cooldown between attempts
        
        current_year = datetime.now().year
        current_month = datetime.now().month
        

        if 1 <= current_month <= 7:
            self.current_season = f"{current_year-1}-{str(current_year)[2:]}"
        else:
            self.current_season = f"{current_year}-{str(current_year+1)[2:]}"
            
        print(f"Current season set to: {self.current_season}")
        self.create_tables()

        self.kick_precompute_if_stale()

        self._auto_jobs = None
        self._start_auto_jobs()

    def _start_auto_jobs(self):
        try:
            cfg = load_config(db_path=self.db_name, season=self.current_season, models_dir=self.ml_predictor.model_dir)

            def _on_models_updated():
                try:
                    # Reload trained models into the running app
                    self.ml_predictor._load_trained_models()
                except Exception:
                    return

            def _on_precompute_updated():
                try:
                    self.precomputed.refresh(force=True)
                except Exception:
                    return

            self._auto_jobs = BackgroundJobRunner(cfg, on_models_updated=_on_models_updated, on_precompute_updated=_on_precompute_updated)
            self._auto_jobs.start()
        except Exception:
            return

    def kick_precompute_if_stale(self, force: bool = False) -> bool:
        try:
            if not force and self.precomputed.is_fresh():
                return False

            now = int(time.time())
            if not force and (now - int(self._precompute_last_attempt) < self._precompute_min_interval_seconds):
                return False

            with self._precompute_lock:
                if self._precompute_in_flight:
                    return False
                self._precompute_in_flight = True
                self._precompute_last_attempt = now

            def _run():
                try:
                    print("[precompute] auto-update starting...")
                    summary = update_precomputed(db_path=self.db_name, season=self.current_season)
                    # Refresh in-process cache so requests immediately see new data
                    self.precomputed.refresh(force=True)
                    print("[precompute] auto-update done:", summary)
                except Exception as e:
                    print("[precompute] auto-update failed:", e)
                finally:
                    with self._precompute_lock:
                        self._precompute_in_flight = False

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            return True
        except Exception:
            return False
        
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

        # Simple TTL cache for expensive API computations (player stats, etc.)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key TEXT PRIMARY KEY,
                cache_value TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
        ''')

        # Incremental training state: track last processed game date per player/season
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incremental_training_state (
                season TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                last_game_date TEXT,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (season, player_id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def _cache_get(self, cache_key, ttl_seconds):
        try:
            conn = self.get_db()
            cursor = conn.cursor()
            cursor.execute('SELECT cache_value, created_at FROM api_cache WHERE cache_key = ?', (cache_key,))
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            value_json, created_at = row
            if int(time.time()) - int(created_at) > int(ttl_seconds):
                return None
            return json.loads(value_json)
        except Exception:
            return None

    def _cache_set(self, cache_key, value):
        try:
            conn = self.get_db()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR REPLACE INTO api_cache (cache_key, cache_value, created_at) VALUES (?, ?, ?)',
                (cache_key, json.dumps(value), int(time.time()))
            )
            conn.commit()
            conn.close()
        except Exception:
            return

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
            cache_key = f"player_stats:{player_id}:{self.current_season}"
            cached = self._cache_get(cache_key, ttl_seconds=6 * 60 * 60)  # 6 hours
            if cached:
                return cached

            current_year = datetime.now().year
            current_month = datetime.now().month
            
            if 1 <= current_month <= 7:
                current_season = f"{current_year-1}-{str(current_year)[2:]}"
                previous_season = f"{current_year-2}-{str(current_year-1)[2:]}"
            else:
                current_season = f"{current_year}-{str(current_year+1)[2:]}"
                previous_season = f"{current_year-1}-{str(current_year)[2:]}"
            
            # Start with current season; only fall back if we don't have enough games
            seasons = [current_season]
            print(f"Fetching seasons: {seasons} (fallback to previous if needed)")
            
            all_games = []
            
            for season in seasons:
                try:
                    gamelog = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season
                    )
                    games = gamelog.get_data_frames()[0]
                    print(f"Found {len(games)} games for {season}")
                    all_games.append(games)
                except Exception as e:
                    print(f"Error fetching {season} data: {e}")
                    continue

            # Fallback to previous season only when necessary
            if all_games and len(all_games[0]) < 10:
                try:
                    gamelog = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=previous_season
                    )
                    games = gamelog.get_data_frames()[0]
                    print(f"Fallback: found {len(games)} games for {previous_season}")
                    all_games.append(games)
                except Exception as e:
                    print(f"Error fetching fallback season data: {e}")
            
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
            
            # Tier 1 additions: shooting efficiency & volume
            stats['shooting'] = {
                'fg_pct_recent': float(games_df.head(5)['FG_PCT'].mean()) if 'FG_PCT' in games_df.columns else 0.0,
                'fg3_pct_recent': float(games_df.head(5)['FG3_PCT'].mean()) if 'FG3_PCT' in games_df.columns else 0.0,
                'ft_pct_recent': float(games_df.head(5)['FT_PCT'].mean()) if 'FT_PCT' in games_df.columns else 0.0,
                'fga_per_game': float(games_df['FGA'].mean()) if 'FGA' in games_df.columns else 0.0,
                'fg3a_per_game': float(games_df['FG3A'].mean()) if 'FG3A' in games_df.columns else 0.0,
                'fta_per_game': float(games_df['FTA'].mean()) if 'FTA' in games_df.columns else 0.0,
            }
            
            # Tier 1: rebounding split
            stats['rebounding'] = {
                'oreb_per_game': float(games_df['OREB'].mean()) if 'OREB' in games_df.columns else 0.0,
                'dreb_per_game': float(games_df['DREB'].mean()) if 'DREB' in games_df.columns else 0.0,
            }
            
            # Tier 1: impact & context
            stats['impact'] = {
                'plus_minus_avg': float(games_df['PLUS_MINUS'].mean()) if 'PLUS_MINUS' in games_df.columns else 0.0,
                'fouls_per_game': float(games_df['PF'].mean()) if 'PF' in games_df.columns else 0.0,
                'win_rate_last10': float((games_df.head(10)['WL'] == 'W').sum() / min(10, len(games_df))) if 'WL' in games_df.columns else 0.5,
            }
            
            # Tier 2 Quick Wins: momentum trends (per prop, calculate later)
            stats['momentum'] = {}
            for prop_col, prop_name in [('PTS', 'points'), ('AST', 'assists'), ('REB', 'rebounds'), 
                                         ('STL', 'steals'), ('BLK', 'blocks'), ('TOV', 'turnovers'), ('FG3M', 'three_pointers')]:
                if prop_col in games_df.columns:
                    vals = games_df[prop_col].values
                    stats['momentum'][prop_name] = {
                        'last_3_trend': float(np.polyfit(range(min(3, len(vals))), vals[:min(3, len(vals))], 1)[0]) if len(vals) >= 2 else 0.0,
                        'last_5_trend': float(np.polyfit(range(min(5, len(vals))), vals[:min(5, len(vals))], 1)[0]) if len(vals) >= 3 else 0.0,
                        'last_10_trend': float(np.polyfit(range(min(10, len(vals))), vals[:min(10, len(vals))], 1)[0]) if len(vals) >= 5 else 0.0,
                        'above_avg_last5': int((vals[:min(5, len(vals))] > np.mean(vals)).sum()) if len(vals) >= 5 else 0,
                    }
            
            # Tier 2 Quick Wins: schedule/fatigue features
            dates = pd.to_datetime(games_df['GAME_DATE'])
            stats['schedule'] = {
                'is_back_to_back': bool((dates.iloc[0] - dates.iloc[1]).days == 1) if len(dates) >= 2 else False,
                'days_since_last_game': int((datetime.now() - dates.iloc[0]).days) if len(dates) > 0 else 7,
                'games_in_last_7_days': int(((datetime.now() - dates).dt.days <= 7).sum()),
            }
            
            # Fetch advanced metrics (cached separately)
            stats['advanced_metrics'] = self._get_advanced_metrics(player_id, season)
            stats['shot_location'] = self._get_shot_location_data(player_id, season)
            stats['clutch_stats'] = self._get_clutch_stats(player_id, season)
            stats['lineup_context'] = self._get_lineup_context(player_id, season)
            stats['defensive_metrics'] = self._get_defensive_metrics(player_id, season)
            stats['play_type_data'] = self._get_play_type_data(player_id, season)
            # Tier 6: Shot quality, touch data
            stats['shot_quality'] = self._get_shot_quality_metrics(player_id, season)
            stats['touch_usage'] = self._get_touch_usage_data(player_id, season)
            
            # Tier 4: Calculate efficiency metrics from game logs
            stats['calculated_efficiency'] = self._calculate_efficiency_metrics(games_df)
            stats['opponent_adjusted'] = self._calculate_opponent_adjusted_stats(games_df)
            
            # Tier 5: Calculate rotation patterns from game logs
            stats['rotation_patterns'] = self._calculate_rotation_patterns(games_df)
            
            stats['trends'] = self._calculate_trends(games_df)
            
            self._cache_set(cache_key, stats)
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

    def _get_advanced_metrics(self, player_id, season):
        """Fetch advanced player metrics from PlayerDashboardByGeneralSplits"""
        try:
            cache_key = f"advanced_metrics:{player_id}:{season}"
            cached = self._cache_get(cache_key, ttl_seconds=12 * 60 * 60)  # 12 hours
            if cached:
                return cached
            
            time.sleep(0.6)  # Rate limit
            dashboard = PlayerDashboardByGeneralSplits(player_id=player_id, season=season)
            overall = dashboard.get_data_frames()[0]
            
            if overall.empty:
                return {}
            
            row = overall.iloc[0]
            metrics = {
                'usage_rate': float(row.get('USG_PCT', 0.0) or 0.0),
                'true_shooting_pct': float(row.get('TS_PCT', 0.0) or 0.0),
                'effective_fg_pct': float(row.get('EFG_PCT', 0.0) or 0.0),
                'assist_percentage': float(row.get('AST_PCT', 0.0) or 0.0),
                'rebound_percentage': float(row.get('REB_PCT', 0.0) or 0.0),
                'pie': float(row.get('PIE', 0.0) or 0.0),
                # Additional available fields from PlayerDashboardByGeneralSplits
                'off_rating': float(row.get('OFF_RATING', 0.0) or 0.0),
                'def_rating': float(row.get('DEF_RATING', 0.0) or 0.0),
                'net_rating': float(row.get('NET_RATING', 0.0) or 0.0),
                'pace': float(row.get('PACE', 0.0) or 0.0),
                'fta_rate': float(row.get('FTA_RATE', 0.0) or 0.0),
                'pct_fga_2pt': float(row.get('PCT_FGA_2PT', 0.0) or 0.0),
                'pct_fga_3pt': float(row.get('PCT_FGA_3PT', 0.0) or 0.0),
                'pct_pts_2pt': float(row.get('PCT_PTS_2PT', 0.0) or 0.0),
                'pct_pts_2pt_mr': float(row.get('PCT_PTS_2PT_MR', 0.0) or 0.0),
                'pct_pts_3pt': float(row.get('PCT_PTS_3PT', 0.0) or 0.0),
                'pct_pts_fb': float(row.get('PCT_PTS_FB', 0.0) or 0.0),
                'pct_pts_ft': float(row.get('PCT_PTS_FT', 0.0) or 0.0),
                'pct_pts_off_tov': float(row.get('PCT_PTS_OFF_TOV', 0.0) or 0.0),
                'pct_pts_in_paint': float(row.get('PCT_PTS_IN_PAINT', 0.0) or 0.0),
                'pct_ast_2pm': float(row.get('PCT_AST_2PM', 0.0) or 0.0),
                'pct_ast_3pm': float(row.get('PCT_AST_3PM', 0.0) or 0.0),
                'pct_uast_2pm': float(row.get('PCT_UAST_2PM', 0.0) or 0.0),
                'pct_uast_3pm': float(row.get('PCT_UAST_3PM', 0.0) or 0.0),
            }
            
            self._cache_set(cache_key, metrics)
            return metrics
        except Exception as e:
            print(f"Error fetching advanced metrics: {e}")
            return {}
    
    def _get_shot_location_data(self, player_id, season):
        """Fetch shot location data from PlayerDashboardByShootingSplits"""
        try:
            cache_key = f"shot_location:{player_id}:{season}"
            cached = self._cache_get(cache_key, ttl_seconds=12 * 60 * 60)
            if cached:
                return cached
            
            time.sleep(0.6)  # Rate limit
            dashboard = PlayerDashboardByShootingSplits(player_id=player_id, season=season)
            shot_areas = dashboard.get_data_frames()[0]
            
            if shot_areas.empty:
                return {}
            
            # Parse shot area data
            metrics = {
                'rim_fga_per_game': 0.0,
                'paint_fga_per_game': 0.0,
                'mid_range_fga_per_game': 0.0,
                'corner_3_pct': 0.0,
                'above_break_3_pct': 0.0,
            }
            
            for _, row in shot_areas.iterrows():
                shot_area = str(row.get('GROUP_VALUE', '')).lower()
                fga = float(row.get('FGA', 0) or 0)
                gp = float(row.get('GP', 1) or 1)
                fg_pct = float(row.get('FG_PCT', 0.0) or 0.0)
                
                if 'restricted area' in shot_area or 'less than' in shot_area:
                    metrics['rim_fga_per_game'] = fga / max(gp, 1)
                elif 'paint' in shot_area and 'non' in shot_area:
                    metrics['paint_fga_per_game'] = fga / max(gp, 1)
                elif 'mid-range' in shot_area or '10-16' in shot_area:
                    metrics['mid_range_fga_per_game'] = fga / max(gp, 1)
                elif 'corner 3' in shot_area:
                    metrics['corner_3_pct'] = fg_pct
                elif 'above the break 3' in shot_area:
                    metrics['above_break_3_pct'] = fg_pct
            
            self._cache_set(cache_key, metrics)
            return metrics
        except Exception as e:
            print(f"Error fetching shot location data: {e}")
            return {}
    
    def _get_clutch_stats(self, player_id, season):
        """Fetch clutch performance from PlayerDashboardByClutch"""
        try:
            cache_key = f"clutch_stats:{player_id}:{season}"
            cached = self._cache_get(cache_key, ttl_seconds=12 * 60 * 60)
            if cached:
                return cached
            
            time.sleep(0.6)  # Rate limit
            dashboard = PlayerDashboardByClutch(player_id=player_id, season=season)
            clutch_df = dashboard.get_data_frames()[0]
            
            if clutch_df.empty:
                return {}
            
            # Find clutch row (last 5 min, score within 5)
            clutch_row = None
            for _, row in clutch_df.iterrows():
                if 'Last 5 Minutes' in str(row.get('GROUP_VALUE', '')):
                    clutch_row = row
                    break
            
            if clutch_row is None and not clutch_df.empty:
                clutch_row = clutch_df.iloc[0]
            
            if clutch_row is not None:
                gp = float(clutch_row.get('GP', 1) or 1)
                metrics = {
                    'clutch_pts_per_game': float(clutch_row.get('PTS', 0) or 0) / max(gp, 1),
                    'clutch_fg_pct': float(clutch_row.get('FG_PCT', 0.0) or 0.0),
                    'clutch_minutes_per_game': float(clutch_row.get('MIN', 0) or 0) / max(gp, 1),
                }
            else:
                metrics = {}
            
            self._cache_set(cache_key, metrics)
            return metrics
        except Exception as e:
            print(f"Error fetching clutch stats: {e}")
            return {}
    
    def _get_lineup_context(self, player_id, season):
        """Fetch lineup context (simplified - full lineup data is expensive)"""
        try:
            cache_key = f"lineup_context:{player_id}:{season}"
            cached = self._cache_get(cache_key, ttl_seconds=24 * 60 * 60)  # 24 hours
            if cached:
                return cached
            
            # For now, use team-level on/off court data from advanced metrics
            # Full lineup analysis would require LeagueLineups which is very expensive
            time.sleep(0.6)
            dashboard = PlayerDashboardByGeneralSplits(player_id=player_id, season=season)
            overall = dashboard.get_data_frames()[0]
            
            if overall.empty:
                return {}
            
            row = overall.iloc[0]
            metrics = {
                'on_court_plus_minus': float(row.get('PLUS_MINUS', 0.0) or 0.0),
                'off_court_plus_minus': 0.0,  # Not available in this endpoint
                'net_rating': float(row.get('NET_RATING', 0.0) or 0.0),
                'top_lineup_minutes_pct': 0.0,  # Would need expensive lineup query
            }
            
            self._cache_set(cache_key, metrics)
            return metrics
        except Exception as e:
            print(f"Error fetching lineup context: {e}")
            return {}
    
    def _get_shot_quality_metrics(self, player_id, season):
        """Fetch shot quality metrics from NBA tracking data (Tier 6)"""
        try:
            cache_key = f"shot_quality:{player_id}:{season}"
            cached = self._cache_get(cache_key, ttl_seconds=24 * 60 * 60)
            if cached:
                return cached
            
            time.sleep(0.6)
            
            # Get shot dashboard data
            try:
                from nba_api.stats.endpoints import PlayerDashboardByShootingSplits
                shot_dashboard = PlayerDashboardByShootingSplits(player_id=player_id, season=season)
                shot_data = shot_dashboard.get_data_frames()[0]
            except Exception:
                shot_data = pd.DataFrame()
            
            # Calculate shot quality metrics
            metrics = {
                'avg_shot_distance': 0.0,
                'contested_shot_pct': 0.5,  # Default middle value
                'open_shot_pct': 0.3,
                'wide_open_shot_pct': 0.2,
                'catch_and_shoot_pct': 0.3,
                'pull_up_shot_pct': 0.3,
                'paint_touch_frequency': 0.0,
                'corner_three_pct': 0.0,
                'above_break_three_pct': 0.0,
                'restricted_area_fg_pct': 0.0,
                'mid_range_frequency': 0.0,
                'shot_quality_vs_expected': 0.0,
                'avg_shot_clock_time': 12.0,  # Default mid-clock
                'late_clock_shot_frequency': 0.15,
                'early_clock_shot_frequency': 0.25,
            }
            
            # Extract from shot data if available
            if not shot_data.empty:
                # Try to find specific shot types in the data
                for _, row in shot_data.iterrows():
                    shot_type = str(row.get('GROUP_VALUE', '')).lower()
                    fga = float(row.get('FGA', 0) or 0)
                    fgpct = float(row.get('FG_PCT', 0) or 0)
                    
                    if 'corner' in shot_type and '3' in shot_type:
                        metrics['corner_three_pct'] = fgpct
                    elif 'above' in shot_type or 'break' in shot_type:
                        metrics['above_break_three_pct'] = fgpct
                    elif 'restricted' in shot_type or 'paint' in shot_type:
                        metrics['restricted_area_fg_pct'] = fgpct
                    elif 'mid' in shot_type or 'range' in shot_type:
                        metrics['mid_range_frequency'] = fga
            
            self._cache_set(cache_key, metrics)
            return metrics
        except Exception as e:
            print(f"Error fetching shot quality metrics: {e}")
            return {}
    
    def _get_touch_usage_data(self, player_id, season):
        """Fetch touch and usage details from NBA tracking (Tier 6)"""
        try:
            cache_key = f"touch_usage:{player_id}:{season}"
            cached = self._cache_get(cache_key, ttl_seconds=24 * 60 * 60)
            if cached:
                return cached
            
            time.sleep(0.6)
            
            # Get tracking data (touches, dribbles, etc.)
            # Note: This data is available but endpoints vary by season
            # Using simplified approach with available data
            metrics = {
                'touches_per_game': 0.0,
                'avg_dribbles_per_touch': 0.0,
                'avg_seconds_per_touch': 0.0,
                'elbow_touches_per_game': 0.0,
                'post_touches_per_game': 0.0,
                'paint_touches_per_game': 0.0,
                'front_court_touches_per_game': 0.0,
                'time_of_possession_per_game': 0.0,
                'touches_per_possession': 0.0,
                'avg_points_per_touch': 0.0,
            }
            
            # Try to get player dashboard for usage indicators
            try:
                dashboard = PlayerDashboardByGeneralSplits(player_id=player_id, season=season)
                overall = dashboard.get_data_frames()[0]
                
                if not overall.empty:
                    row = overall.iloc[0]
                    # Estimate touches from usage and possessions
                    usage_pct = float(row.get('USG_PCT', 0) or 0)
                    pace = float(row.get('PACE', 100) or 100)
                    mins = float(row.get('MIN', 0) or 0)
                    pts = float(row.get('PTS', 0) or 0)
                    
                    # Rough estimates based on usage
                    metrics['touches_per_game'] = usage_pct * 0.5  # Rough estimate
                    metrics['time_of_possession_per_game'] = (usage_pct / 100.0) * mins * 0.3
                    if metrics['touches_per_game'] > 0:
                        metrics['avg_points_per_touch'] = pts / max(metrics['touches_per_game'], 1.0)
            except Exception:
                pass
            
            self._cache_set(cache_key, metrics)
            return metrics
        except Exception as e:
            print(f"Error fetching touch/usage data: {e}")
            return {}
    
    def _get_defensive_metrics(self, player_id, season):
        """Fetch advanced defensive metrics (Note: Not available in free NBA API)"""
        try:
            # Advanced defensive metrics like deflections and contested shots
            # are not available in the free NBA API endpoints
            # These would require paid access to Second Spectrum or Synergy data
            cache_key = f"defensive_metrics:{player_id}:{season}"
            cached = self._cache_get(cache_key, ttl_seconds=12 * 60 * 60)
            if cached:
                return cached
            
            # Return placeholders for now
            metrics = {
                'def_fg_pct_allowed': 0.0,
                'def_rating_individual': 0.0,
                'deflections_per_game': 0.0,
                'contested_shots_per_game': 0.0,
            }
            
            self._cache_set(cache_key, metrics)
            return metrics
        except Exception as e:
            print(f"Error fetching defensive metrics: {e}")
            return {}
    
    def _get_play_type_data(self, player_id, season):
        """Fetch play type data (Note: This endpoint may not exist in nba_api, using approximation)"""
        try:
            cache_key = f"play_type_data:{player_id}:{season}"
            cached = self._cache_get(cache_key, ttl_seconds=12 * 60 * 60)
            if cached:
                return cached
            
            # PlayType data is not directly available in free NBA API
            # We'll approximate using general splits and shooting data
            time.sleep(0.6)
            dashboard = PlayerDashboardByGeneralSplits(player_id=player_id, season=season)
            overall = dashboard.get_data_frames()[0]
            
            if overall.empty:
                return {}
            
            # Approximate play type percentages based on available data
            # These would ideally come from Synergy or Second Spectrum
            metrics = {
                'pnr_ball_handler_pct': 0.0,  # Not available in free API
                'pnr_roll_man_pct': 0.0,
                'isolation_pct': 0.0,
                'spot_up_pct': 0.0,
                'post_up_pct': 0.0,
                'transition_pct': 0.0,
            }
            
            self._cache_set(cache_key, metrics)
            return metrics
        except Exception as e:
            print(f"Error fetching play type data: {e}")
            return {}
    
    def _calculate_efficiency_metrics(self, games_df):
        """Calculate efficiency metrics from game logs (Tier 4)"""
        try:
            if games_df.empty:
                return {}
            
            # Points per shot
            total_fga = games_df['FGA'].sum() if 'FGA' in games_df.columns else 1
            total_pts = games_df['PTS'].sum() if 'PTS' in games_df.columns else 0
            points_per_shot = total_pts / max(total_fga, 1)
            
            # AST/TOV ratio
            total_ast = games_df['AST'].sum() if 'AST' in games_df.columns else 0
            total_tov = games_df['TOV'].sum() if 'TOV' in games_df.columns else 1
            ast_to_tov_ratio = total_ast / max(total_tov, 1)
            
            # Rebound rate per 36
            total_reb = games_df['REB'].sum() if 'REB' in games_df.columns else 0
            total_min = games_df['MIN'].sum() if 'MIN' in games_df.columns else 1
            reb_rate_per_36 = (total_reb / max(total_min, 1)) * 36
            
            # Scoring efficiency trend (last 5 vs season)
            recent_pts_per_shot = games_df.head(5)['PTS'].sum() / max(games_df.head(5)['FGA'].sum(), 1) if 'FGA' in games_df.columns else 0
            scoring_efficiency_trend = recent_pts_per_shot - points_per_shot
            
            # Usage trend (FGA trend)
            recent_usage = games_df.head(5)['FGA'].mean() if 'FGA' in games_df.columns else 0
            season_usage = games_df['FGA'].mean() if 'FGA' in games_df.columns else 0
            usage_trend = recent_usage - season_usage
            
            # Minutes volatility (std dev)
            minutes_volatility = float(games_df['MIN'].std()) if 'MIN' in games_df.columns else 0.0
            
            # Blowout & close game percentages
            plus_minus_col = 'PLUS_MINUS' if 'PLUS_MINUS' in games_df.columns else None
            if plus_minus_col:
                blowout_games = (games_df[plus_minus_col].abs() > 15).sum()
                close_games = (games_df[plus_minus_col].abs() <= 5).sum()
                total_games = len(games_df)
                blowout_game_pct = blowout_games / max(total_games, 1)
                close_game_pct = close_games / max(total_games, 1)
            else:
                blowout_game_pct = 0.0
                close_game_pct = 0.0
            
            return {
                'points_per_shot': float(points_per_shot),
                'ast_to_tov_ratio': float(ast_to_tov_ratio),
                'reb_rate_per_36': float(reb_rate_per_36),
                'scoring_efficiency_trend': float(scoring_efficiency_trend),
                'usage_trend': float(usage_trend),
                'minutes_volatility': float(minutes_volatility),
                'blowout_game_pct': float(blowout_game_pct),
                'close_game_pct': float(close_game_pct),
            }
        except Exception as e:
            print(f"Error calculating efficiency metrics: {e}")
            return {}
    
    def _calculate_opponent_adjusted_stats(self, games_df):
        """Calculate opponent-adjusted stats from game logs (Tier 4)"""
        try:
            if games_df.empty:
                return {}
            
            # For now, we'll use simplified logic
            # Ideally, we'd fetch opponent defensive ratings and adjust
            
            # Consistency score (inverse of coefficient of variation)
            pts_mean = games_df['PTS'].mean() if 'PTS' in games_df.columns else 0
            pts_std = games_df['PTS'].std() if 'PTS' in games_df.columns else 0
            consistency_score = 1.0 - (pts_std / max(pts_mean, 1))
            
            # Ceiling game frequency (games with 1.5x average or more)
            ceiling_threshold = pts_mean * 1.5
            ceiling_games = (games_df['PTS'] >= ceiling_threshold).sum() if 'PTS' in games_df.columns else 0
            ceiling_game_frequency = ceiling_games / max(len(games_df), 1)
            
            # For top/bottom defenses, we'd need team defensive ratings
            # Using placeholders for now
            return {
                'pts_vs_top10_defenses': float(pts_mean),  # Placeholder
                'pts_vs_bottom10_defenses': float(pts_mean),  # Placeholder
                'consistency_score': float(max(0.0, consistency_score)),
                'ceiling_game_frequency': float(ceiling_game_frequency),
            }
        except Exception as e:
            print(f"Error calculating opponent-adjusted stats: {e}")
            return {}
    
    def _calculate_player_streaks(self, games_df, prop_type, line):
        """Calculate player streaks vs betting lines (Tier 5)"""
        try:
            if games_df.empty or len(games_df) < 3:
                return {
                    'consecutive_over_games': 0,
                    'consecutive_under_games': 0,
                    'hot_hand_indicator': 0.0,
                    'recent_variance_spike': 0.0,
                }
            
            # Handle simple DataFrame with 'stat' column
            if 'stat' in games_df.columns:
                stat_col = 'stat'
            else:
                # Map prop_type to stat column for full game log
                stat_col_map = {
                    'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB',
                    'three_pointers': 'FG3M', 'steals': 'STL', 'blocks': 'BLK',
                    'turnovers': 'TOV'
                }
                stat_col = stat_col_map.get(prop_type, 'PTS')
            
            if stat_col not in games_df.columns:
                return {'consecutive_over_games': 0, 'consecutive_under_games': 0, 
                       'hot_hand_indicator': 0.0, 'recent_variance_spike': 0.0}
            
            # Calculate consecutive overs/unders
            recent_games = games_df.head(10)  # Last 10 games
            over_under = (recent_games[stat_col] > line).astype(int)
            
            # Count consecutive from most recent
            consecutive_over = 0
            consecutive_under = 0
            for val in over_under.values:
                if val == 1:
                    consecutive_over += 1
                    break
                consecutive_under += 1
            
            # Continue counting if streak continues
            if consecutive_over > 0:
                for val in over_under.values[1:]:
                    if val == 1:
                        consecutive_over += 1
                    else:
                        break
            elif consecutive_under > 0:
                for val in over_under.values[1:]:
                    if val == 0:
                        consecutive_under += 1
                    else:
                        break
            
            # Hot hand indicator (hit rate in last 5 > season rate)
            last5_hit_rate = float((games_df.head(5)[stat_col] > line).sum() / 5) if len(games_df) >= 5 else 0.0
            season_hit_rate = float((games_df[stat_col] > line).sum() / len(games_df))
            hot_hand_indicator = max(0.0, last5_hit_rate - season_hit_rate)
            
            # Recent variance spike (last 3 std / season std)
            season_std = games_df[stat_col].std()
            recent_std = games_df.head(3)[stat_col].std() if len(games_df) >= 3 else season_std
            recent_variance_spike = (recent_std / max(season_std, 0.1)) - 1.0 if season_std > 0 else 0.0
            
            return {
                'consecutive_over_games': int(consecutive_over),
                'consecutive_under_games': int(consecutive_under),
                'hot_hand_indicator': float(hot_hand_indicator),
                'recent_variance_spike': float(min(max(recent_variance_spike, -1.0), 2.0)),
            }
        except Exception as e:
            print(f"Error calculating player streaks: {e}")
            return {'consecutive_over_games': 0, 'consecutive_under_games': 0, 
                   'hot_hand_indicator': 0.0, 'recent_variance_spike': 0.0}
    
    def _calculate_game_importance(self, team_id, opponent_team_id, season):
        """Calculate game importance indicators (Tier 5)"""
        try:
            # Simplified version - would need standings API for full implementation
            # For now, use placeholders with smart defaults
            
            import datetime
            today = datetime.datetime.now()
            month = today.month
            day = today.day
            
            # Playoff seeding impact (higher late season)
            if month >= 4:  # Late season
                playoff_seeding_impact = 0.8
            elif month >= 3:  # Mid season
                playoff_seeding_impact = 0.5
            else:
                playoff_seeding_impact = 0.2
            
            # Tanking indicator (late season, bottom teams)
            # Would need standings to implement properly
            tanking_indicator = 0.0
            
            # Must win situation (simplified - late season indicator)
            must_win_situation = 1.0 if (month == 4 and day > 7) else 0.0
            
            # Games back from playoff (would need standings)
            games_back_from_playoff = 0.0
            
            return {
                'playoff_seeding_impact': float(playoff_seeding_impact),
                'tanking_indicator': float(tanking_indicator),
                'must_win_situation': float(must_win_situation),
                'games_back_from_playoff': float(games_back_from_playoff),
            }
        except Exception as e:
            print(f"Error calculating game importance: {e}")
            return {'playoff_seeding_impact': 0.5, 'tanking_indicator': 0.0,
                   'must_win_situation': 0.0, 'games_back_from_playoff': 0.0}
    
    def _calculate_minutes_fatigue(self, minutes_list):
        """
        Calculate minutes fatigue metrics (Tier 10).
        Cumulative workload indicators from recent games.
        """
        try:
            if not minutes_list or len(minutes_list) == 0:
                return {
                    'minutes_last_3_games': 0.0,
                    'minutes_last_5_games': 0.0,
                    'minutes_last_7_games': 0.0,
                    'avg_minutes_last_3': 0.0,
                    'minutes_fatigue_score': 0.0,
                }
            
            # Convert to float and handle None values
            minutes = [float(m) if m is not None else 0.0 for m in minutes_list]
            
            # Calculate cumulative minutes
            minutes_last_3 = sum(minutes[:3]) if len(minutes) >= 3 else sum(minutes)
            minutes_last_5 = sum(minutes[:5]) if len(minutes) >= 5 else sum(minutes)
            minutes_last_7 = sum(minutes[:7]) if len(minutes) >= 7 else sum(minutes)
            
            # Average minutes in last 3 games
            avg_minutes_last_3 = minutes_last_3 / min(3, len(minutes)) if len(minutes) > 0 else 0.0
            
            # Normalized fatigue score (0-1 scale)
            # Formula: (avg_min_last_3 / 48) * (1 + (games_in_7_days / 4))
            # Higher score = more fatigue
            # Typical player plays ~35 min/game, max is 48
            # 7 games in 7 days would be max fatigue
            avg_min_normalized = min(avg_minutes_last_3 / 48.0, 1.0)  # Cap at 1.0
            games_count = min(len(minutes), 7)
            games_factor = games_count / 7.0  # Normalize by max games in 7 days
            fatigue_score = avg_min_normalized * (0.7 + 0.3 * games_factor)
            fatigue_score = min(fatigue_score, 1.0)  # Cap at 1.0
            
            return {
                'minutes_last_3_games': float(minutes_last_3),
                'minutes_last_5_games': float(minutes_last_5),
                'minutes_last_7_games': float(minutes_last_7),
                'avg_minutes_last_3': float(avg_minutes_last_3),
                'minutes_fatigue_score': float(fatigue_score),
            }
        except Exception as e:
            print(f"Error calculating minutes fatigue: {e}")
            return {
                'minutes_last_3_games': 0.0,
                'minutes_last_5_games': 0.0,
                'minutes_last_7_games': 0.0,
                'avg_minutes_last_3': 0.0,
                'minutes_fatigue_score': 0.0,
            }
    
    def _calculate_rotation_patterns(self, games_df):
        """Calculate rotation patterns from game logs (Tier 5)"""
        try:
            if games_df.empty:
                return {
                    'fourth_quarter_usage_rate': 0.0,
                    'garbage_time_minutes_pct': 0.0,
                    'typical_substitution_minute': 36.0,
                    'crunch_time_usage': 0.0,
                }
            
            # Note: Full implementation would require play-by-play data
            # Using simplified estimates from game logs
            
            avg_minutes = games_df['MIN'].mean() if 'MIN' in games_df.columns else 0.0
            
            # Fourth quarter usage (estimate: higher if plays full games)
            fourth_quarter_usage_rate = 0.25 if avg_minutes > 30 else 0.18
            
            # Garbage time % (estimate: higher in blowouts)
            if 'PLUS_MINUS' in games_df.columns:
                blowout_games = (games_df['PLUS_MINUS'].abs() > 15).sum()
                garbage_time_minutes_pct = (blowout_games / max(len(games_df), 1)) * 0.15
            else:
                garbage_time_minutes_pct = 0.05
            
            # Typical substitution minute (estimate from avg minutes)
            typical_substitution_minute = min(48.0, avg_minutes + 3.0)
            
            # Crunch time usage (estimate: higher for starters)
            crunch_time_usage = 0.28 if avg_minutes > 28 else 0.15
            
            return {
                'fourth_quarter_usage_rate': float(fourth_quarter_usage_rate),
                'garbage_time_minutes_pct': float(garbage_time_minutes_pct),
                'typical_substitution_minute': float(typical_substitution_minute),
                'crunch_time_usage': float(crunch_time_usage),
            }
        except Exception as e:
            print(f"Error calculating rotation patterns: {e}")
            return {'fourth_quarter_usage_rate': 0.2, 'garbage_time_minutes_pct': 0.05,
                   'typical_substitution_minute': 36.0, 'crunch_time_usage': 0.2}
    
    def _get_specific_matchup_history(self, player_id, opponent_team_id, season):
        """Get specific matchup history (Tier 5)"""
        try:
            cache_key = f"matchup_history:{player_id}:{opponent_team_id}:{season}"
            cached = self._cache_get(cache_key, ttl_seconds=24 * 60 * 60)
            if cached:
                return cached
            
            # Use PlayerVsPlayer endpoint if available
            # For now, simplified version
            time.sleep(0.6)
            
            try:
                # Get career stats vs this team
                gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
                games = gamelog.get_data_frames()[0]
                
                # Filter for games vs this opponent
                # Match opponent team abbreviation in MATCHUP column
                # This is simplified - would need team ID to abbreviation mapping
                vs_team_games = games[games['MATCHUP'].str.contains(str(opponent_team_id)) if 'MATCHUP' in games.columns else False]
                
                career_vs_defender = float(vs_team_games['PTS'].mean()) if not vs_team_games.empty and 'PTS' in vs_team_games.columns else 0.0
                recent_vs_defender = float(vs_team_games.head(3)['PTS'].mean()) if len(vs_team_games) >= 3 and 'PTS' in vs_team_games.columns else career_vs_defender
                
            except Exception:
                career_vs_defender = 0.0
                recent_vs_defender = 0.0
            
            # Player vs arena (would need arena/location data)
            player_vs_arena = 0.0
            
            metrics = {
                'career_vs_defender': float(career_vs_defender),
                'recent_vs_defender': float(recent_vs_defender),
                'player_vs_arena': float(player_vs_arena),
            }
            
            self._cache_set(cache_key, metrics)
            return metrics
        except Exception as e:
            print(f"Error fetching specific matchup history: {e}")
            return {'career_vs_defender': 0.0, 'recent_vs_defender': 0.0, 'player_vs_arena': 0.0}

    def _get_opponent_recent_games(self, team_id, season=None, last_n=10):
        """Get opponent's recent games for calculating recent form"""
        try:
            season = season or self.current_season
            # Use team context which should have recent defensive rating
            # For now, return empty and use season averages as fallback
            # Full implementation would fetch TeamGameLog and calculate def ratings
            return []
        except Exception as e:
            print(f"Error getting opponent recent games: {e}")
            return []
    
    def _get_team_recent_games(self, team_id, season=None, last_n=10):
        """Get team's recent games for calculating win/loss streaks"""
        try:
            season = season or self.current_season
            team_log = TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
            time.sleep(0.6)
            
            if team_log.empty:
                return []
            
            team_log['GAME_DATE'] = pd.to_datetime(team_log['GAME_DATE'])
            team_log = team_log.sort_values('GAME_DATE', ascending=False).head(last_n)
            
            games = []
            for _, row in team_log.iterrows():
                games.append({
                    'win': row.get('WL', 'L') == 'W' if 'WL' in row else False,
                })
            return games
        except Exception as e:
            print(f"Error getting team recent games: {e}")
            return []
    
    def analyze_prop_bet(self, player_id, prop_type, line, opponent_team_id, stats=None, game_location='auto'):
        """Analyze prop bet for given player and line"""
        try:
            stats = stats or self.get_player_stats(player_id)
            if not stats:
                return {
                    'success': False,
                    'error': 'Unable to retrieve player stats'
                }

            # Get more context (order matters: we pass opponent_context to avoid extra expensive calls)
            opponent_context = self.ml_predictor.get_team_context(opponent_team_id)
            player_context = self.ml_predictor.get_player_context(
                player_id,
                opponent_team_id,
                opponent_context=opponent_context
            )
            team_id = None
            if player_context and player_context.get('team_id'):
                team_id = int(player_context['team_id'])
            else:
            team_id = self._get_player_team_id(player_id)
            team_context = self.ml_predictor.get_team_context(team_id) if team_id else None

            # Precomputed daily datasets (DVP by position + special defenders)
            pre = self.precomputed.refresh()
            dvp_map = pre.get('dvp', {})
            dvp_pos_avgs = pre.get('dvp_pos_avgs', {})
            defenders_map = pre.get('defenders', {})

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

            # Map player's position to DVP position buckets (FantasyPros uses PG/SG/SF/PF/C).
            raw_pos = ((player_context or {}).get('position') or '').upper()
            if 'C' in raw_pos and 'G' not in raw_pos:
                dvp_pos = 'C'
                pos_group = 'C'
            elif 'G' in raw_pos and 'F' not in raw_pos:
                dvp_pos = 'SG'  # generic guard
                pos_group = 'G'
            elif 'F' in raw_pos and 'C' not in raw_pos and 'G' not in raw_pos:
                dvp_pos = 'SF'  # generic forward
                pos_group = 'F'
            elif 'C' in raw_pos:
                dvp_pos = 'C'
                pos_group = 'C'
            elif 'G' in raw_pos:
                dvp_pos = 'SG'
                pos_group = 'G'
            else:
                dvp_pos = 'SF'
                pos_group = 'F'

            dvp = dvp_map.get((int(opponent_team_id), dvp_pos))
            dvp_avg = dvp_pos_avgs.get(dvp_pos, {})
            dvp_deltas = {}
            dvp_gp = 0
            if dvp:
                dvp_gp = int(dvp.get('gp', 0) or 0)
                for k in ['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov', 'fd_pts']:
                    dvp_deltas[f'dvp_{k}_delta'] = float(dvp.get(k, 0.0) - float(dvp_avg.get(k, 0.0) or 0.0))

            special_defenders = defenders_map.get((int(opponent_team_id), pos_group), [])
            primary_defender = special_defenders[0] if special_defenders else None
            defender_score01 = float((primary_defender or {}).get('score01', 0.0) or 0.0)

            # Location inference from matchup strings (aligned with `values`)
            matchups = stats.get('matchups', []) or []
            def infer_is_home(m):
                m = str(m or '')
                if ' vs ' in m:
                    return True
                if ' @ ' in m:
                    return False
                return None

            is_home_flags = [infer_is_home(m) for m in matchups[:len(values)]]
            home_values = [v for v, h in zip(values, is_home_flags) if h is True]
            away_values = [v for v, h in zip(values, is_home_flags) if h is False]
            home_avg = float(np.mean(home_values)) if home_values else float(stat_data.get('avg', 0))
            away_avg = float(np.mean(away_values)) if away_values else float(stat_data.get('avg', 0))

            is_home_game = None
            if game_location == 'home':
                is_home_game = True
            elif game_location == 'away':
                is_home_game = False

            # Recent road-trip proxy: consecutive away games in the last N logs
            away_streak = 0
            for h in is_home_flags:
                if h is False:
                    away_streak += 1
                else:
                    break

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
                'edge': edge,
                'is_home_game': is_home_game,
                'home_split_avg': float(home_avg),
                'away_split_avg': float(away_avg),
                'home_split_n': len(home_values),
                'away_split_n': len(away_values),
                'recent_away_streak': int(away_streak)
            }
            features.update({
                'dvp_position': dvp_pos,
                'dvp_gp': int(dvp_gp),
                **dvp_deltas,
                'primary_defender_name': (primary_defender or {}).get('player_name'),
                'primary_defender_score01': defender_score01,
            })

            # Add richer, fast-to-compute context features
            minutes = stats.get('minutes', []) or []
            mins_last5 = float(np.mean(minutes[:5])) if len(minutes) >= 5 else float(np.mean(minutes)) if minutes else 0.0
            mins_season = float(np.mean(minutes)) if minutes else 0.0

            # Calculate minutes fatigue (cumulative workload) - Tier 10
            minutes_fatigue = self._calculate_minutes_fatigue(minutes)
            
            # Player rest proxy from last game date (team rest is not always available cheaply)
            rest_days_player = 2
            try:
                last_game_date = stats.get('last_game_date')
                if last_game_date:
                    last_dt = datetime.strptime(last_game_date, '%Y-%m-%d')
                    rest_days_player = max(0, (datetime.now() - last_dt).days)
            except Exception:
                rest_days_player = 2

            # Extract Tier 1 features
            shooting = stats.get('shooting', {})
            rebounding = stats.get('rebounding', {})
            impact = stats.get('impact', {})
            
            # Extract Tier 2 Quick Win features
            momentum = stats.get('momentum', {}).get(prop_type, {})
            schedule = stats.get('schedule', {})

            team_style = (team_context or {}).get('style', {}) or {}
            opp_style = (opponent_context or {}).get('style', {}) or {}
            team_base = (team_context or {}).get('base', {}) or {}
            opp_base = (opponent_context or {}).get('base', {}) or {}
            league_avgs = (opponent_context or {}).get('league_avgs', {}) or {}
            features.update({
                'mins_last5': mins_last5,
                'mins_season': mins_season,
                'rest_days': int(rest_days_player),
                # Tier 10: Minutes Fatigue
                **minutes_fatigue,
                'team_pace': float((team_context or {}).get('pace', 100.0)),
                'team_off_rating': float((team_context or {}).get('offensive_rating', 110.0)),
                'team_def_rating': float((team_context or {}).get('defensive_rating', 110.0)),
                'opp_pace': float((opponent_context or {}).get('pace', 100.0)),
                'opp_off_rating': float((opponent_context or {}).get('offensive_rating', 110.0)),
                'opp_def_rating': float((opponent_context or {}).get('defensive_rating', 110.0)),
                'team_key_players_out': int(((team_context or {}).get('injuries', {}) or {}).get('key_players_out', 0)),
                'opp_injury_impact': float((opponent_context or {}).get('injury_impact', 0.0)),
                'opp_key_players_out': int((((opponent_context or {}).get('injuries', {}) or {}).get('key_players_out', 0)) or 0),

                # Transition / style
                'team_pts_fb': float(team_style.get('pts_fb', 0.0)),
                'opp_pts_fb_allowed': float(opp_style.get('opp_pts_fb', 0.0)),
                'team_pts_off_tov': float(team_style.get('pts_off_tov', 0.0)),
                'opp_pts_off_tov_allowed': float(opp_style.get('opp_pts_off_tov', 0.0)),
                'opp_pts_paint': float(opp_style.get('pts_paint', 0.0)),

                # Opponent base profile (opportunities)
                'opp_fga': float(opp_base.get('fga', 0.0)),
                'opp_fg_pct': float(opp_base.get('fg_pct', 0.47)),
                'opp_fg3a': float(opp_base.get('fg3a', 0.0)),
                'opp_fg3_pct': float(opp_base.get('fg3_pct', 0.36)),
                'opp_tov': float(opp_base.get('tov', 0.0)),
                'opp_stl': float(opp_base.get('stl', 0.0)),
                'opp_blk': float(opp_base.get('blk', 0.0)),

                # League average baselines (for normalization)
                'lg_pts_fb': float(league_avgs.get('pts_fb', 12.0)),
                'lg_opp_pts_fb': float(league_avgs.get('opp_pts_fb', 12.0)),
                'lg_fga': float(league_avgs.get('fga', 86.0)),
                'lg_fg_pct': float(league_avgs.get('fg_pct', 0.47)),
                'lg_tov': float(league_avgs.get('tov', 14.0)),
                'lg_stl': float(league_avgs.get('stl', 7.0)),
                'lg_pts_off_tov': float(league_avgs.get('pts_off_tov', 16.0)),
                'lg_opp_pts_off_tov': float(league_avgs.get('opp_pts_off_tov', 16.0)),
                'lg_fg3a': float(league_avgs.get('fg3a', 35.0)),
            })

            mh = (player_context or {}).get('matchup_history') or {}
            features.update({
                'vs_team_avg': float(mh.get('avg_points', 0.0) or 0.0),
                'matchup_games': int(mh.get('games_played', 0) or 0),
            })

            # Add Tier 1 features: shooting efficiency & volume
            features.update({
                'fg_pct_recent': float(shooting.get('fg_pct_recent', 0.0) or 0.0),
                'fg3_pct_recent': float(shooting.get('fg3_pct_recent', 0.0) or 0.0),
                'ft_pct_recent': float(shooting.get('ft_pct_recent', 0.0) or 0.0),
                'fga_per_game': float(shooting.get('fga_per_game', 0.0) or 0.0),
                'fg3a_per_game': float(shooting.get('fg3a_per_game', 0.0) or 0.0),
                'fta_per_game': float(shooting.get('fta_per_game', 0.0) or 0.0),
                # Rebounding split
                'oreb_per_game': float(rebounding.get('oreb_per_game', 0.0) or 0.0),
                'dreb_per_game': float(rebounding.get('dreb_per_game', 0.0) or 0.0),
                # Impact & context
                'plus_minus_avg': float(impact.get('plus_minus_avg', 0.0) or 0.0),
                'fouls_per_game': float(impact.get('fouls_per_game', 0.0) or 0.0),
                'win_rate_last10': float(impact.get('win_rate_last10', 0.5) or 0.5),
                # Tier 2 Quick Wins: momentum
                'last_3_games_trend': float(momentum.get('last_3_trend', 0.0) or 0.0),
                'last_5_games_trend': float(momentum.get('last_5_trend', 0.0) or 0.0),
                'last_10_games_trend': float(momentum.get('last_10_trend', 0.0) or 0.0),
                'games_above_season_avg_last5': int(momentum.get('above_avg_last5', 0) or 0),
                # Tier 2 Quick Wins: schedule/fatigue
                'is_back_to_back': float(1.0 if schedule.get('is_back_to_back', False) else 0.0),
                'days_since_last_game': int(schedule.get('days_since_last_game', 7) or 7),
                'games_in_last_7_days': int(schedule.get('games_in_last_7_days', 0) or 0),
            })

            # Extract Tier 2: Advanced metrics, shot location, clutch, lineup
            advanced = stats.get('advanced_metrics', {})
            shot_loc = stats.get('shot_location', {})
            clutch = stats.get('clutch_stats', {})
            lineup = stats.get('lineup_context', {})
            
            features.update({
                # Advanced player metrics
                'usage_rate': float(advanced.get('usage_rate', 0.0) or 0.0),
                'true_shooting_pct': float(advanced.get('true_shooting_pct', 0.0) or 0.0),
                'effective_fg_pct': float(advanced.get('effective_fg_pct', 0.0) or 0.0),
                'assist_percentage': float(advanced.get('assist_percentage', 0.0) or 0.0),
                'rebound_percentage': float(advanced.get('rebound_percentage', 0.0) or 0.0),
                'pie': float(advanced.get('pie', 0.0) or 0.0),
                # Tier 10: Player-Level Advanced Metrics (from PlayerDashboardByGeneralSplits)
                'player_off_rating': float(advanced.get('off_rating', 0.0) or 0.0),
                'player_def_rating': float(advanced.get('def_rating', 0.0) or 0.0),
                'player_pace': float(advanced.get('pace', 0.0) or 0.0),
                'fta_rate_player': float(advanced.get('fta_rate', 0.0) or 0.0),
                'pct_fga_2pt': float(advanced.get('pct_fga_2pt', 0.0) or 0.0),
                'pct_fga_3pt': float(advanced.get('pct_fga_3pt', 0.0) or 0.0),
                'pct_pts_in_paint': float(advanced.get('pct_pts_in_paint', 0.0) or 0.0),
                'pct_pts_off_tov': float(advanced.get('pct_pts_off_tov', 0.0) or 0.0),
                'pct_pts_fb': float(advanced.get('pct_pts_fb', 0.0) or 0.0),
                # Shot location
                'rim_fga_per_game': float(shot_loc.get('rim_fga_per_game', 0.0) or 0.0),
                'paint_fga_per_game': float(shot_loc.get('paint_fga_per_game', 0.0) or 0.0),
                'mid_range_fga_per_game': float(shot_loc.get('mid_range_fga_per_game', 0.0) or 0.0),
                'corner_3_pct': float(shot_loc.get('corner_3_pct', 0.0) or 0.0),
                'above_break_3_pct': float(shot_loc.get('above_break_3_pct', 0.0) or 0.0),
                # Clutch performance
                'clutch_pts_per_game': float(clutch.get('clutch_pts_per_game', 0.0) or 0.0),
                'clutch_fg_pct': float(clutch.get('clutch_fg_pct', 0.0) or 0.0),
                'clutch_minutes_per_game': float(clutch.get('clutch_minutes_per_game', 0.0) or 0.0),
                # Lineup context
                'on_court_plus_minus': float(lineup.get('on_court_plus_minus', 0.0) or 0.0),
                'off_court_plus_minus': float(lineup.get('off_court_plus_minus', 0.0) or 0.0),
                'net_rating': float(lineup.get('net_rating', 0.0) or 0.0),
                'top_lineup_minutes_pct': float(lineup.get('top_lineup_minutes_pct', 0.0) or 0.0),
            })
            
            # Extract Tier 6: Shot Quality, Touch/Usage Data, Full Lineup Context
            shot_quality = stats.get('shot_quality', {})
            touch_usage = stats.get('touch_usage', {})
            
            features.update({
                # Shot Quality Metrics
                'avg_shot_distance': float(shot_quality.get('avg_shot_distance', 0.0) or 0.0),
                'contested_shot_pct': float(shot_quality.get('contested_shot_pct', 0.5) or 0.5),
                'open_shot_pct': float(shot_quality.get('open_shot_pct', 0.3) or 0.3),
                'wide_open_shot_pct': float(shot_quality.get('wide_open_shot_pct', 0.2) or 0.2),
                'catch_and_shoot_pct': float(shot_quality.get('catch_and_shoot_pct', 0.3) or 0.3),
                'pull_up_shot_pct': float(shot_quality.get('pull_up_shot_pct', 0.3) or 0.3),
                'paint_touch_frequency': float(shot_quality.get('paint_touch_frequency', 0.0) or 0.0),
                'corner_three_pct': float(shot_quality.get('corner_three_pct', 0.0) or 0.0),
                'above_break_three_pct': float(shot_quality.get('above_break_three_pct', 0.0) or 0.0),
                'restricted_area_fg_pct': float(shot_quality.get('restricted_area_fg_pct', 0.0) or 0.0),
                'mid_range_frequency': float(shot_quality.get('mid_range_frequency', 0.0) or 0.0),
                'shot_quality_vs_expected': float(shot_quality.get('shot_quality_vs_expected', 0.0) or 0.0),
                'avg_shot_clock_time': float(shot_quality.get('avg_shot_clock_time', 12.0) or 12.0),
                'late_clock_shot_frequency': float(shot_quality.get('late_clock_shot_frequency', 0.15) or 0.15),
                'early_clock_shot_frequency': float(shot_quality.get('early_clock_shot_frequency', 0.25) or 0.25),
                # Touch & Usage Details
                'touches_per_game': float(touch_usage.get('touches_per_game', 0.0) or 0.0),
                'avg_dribbles_per_touch': float(touch_usage.get('avg_dribbles_per_touch', 0.0) or 0.0),
                'avg_seconds_per_touch': float(touch_usage.get('avg_seconds_per_touch', 0.0) or 0.0),
                'elbow_touches_per_game': float(touch_usage.get('elbow_touches_per_game', 0.0) or 0.0),
                'post_touches_per_game': float(touch_usage.get('post_touches_per_game', 0.0) or 0.0),
                'paint_touches_per_game': float(touch_usage.get('paint_touches_per_game', 0.0) or 0.0),
                'front_court_touches_per_game': float(touch_usage.get('front_court_touches_per_game', 0.0) or 0.0),
                'time_of_possession_per_game': float(touch_usage.get('time_of_possession_per_game', 0.0) or 0.0),
                'touches_per_possession': float(touch_usage.get('touches_per_possession', 0.0) or 0.0),
                'avg_points_per_touch': float(touch_usage.get('avg_points_per_touch', 0.0) or 0.0),
                # Extended Lineup Context (from _get_lineup_context - need to expand)
                'net_rating_with_starters': float(lineup.get('net_rating', 0.0) or 0.0),  # Using net_rating as proxy
                'usage_rate_with_star_out': float(advanced.get('usage_rate', 0.0) or 0.0),  # Default to overall usage
                'minutes_with_starting_lineup_pct': 0.65,  # Default estimate for starters
                'five_man_unit_net_rating': float(lineup.get('net_rating', 0.0) or 0.0),  # Proxy
                'on_court_net_rating': float(lineup.get('net_rating', 0.0) or 0.0),
                'off_court_net_rating': 0.0,  # Not easily available
                'on_off_differential': float(lineup.get('net_rating', 0.0) or 0.0),
                'lineups_played_count': 1.0,  # Default
            })
            
            # Extract Tier 3: Travel, arena, opponent history
            # Travel metrics (calculate from last game location)
            player_team_id = team_id or self._get_player_team_id(player_id)
            last_game_team_id = None  # Would need to track from game logs
            
            # Arena data
            if game_location == 'home':
                arena_info = get_arena_info(player_team_id)
                home_adv = get_home_court_advantage(player_team_id)
            elif game_location == 'away':
                arena_info = get_arena_info(opponent_team_id)
                home_adv = 0.0
            else:  # auto
                arena_info = get_arena_info(player_team_id)  # assume home for now
                home_adv = get_home_court_advantage(player_team_id) * 0.5
            
            tz_change, coast_to_coast, travel_dist = calculate_travel_metrics(
                last_game_team_id, 
                opponent_team_id if game_location == 'away' else player_team_id,
                game_location == 'home'
            )
            
            # Opponent history enhancement (from existing matchup data)
            vs_team_games = mh.get('games_played', 0) or 0
            vs_team_wins = mh.get('wins', 0) or 0
            
            features.update({
                # Time zone & travel
                'time_zone_change': int(tz_change),
                'travel_distance': float(travel_dist),
                'coast_to_coast': float(1.0 if coast_to_coast else 0.0),
                # Arena/environmental
                'arena_altitude': float(arena_info.get('altitude', 0) or 0),
                'arena_capacity': float(arena_info.get('capacity', 18000) or 18000),
                'home_court_advantage_rating': float(home_adv),
                # Opponent history enhancement
                'vs_team_last_season_avg': float(mh.get('avg_points', 0.0) or 0.0),  # Already have this
                'vs_team_home_away_split': 0.0,  # Would need home/away split in matchup data
                'vs_team_win_pct': float(vs_team_wins / max(vs_team_games, 1)),
                # Historical model performance (placeholder - would track over time)
                'model_accuracy_player': 0.5,  # Default 50%
                'avg_prediction_error_player': 0.0,
                'calibration_score_player': 0.0,
            })
            
            # Tier 7: Time-Series Features (Quick Win #2)
            # Calculate from game log values (need to get full history, not just last 20)
            try:
                # Get values for the current prop type
                prop_values = values if values else []
                if len(prop_values) > 0:
                    # Convert to pandas Series for easier calculations
                    values_series = pd.Series(prop_values)
                    
                    # Rolling averages
                    rolling_7 = values_series.rolling(window=7, min_periods=1).mean().iloc[-1] if len(values_series) >= 1 else float(stat_data.get('last5_avg', 0))
                    rolling_14 = values_series.rolling(window=14, min_periods=1).mean().iloc[-1] if len(values_series) >= 1 else float(stat_data.get('avg', 0))
                    rolling_30 = values_series.rolling(window=30, min_periods=1).mean().iloc[-1] if len(values_series) >= 1 else float(stat_data.get('avg', 0))
                    
                    # Exponential weighted moving averages
                    ewm_03 = values_series.ewm(alpha=0.3, adjust=False).mean().iloc[-1] if len(values_series) >= 1 else float(stat_data.get('last5_avg', 0))
                    ewm_05 = values_series.ewm(alpha=0.5, adjust=False).mean().iloc[-1] if len(values_series) >= 1 else float(stat_data.get('last5_avg', 0))
                    
                    # Trend slopes (linear regression)
                    trend_10 = 0.0
                    trend_5 = 0.0
                    if len(values_series) >= 5:
                        last_10 = values_series.tail(10).values
                        if len(last_10) >= 3:
                            x = np.arange(len(last_10))
                            trend_10 = float(np.polyfit(x, last_10, 1)[0])
                    if len(values_series) >= 3:
                        last_5 = values_series.tail(5).values
                        if len(last_5) >= 2:
                            x = np.arange(len(last_5))
                            trend_5 = float(np.polyfit(x, last_5, 1)[0])
                    
                    # Volatility ratio (std/mean)
                    volatility_ratio = float(np.std(values_series) / (np.mean(values_series) + 1e-6)) if len(values_series) > 1 else 0.0
                    
                    # Momentum score: (last_5_avg - prev_5_avg) / prev_5_avg
                    momentum_score = 0.0
                    if len(values_series) >= 10:
                        last_5_avg = float(values_series.tail(5).mean())
                        prev_5_avg = float(values_series.tail(10).head(5).mean())
                        momentum_score = float((last_5_avg - prev_5_avg) / (prev_5_avg + 1e-6))
                    elif len(values_series) >= 5:
                        last_5_avg = float(values_series.tail(5).mean())
                        prev_avg = float(stat_data.get('avg', last_5_avg))
                        momentum_score = float((last_5_avg - prev_avg) / (prev_avg + 1e-6))
                    
                    # Games above season average
                    season_avg_val = float(stat_data.get('avg', 0))
                    games_above_7day = int((values_series.tail(7) > season_avg_val).sum()) if len(values_series) >= 1 else 0
                    games_above_14day = int((values_series.tail(14) > season_avg_val).sum()) if len(values_series) >= 1 else 0
                else:
                    # Defaults if no values
                    rolling_7 = float(stat_data.get('last5_avg', 0))
                    rolling_14 = float(stat_data.get('avg', 0))
                    rolling_30 = float(stat_data.get('avg', 0))
                    ewm_03 = float(stat_data.get('last5_avg', 0))
                    ewm_05 = float(stat_data.get('last5_avg', 0))
                    trend_10 = 0.0
                    trend_5 = 0.0
                    volatility_ratio = 0.0
                    momentum_score = 0.0
                    games_above_7day = 0
                    games_above_14day = 0
                
                features.update({
                    'rolling_7day_avg': rolling_7,
                    'rolling_14day_avg': rolling_14,
                    'rolling_30day_avg': rolling_30,
                    'ewm_alpha_0.3': ewm_03,  # Note: dot in feature name
                    'ewm_alpha_0.5': ewm_05,  # Note: dot in feature name
                    'trend_slope_10games': trend_10,
                    'trend_slope_5games': trend_5,
                    'volatility_ratio': volatility_ratio,
                    'momentum_score': momentum_score,
                    'games_above_season_avg_7day': games_above_7day,
                    'games_above_season_avg_14day': games_above_14day,
                })
            except Exception as e:
                # Fallback to defaults on error
                print(f"Error calculating time-series features: {e}")
                features.update({
                    'rolling_7day_avg': float(stat_data.get('last5_avg', 0)),
                    'rolling_14day_avg': float(stat_data.get('avg', 0)),
                    'rolling_30day_avg': float(stat_data.get('avg', 0)),
                    'ewm_alpha_0.3': float(stat_data.get('last5_avg', 0)),  # Note: dot in feature name
                    'ewm_alpha_0.5': float(stat_data.get('last5_avg', 0)),  # Note: dot in feature name
                    'trend_slope_10games': 0.0,
                    'trend_slope_5games': 0.0,
                    'volatility_ratio': 0.0,
                    'momentum_score': 0.0,
                    'games_above_season_avg_7day': 0,
                    'games_above_season_avg_14day': 0,
                })
            
            # Tier 7: Enhanced Matchup Features (Quick Win #5)
            try:
                # Enhanced head-to-head (already have vs_team_avg, enhance it)
                head_to_head_avg = float(mh.get('avg_points', 0.0) or 0.0)  # Use existing, but could be prop-specific
                head_to_head_games = int(mh.get('games_played', 0) or 0)
                
                # Position-specific DVP (enhance existing position matchup)
                position_matchup = (player_context or {}).get('position_matchup') or {}
                position_vs_position_dvp = float(position_matchup.get('defensive_rating', 110.0) or 110.0)
                
                # Matchup pace (pace when these teams play - use team pace as proxy)
                matchup_pace = float((team_context or {}).get('pace', 100.0) or 100.0)
                
                # Defender switching frequency (placeholder - would need play-by-play data)
                # Estimate based on opponent defensive style
                defender_switching_frequency = 0.5  # Default 50% (would need actual data)
                
                # Historical game script (blowout vs close game impact)
                # Use matchup history success rate as proxy
                historical_game_script_avg = float(mh.get('success_rate', 0.5) or 0.5)
                
                features.update({
                    'head_to_head_avg': head_to_head_avg,
                    'head_to_head_games': head_to_head_games,
                    'position_vs_position_dvp': position_vs_position_dvp,
                    'matchup_pace': matchup_pace,
                    'defender_switching_frequency': defender_switching_frequency,
                    'historical_game_script_avg': historical_game_script_avg,
                })
            except Exception as e:
                # Fallback to defaults on error
                print(f"Error calculating enhanced matchup features: {e}")
                features.update({
                    'head_to_head_avg': float(mh.get('avg_points', 0.0) or 0.0),
                    'head_to_head_games': int(mh.get('games_played', 0) or 0),
                    'position_vs_position_dvp': 110.0,
                    'matchup_pace': float((team_context or {}).get('pace', 100.0) or 100.0),
                    'defender_switching_frequency': 0.5,
                    'historical_game_script_avg': 0.5,
                })
            
            # Tier 8: Relative Rest Advantage
            try:
                player_rest = int(features.get('rest_days', 2) or 2)
                # Opponent rest might not be in context, use default or calculate
                opponent_rest = int((opponent_context or {}).get('rest_days', 2) or 2)
                # If not available, assume 2 days (typical rest)
                if opponent_rest == 2 and opponent_context:
                    # Try to get from opponent context if available
                    opponent_rest = int((opponent_context or {}).get('rest_days', 2) or 2)
                
                rest_advantage = player_rest - opponent_rest
                rest_advantage_abs = abs(rest_advantage)
                both_teams_rested = 1.0 if (player_rest >= 2 and opponent_rest >= 2) else 0.0
                
                features.update({
                    'rest_advantage': float(rest_advantage),
                    'rest_advantage_abs': float(rest_advantage_abs),
                    'both_teams_rested': both_teams_rested,
                })
            except Exception as e:
                print(f"Error calculating rest advantage: {e}")
                features.update({
                    'rest_advantage': 0.0,
                    'rest_advantage_abs': 0.0,
                    'both_teams_rested': 0.0,
                })
            
            # Tier 8: Opponent Recent Form
            try:
                # Get opponent's recent games (last 10)
                opp_recent_games = self._get_opponent_recent_games(opponent_team_id, season=self.current_season, last_n=10)
                
                if opp_recent_games and len(opp_recent_games) > 0:
                    opp_def_ratings = [g.get('def_rating', 110.0) for g in opp_recent_games if g.get('def_rating')]
                    opp_paces = [g.get('pace', 100.0) for g in opp_recent_games if g.get('pace')]
                    opp_wins = [1 if g.get('win', False) else 0 for g in opp_recent_games]
                    
                    opp_def_last5 = float(np.mean(opp_def_ratings[:5])) if len(opp_def_ratings) >= 5 else float((opponent_context or {}).get('defensive_rating', 110.0) or 110.0)
                    opp_def_last10 = float(np.mean(opp_def_ratings)) if len(opp_def_ratings) > 0 else float((opponent_context or {}).get('defensive_rating', 110.0) or 110.0)
                    
                    # Calculate trend (slope of defensive rating over time)
                    opp_def_trend = 0.0
                    if len(opp_def_ratings) >= 3:
                        x = np.arange(len(opp_def_ratings))
                        opp_def_trend = float(np.polyfit(x, opp_def_ratings, 1)[0])
                    
                    opp_pace_last5 = float(np.mean(opp_paces[:5])) if len(opp_paces) >= 5 else float((opponent_context or {}).get('pace', 100.0) or 100.0)
                    opp_win_rate_last10 = float(np.mean(opp_wins)) if len(opp_wins) > 0 else 0.5
                else:
                    # Fallback to season averages
                    opp_def_last5 = float((opponent_context or {}).get('defensive_rating', 110.0) or 110.0)
                    opp_def_last10 = opp_def_last5
                    opp_def_trend = 0.0
                    opp_pace_last5 = float((opponent_context or {}).get('pace', 100.0) or 100.0)
                    opp_win_rate_last10 = 0.5
                
                features.update({
                    'opp_def_rating_last5': opp_def_last5,
                    'opp_def_rating_last10': opp_def_last10,
                    'opp_def_rating_trend': opp_def_trend,
                    'opp_pace_last5': opp_pace_last5,
                    'opp_win_rate_last10': opp_win_rate_last10,
                })
            except Exception as e:
                print(f"Error calculating opponent recent form: {e}")
                features.update({
                    'opp_def_rating_last5': float((opponent_context or {}).get('defensive_rating', 110.0) or 110.0),
                    'opp_def_rating_last10': float((opponent_context or {}).get('defensive_rating', 110.0) or 110.0),
                    'opp_def_rating_trend': 0.0,
                    'opp_pace_last5': float((opponent_context or {}).get('pace', 100.0) or 100.0),
                    'opp_win_rate_last10': 0.5,
                })
            
            # Tier 8: Player Age & Experience
            try:
                player_info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
                time.sleep(0.6)  # Rate limiting
                
                # Calculate age from birthdate
                birthdate_str = player_info['BIRTHDATE'].iloc[0] if 'BIRTHDATE' in player_info.columns else None
                if birthdate_str:
                    try:
                        birthdate = datetime.strptime(birthdate_str, '%Y-%m-%dT%H:%M:%S')
                        age = (datetime.now() - birthdate).days / 365.25
                    except:
                        age = 25.0  # Default
                else:
                    age = 25.0
                
                years_exp = float(player_info['SEASON_EXP'].iloc[0]) if 'SEASON_EXP' in player_info.columns else 0.0
                is_rookie = 1.0 if years_exp == 1 else 0.0
                is_veteran = 1.0 if years_exp >= 10 else 0.0
                
                features.update({
                    'player_age': float(age),
                    'years_experience': float(years_exp),
                    'is_rookie': is_rookie,
                    'is_veteran': is_veteran,
                })
            except Exception as e:
                print(f"Error calculating player age/experience: {e}")
                features.update({
                    'player_age': 25.0,
                    'years_experience': 5.0,
                    'is_rookie': 0.0,
                    'is_veteran': 0.0,
                })
            
            # Tier 8: Game Script Prediction
            try:
                team_off_rating = float((team_context or {}).get('offensive_rating', 110.0) or 110.0)
                opp_def_rating = float((opponent_context or {}).get('defensive_rating', 110.0) or 110.0)
                
                # Rough point spread estimate (offensive rating - opponent defensive rating)
                point_spread = team_off_rating - opp_def_rating
                
                # Game script: positive = blowout win, negative = blowout loss, near 0 = close
                expected_game_script = point_spread
                
                # Blowout probability (sigmoid function)
                blowout_probability = 1.0 / (1.0 + np.exp(-point_spread / 10.0))
                
                # Close game probability (inverse of blowout probability, centered around 0)
                close_game_probability = 1.0 - abs(blowout_probability - 0.5) * 2.0
                close_game_probability = max(0.0, min(1.0, close_game_probability))
                
                features.update({
                    'expected_game_script': float(expected_game_script),
                    'blowout_probability': float(blowout_probability),
                    'close_game_probability': float(close_game_probability),
                })
            except Exception as e:
                print(f"Error calculating game script: {e}")
                features.update({
                    'expected_game_script': 0.0,
                    'blowout_probability': 0.5,
                    'close_game_probability': 0.5,
                })
            
            # Tier 8: Quarter-Specific Performance
            try:
                # Get quarter splits from game logs if available
                # For now, use clutch stats as proxy for 4th quarter performance
                clutch_stats = stats.get('clutch_stats', {})
                first_quarter_avg = 0.0  # Would need quarter splits
                fourth_quarter_avg = float(clutch_stats.get('clutch_pts_per_game', 0.0) or 0.0) * 0.3  # Rough estimate
                clutch_performance_score = float(clutch_stats.get('clutch_fg_pct', 0.45) or 0.45)
                
                # If we have game logs with quarter data, calculate properly
                # For now, use estimates based on season average
                season_avg_pts = float(stat_data.get('avg', 0) or 0)
                # Rough estimates: 1st quarter ~25% of points, 4th quarter ~25% of points
                if first_quarter_avg == 0.0:
                    first_quarter_avg = season_avg_pts * 0.25
                if fourth_quarter_avg == 0.0:
                    fourth_quarter_avg = season_avg_pts * 0.25
                
                features.update({
                    'first_quarter_avg': float(first_quarter_avg),
                    'fourth_quarter_avg': float(fourth_quarter_avg),
                    'clutch_performance_score': float(clutch_performance_score),
                })
            except Exception as e:
                print(f"Error calculating quarter-specific performance: {e}")
                season_avg_pts = float(stat_data.get('avg', 0) or 0)
                features.update({
                    'first_quarter_avg': float(season_avg_pts * 0.25),
                    'fourth_quarter_avg': float(season_avg_pts * 0.25),
                    'clutch_performance_score': 0.45,
                })
            
            # Tier 8: Enhanced Shot Selection Quality
            try:
                shot_quality = stats.get('shot_quality', {})
                open_shot_pct = float(shot_quality.get('open_shot_pct', 0.3) or 0.3)
                contested_shot_pct = float(shot_quality.get('contested_shot_pct', 0.5) or 0.5)
                
                # Shot selection rating: higher = better shot selection (more open shots)
                shot_selection_rating = open_shot_pct / (open_shot_pct + contested_shot_pct + 1e-6)
                
                # Bad shot frequency: frequency of contested shots
                bad_shot_frequency = contested_shot_pct
                
                # Shot clock management: performance in different shot clock situations
                late_clock_freq = float(shot_quality.get('late_clock_shot_frequency', 0.15) or 0.15)
                early_clock_freq = float(shot_quality.get('early_clock_shot_frequency', 0.25) or 0.25)
                # Better shot clock management = more early clock shots, fewer late clock shots
                shot_clock_management = early_clock_freq - late_clock_freq
                
                features.update({
                    'shot_selection_rating': float(shot_selection_rating),
                    'bad_shot_frequency': float(bad_shot_frequency),
                    'shot_clock_management': float(shot_clock_management),
                })
            except Exception as e:
                print(f"Error calculating shot selection quality: {e}")
                features.update({
                    'shot_selection_rating': 0.5,
                    'bad_shot_frequency': 0.5,
                    'shot_clock_management': 0.0,
                })
            
            # Tier 8: Team Chemistry Metrics
            try:
                # Calculate from roster stability and lineup continuity
                # For now, use win/loss streaks and lineup stability as proxies
                team_recent_games = self._get_team_recent_games(team_id, season=self.current_season, last_n=10)
                
                if team_recent_games and len(team_recent_games) > 0:
                    wins = [1 if g.get('win', False) else 0 for g in team_recent_games]
                    
                    # Calculate win streak
                    team_win_streak = 0
                    for i in range(len(wins)-1, -1, -1):
                        if wins[i] == 1:
                            team_win_streak += 1
                        else:
                            break
                    
                    # Calculate loss streak
                    team_loss_streak = 0
                    for i in range(len(wins)-1, -1, -1):
                        if wins[i] == 0:
                            team_loss_streak += 1
                        else:
                            break
                else:
                    team_win_streak = 0
                    team_loss_streak = 0
                
                # Teammate chemistry score: placeholder (would need to track how long teammates played together)
                teammate_chemistry_score = 0.7  # Default estimate
                
                # Lineup continuity: placeholder (would need to track lineup stability)
                lineup_continuity = 0.7  # Default estimate
                
                features.update({
                    'teammate_chemistry_score': float(teammate_chemistry_score),
                    'lineup_continuity': float(lineup_continuity),
                    'team_win_streak': float(team_win_streak),
                    'team_loss_streak': float(team_loss_streak),
                })
            except Exception as e:
                print(f"Error calculating team chemistry: {e}")
                features.update({
                    'teammate_chemistry_score': 0.7,
                    'lineup_continuity': 0.7,
                    'team_win_streak': 0.0,
                    'team_loss_streak': 0.0,
                })
            
            # Tier 8: Enhanced Defensive Matchup Quality
            try:
                primary_defender = (player_context or {}).get('primary_defender') or {}
                defender_score01 = float(features.get('primary_defender_score01', 0.0) or 0.0)
                
                # Enhanced defender metrics
                primary_defender_rating = defender_score01 * 120.0  # Scale 0-1 to 0-120 defensive rating
                primary_defender_age = 28.0  # Default (would need to fetch defender info)
                defender_size_mismatch = 0.0  # Default (would need height/weight data)
                defender_recent_form = 0.5  # Default (would need defender recent games)
                
                # If we have primary defender info, enhance it
                if primary_defender and 'player_id' in primary_defender:
                    try:
                        defender_info = CommonPlayerInfo(player_id=primary_defender['player_id']).get_data_frames()[0]
                        time.sleep(0.6)
                        birthdate_str = defender_info['BIRTHDATE'].iloc[0] if 'BIRTHDATE' in defender_info.columns else None
                        if birthdate_str:
                            try:
                                birthdate = datetime.strptime(birthdate_str, '%Y-%m-%dT%H:%M:%S')
                                defender_age = (datetime.now() - birthdate).days / 365.25
                                primary_defender_age = float(defender_age)
                            except:
                                pass
                    except:
                        pass
                
                features.update({
                    'primary_defender_rating': float(primary_defender_rating),
                    'primary_defender_age': float(primary_defender_age),
                    'defender_size_mismatch': float(defender_size_mismatch),
                    'defender_recent_form': float(defender_recent_form),
                })
            except Exception as e:
                print(f"Error calculating enhanced defensive matchup: {e}")
                features.update({
                    'primary_defender_rating': 110.0,
                    'primary_defender_age': 28.0,
                    'defender_size_mismatch': 0.0,
                    'defender_recent_form': 0.5,
                })
            
            # Tier 9: Pace-Adjusted Stats (Per 100 Possessions)
            try:
                # Get team pace for normalization
                team_pace_val = float((team_context or {}).get('pace', 100.0) or 100.0)
                
                # Calculate per-100 stats (normalize by pace)
                # Formula: (stat per game) * (100 / team_pace) * (48 / avg_minutes)
                avg_minutes = float(features.get('mins_season', 25.0) or 25.0)
                pace_factor = (100.0 / team_pace_val) * (48.0 / max(avg_minutes, 1.0))
                
                season_avg_pts = float(stat_data.get('avg', 0) or 0)
                season_avg_ast = float((stats.get('assists', {}) or {}).get('avg', 0) or 0)
                season_avg_reb = float((stats.get('rebounds', {}) or {}).get('avg', 0) or 0)
                season_avg_stl = float((stats.get('steals', {}) or {}).get('avg', 0) or 0)
                season_avg_blk = float((stats.get('blocks', {}) or {}).get('avg', 0) or 0)
                season_avg_tov = float((stats.get('turnovers', {}) or {}).get('avg', 0) or 0)
                
                features.update({
                    'pts_per_100': float(season_avg_pts * pace_factor),
                    'ast_per_100': float(season_avg_ast * pace_factor),
                    'reb_per_100': float(season_avg_reb * pace_factor),
                    'stl_per_100': float(season_avg_stl * pace_factor),
                    'blk_per_100': float(season_avg_blk * pace_factor),
                    'tov_per_100': float(season_avg_tov * pace_factor),
                })
            except Exception as e:
                print(f"Error calculating pace-adjusted stats: {e}")
                features.update({
                    'pts_per_100': 0.0,
                    'ast_per_100': 0.0,
                    'reb_per_100': 0.0,
                    'stl_per_100': 0.0,
                    'blk_per_100': 0.0,
                    'tov_per_100': 0.0,
                })
            
            # Tier 9: Free Throw Rate & Foul Drawing
            try:
                fga_per_game = float(features.get('fga_per_game', 0) or 0)
                fta_per_game = float(features.get('fta_per_game', 0) or 0)
                
                # Free throw rate: FTA per FGA
                ft_rate = float(fta_per_game / (fga_per_game + 1e-6))
                
                # Fouls drawn: Estimate from FTA (roughly 2 FTA per foul drawn)
                fouls_drawn_per_game = float(fta_per_game / 2.0)
                
                # And-1 frequency: Estimate from FTA and FGM (rough estimate)
                fgm_per_game = fga_per_game * float(features.get('fg_pct_recent', 0.45) or 0.45)
                and_one_frequency = float(min(fta_per_game / max(fgm_per_game, 1), 0.3))  # Cap at 30%
                
                # Foul drawing ability: Fouls drawn per 36 minutes
                avg_minutes = float(features.get('mins_season', 25.0) or 25.0)
                foul_drawing_ability = float((fouls_drawn_per_game / max(avg_minutes, 1)) * 36.0)
                
                features.update({
                    'ft_rate': ft_rate,
                    'fouls_drawn_per_game': fouls_drawn_per_game,
                    'ft_attempts_per_game': fta_per_game,  # Already have, but keep for consistency
                    'and_one_frequency': and_one_frequency,
                    'foul_drawing_ability': foul_drawing_ability,
                })
            except Exception as e:
                print(f"Error calculating free throw rate: {e}")
                features.update({
                    'ft_rate': 0.3,
                    'fouls_drawn_per_game': 2.0,
                    'ft_attempts_per_game': 4.0,
                    'and_one_frequency': 0.1,
                    'foul_drawing_ability': 3.0,
                })
            
            # Tier 9: Rebounding Rates & Positioning
            try:
                oreb_per_game = float(features.get('oreb_per_game', 0) or 0)
                dreb_per_game = float(features.get('dreb_per_game', 0) or 0)
                total_reb_per_game = oreb_per_game + dreb_per_game
                
                # Estimate total rebound opportunities (league average ~45 per game per team)
                # Offensive rebound rate: OREB / (team offensive rebound opportunities)
                # Defensive rebound rate: DREB / (opponent missed shots)
                # Simplified: use league averages for opportunities
                team_oreb_opps = 45.0  # Estimated team offensive rebound opportunities
                opp_missed_shots = 50.0  # Estimated opponent missed shots
                
                oreb_rate = float(oreb_per_game / (team_oreb_opps + 1e-6))
                dreb_rate = float(dreb_per_game / (opp_missed_shots + 1e-6))
                total_reb_rate = float(total_reb_per_game / (team_oreb_opps + opp_missed_shots + 1e-6))
                
                # Rebound contested percentage (estimate: higher for better rebounders)
                rebound_contested_pct = float(min(total_reb_rate * 2.0, 1.0))  # Estimate
                
                # Rebound positioning score (estimate: based on rebounding rate)
                rebound_positioning_score = float(total_reb_rate)
                
                features.update({
                    'oreb_rate': oreb_rate,
                    'dreb_rate': dreb_rate,
                    'total_reb_rate': total_reb_rate,
                    'rebound_contested_pct': rebound_contested_pct,
                    'rebound_positioning_score': rebound_positioning_score,
                })
            except Exception as e:
                print(f"Error calculating rebounding rates: {e}")
                features.update({
                    'oreb_rate': 0.1,
                    'dreb_rate': 0.2,
                    'total_reb_rate': 0.15,
                    'rebound_contested_pct': 0.3,
                    'rebound_positioning_score': 0.15,
                })
            
            # Tier 9: Points in the Paint
            try:
                shot_quality = stats.get('shot_quality', {})
                shot_location = stats.get('shot_location', {})
                
                # Paint points (estimate from shot location data)
                paint_fga = float(shot_location.get('paint_fga_per_game', 0.0) or 0.0)
                rim_fga = float(shot_location.get('rim_fga_per_game', 0.0) or 0.0)
                paint_fg_pct = float(shot_quality.get('restricted_area_fg_pct', 0.6) or 0.6)
                
                paint_attempts_per_game = paint_fga + rim_fga
                paint_pts_per_game = float(paint_attempts_per_game * paint_fg_pct * 2.0)  # 2 points per made shot
                
                # Paint touch to points ratio
                paint_touches = float(features.get('paint_touches_per_game', 0.0) or 0.0)
                paint_touch_to_points = float(paint_pts_per_game / (paint_touches + 1e-6))
                
                # Restricted area attempts
                restricted_area_attempts = rim_fga
                
                features.update({
                    'paint_pts_per_game': paint_pts_per_game,
                    'paint_attempts_per_game': paint_attempts_per_game,
                    'paint_fg_pct': paint_fg_pct,
                    'paint_touch_to_points': paint_touch_to_points,
                    'restricted_area_attempts': restricted_area_attempts,
                })
            except Exception as e:
                print(f"Error calculating paint stats: {e}")
                season_avg_pts = float(stat_data.get('avg', 0) or 0)
                features.update({
                    'paint_pts_per_game': float(season_avg_pts * 0.4),  # Estimate 40% from paint
                    'paint_attempts_per_game': 5.0,
                    'paint_fg_pct': 0.6,
                    'paint_touch_to_points': 0.5,
                    'restricted_area_attempts': 3.0,
                })
            
            # Tier 9: Game Situation Performance
            try:
                # Use plus/minus and win/loss to estimate game situation performance
                plus_minus_avg = float(features.get('plus_minus_avg', 0.0) or 0.0)
                win_rate = float(features.get('win_rate_last10', 0.5) or 0.5)
                
                # Estimate performance in different situations
                # When leading: players might play more conservatively (slightly lower stats)
                # When trailing: players might play more aggressively (slightly higher stats)
                performance_when_leading = float(season_avg_pts * (1.0 - 0.05 * (1.0 - win_rate)))  # Slight decrease when leading
                performance_when_trailing = float(season_avg_pts * (1.0 + 0.05 * (1.0 - win_rate)))  # Slight increase when trailing
                performance_when_tied = float(season_avg_pts)  # Baseline
                
                # Overtime performance (estimate: similar to clutch performance)
                clutch_pts = float(features.get('clutch_pts_per_game', 0.0) or 0.0)
                performance_in_overtime = float(clutch_pts * 1.2) if clutch_pts > 0 else float(season_avg_pts * 1.1)
                
                # Performance by score differential (estimate from plus/minus)
                # Positive plus/minus = team usually leading = slightly lower individual stats
                # Negative plus/minus = team usually trailing = slightly higher individual stats
                performance_by_score_differential = float(season_avg_pts * (1.0 + plus_minus_avg * 0.01))
                
                features.update({
                    'performance_when_leading': performance_when_leading,
                    'performance_when_trailing': performance_when_trailing,
                    'performance_when_tied': performance_when_tied,
                    'performance_in_overtime': performance_in_overtime,
                    'performance_by_score_differential': performance_by_score_differential,
                })
            except Exception as e:
                print(f"Error calculating game situation performance: {e}")
                season_avg_pts = float(stat_data.get('avg', 0) or 0)
                features.update({
                    'performance_when_leading': float(season_avg_pts * 0.95),
                    'performance_when_trailing': float(season_avg_pts * 1.05),
                    'performance_when_tied': float(season_avg_pts),
                    'performance_in_overtime': float(season_avg_pts * 1.1),
                    'performance_by_score_differential': float(season_avg_pts),
                })
            
            # Extract Tier 4: Calculated Efficiency, Opponent Trends, Context, Teammate, Defensive
            calc_eff = stats.get('calculated_efficiency', {})
            opp_adj = stats.get('opponent_adjusted', {})
            def_metrics = stats.get('defensive_metrics', {})
            play_type = stats.get('play_type_data', {})
            
            # Extract Tier 5: Player Streaks, Game Importance, Rotation, Matchup History
            # Note: player_streaks needs game log data, which we don't have here
            # We'll calculate it from the stats we have
            try:
                # Rebuild game DataFrame from stats for streaks calculation
                stat_values = []
                prop_stat_map = {
                    'points': stats.get('points', {}), 'assists': stats.get('assists', {}),
                    'rebounds': stats.get('rebounds', {}), 'three_pointers': stats.get('three_pointers', {}),
                    'steals': stats.get('steals', {}), 'blocks': stats.get('blocks', {}),
                    'turnovers': stats.get('turnovers', {})
                }
                prop_data = prop_stat_map.get(prop_type, {})
                if prop_data and 'values' in prop_data:
                    stat_values = prop_data['values']
                
                # Build simple DataFrame for streaks
                if stat_values:
                    streak_df = pd.DataFrame({'stat': stat_values})
                    player_streaks = self._calculate_player_streaks(streak_df, prop_type, line)
                else:
                    player_streaks = {}
            except Exception:
                player_streaks = {}
            
            game_importance = self._calculate_game_importance(team_id, opponent_team_id, self.current_season)
            rotation_patterns = stats.get('rotation_patterns', {})
            matchup_history = self._get_specific_matchup_history(player_id, opponent_team_id, self.current_season)
            
            features.update({
                # Calculated efficiency metrics
                'points_per_shot': float(calc_eff.get('points_per_shot', 0.0) or 0.0),
                'ast_to_tov_ratio': float(calc_eff.get('ast_to_tov_ratio', 0.0) or 0.0),
                'reb_rate_per_36': float(calc_eff.get('reb_rate_per_36', 0.0) or 0.0),
                'scoring_efficiency_trend': float(calc_eff.get('scoring_efficiency_trend', 0.0) or 0.0),
                'usage_trend': float(calc_eff.get('usage_trend', 0.0) or 0.0),
                'minutes_volatility': float(calc_eff.get('minutes_volatility', 0.0) or 0.0),
                'blowout_game_pct': float(calc_eff.get('blowout_game_pct', 0.0) or 0.0),
                'close_game_pct': float(calc_eff.get('close_game_pct', 0.0) or 0.0),
                # Opponent defensive trends
                'opp_def_rating_last5': float(opponent_context.get('defensive_rating', 110.0) or 110.0),
                'opp_def_rating_trend': 0.0,  # Placeholder
                'opp_def_rating_home_away_split': 0.0,  # Placeholder
                'opp_blocks_per_game_last5': float(opp_base.get('blk', 0.0) or 0.0),
                'opp_steals_per_game_last5': float(opp_base.get('stl', 0.0) or 0.0),
                # Game context features
                'days_rest_opponent': 2,  # Placeholder
                'opponent_back_to_back': 0,  # Placeholder
                'playoff_implications': 0,  # Placeholder
                'rivalry_game': 0,  # Placeholder
                'national_tv_game': 0,  # Placeholder
                'season_phase': 0,  # Placeholder: 0=early, 1=mid, 2=late
                # Teammate impact features
                'primary_teammate_out': 0,  # Placeholder
                'secondary_teammate_out': 0,  # Placeholder
                'new_teammate_games': 0,  # Placeholder
                'lineup_stability_score': 1.0,  # Placeholder
                'bench_strength': 0.0,  # Placeholder
                # Opponent-adjusted stats
                'pts_vs_top10_defenses': float(opp_adj.get('pts_vs_top10_defenses', 0.0) or 0.0),
                'pts_vs_bottom10_defenses': float(opp_adj.get('pts_vs_bottom10_defenses', 0.0) or 0.0),
                'consistency_score': float(opp_adj.get('consistency_score', 0.0) or 0.0),
                'ceiling_game_frequency': float(opp_adj.get('ceiling_game_frequency', 0.0) or 0.0),
                # Advanced defensive metrics
                'def_fg_pct_allowed': float(def_metrics.get('def_fg_pct_allowed', 0.0) or 0.0),
                'def_rating_individual': float(def_metrics.get('def_rating_individual', 0.0) or 0.0),
                'deflections_per_game': float(def_metrics.get('deflections_per_game', 0.0) or 0.0),
                'contested_shots_per_game': float(def_metrics.get('contested_shots_per_game', 0.0) or 0.0),
                # Play type data
                'pnr_ball_handler_pct': float(play_type.get('pnr_ball_handler_pct', 0.0) or 0.0),
                'pnr_roll_man_pct': float(play_type.get('pnr_roll_man_pct', 0.0) or 0.0),
                'isolation_pct': float(play_type.get('isolation_pct', 0.0) or 0.0),
                'spot_up_pct': float(play_type.get('spot_up_pct', 0.0) or 0.0),
                'post_up_pct': float(play_type.get('post_up_pct', 0.0) or 0.0),
                'transition_pct': float(play_type.get('transition_pct', 0.0) or 0.0),
            })
            
            # Extract Tier 5: Player Streaks, Game Importance, Rotation, Matchup
            features.update({
                # Player Streaks
                'consecutive_over_games': int(player_streaks.get('consecutive_over_games', 0) or 0),
                'consecutive_under_games': int(player_streaks.get('consecutive_under_games', 0) or 0),
                'hot_hand_indicator': float(player_streaks.get('hot_hand_indicator', 0.0) or 0.0),
                'recent_variance_spike': float(player_streaks.get('recent_variance_spike', 0.0) or 0.0),
                # Game Importance
                'playoff_seeding_impact': float(game_importance.get('playoff_seeding_impact', 0.5) or 0.5),
                'tanking_indicator': float(game_importance.get('tanking_indicator', 0.0) or 0.0),
                'must_win_situation': float(game_importance.get('must_win_situation', 0.0) or 0.0),
                'games_back_from_playoff': float(game_importance.get('games_back_from_playoff', 0.0) or 0.0),
                # Rotation Patterns
                'fourth_quarter_usage_rate': float(rotation_patterns.get('fourth_quarter_usage_rate', 0.2) or 0.2),
                'garbage_time_minutes_pct': float(rotation_patterns.get('garbage_time_minutes_pct', 0.05) or 0.05),
                'typical_substitution_minute': float(rotation_patterns.get('typical_substitution_minute', 36.0) or 36.0),
                'crunch_time_usage': float(rotation_patterns.get('crunch_time_usage', 0.2) or 0.2),
                # Specific Matchup History
                'career_vs_defender': float(matchup_history.get('career_vs_defender', 0.0) or 0.0),
                'recent_vs_defender': float(matchup_history.get('recent_vs_defender', 0.0) or 0.0),
                'player_vs_arena': float(matchup_history.get('player_vs_arena', 0.0) or 0.0),
            })

            # ML prediction
            ml_prediction = self.ml_predictor.predict(features, line, prop_type=prop_type)
            if not ml_prediction:
                ml_prediction = {
                    'over_probability': hit_rate,
                    'predicted_value': stat_data.get('avg', line),
                    'recommendation': 'PASS',
                    'confidence': 'LOW',
                    'edge': edge
                }

            # Calibrate probability: blend model prob with a smoothed empirical hit-rate.
            # Beta(1,1) prior => posterior mean (hits+1)/(n+2)
            n = len(values)
            posterior_mean = (hits + 1) / (n + 2) if n > 0 else 0.5
            model_prob = float(ml_prediction.get('over_probability', hit_rate))
            calibrated_prob = 0.65 * model_prob + 0.35 * float(posterior_mean)
            calibrated_prob = float(min(max(calibrated_prob, 0.01), 0.99))
            ml_prediction['over_probability_raw'] = model_prob
            ml_prediction['over_probability'] = calibrated_prob
            ml_prediction['factors'] = {
                **(ml_prediction.get('factors', {}) or {}),
                'empirical_hit_rate': float(hit_rate),
                'empirical_posterior_mean': float(posterior_mean),
                'calibration_blend': {'model': 0.65, 'empirical': 0.35}
            }

            # Model provenance + freshness
            model_source = (ml_prediction.get('factors') or {}).get('model_source') or 'heuristic'
            model_info = self.ml_predictor.get_model_artifact_info(prop_type=prop_type, model_source=model_source)
            prop_meta = ((self.ml_predictor.model_metadata or {}).get("props") or {}).get(prop_type, {})
            pre_meta = pre or {}
            dvp_ts = int((pre_meta.get("dvp_meta") or {}).get("updated_at") or 0)
            def_ts = int((pre_meta.get("defenders_meta") or {}).get("updated_at") or 0)

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
                'factor_breakdown': ml_prediction.get('factors', {}),
                'factor_analysis': ml_prediction.get('factor_analysis', {}),  # Comprehensive factor analysis
                'model_used': model_info or {'source': model_source},
                'model_metrics': {
                    'walk_forward': prop_meta.get('walk_forward'),
                    'rmse': prop_meta.get('rmse'),
                } if prop_meta else None,
                'precomputed_freshness': {
                    'dvp_updated_at': dvp_ts,
                    'defenders_updated_at': def_ts,
                    'updated_at': max(dvp_ts, def_ts),
                },
                'player_stats': stats,
                'context': {
                    'player': player_context,
                    'team': team_context,
                    'opponent': {
                        **(opponent_context or {}),
                        'dvp_position': dvp_pos,
                        'dvp': dvp,
                        'dvp_deltas': dvp_deltas,
                        'special_defenders': special_defenders,
                        'primary_defender': primary_defender
                    }
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
    
