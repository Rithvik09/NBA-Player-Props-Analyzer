import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, CommonPlayerInfo
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
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retrain_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_retrain_at TEXT,
                samples_at_last_retrain INTEGER DEFAULT 0,
                last_auc REAL,
                last_rmse REAL
            )
        ''')
        cursor.execute('''
            INSERT OR IGNORE INTO retrain_meta (id, samples_at_last_retrain)
            VALUES (1, 0)
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                player_name TEXT,
                prop_type TEXT NOT NULL,
                line REAL NOT NULL,
                is_home INTEGER,
                location_detected INTEGER DEFAULT 0,
                predicted_value REAL,
                over_probability REAL,
                recommendation TEXT,
                confidence TEXT,
                edge REAL,
                season_avg REAL,
                last5_avg REAL,
                location_avg REAL,
                actual_result REAL,
                actual_outcome TEXT,
                correct INTEGER,
                notes TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def log_prediction(self, player_id, player_name, prop_type, line, is_home,
                       location_detected, predicted_value, over_probability,
                       recommendation, confidence, edge, season_avg, last5_avg, location_avg):
        """Save a prediction to the database. Returns the log id."""
        try:
            conn = self.get_db()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO prediction_logs
                (timestamp, player_id, player_name, prop_type, line, is_home,
                 location_detected, predicted_value, over_probability,
                 recommendation, confidence, edge, season_avg, last5_avg, location_avg)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                player_id, player_name, prop_type, line,
                (1 if is_home else 0) if is_home is not None else None,
                1 if location_detected else 0,
                predicted_value, over_probability,
                recommendation, confidence, edge,
                season_avg, last5_avg, location_avg
            ))
            log_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return log_id
        except Exception as e:
            print(f"Error logging prediction: {e}")
            return None

    def update_actual_result(self, log_id, actual_result, notes=None):
        """Record the actual stat value after the game and compute correctness."""
        try:
            conn = self.get_db()
            cursor = conn.cursor()

            cursor.execute('SELECT line, recommendation FROM prediction_logs WHERE id = ?', (log_id,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False, 'Log entry not found'

            line, recommendation = row
            actual_outcome = 'OVER' if actual_result > line else 'UNDER'

            # A prediction is correct if rec was OVER/LEAN OVER and it went over,
            # or rec was UNDER/LEAN UNDER and it went under.
            # PASS predictions are logged but not counted as correct/incorrect.
            if 'OVER' in recommendation:
                correct = 1 if actual_outcome == 'OVER' else 0
            elif 'UNDER' in recommendation:
                correct = 1 if actual_outcome == 'UNDER' else 0
            else:
                correct = None  # PASS — not graded

            cursor.execute('''
                UPDATE prediction_logs
                SET actual_result = ?, actual_outcome = ?, correct = ?, notes = ?
                WHERE id = ?
            ''', (actual_result, actual_outcome, correct, notes, log_id))
            conn.commit()
            conn.close()
            return True, actual_outcome
        except Exception as e:
            print(f"Error updating actual result: {e}")
            return False, str(e)

    def auto_grade_pending(self):
        """
        Find all ungraded predictions from previous days, fetch the actual
        stat from the player's game log, and auto-grade them.
        Returns a dict with counts: {graded, skipped, errors}
        """
        PROP_COL_MAP = {
            'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB',
            'steals': 'STL', 'blocks': 'BLK', 'turnovers': 'TOV',
            'three_pointers': 'FG3M',
            # Combined props — summed
            'pts_reb': ['PTS', 'REB'], 'pts_ast': ['PTS', 'AST'],
            'ast_reb': ['AST', 'REB'], 'pts_ast_reb': ['PTS', 'AST', 'REB'],
            'stl_blk': ['STL', 'BLK'],
            # Special — computed
            'double_double': 'DD', 'triple_double': 'TD',
        }

        today = datetime.now().date()

        try:
            conn = self.get_db()
            cursor = conn.cursor()
            # Fetch ungraded predictions from before today.
            # Include PASS rows (correct IS NULL) that haven't had actual_result filled in yet.
            # Exclude already-graded PASS rows (actual_result IS NOT NULL) to avoid re-processing.
            cursor.execute('''
                SELECT id, player_id, prop_type, line, timestamp
                FROM prediction_logs
                WHERE actual_result IS NULL
                AND date(timestamp) < ?
            ''', (today.isoformat(),))
            pending = cursor.fetchall()
            conn.close()
        except Exception as e:
            print(f"auto_grade_pending: DB error: {e}")
            return {'graded': 0, 'skipped': 0, 'errors': 1}

        graded = skipped = errors = 0
        # Cache game logs per player to avoid redundant API calls
        gamelog_cache = {}

        for log_id, player_id, prop_type, line, timestamp in pending:
            try:
                pred_date = datetime.fromisoformat(timestamp).date()

                # Determine season string for that date
                year = pred_date.year
                month = pred_date.month
                if 1 <= month <= 7:
                    season = f"{year-1}-{str(year)[2:]}"
                else:
                    season = f"{year}-{str(year+1)[2:]}"

                cache_key = (player_id, season)
                if cache_key not in gamelog_cache:
                    games = playergamelog.PlayerGameLog(
                        player_id=player_id, season=season
                    ).get_data_frames()[0]
                    time.sleep(0.6)
                    games['_date'] = pd.to_datetime(games['GAME_DATE']).dt.date
                    gamelog_cache[cache_key] = games

                games = gamelog_cache[cache_key]

                # Match the game on the prediction date (allow +1 day for late-night games)
                row = games[games['_date'] == pred_date]
                if row.empty:
                    row = games[games['_date'] == pred_date + timedelta(days=1)]
                if row.empty:
                    skipped += 1
                    continue

                row = row.iloc[0]
                col = PROP_COL_MAP.get(prop_type)

                if col is None:
                    skipped += 1
                    continue
                elif isinstance(col, list):
                    actual = float(sum(row[c] for c in col))
                elif col == 'DD':
                    stats = [row['PTS'], row['REB'], row['AST'], row['STL'], row['BLK']]
                    actual = 1.0 if sum(s >= 10 for s in stats) >= 2 else 0.0
                elif col == 'TD':
                    stats = [row['PTS'], row['REB'], row['AST'], row['STL'], row['BLK']]
                    actual = 1.0 if sum(s >= 10 for s in stats) >= 3 else 0.0
                else:
                    actual = float(row[col])

                success, _ = self.update_actual_result(log_id, actual, notes='auto-graded')
                if success:
                    graded += 1
                else:
                    errors += 1

            except Exception as e:
                print(f"auto_grade_pending: error on log {log_id}: {e}")
                errors += 1

        print(f"auto_grade_pending: graded={graded}, skipped={skipped}, errors={errors}")
        return {'graded': graded, 'skipped': skipped, 'errors': errors}

    def get_log_training_samples(self):
        """
        Convert all graded prediction_logs rows into training samples compatible
        with EnhancedMLPredictor.train(). These use real sportsbook lines and real
        outcomes — higher quality signal than simulated rolling-average lines.
        """
        try:
            conn = self.get_db()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT season_avg, last5_avg, location_avg, predicted_value,
                       over_probability, edge, is_home, line, actual_result
                FROM prediction_logs
                WHERE correct IN (0, 1) AND actual_result IS NOT NULL
            ''')
            rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            print(f"get_log_training_samples error: {e}")
            return []

        samples = []
        for row in rows:
            (season_avg, last5_avg, location_avg, predicted_value,
             over_prob, edge, is_home, line, actual_result) = row

            season_avg  = season_avg  or 0.0
            last5_avg   = last5_avg   or 0.0
            location_avg = location_avg or season_avg
            edge         = edge        or 0.0

            # Reconstruct a feature dict matching what prepare_features produces
            features = {
                'recent_avg':   float(last5_avg),
                'season_avg':   float(season_avg),
                'stddev':       abs(float(season_avg) - float(last5_avg)),
                'max_recent':   float(max(season_avg, last5_avg)),
                'min_recent':   float(min(season_avg, last5_avg)),
                'games_played': 20,
                # Use 0.5 (neutral) — we don't store historical hit rate in the log.
                # Using the actual outcome here would be data leakage (label → feature).
                'hit_rate':     0.5,
                'edge':         float(edge),
                'is_home':      float(is_home) if is_home is not None else 0.5,
                'location_avg': float(location_avg),
            }
            samples.append({
                'features': features,
                'result':   float(actual_result),
                'line':     float(line),
            })
        return samples

    def retrain(self, min_new_samples=50, num_players=100, num_seasons=3):
        """
        Full retraining pipeline:
          1. Check how many new graded log samples exist since last retrain
          2. If >= min_new_samples, proceed
          3. Collect fresh historical data via DataCollector
          4. Merge with graded prediction_logs samples (weighted 2× for real lines)
          5. Retrain both models and save
          6. Update retrain_meta
        Returns dict with status and metrics.
        """
        from .data_collector import TrainingDataCollector

        try:
            conn = self.get_db()
            cursor = conn.cursor()
            cursor.execute('SELECT samples_at_last_retrain FROM retrain_meta WHERE id = 1')
            row = cursor.fetchone()
            prev_count = row[0] if row else 0

            cursor.execute('SELECT COUNT(*) FROM prediction_logs WHERE correct IN (0, 1)')
            current_count = cursor.fetchone()[0]
            conn.close()

            new_samples = current_count - prev_count
            print(f"retrain: {new_samples} new graded samples since last retrain (need {min_new_samples})")

            if new_samples < min_new_samples:
                return {
                    'status': 'skipped',
                    'reason': f'Only {new_samples} new graded samples, need {min_new_samples}',
                    'new_samples': new_samples
                }

            # Collect fresh historical data
            print("retrain: collecting historical training data...")
            collector = TrainingDataCollector()
            player_ids = collector.get_active_player_ids(n=num_players)
            seasons = collector._get_seasons(num_seasons=num_seasons)
            historical = collector.collect_bulk(player_ids, seasons=seasons)
            print(f"retrain: {len(historical)} historical samples collected")

            # Get graded log samples — duplicate them for higher weight
            # (real sportsbook lines are more accurate signal than simulated lines)
            log_samples = self.get_log_training_samples()
            print(f"retrain: {len(log_samples)} graded log samples (weighted 2×)")
            weighted_log = log_samples * 2

            all_samples = historical + weighted_log
            print(f"retrain: total training samples = {len(all_samples)}")

            if len(all_samples) < 500:
                return {
                    'status': 'skipped',
                    'reason': f'Only {len(all_samples)} total samples, need 500',
                }

            # Train — returns {'auc': float, 'rmse': float}
            metrics = self.ml_predictor.train(all_samples)

            # Update metadata
            conn = self.get_db()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE retrain_meta
                SET last_retrain_at = ?, samples_at_last_retrain = ?,
                    last_auc = ?, last_rmse = ?
                WHERE id = 1
            ''', (datetime.now().isoformat(), current_count,
                  metrics.get('auc'), metrics.get('rmse')))
            conn.commit()
            conn.close()

            return {
                'status': 'retrained',
                'historical_samples': len(historical),
                'log_samples': len(log_samples),
                'total_samples': len(all_samples),
                'new_graded_since_last': new_samples,
                'auc': metrics.get('auc'),
                'rmse': metrics.get('rmse'),
            }

        except Exception as e:
            print(f"retrain error: {e}")
            import traceback; traceback.print_exc()
            return {'status': 'error', 'error': str(e)}

    def get_retrain_meta(self):
        """Return metadata about the last retraining run."""
        try:
            conn = self.get_db()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM retrain_meta WHERE id = 1')
            cols = [d[0] for d in cursor.description]
            row = cursor.fetchone()
            conn.close()
            return dict(zip(cols, row)) if row else {}
        except Exception as e:
            return {}

    def get_prediction_logs(self, limit=50, prop_type=None, player_id=None, graded_only=False):
        """Fetch prediction logs with optional filters."""
        try:
            conn = self.get_db()
            cursor = conn.cursor()

            conditions = []
            params = []
            if prop_type:
                conditions.append('prop_type = ?')
                params.append(prop_type)
            if player_id:
                conditions.append('player_id = ?')
                params.append(player_id)
            if graded_only:
                conditions.append('correct IN (0, 1)')

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ''
            cursor.execute(f'''
                SELECT * FROM prediction_logs {where}
                ORDER BY timestamp DESC LIMIT ?
            ''', params + [limit])

            cols = [d[0] for d in cursor.description]
            rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            print(f"Error fetching logs: {e}")
            return []

    def get_accuracy_stats(self):
        """Compute overall and per-prop accuracy stats."""
        try:
            conn = self.get_db()
            cursor = conn.cursor()

            # Overall stats — only count predictions with a directional rec (OVER/UNDER),
            # not PASS. correct=NULL means PASS (ungraded by design), correct=0/1 means graded.
            cursor.execute('''
                SELECT
                    COUNT(*) as total_graded,
                    SUM(correct) as total_correct,
                    ROUND(100.0 * SUM(correct) / NULLIF(COUNT(*), 0), 1) as accuracy_pct,
                    COUNT(CASE WHEN confidence = 'HIGH' THEN 1 END) as high_conf_count,
                    SUM(CASE WHEN confidence = 'HIGH' THEN correct ELSE 0 END) as high_conf_correct,
                    COUNT(CASE WHEN confidence = 'MEDIUM' THEN 1 END) as med_conf_count,
                    SUM(CASE WHEN confidence = 'MEDIUM' THEN correct ELSE 0 END) as med_conf_correct
                FROM prediction_logs
                WHERE correct IN (0, 1)
            ''')
            overall = dict(zip([d[0] for d in cursor.description], cursor.fetchone()))

            # Per prop type
            cursor.execute('''
                SELECT prop_type,
                    COUNT(*) as total,
                    SUM(correct) as correct,
                    ROUND(100.0 * SUM(correct) / NULLIF(COUNT(*), 0), 1) as accuracy_pct,
                    ROUND(AVG(edge) * 100, 1) as avg_edge_pct
                FROM prediction_logs
                WHERE correct IN (0, 1)
                GROUP BY prop_type
                ORDER BY accuracy_pct DESC
            ''')
            cols = [d[0] for d in cursor.description]
            by_prop = [dict(zip(cols, row)) for row in cursor.fetchall()]

            # Per player
            cursor.execute('''
                SELECT player_name, player_id,
                    COUNT(*) as total,
                    SUM(correct) as correct,
                    ROUND(100.0 * SUM(correct) / NULLIF(COUNT(*), 0), 1) as accuracy_pct
                FROM prediction_logs
                WHERE correct IN (0, 1)
                GROUP BY player_id
                ORDER BY total DESC
                LIMIT 10
            ''')
            cols = [d[0] for d in cursor.description]
            by_player = [dict(zip(cols, row)) for row in cursor.fetchall()]

            # Recent form: last 20 graded directional picks (not PASS)
            cursor.execute('''
                SELECT correct FROM prediction_logs
                WHERE correct IN (0, 1)
                ORDER BY timestamp DESC LIMIT 20
            ''')
            recent = [r[0] for r in cursor.fetchall()]
            recent_accuracy = round(100.0 * sum(recent) / len(recent), 1) if recent else None

            # Total predictions (including ungraded)
            cursor.execute('SELECT COUNT(*) FROM prediction_logs')
            total_predictions = cursor.fetchone()[0]

            conn.close()
            return {
                'overall': overall,
                'by_prop': by_prop,
                'by_player': by_player,
                'recent_accuracy': recent_accuracy,
                'total_predictions': total_predictions,
            }
        except Exception as e:
            print(f"Error computing accuracy stats: {e}")
            return {}

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
                    if not games.empty:
                        all_games.append(games)
                except Exception as e:
                    print(f"Error fetching {season} data: {e}")
                    continue

            if not all_games:
                raise Exception("Could not fetch any game data")

            games_df = pd.concat(all_games, ignore_index=True)
            # Ensure numeric columns have correct dtype (empty-season concat can leave them as object)
            for col in ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FG3M', 'FGA', 'FGM', 'FTA', 'FTM', 'OREB', 'DREB', 'PLUS_MINUS']:
                if col in games_df.columns:
                    games_df[col] = pd.to_numeric(games_df[col], errors='coerce').fillna(0)
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

            # Merge trend direction into individual stat dicts so the frontend can access it
            trend_key_map = {
                'points': 'pts', 'assists': 'ast', 'rebounds': 'reb',
                'steals': 'stl', 'blocks': 'blk', 'turnovers': 'tov', 'three_pointers': 'fg3m'
            }
            for stat_name, trend_key in trend_key_map.items():
                if trend_key in stats['trends'] and stat_name in stats:
                    stats[stat_name]['direction'] = stats['trends'][trend_key]['direction']

            return stats
            
        except Exception as e:
            print(f"Error getting player stats: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_stat_dict(self, df, column):
        values = df[column].tolist()
        home_df = df[~df['MATCHUP'].str.contains('@')]
        away_df = df[df['MATCHUP'].str.contains('@')]
        return {
            'values': values,
            'avg': float(df[column].mean()),
            'last5_avg': float(df[column].head(5).mean()),
            'max': float(df[column].max()),
            'min': float(df[column].min()),
            'home_avg': float(home_df[column].mean()) if not home_df.empty else float(df[column].mean()),
            'away_avg': float(away_df[column].mean()) if not away_df.empty else float(df[column].mean()),
            'home_games': len(home_df),
            'away_games': len(away_df),
        }

    def _get_combined_stat_dict(self, df, columns):
        combined = df[columns].sum(axis=1)
        home_df = df[~df['MATCHUP'].str.contains('@')]
        away_df = df[df['MATCHUP'].str.contains('@')]
        home_combined = home_df[columns].sum(axis=1)
        away_combined = away_df[columns].sum(axis=1)
        return {
            'values':      combined.tolist(),
            'avg':         float(combined.mean()),
            'last5_avg':   float(combined.head(5).mean()),
            'home_avg':    float(home_combined.mean()) if not home_combined.empty else float(combined.mean()),
            'away_avg':    float(away_combined.mean()) if not away_combined.empty else float(combined.mean()),
            'home_games':  len(home_df),
            'away_games':  len(away_df),
        }

    def _get_double_double_stats(self, df):
        #Calculate double-double stats
        stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        double_doubles = df[stats].apply(lambda x: sum(x >= 10) >= 2, axis=1)
        return {
            'values': double_doubles.astype(int).tolist(),
            'avg': float(double_doubles.mean()),
            'last5_avg': float(double_doubles.head(5).mean())
        }

    def _get_triple_double_stats(self, df):
        #Calculate triple-double stats
        stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        triple_doubles = df[stats].apply(lambda x: sum(x >= 10) >= 3, axis=1)
        return {
            'values': triple_doubles.astype(int).tolist(),
            'avg': float(triple_doubles.mean()),
            'last5_avg': float(triple_doubles.head(5).mean())
        }

    def _calculate_trends(self, df):
        #Calculate performance trends
        trends = {}
        stats = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FG3M']
        
        for stat in stats:
            values = df[stat].values
            if len(values) >= 5:
                # Reverse so index 0 = oldest, index 4 = most recent (data is sorted desc)
                recent_values = values[:5][::-1]
                z = np.polyfit(range(len(recent_values)), recent_values, 1)
                slope = z[0]
                
                trends[stat.lower()] = {
                    'slope': float(slope),
                    'direction': 'Increasing' if slope > 0.1 else 'Decreasing' if slope < -0.1 else 'Stable',
                    'strength': abs(float(slope))
                }
        
        return trends

    def _detect_home_away(self, team_id, opponent_team_id):
        """
        Auto-detect home/away by checking today's NBA schedule.
        Returns True if team_id is the home team, False if away.
        Returns None if no game found between these two teams today.
        """
        try:
            from nba_api.stats.endpoints import ScoreboardV2
            today = datetime.now().strftime('%Y-%m-%d')
            games = ScoreboardV2(game_date=today).get_data_frames()[0]
            time.sleep(0.6)
            for _, game in games.iterrows():
                home = int(game['HOME_TEAM_ID'])
                visitor = int(game['VISITOR_TEAM_ID'])
                if home == int(team_id) and visitor == int(opponent_team_id):
                    return True
                if visitor == int(team_id) and home == int(opponent_team_id):
                    return False
            return None  # No game found today between these teams
        except Exception as e:
            print(f"Could not detect home/away from schedule: {e}")
            return None

    def analyze_prop_bet(self, player_id, prop_type, line, opponent_team_id, is_home=None):
        """Analyze prop bet for given player and line"""
        try:
            stats = stats or self.get_player_stats(player_id)
            if not stats:
                return {
                    'success': False,
                    'error': 'Unable to retrieve player stats'
                }

            # Get more context
            player_context = self.ml_predictor.get_player_context(player_id, opponent_team_id)
            team_id = (player_context.get('team_id') if player_context else None) or self._get_player_team_id(player_id)

            # Auto-detect home/away from today's schedule if not explicitly provided
            # location_known = True only when we're confident (auto-detected OR user-set)
            user_set_location = is_home is not None   # user explicitly chose Home/Away
            location_detected = False
            if is_home is None and team_id:
                detected = self._detect_home_away(team_id, opponent_team_id)
                if detected is not None:
                    is_home = detected
                    location_detected = True
                # else: is_home stays None — no game today, location unknown
            # is_home=None means unknown; don't guess True/False
            location_known = location_detected or user_set_location
            team_context = self.ml_predictor.get_team_context(team_id) if team_id else None

            # Precomputed daily datasets (DVP by position + special defenders)
            pre = self.precomputed.refresh()
            dvp_map = pre.get('dvp', {})
            dvp_pos_avgs = pre.get('dvp_pos_avgs', {})
            defenders_map = pre.get('defenders', {})

            prop_trend_key_map = {
                'points': 'pts', 'assists': 'ast', 'rebounds': 'reb',
                'steals': 'stl', 'blocks': 'blk', 'turnovers': 'tov', 'three_pointers': 'fg3m'
            }

            if prop_type in ['double_double', 'triple_double']:
                stat_data = stats.get(prop_type, {})
                trend = {}
            elif prop_type in prop_trend_key_map:
                stat_data = stats.get(prop_type, {})
                trend = stats.get('trends', {}).get(prop_trend_key_map[prop_type], {})
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

            features = self.ml_predictor.prepare_features(
                stat_data, player_context, team_context, opponent_context
            )
            features['hit_rate'] = hit_rate
            features['edge'] = edge
            # 0.5 = unknown location; 1.0 = home; 0.0 = away
            features['is_home'] = 0.5 if not location_known else (1.0 if is_home else 0.0)

            # Compute location_avg before predict() so it can be used as a feature
            if location_known and is_home is not None:
                location_key = 'home_avg' if is_home else 'away_avg'
                location_avg_val = float(stat_data.get(location_key, stat_data.get('avg', 0)) or stat_data.get('avg', 0))
                # Override recent/season averages with location-specific values
                if stat_data.get(location_key) is not None:
                    features['recent_avg'] = location_avg_val
                    features['season_avg'] = location_avg_val
            else:
                location_avg_val = float(stat_data.get('avg', 0))
            features['location_avg'] = location_avg_val

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

            location_avg = location_avg_val  # already computed above
            if location_known and is_home is not None:
                location_games = int(stat_data.get('home_games' if is_home else 'away_games', len(values)))
            else:
                location_games = len(values)

            # Get player name for logging
            player_name = None
            if player_context and player_context.get('position'):
                try:
                    from nba_api.stats.static import players as nba_players_static
                    player_info = next((p for p in nba_players_static.get_players() if p['id'] == player_id), None)
                    player_name = player_info['full_name'] if player_info else str(player_id)
                except Exception as e:
                    print(f"Warning: could not resolve player name for id {player_id}: {e}")
                    player_name = str(player_id)

            log_id = self.log_prediction(
                player_id=player_id,
                player_name=player_name,
                prop_type=prop_type,
                line=line,
                is_home=is_home,
                location_detected=location_detected,
                predicted_value=ml_prediction.get('predicted_value', stat_data.get('avg', 0)),
                over_probability=ml_prediction.get('over_probability', hit_rate),
                recommendation=ml_prediction.get('recommendation', 'PASS'),
                confidence=ml_prediction.get('confidence', 'LOW'),
                edge=ml_prediction.get('edge', edge),
                season_avg=stat_data.get('avg', 0),
                last5_avg=stat_data.get('last5_avg', 0),
                location_avg=location_avg,
            )

            return {
                'success': True,
                'log_id': log_id,
                'hit_rate': hit_rate,
                'average': stat_data.get('avg', 0),
                'last5_average': stat_data.get('last5_avg', 0),
                'location_avg': location_avg,
                'location_games': location_games,
                'is_home': is_home,
                'location_detected': location_detected,
                'times_hit': hits,
                'total_games': len(values),
                'edge': ml_prediction.get('edge', edge),
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
    
