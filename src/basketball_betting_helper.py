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
                notes TEXT,
                hit_rate REAL,
                model_version TEXT
            )
        ''')

        # migrate existing DBs that don't have these columns yet
        for col, typedef in [('hit_rate', 'REAL'), ('model_version', 'TEXT')]:
            try:
                cursor.execute(f'ALTER TABLE prediction_logs ADD COLUMN {col} {typedef}')
            except Exception:
                pass  # column already exists

        conn.commit()
        conn.close()

    def log_prediction(self, player_id, player_name, prop_type, line, is_home,
                       location_detected, predicted_value, over_probability,
                       recommendation, confidence, edge, season_avg, last5_avg, location_avg,
                       hit_rate=None, model_version=None):
        """Saves a prediction to the DB. Returns the new row id."""
        conn = self.get_db()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO prediction_logs
                (timestamp, player_id, player_name, prop_type, line, is_home,
                 location_detected, predicted_value, over_probability,
                 recommendation, confidence, edge, season_avg, last5_avg, location_avg,
                 hit_rate, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                player_id, player_name, prop_type, line,
                (1 if is_home else 0) if is_home is not None else None,
                1 if location_detected else 0,
                predicted_value, over_probability,
                recommendation, confidence, edge,
                season_avg, last5_avg, location_avg,
                hit_rate, model_version
            ))
            log_id = cursor.lastrowid
            conn.commit()
            return log_id
        except Exception as e:
            print(f"error logging prediction: {e}")
            return None
        finally:
            conn.close()

    def update_actual_result(self, log_id, actual_result, notes=None):
        """Records the actual game result and marks the prediction correct/incorrect."""
        conn = self.get_db()
        try:
            cursor = conn.cursor()

            cursor.execute('SELECT line, recommendation FROM prediction_logs WHERE id = ?', (log_id,))
            row = cursor.fetchone()
            if not row:
                return False, 'Log entry not found'

            line, recommendation = row
            actual_outcome = 'OVER' if actual_result > line else 'UNDER'

            # OVER/LEAN OVER must hit over, UNDER/LEAN UNDER must hit under, PASS is ungraded
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
            return True, actual_outcome
        except Exception as e:
            print(f"error updating actual result: {e}")
            return False, str(e)
        finally:
            conn.close()

    def auto_grade_pending(self):
        """Grades all ungraded predictions from previous days using the player's actual game log."""
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

        conn = self.get_db()
        try:
            cursor = conn.cursor()
            # grab anything without an actual_result from before today (includes PASSes)
            cursor.execute('''
                SELECT id, player_id, prop_type, line, timestamp
                FROM prediction_logs
                WHERE actual_result IS NULL
                AND date(timestamp) < ?
            ''', (today.isoformat(),))
            pending = cursor.fetchall()
        except Exception as e:
            print(f"auto_grade_pending db error: {e}")
            return {'graded': 0, 'skipped': 0, 'errors': 1}
        finally:
            conn.close()

        graded = skipped = errors = 0
        gamelog_cache = {}  # avoid hitting the API twice for the same player/season

        for log_id, player_id, prop_type, line, timestamp in pending:
            try:
                pred_date = datetime.fromisoformat(timestamp).date()

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

                # +1 day tolerance for late-night games that log the next day
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
        """Turns graded prediction_logs rows into training samples for the ML model."""
        conn = self.get_db()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT season_avg, last5_avg, location_avg, predicted_value,
                       over_probability, edge, is_home, hit_rate, line, actual_result,
                       prop_type
                FROM prediction_logs
                WHERE correct IN (0, 1) AND actual_result IS NOT NULL
            ''')
            rows = cursor.fetchall()
        except Exception as e:
            print(f"get_log_training_samples error: {e}")
            return []
        finally:
            conn.close()

        samples = []
        for row in rows:
            (season_avg, last5_avg, location_avg, predicted_value,
             over_prob, edge, is_home, stored_hit_rate, line, actual_result,
             prop_type) = row

            season_avg  = season_avg  or 0.0
            last5_avg   = last5_avg   or 0.0
            location_avg = location_avg or season_avg
            edge         = edge        or 0.0

            # reconstruct training features from stored log columns
            # stddev: we don't have raw game values, so proxy with how much recent form deviates from season avg
            _spread = abs(float(season_avg) - float(last5_avg))
            # max_recent must be >= last5_avg; min_recent must be <= last5_avg — use spread as an estimate
            features = {
                'recent_avg':   float(last5_avg),
                'season_avg':   float(season_avg),
                'stddev':       _spread,
                'max_recent':   float(last5_avg) + _spread,       # approx: avg + one spread unit
                'min_recent':   max(0.0, float(last5_avg) - _spread),  # approx: avg - one spread unit, floor 0
                'games_played': 20,
                'hit_rate':     float(stored_hit_rate) if stored_hit_rate is not None else 0.5,
                'edge':         float(edge),
                'is_home':      float(is_home) if is_home is not None else 0.5,
                'location_avg': float(location_avg),
            }
            samples.append({
                'features':  features,
                'result':    float(actual_result),
                'line':      float(line),
                'prop_type': prop_type,
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

            print("retrain: collecting historical data...")
            collector = TrainingDataCollector()
            player_ids = collector.get_active_player_ids(n=num_players)
            seasons = collector._get_seasons(num_seasons=num_seasons)
            historical = collector.collect_bulk(player_ids, seasons=seasons)
            print(f"retrain: {len(historical)} historical samples collected")

            # double up log samples since they use real lines (higher quality)
            log_samples = self.get_log_training_samples()
            print(f"retrain: {len(log_samples)} graded log samples (2x weight)")
            weighted_log = log_samples * 2

            all_samples = historical + weighted_log
            print(f"retrain: total training samples = {len(all_samples)}")

            if len(all_samples) < 500:
                return {
                    'status': 'skipped',
                    'reason': f'Only {len(all_samples)} total samples, need 500',
                }

            metrics = self.ml_predictor.train(all_samples)

            # update retrain_meta
            conn = self.get_db()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE retrain_meta
                    SET last_retrain_at = ?, samples_at_last_retrain = ?,
                        last_auc = ?, last_rmse = ?
                    WHERE id = 1
                ''', (datetime.now().isoformat(), current_count,
                      metrics.get('auc'), metrics.get('rmse')))
                conn.commit()
            finally:
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
            print(f"error fetching logs: {e}")
            return []

    def get_accuracy_stats(self):
        try:
            conn = self.get_db()
            cursor = conn.cursor()

            # only count OVER/UNDER picks (correct=0/1); PASS rows have correct=NULL
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

            # last 20 directional picks for recent form
            cursor.execute('''
                SELECT correct FROM prediction_logs
                WHERE correct IN (0, 1)
                ORDER BY timestamp DESC LIMIT 20
            ''')
            recent = [r[0] for r in cursor.fetchall()]
            recent_accuracy = round(100.0 * sum(recent) / len(recent), 1) if recent else None

            # all predictions, even ungraded ones
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
            print(f"accuracy stats error: {e}")
            return {}

    def get_bias_report(self):
        """Queries graded logs to surface systematic over/under-estimation by prop type and location."""
        try:
            conn = self.get_db()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT prop_type,
                    AVG(predicted_value - actual_result) as avg_error,
                    AVG(ABS(predicted_value - actual_result)) as mae,
                    COUNT(*) as n,
                    ROUND(100.0 * SUM(correct) / NULLIF(COUNT(*), 0), 1) as accuracy_pct,
                    AVG(over_probability) as avg_prob,
                    AVG(CASE WHEN actual_result > line THEN 1.0 ELSE 0.0 END) as actual_over_rate
                FROM prediction_logs
                WHERE correct IN (0, 1)
                  AND predicted_value IS NOT NULL
                  AND actual_result IS NOT NULL
                GROUP BY prop_type
                HAVING COUNT(*) >= 10
                ORDER BY ABS(AVG(predicted_value - actual_result)) DESC
            ''')
            cols = [d[0] for d in cursor.description]
            by_prop = [dict(zip(cols, r)) for r in cursor.fetchall()]

            cursor.execute('''
                SELECT
                    CASE WHEN is_home = 1 THEN 'home'
                         WHEN is_home = 0 THEN 'away'
                         ELSE 'unknown' END as location,
                    AVG(predicted_value - actual_result) as avg_error,
                    COUNT(*) as n,
                    ROUND(100.0 * SUM(correct) / NULLIF(COUNT(*), 0), 1) as accuracy_pct,
                    AVG(over_probability) as avg_prob,
                    AVG(CASE WHEN actual_result > line THEN 1.0 ELSE 0.0 END) as actual_over_rate
                FROM prediction_logs
                WHERE correct IN (0, 1)
                  AND predicted_value IS NOT NULL
                  AND actual_result IS NOT NULL
                GROUP BY is_home
                HAVING COUNT(*) >= 5
            ''')
            cols = [d[0] for d in cursor.description]
            by_location = [dict(zip(cols, r)) for r in cursor.fetchall()]

            cursor.execute('''
                SELECT confidence,
                    COUNT(*) as n,
                    ROUND(100.0 * SUM(correct) / NULLIF(COUNT(*), 0), 1) as accuracy_pct,
                    AVG(over_probability) as avg_prob,
                    AVG(CASE WHEN actual_result > line THEN 1.0 ELSE 0.0 END) as actual_over_rate
                FROM prediction_logs
                WHERE correct IN (0, 1)
                  AND actual_result IS NOT NULL
                GROUP BY confidence
            ''')
            cols = [d[0] for d in cursor.description]
            by_confidence = [dict(zip(cols, r)) for r in cursor.fetchall()]

            conn.close()
            return {
                'by_prop': by_prop,
                'by_location': by_location,
                'by_confidence': by_confidence,
            }
        except Exception as e:
            print(f"bias report error: {e}")
            return {}

    def get_confidence_calibration_data(self):
        """Returns graded rows needed to recalibrate the confidence thresholds."""
        try:
            conn = self.get_db()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT over_probability, edge, correct
                FROM prediction_logs
                WHERE correct IN (0, 1)
                  AND over_probability IS NOT NULL
                  AND edge IS NOT NULL
            ''')
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    'over_probability': float(r[0]),
                    'edge': float(r[1]),
                    'correct': int(r[2]),
                    # replicate how confidence_score is computed in the model
                    'confidence_score': 0.7 * abs(float(r[0]) - 0.5) + 0.3 * abs(float(r[1])),
                }
                for r in rows
            ]
        except Exception as e:
            print(f"confidence calibration data error: {e}")
            return []

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
            try:
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
            finally:
                conn.close()

            return suggestions
            
        except Exception as e:
            print(f"player suggestions error: {e}")
            return []

    def get_player_stats(self, player_id):
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
            print(f"fetching seasons: {seasons}")
            
            all_games = []
            
            for season in seasons:
                try:
                    gamelog = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season
                    )
                    time.sleep(0.5)
                    games = gamelog.get_data_frames()[0]
                    print(f"{season}: {len(games)} games")
                    if not games.empty:
                        all_games.append(games)
                except Exception as e:
                    print(f"error fetching {season}: {e}")
                    continue

            if not all_games:
                raise Exception("Could not fetch any game data")

            games_df = pd.concat(all_games, ignore_index=True)
            # multi-season concat can leave numeric cols as object dtype
            for col in ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FG3M', 'FGA', 'FGM', 'FTA', 'FTM', 'OREB', 'DREB', 'PLUS_MINUS']:
                if col in games_df.columns:
                    games_df[col] = pd.to_numeric(games_df[col], errors='coerce').fillna(0)
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            games_df = games_df.sort_values('GAME_DATE', ascending=False)
            
            games_df = games_df.head(20)
            print(f"using {len(games_df)} most recent games")
            
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

            trend_key_map = {
                'points': 'pts', 'assists': 'ast', 'rebounds': 'reb',
                'steals': 'stl', 'blocks': 'blk', 'turnovers': 'tov', 'three_pointers': 'fg3m'
            }
            for stat_name, trend_key in trend_key_map.items():
                if trend_key in stats['trends'] and stat_name in stats:
                    stats[stat_name]['direction'] = stats['trends'][trend_key]['direction']

            return stats
            
        except Exception as e:
            print(f"error getting player stats: {e}")
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

    def _get_double_double_stats(self, df):  # 2 stats >= 10
        stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        double_doubles = df[stats].apply(lambda x: sum(x >= 10) >= 2, axis=1)
        return {
            'values': double_doubles.astype(int).tolist(),
            'avg': float(double_doubles.mean()),
            'last5_avg': float(double_doubles.head(5).mean())
        }

    def _get_triple_double_stats(self, df):  # 3 stats >= 10
        stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        triple_doubles = df[stats].apply(lambda x: sum(x >= 10) >= 3, axis=1)
        return {
            'values': triple_doubles.astype(int).tolist(),
            'avg': float(triple_doubles.mean()),
            'last5_avg': float(triple_doubles.head(5).mean())
        }

    def _calculate_trends(self, df):
        trends = {}
        stats = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FG3M']

        for stat in stats:
            values = df[stat].values
            if len(values) >= 5:
                recent_values = values[:5][::-1]  # flip to oldest-first for polyfit
                z = np.polyfit(range(len(recent_values)), recent_values, 1)
                slope = z[0]
                
                trends[stat.lower()] = {
                    'slope': float(slope),
                    'direction': 'Increasing' if slope > 0.1 else 'Decreasing' if slope < -0.1 else 'Stable',
                    'strength': abs(float(slope))
                }
        
        return trends

    def _detect_home_away(self, team_id, opponent_team_id):
        """Checks today's scoreboard to figure out if team_id is home or away. Returns None if no game."""
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
            return None
        except Exception as e:
            print(f"home/away detection failed: {e}")
            return None

    def analyze_prop_bet(self, player_id, prop_type, line, opponent_team_id, is_home=None):
        try:
            stats = self.get_player_stats(player_id)
            if not stats:
                return {
                    'success': False,
                    'error': 'Unable to retrieve player stats'
                }

            player_context = self.ml_predictor.get_player_context(player_id, opponent_team_id)
            team_id = (player_context.get('team_id') if player_context else None) or self._get_player_team_id(player_id)

            user_set_location = is_home is not None
            location_detected = False
            if is_home is None and team_id:
                detected = self._detect_home_away(team_id, opponent_team_id)
                if detected is not None:
                    is_home = detected
                    location_detected = True
            location_known = location_detected or user_set_location
            team_context = self.ml_predictor.get_team_context(team_id) if team_id else None
            opponent_context = self.ml_predictor.get_team_context(opponent_team_id)

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

            hits = sum(1 for x in values if x > line)
            hit_rate = hits / len(values) if values else 0
            # use last5_avg to match training (data_collector computes edge from recent_avg = mean(last5))
            last5_avg_val = stat_data.get('last5_avg', stat_data.get('avg', 0))
            edge = ((last5_avg_val - line) / line) if line > 0 else 0

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
                    # keep edge consistent with the location-adjusted baseline
                    features['edge'] = ((location_avg_val - line) / line) if line > 0 else 0
            else:
                location_avg_val = float(stat_data.get('avg', 0))
            features['location_avg'] = location_avg_val

            ml_prediction = self.ml_predictor.predict(features, line, prop_type=prop_type)
            if not ml_prediction:
                ml_prediction = {
                    'over_probability': hit_rate,
                    'predicted_value': stat_data.get('avg', line),
                    'recommendation': 'PASS',
                    'confidence': 'LOW',
                    'edge': edge
                }

            location_avg = location_avg_val
            if location_known and is_home is not None:
                location_games = int(stat_data.get('home_games' if is_home else 'away_games', len(values)))
            else:
                location_games = len(values)

            player_name = None
            if player_context and player_context.get('position'):
                try:
                    from nba_api.stats.static import players as nba_players_static
                    player_info = next((p for p in nba_players_static.get_players() if p['id'] == player_id), None)
                    player_name = player_info['full_name'] if player_info else str(player_id)
                except Exception as e:
                    print(f"couldn't resolve player name for {player_id}: {e}")
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
                hit_rate=hit_rate,
                model_version=self.ml_predictor.model_version,
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
                'context': {
                    'player': player_context,
                    'team': team_context,
                    'opponent': opponent_context
                }
            }

        except Exception as e:
            print(f"analyze_prop_bet error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_player_team_id(self, player_id):
        try:
            player_info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            time.sleep(0.6)
            return int(player_info['TEAM_ID'].iloc[0])
        except Exception as e:
            print(f"couldn't get team id for player {player_id}: {e}")
            return None
    
