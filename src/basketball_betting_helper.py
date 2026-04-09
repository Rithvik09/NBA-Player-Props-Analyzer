import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, CommonPlayerInfo, TeamGameLog
from nba_api.stats.static import players
import sqlite3
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from .models import EnhancedMLPredictor
from .incremental_models import IncrementalModelManager
from .ml_features import build_feature_vector, build_classifier_vector
from .precomputed_store import PrecomputedStore


def _compute_calendar_features(ref_date=None):
    """
    Compute season-phase features from current date.
    NBA season typically runs October through April (~82 games = ~200 days).
    """
    from datetime import date
    today = ref_date or date.today()
    year = today.year if today.month >= 10 else today.year - 1
    season_start = date(year, 10, 18)
    season_end   = date(year + 1, 4, 15)
    total_days   = max((season_end - season_start).days, 1)
    elapsed_days = max((today - season_start).days, 0)

    phase = min(1.0, elapsed_days / total_days)
    games_remaining = max(0.0, 82.0 * (1.0 - phase))

    return {
        'days_into_season':       float(elapsed_days),
        'season_phase_numeric':   float(phase),
        'games_remaining_approx': float(games_remaining),
    }


def _compute_injury_trajectory(game_log):
    """
    Given a list of recent game dicts (ordered most-recent first), estimate:
    - games_since_return: how many games since last injury gap
    - missed_games_before_return: estimated games missed in last gap
    Returns (games_since_return, missed_games_before_return).
    """
    if not game_log or len(game_log) < 2:
        return 0.0, 0.0

    try:
        from datetime import datetime
        dates = []
        for g in game_log:
            d = g.get('GAME_DATE') or g.get('game_date') or ''
            if d:
                try:
                    dates.append(datetime.strptime(str(d)[:10], '%Y-%m-%d'))
                except Exception:
                    pass

        if len(dates) < 2:
            return 0.0, 0.0

        dates = sorted(dates, reverse=True)

        games_since_return = 0
        missed_games = 0.0
        found_gap = False

        for i in range(len(dates) - 1):
            gap = (dates[i] - dates[i + 1]).days
            if gap > 5 and not found_gap:
                missed_games = max(0.0, (gap - 2) / 2.0)
                found_gap = True
                break
            elif not found_gap:
                games_since_return += 1

        return float(games_since_return), float(missed_games)
    except Exception:
        return 0.0, 0.0


_defender_active_cache: dict = {}   # player_id -> (result: bool, fetched_at: float)
_DEFENDER_CACHE_TTL = 1800           # 30 minutes


def _check_defender_active(player_id):
    """
    Check if a player (defender) has appeared in the last 10 days.
    Result is cached for 30 minutes so predictions don't each make an API call.
    """
    import time as _time_mod
    now = _time_mod.time()
    cached = _defender_active_cache.get(player_id)
    if cached is not None and (now - cached[1]) < _DEFENDER_CACHE_TTL:
        return cached[0]

    try:
        from nba_api.stats.endpoints import playergamelog as _pgl
        from datetime import datetime

        current_year = datetime.now().year
        current_month = datetime.now().month
        latest_year = current_year - 1 if 1 <= current_month <= 7 else current_year
        season = f"{latest_year}-{str(latest_year + 1)[2:]}"

        logs = _pgl.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        _time_mod.sleep(0.4)

        if logs.empty:
            result = False
        else:
            most_recent = str(logs.iloc[0]['GAME_DATE'])
            try:
                game_date = datetime.strptime(most_recent[:10], '%Y-%m-%d')
                result = (datetime.now() - game_date).days <= 10
            except Exception:
                result = True
    except Exception:
        result = True

    _defender_active_cache[player_id] = (result, now)
    return result


class BasketballBettingHelper:
    def __init__(self, db_name='basketball_data.db'):
        self.db_name = db_name
        self.ml_predictor = EnhancedMLPredictor()
        self._incremental_mm = IncrementalModelManager('models')
        self._precomputed = PrecomputedStore(db_name)
        
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

            # Load precomputed DVP / defender data so training samples include
            # the same matchup-context features that inference uses at prediction time.
            try:
                _pre_data  = self._precomputed.refresh(force=True)
                _dvp_map   = _pre_data.get('dvp') or {}
                _dvp_avgs  = _pre_data.get('dvp_pos_avgs') or {}
                _defs_map  = _pre_data.get('defenders') or {}
                print(f"retrain: precomputed DVP loaded — "
                      f"{len(_dvp_map)} team/pos entries, "
                      f"{len(_defs_map)} defender entries")
            except Exception as _pre_err:
                print(f"retrain: could not load precomputed data ({_pre_err}), "
                      "DVP features will be zero-filled")
                _dvp_map  = {}
                _dvp_avgs = {}
                _defs_map = {}

            historical = collector.collect_bulk(
                player_ids, seasons=seasons,
                dvp_map=_dvp_map,
                dvp_pos_avgs=_dvp_avgs,
                defenders_map=_defs_map,
            )
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

            # efficiency scalars — used as ML features
            stats['efficiency'] = {
                'avg_minutes':    float(pd.to_numeric(games_df['MIN'], errors='coerce').fillna(0).mean()),
                'recent_minutes': float(pd.to_numeric(games_df['MIN'], errors='coerce').fillna(0).head(5).mean()),
                'fg_pct':         float(pd.to_numeric(games_df['FG_PCT'], errors='coerce').fillna(0).mean()),
                'recent_fg_pct':  float(pd.to_numeric(games_df['FG_PCT'], errors='coerce').fillna(0).head(5).mean()),
                'ft_pct':         float(pd.to_numeric(games_df['FT_PCT'], errors='coerce').fillna(0).mean()),
                'usage_rate':     float(
                    (pd.to_numeric(games_df['FGA'], errors='coerce').fillna(0) +
                     0.44 * pd.to_numeric(games_df['FTA'], errors='coerce').fillna(0) +
                     pd.to_numeric(games_df['TOV'], errors='coerce').fillna(0)).mean()
                ),
            }

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

            # ---- inject efficiency + trend scalars into stat_data so
            #      prepare_features() picks them up automatically ----
            efficiency = stats.get('efficiency', {})
            stat_data = dict(stat_data)   # shallow copy — don't mutate the cached stats object
            stat_data.update({
                'avg_minutes':    efficiency.get('avg_minutes',    24.0),
                'recent_minutes': efficiency.get('recent_minutes', 24.0),
                'fg_pct':         efficiency.get('fg_pct',          0.45),
                'recent_fg_pct':  efficiency.get('recent_fg_pct',   0.45),
                'ft_pct':         efficiency.get('ft_pct',           0.75),
                'usage_rate':     efficiency.get('usage_rate',       18.0),
            })

            # trend slope for this specific prop (already computed in get_player_stats)
            _trend_key_map = {
                'points': 'pts', 'assists': 'ast', 'rebounds': 'reb',
                'steals': 'stl', 'blocks': 'blk', 'turnovers': 'tov', 'three_pointers': 'fg3m',
            }
            _tkey = _trend_key_map.get(prop_type)
            stat_data['trend_slope'] = float(
                stats.get('trends', {}).get(_tkey, {}).get('slope', 0.0)
            ) if _tkey else 0.0

            # b2b flag — derived from current team context rest_days
            _rest = int(team_context.get('rest_days', 2)) if team_context else 2
            stat_data['b2b_flag'] = int(_rest <= 1)

            # ---- extended game-log features (computed from values array) ----
            try:
                _vals_ext = list(stat_data.get('values') or [])
                if len(_vals_ext) >= 3:
                    _arr = np.array(_vals_ext[::-1], dtype=float)  # most-recent-first -> flip to chronological
                    _seas_avg_e = float(np.mean(_arr))
                    _seas_std_e = float(np.std(_arr)) if len(_arr) > 1 else 1.0
                    _last5_e  = _arr[-5:]  if len(_arr) >= 5  else _arr
                    _last10_e = _arr[-10:] if len(_arr) >= 10 else _arr
                    _last3_e  = _arr[-3:]  if len(_arr) >= 3  else _arr

                    def _slope_e(a):
                        if len(a) < 2:
                            return 0.0
                        try:
                            return float(np.polyfit(range(len(a)), a, 1)[0])
                        except Exception:
                            return 0.0

                    stat_data.setdefault('last_3_games_trend',  _slope_e(_last3_e))
                    stat_data.setdefault('last_5_games_trend',  _slope_e(_last5_e))
                    stat_data.setdefault('last_10_games_trend', _slope_e(_last10_e))
                    stat_data.setdefault('games_above_season_avg_last5', float(np.sum(_last5_e > _seas_avg_e)))
                    stat_data.setdefault('consistency_score', max(0.0, 1.0 - (_seas_std_e / max(_seas_avg_e, 0.1))))
                    stat_data.setdefault('ceiling_game_frequency', float(np.mean(_arr > _seas_avg_e * 1.5)))
                    _rec3_std_e = float(np.std(_last3_e)) if len(_last3_e) >= 2 else 0.0
                    stat_data.setdefault('recent_variance_spike', float(_rec3_std_e / max(_seas_std_e, 0.1) - 1.0))
                    stat_data.setdefault('blowout_game_pct', 0.2)
                    stat_data.setdefault('close_game_pct', 0.3)
            except Exception:
                pass

            # zero-fill remaining game-log features if not already set
            for _k, _default in [
                ('fg3_pct_recent', 0.33), ('fga_per_game', 15.0), ('fg3a_per_game', 5.0),
                ('fta_per_game', 4.0), ('oreb_per_game', 1.0), ('dreb_per_game', 3.0),
                ('plus_minus_avg', 0.0), ('fouls_per_game', 2.0), ('win_rate_last10', 0.5),
                ('points_per_shot', 0.5), ('ast_to_tov_ratio', 1.5), ('reb_rate_per_36', 0.0),
                ('scoring_efficiency_trend', 0.0), ('usage_trend', 0.0), ('minutes_volatility', 3.0),
                ('days_since_last_game', 2.0), ('games_in_last_7_days', 3.0),
            ]:
                stat_data.setdefault(_k, _default)

            # ---- DVP (Defense vs Position) + primary defender ----
            # Inject into stat_data so prepare_features() picks them up
            _pre  = {}
            _dvp  = {}
            _defs = []
            try:
                _pre  = self._precomputed.refresh()
                _pos  = str((player_context or {}).get('position', '') or '')
                _dvp_pos, _pos_group = self._position_keys(_pos)
                _dvp     = _pre['dvp'].get((int(opponent_team_id), _dvp_pos), {})
                _dvp_avg = _pre['dvp_pos_avgs'].get(_dvp_pos, {})
                stat_data['dvp_gp'] = int(_dvp.get('gp', 0))
                for _k in ('pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov'):
                    stat_data[f'dvp_{_k}_delta'] = (
                        float(_dvp.get(_k, 0.0)) - float(_dvp_avg.get(_k, 0.0))
                    )
                _defs = _pre['defenders'].get((int(opponent_team_id), _pos_group), [])
                stat_data['primary_defender_score01'] = float(
                    (_defs[0] if _defs else {}).get('score01', 0.0) or 0.0
                )
            except Exception as _dvp_err:
                print(f"DVP lookup failed (non-fatal): {_dvp_err}")
                # defaults: zero deltas = league-average defence, no elite defender
                for _k in ('pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov'):
                    stat_data.setdefault(f'dvp_{_k}_delta', 0.0)
                stat_data.setdefault('dvp_gp', 0)
                stat_data.setdefault('primary_defender_score01', 0.0)

            # ---- NEW FEATURES: referee, rolling DVP, foul rates, injury trajectory, calendar ----
            try:
                _opp_id_int = int(opponent_team_id) if opponent_team_id else None
                if _opp_id_int:
                    _dvp_roll = _pre.get('dvp_rolling', {})
                    for _w, _key_suffix in ((5, 'last5'), (10, 'last10')):
                        _roll = _dvp_roll.get((_opp_id_int, _w), {})
                        for _stat in ('pts', 'reb', 'ast', 'fg3m'):
                            _roll_val = float(_roll.get(_stat, _dvp.get(_stat, 0.0)))
                            _season_val = float(_dvp.get(_stat, 0.0))
                            stat_data[f'dvp_{_stat}_delta_{_key_suffix}'] = _roll_val - _season_val

                # Opponent foul rates
                _foul_data = _pre.get('team_foul', {}).get(_opp_id_int, {}) if _opp_id_int else {}
                stat_data['opp_foul_rate_per48'] = float(_foul_data.get('foul_rate_season', 20.0))
                stat_data['opp_foul_rate_last5'] = float(_foul_data.get('foul_rate_last5', 20.0))

                # Implied game total: avg possessions × scoring rate proxy
                # Use team_context/opponent_context which carry current-season pace
                _t_pace = float((team_context or {}).get('pace', 100.0))
                _o_pace = float((opponent_context or {}).get('pace', 100.0))
                _avg_pace = (_t_pace + _o_pace) / 2.0
                # Each team uses ~avg_pace possessions; ~1.1 pts/possession baseline
                stat_data['implied_game_total'] = float(_avg_pace * 2.0 * 1.1)

                # Referee features
                _ref_features = self._get_referee_features(game_id=None, precomputed=_pre)
                stat_data.update(_ref_features)

            except Exception as _new_feat_err:
                for _k in ('dvp_pts_delta_last5', 'dvp_pts_delta_last10', 'dvp_reb_delta_last5',
                           'dvp_ast_delta_last5', 'dvp_fg3m_delta_last5'):
                    stat_data.setdefault(_k, 0.0)
                stat_data.setdefault('opp_foul_rate_per48', 20.0)
                stat_data.setdefault('opp_foul_rate_last5', 20.0)
                stat_data.setdefault('implied_game_total', 220.0)
                stat_data.setdefault('ref_foul_rate', 0.0)
                stat_data.setdefault('ref_home_bias', 0.5)
                stat_data.setdefault('ref_pace_tendency', 0.0)

            # ---- team style + opponent baseline (from team_stats) ----
            try:
                _opp_stats      = _pre.get('team_stats', {}).get(int(opponent_team_id), {}) if opponent_team_id else {}
                _team_stats_own = _pre.get('team_stats', {}).get(int(team_id), {}) if team_id else {}

                stat_data['team_pts_fb']             = float(_team_stats_own.get('pts_fb', 12.0))
                stat_data['team_pts_off_tov']        = float(_team_stats_own.get('pts_off_tov', 16.0))
                stat_data['opp_pts_fb_allowed']      = float(_opp_stats.get('opp_pts_fb', 12.0))
                stat_data['opp_pts_off_tov_allowed'] = float(_opp_stats.get('opp_pts_off_tov', 16.0))
                stat_data['opp_pts_paint']           = float(_opp_stats.get('opp_pts_paint', 44.0))
                stat_data['opp_fga']                 = float(_opp_stats.get('opp_fga', 86.0))
                stat_data['opp_fg_pct']              = float(_opp_stats.get('opp_fg_pct', 0.47))
                stat_data['opp_fg3a']                = float(_opp_stats.get('opp_fg3a', 35.0))
                stat_data['opp_fg3_pct']             = float(_opp_stats.get('opp_fg3_pct', 0.36))
                stat_data['opp_tov']                 = float(_opp_stats.get('opp_tov', 14.0))
                stat_data['opp_stl']                 = float(_opp_stats.get('opp_stl', 7.0))
                stat_data['opp_blk']                 = float(_opp_stats.get('opp_blk', 5.0))
                stat_data['opp_off_rating']          = float(_opp_stats.get('opp_off_rating',
                                                           (opponent_context or {}).get('offensive_rating', 110.0)))
                stat_data['opp_def_rating_last5']      = float(_opp_stats.get('opp_def_rating_last5', 110.0))
                stat_data['opp_blocks_per_game_last5'] = float(_opp_stats.get('opp_blk_last5', 5.0))
                stat_data['opp_steals_per_game_last5'] = float(_opp_stats.get('opp_stl_last5', 7.0))
                # League averages (stored per-row; use _opp_stats as source)
                _lg = _opp_stats
                stat_data['lg_pts_fb']          = float(_lg.get('lg_pts_fb', 12.0))
                stat_data['lg_opp_pts_fb']      = float(_lg.get('lg_pts_fb', 12.0))
                stat_data['lg_pts_off_tov']     = float(_lg.get('lg_pts_off_tov', 16.0))
                stat_data['lg_opp_pts_off_tov'] = float(_lg.get('lg_pts_off_tov', 16.0))
                stat_data['lg_fga']             = float(_lg.get('lg_fga', 86.0))
                stat_data['lg_fg_pct']          = float(_lg.get('lg_fg_pct', 0.47))
                stat_data['lg_fg3a']            = float(_lg.get('lg_fg3a', 35.0))
                stat_data['lg_tov']             = float(_lg.get('lg_tov', 14.0))
                stat_data['lg_stl']             = float(_lg.get('lg_stl', 7.0))
            except Exception as _ts_err:
                # safe defaults for all team-stats features
                for _k, _d in [
                    ('team_pts_fb', 12.0), ('team_pts_off_tov', 16.0),
                    ('opp_pts_fb_allowed', 12.0), ('opp_pts_off_tov_allowed', 16.0),
                    ('opp_pts_paint', 44.0), ('opp_fga', 86.0), ('opp_fg_pct', 0.47),
                    ('opp_fg3a', 35.0), ('opp_fg3_pct', 0.36), ('opp_tov', 14.0),
                    ('opp_stl', 7.0), ('opp_blk', 5.0), ('opp_off_rating', 110.0),
                    ('opp_def_rating_last5', 110.0), ('opp_blocks_per_game_last5', 5.0),
                    ('opp_steals_per_game_last5', 7.0), ('lg_pts_fb', 12.0), ('lg_opp_pts_fb', 12.0),
                    ('lg_pts_off_tov', 16.0), ('lg_opp_pts_off_tov', 16.0), ('lg_fga', 86.0),
                    ('lg_fg_pct', 0.47), ('lg_fg3a', 35.0), ('lg_tov', 14.0), ('lg_stl', 7.0),
                ]:
                    stat_data.setdefault(_k, _d)

            # ---- player-specific precomputed features ----
            try:
                _pid_int = int(player_id)
                _opp_int = int(opponent_team_id) if opponent_team_id else None
                _pre_data = _pre  # already populated above

                # Advanced stats
                _adv = _pre_data.get('player_advanced', {}).get(_pid_int, {})
                for _k, _dk in [
                    ('usg_pct_official', 0.18), ('ts_pct_official', 0.55),
                    ('efg_pct_official', 0.50), ('ast_pct_official', 0.15),
                    ('oreb_pct_official', 0.05), ('dreb_pct_official', 0.15),
                    ('reb_pct_official', 0.10), ('pie', 0.10),
                    ('player_off_rating', 110.0), ('player_def_rating', 110.0),
                    ('player_pace', 100.0), ('net_rating_player', 0.0),
                    ('player_age', 26.0), ('player_height_inches', 78.0),
                    ('player_weight', 220.0), ('years_experience', 5.0),
                ]:
                    stat_data[_k] = float(_adv.get(_k, _dk))

                # Clutch
                _clutch = _pre_data.get('player_clutch', {}).get(_pid_int, {})
                for _k, _dk in [
                    ('clutch_pts_per_game', 0.0), ('clutch_fg_pct', 0.45),
                    ('clutch_fg3_pct', 0.33), ('clutch_fta_per_game', 0.0),
                    ('clutch_plus_minus', 0.0), ('clutch_min_per_game', 0.0),
                    ('clutch_games', 0),
                ]:
                    stat_data[_k] = float(_clutch.get(_k, _dk))

                # Hustle
                _hustle = _pre_data.get('player_hustle', {}).get(_pid_int, {})
                for _k, _dk in [
                    ('contested_shots_per_game', 3.0), ('deflections_per_game', 1.0),
                    ('charges_drawn_per_game', 0.1), ('screen_assists_per_game', 0.5),
                ]:
                    stat_data[_k] = float(_hustle.get(_k, _dk))

                # Shot profile
                _sp = _pre_data.get('player_shot_profile', {}).get(_pid_int, {})
                for _k, _dk in [
                    ('open_shot_fg_pct', 0.50), ('open_shot_frequency', 0.30),
                    ('tight_shot_fg_pct', 0.38), ('tight_shot_frequency', 0.15),
                    ('catch_shoot_fg_pct', 0.40), ('catch_shoot_frequency', 0.25),
                    ('pullup_fg_pct', 0.40), ('pullup_frequency', 0.20),
                ]:
                    stat_data[_k] = float(_sp.get(_k, _dk))

                # Play types
                _pt = _pre_data.get('player_play_types', {}).get(_pid_int, {})
                for _k, _dk in [
                    ('iso_poss_pct', 0.0), ('iso_ppp', 0.9),
                    ('pnr_bh_poss_pct', 0.0), ('pnr_bh_ppp', 0.9),
                    ('pnr_roll_poss_pct', 0.0), ('pnr_roll_ppp', 0.9),
                    ('spotup_poss_pct', 0.0), ('spotup_ppp', 1.0),
                    ('transition_poss_pct', 0.0), ('transition_ppp', 1.1),
                    ('postup_poss_pct', 0.0), ('cut_poss_pct', 0.0),
                ]:
                    stat_data[_k] = float(_pt.get(_k, _dk))

                # On/off
                _oo = _pre_data.get('player_on_off', {}).get(_pid_int, {})
                stat_data['on_court_net_rating']  = float(_oo.get('on_court_net_rating', 0.0))
                stat_data['off_court_net_rating'] = float(_oo.get('off_court_net_rating', 0.0))
                stat_data['on_off_differential']  = float(_oo.get('on_off_differential', 0.0))

                # Shot zones (player)
                _sz = _pre_data.get('player_shot_zones', {}).get(_pid_int, {})
                for _k, _dk in [
                    ('rim_fga_pct', 0.25), ('rim_fg_pct', 0.62),
                    ('paint_fga_pct', 0.30), ('paint_fg_pct', 0.55),
                    ('midrange_fga_pct', 0.20), ('midrange_fg_pct', 0.42),
                    ('corner3_fga_pct', 0.10), ('corner3_fg_pct', 0.38),
                    ('above_break3_fga_pct', 0.25), ('above_break3_fg_pct', 0.35),
                ]:
                    stat_data[_k] = float(_sz.get(_k, _dk))

                # Quarter splits
                _qs = _pre_data.get('player_quarter_splits', {}).get(_pid_int, {})
                for _k, _dk in [
                    ('q1_avg', 0.0), ('q2_avg', 0.0), ('q3_avg', 0.0),
                    ('q4_avg', 0.0), ('q4_min_per_game', 0.0),
                ]:
                    stat_data[_k] = float(_qs.get(_k, _dk))

                # Opponent shot zone defense
                _osz = _pre_data.get('team_opp_shot_zones', {}).get(_opp_int, {}) if _opp_int else {}
                for _k, _dk in [
                    ('opp_rim_fg_pct_allowed', 0.62), ('opp_paint_fg_pct_allowed', 0.55),
                    ('opp_midrange_fg_pct_allowed', 0.42), ('opp_corner3_fg_pct_allowed', 0.38),
                    ('opp_above_break3_fg_pct_allowed', 0.35),
                ]:
                    _zone_inner_key = _k.replace('opp_', '', 1)   # 'opp_rim_fg_pct_allowed' → 'rim_fg_pct_allowed'
                    stat_data[_k] = float(_osz.get(_zone_inner_key, _dk))

                # Synergy team defense
                _sd = _pre_data.get('team_synergy_defense', {}).get(_opp_int, {}) if _opp_int else {}
                for _k, _dk in [
                    ('opp_pnr_ppp_allowed', 0.9), ('opp_iso_ppp_allowed', 0.9),
                    ('opp_spotup_ppp_allowed', 1.0), ('opp_transition_ppp_allowed', 1.1),
                    ('opp_postup_ppp_allowed', 0.9),
                ]:
                    stat_data[_k] = float(_sd.get(_k.replace('opp_', '', 1), _dk))  # 'opp_pnr_ppp_allowed' → 'pnr_ppp_allowed'

            except Exception as _player_feat_err:
                # Safe defaults — never crash prediction on new feature failures
                _new_player_defaults = {
                    'usg_pct_official': 0.18, 'ts_pct_official': 0.55, 'efg_pct_official': 0.50,
                    'ast_pct_official': 0.15, 'oreb_pct_official': 0.05, 'dreb_pct_official': 0.15,
                    'reb_pct_official': 0.10, 'pie': 0.10, 'player_off_rating': 110.0,
                    'player_def_rating': 110.0, 'player_pace': 100.0, 'net_rating_player': 0.0,
                    'player_age': 26.0, 'player_height_inches': 78.0, 'player_weight': 220.0,
                    'years_experience': 5.0, 'clutch_pts_per_game': 0.0, 'clutch_fg_pct': 0.45,
                    'clutch_fg3_pct': 0.33, 'clutch_fta_per_game': 0.0, 'clutch_plus_minus': 0.0,
                    'clutch_min_per_game': 0.0, 'clutch_games': 0, 'contested_shots_per_game': 3.0,
                    'deflections_per_game': 1.0, 'charges_drawn_per_game': 0.1,
                    'screen_assists_per_game': 0.5, 'open_shot_fg_pct': 0.50,
                    'open_shot_frequency': 0.30, 'tight_shot_fg_pct': 0.38,
                    'tight_shot_frequency': 0.15, 'catch_shoot_fg_pct': 0.40,
                    'catch_shoot_frequency': 0.25, 'pullup_fg_pct': 0.40, 'pullup_frequency': 0.20,
                    'iso_poss_pct': 0.0, 'iso_ppp': 0.9, 'pnr_bh_poss_pct': 0.0, 'pnr_bh_ppp': 0.9,
                    'pnr_roll_poss_pct': 0.0, 'pnr_roll_ppp': 0.9, 'spotup_poss_pct': 0.0,
                    'spotup_ppp': 1.0, 'transition_poss_pct': 0.0, 'transition_ppp': 1.1,
                    'postup_poss_pct': 0.0, 'cut_poss_pct': 0.0, 'on_court_net_rating': 0.0,
                    'off_court_net_rating': 0.0, 'on_off_differential': 0.0,
                    'rim_fga_pct': 0.25, 'rim_fg_pct': 0.62, 'paint_fga_pct': 0.30,
                    'paint_fg_pct': 0.55, 'midrange_fga_pct': 0.20, 'midrange_fg_pct': 0.42,
                    'corner3_fga_pct': 0.10, 'corner3_fg_pct': 0.38, 'above_break3_fga_pct': 0.25,
                    'above_break3_fg_pct': 0.35, 'q1_avg': 0.0, 'q2_avg': 0.0, 'q3_avg': 0.0,
                    'q4_avg': 0.0, 'q4_min_per_game': 0.0, 'opp_rim_fg_pct_allowed': 0.62,
                    'opp_paint_fg_pct_allowed': 0.55, 'opp_midrange_fg_pct_allowed': 0.42,
                    'opp_corner3_fg_pct_allowed': 0.38, 'opp_above_break3_fg_pct_allowed': 0.35,
                    'opp_pnr_ppp_allowed': 0.9, 'opp_iso_ppp_allowed': 0.9,
                    'opp_spotup_ppp_allowed': 1.0, 'opp_transition_ppp_allowed': 1.1,
                    'opp_postup_ppp_allowed': 0.9,
                }
                for _k, _d in _new_player_defaults.items():
                    stat_data.setdefault(_k, _d)

            # ---- GROUP 1: Player Tracking Stats ----
            try:
                _pid_int2 = int(player_id)
                _tracking = _pre.get('player_tracking', {}).get(_pid_int2, {})
                for _k, _dk in [
                    ('tracking_avg_speed', 4.5), ('tracking_avg_speed_off', 4.8),
                    ('tracking_avg_speed_def', 4.2), ('tracking_dist_miles', 2.5),
                    ('tracking_dist_miles_off', 1.3), ('tracking_dist_miles_def', 1.2),
                    ('tracking_touches_pg', 50.0), ('tracking_time_of_poss_pg', 2.5),
                    ('tracking_avg_drib_per_touch', 1.5), ('tracking_passes_made_pg', 30.0),
                    ('tracking_potential_ast_pg', 5.0), ('tracking_secondary_ast_pg', 1.0),
                ]:
                    stat_data[_k] = float(_tracking.get(_k, _dk))
            except Exception:
                for _k, _dk in [
                    ('tracking_avg_speed', 4.5), ('tracking_avg_speed_off', 4.8),
                    ('tracking_avg_speed_def', 4.2), ('tracking_dist_miles', 2.5),
                    ('tracking_dist_miles_off', 1.3), ('tracking_dist_miles_def', 1.2),
                    ('tracking_touches_pg', 50.0), ('tracking_time_of_poss_pg', 2.5),
                    ('tracking_avg_drib_per_touch', 1.5), ('tracking_passes_made_pg', 30.0),
                    ('tracking_potential_ast_pg', 5.0), ('tracking_secondary_ast_pg', 1.0),
                ]:
                    stat_data.setdefault(_k, _dk)

            # ---- GROUP 2: Team Standings / Game Importance ----
            try:
                _tid_int = int(team_id) if team_id else None
                _opp_tid_int = int(opponent_team_id) if opponent_team_id else None
                _standings = _pre.get('team_standings', {})
                _my_std = _standings.get(_tid_int, {}) if _tid_int else {}
                _opp_std = _standings.get(_opp_tid_int, {}) if _opp_tid_int else {}

                stat_data['team_win_pct']        = float(_my_std.get('win_pct', 0.5))
                stat_data['team_conf_rank']       = float(_my_std.get('conf_rank', 8))
                stat_data['team_games_back']      = float(_my_std.get('games_back', 5.0))
                stat_data['team_current_streak']  = float(_my_std.get('current_streak', 0))
                stat_data['team_l10_wins']        = float(_my_std.get('l10_wins', 5))
                stat_data['team_home_win_pct']    = float(_my_std.get('home_win_pct', 0.5))
                stat_data['opp_win_pct_standings']     = float(_opp_std.get('win_pct', 0.5))
                stat_data['opp_conf_rank']             = float(_opp_std.get('conf_rank', 8))
                stat_data['opp_games_back']            = float(_opp_std.get('games_back', 5.0))
                stat_data['opp_current_streak_standings'] = float(_opp_std.get('current_streak', 0))
                stat_data['opp_l10_wins']              = float(_opp_std.get('l10_wins', 5))
                stat_data['opp_road_win_pct']          = float(_opp_std.get('road_win_pct', 0.5))
                _wpct_diff = stat_data['team_win_pct'] - stat_data['opp_win_pct_standings']
                stat_data['win_pct_diff'] = float(_wpct_diff)
                _my_gb   = abs(float(_my_std.get('games_back', 10.0)))
                _opp_gb  = abs(float(_opp_std.get('games_back', 10.0)))
                stat_data['is_playoff_race_game'] = 1.0 if (_my_gb < 5.0 or _opp_gb < 5.0) else 0.0
            except Exception:
                for _k, _dk in [
                    ('team_win_pct', 0.5), ('team_conf_rank', 8.0), ('team_games_back', 5.0),
                    ('team_current_streak', 0.0), ('team_l10_wins', 5.0), ('team_home_win_pct', 0.5),
                    ('opp_win_pct_standings', 0.5), ('opp_conf_rank', 8.0), ('opp_games_back', 5.0),
                    ('opp_current_streak_standings', 0.0), ('opp_l10_wins', 5.0),
                    ('opp_road_win_pct', 0.5), ('win_pct_diff', 0.0), ('is_playoff_race_game', 0.0),
                ]:
                    stat_data.setdefault(_k, _dk)

            # ---- GROUP 3: Scoring Breakdown by Method ----
            try:
                _pid_int3 = int(player_id)
                _sb = _pre.get('player_scoring_breakdown', {}).get(_pid_int3, {})
                for _k, _dk in [
                    ('pct_pts_3pt', 0.25), ('pct_pts_paint', 0.30), ('pct_pts_ft', 0.15),
                    ('pct_pts_midrange', 0.20), ('pct_uast_fgm', 0.40), ('pct_ast_fgm', 0.60),
                ]:
                    stat_data[_k] = float(_sb.get(_k, _dk))
            except Exception:
                for _k, _dk in [
                    ('pct_pts_3pt', 0.25), ('pct_pts_paint', 0.30), ('pct_pts_ft', 0.15),
                    ('pct_pts_midrange', 0.20), ('pct_uast_fgm', 0.40), ('pct_ast_fgm', 0.60),
                ]:
                    stat_data.setdefault(_k, _dk)

            # ---- GROUP 4: Win/Loss Performance Splits (from game log values + WL) ----
            try:
                _vals_wl = list(stat_data.get('values') or [])
                # Fetch WL alongside game log to compute win/loss splits
                # We pull it fresh from get_player_stats which already has all_games_df
                # Instead compute from scratch via a fresh game log fetch for current season
                _season_avg_val = float(stat_data.get('avg', 0.0) or 0.0)
                try:
                    _gl_wl_df = playergamelog.PlayerGameLog(
                        player_id=player_id, season=self.current_season
                    ).get_data_frames()[0]
                    time.sleep(0.6)
                    _PROP_COL = {
                        'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB',
                        'steals': 'STL', 'blocks': 'BLK', 'turnovers': 'TOV',
                        'three_pointers': 'FG3M',
                    }.get(prop_type, 'PTS')
                    if _PROP_COL in _gl_wl_df.columns and 'WL' in _gl_wl_df.columns:
                        _win_mask  = _gl_wl_df['WL'] == 'W'
                        _loss_mask = _gl_wl_df['WL'] == 'L'
                        _win_vals  = pd.to_numeric(_gl_wl_df.loc[_win_mask, _PROP_COL], errors='coerce').dropna().tolist()
                        _loss_vals = pd.to_numeric(_gl_wl_df.loc[_loss_mask, _PROP_COL], errors='coerce').dropna().tolist()
                        _stat_in_wins   = float(np.mean(_win_vals))  if _win_vals  else _season_avg_val
                        _stat_in_losses = float(np.mean(_loss_vals)) if _loss_vals else _season_avg_val
                        _wl_split = _stat_in_wins - _stat_in_losses
                        _over_rate_wins = float(np.mean([1.0 if v > line else 0.0 for v in _win_vals])) if _win_vals else 0.5
                    else:
                        _stat_in_wins = _stat_in_losses = _season_avg_val
                        _wl_split = 0.0
                        _over_rate_wins = 0.5
                except Exception:
                    _stat_in_wins = _stat_in_losses = _season_avg_val
                    _wl_split = 0.0
                    _over_rate_wins = 0.5
                stat_data['stat_in_wins']            = _stat_in_wins
                stat_data['stat_in_losses']          = _stat_in_losses
                stat_data['win_loss_performance_split'] = _wl_split
                stat_data['over_rate_in_wins']       = _over_rate_wins
            except Exception:
                stat_data.setdefault('stat_in_wins', float(stat_data.get('avg', 0.0) or 0.0))
                stat_data.setdefault('stat_in_losses', float(stat_data.get('avg', 0.0) or 0.0))
                stat_data.setdefault('win_loss_performance_split', 0.0)
                stat_data.setdefault('over_rate_in_wins', 0.5)

            # ---- GROUP 5: Opponent Rest & Schedule Context ----
            try:
                from datetime import date as _date_cls
                _today_str = str(_date_cls.today())
                _opp_cache_key = (int(opponent_team_id), _today_str)
                _opp_rest = BasketballBettingHelper._opp_gamelog_cache.get(_opp_cache_key)
                if _opp_rest is None:
                    try:
                        _opp_gl = TeamGameLog(
                            team_id=int(opponent_team_id),
                            season=self.current_season,
                        ).get_data_frames()[0]
                        time.sleep(0.6)
                        if not _opp_gl.empty:
                            _opp_gl['GAME_DATE'] = pd.to_datetime(_opp_gl['GAME_DATE'])
                            _opp_gl = _opp_gl.sort_values('GAME_DATE', ascending=False)
                            _opp_last_date = _opp_gl.iloc[0]['GAME_DATE'].date()
                            _opp_days_rest_val = (_date_cls.today() - _opp_last_date).days - 1
                        else:
                            _opp_days_rest_val = 2
                    except Exception:
                        _opp_days_rest_val = 2
                    BasketballBettingHelper._opp_gamelog_cache[_opp_cache_key] = _opp_days_rest_val
                    _opp_rest = _opp_days_rest_val

                _player_rest = int(_rest)  # already computed above as _rest
                stat_data['opp_days_rest']   = float(max(0, _opp_rest))
                stat_data['opp_b2b']         = 1.0 if _opp_rest <= 1 else 0.0
                stat_data['rest_advantage']  = float(_player_rest - _opp_rest)
            except Exception:
                stat_data.setdefault('opp_days_rest', 2.0)
                stat_data.setdefault('opp_b2b', 0.0)
                stat_data.setdefault('rest_advantage', 0.0)

            # ---- GROUP 6: Additional Derived Features ----
            try:
                _ast_pct_off  = float(stat_data.get('ast_pct_official', 0.15))
                _usg_pct_off  = float(stat_data.get('usg_pct_official', 0.18))
                stat_data['ast_pct_to_usg_ratio'] = _ast_pct_off / max(_usg_pct_off, 0.01)

                _opp_def_r5   = float(stat_data.get('opp_def_rating_last5', 110.0))
                _opp_fg_pct_v = float(stat_data.get('opp_fg_pct', 0.47))
                stat_data['defensive_burden'] = _opp_def_r5 * (1.0 - _opp_fg_pct_v)

                _pct_3pt_v    = float(stat_data.get('pct_pts_3pt', 0.25))
                _ab3_pct_allow = float(stat_data.get('opp_above_break3_fg_pct_allowed', 0.35))
                stat_data['shot_profile_fit'] = _pct_3pt_v * _ab3_pct_allow

                _recent_avg_v = float(stat_data.get('last5_avg', stat_data.get('avg', 0.0) or 0.0))
                _igt = float(stat_data.get('implied_game_total', 220.0))
                stat_data['pace_adjusted_projection'] = _recent_avg_v * (_igt / 220.0)

                _l3t = float(stat_data.get('last_3_games_trend', 0.0))
                _l5t = float(stat_data.get('last_5_games_trend', 0.0))
                stat_data['form_momentum'] = (_l3t * 3.0 + _l5t * 2.0) / 5.0
            except Exception:
                stat_data.setdefault('ast_pct_to_usg_ratio', 0.83)
                stat_data.setdefault('defensive_burden', 58.0)
                stat_data.setdefault('shot_profile_fit', 0.09)
                stat_data.setdefault('pace_adjusted_projection', 0.0)
                stat_data.setdefault('form_momentum', 0.0)

            # ---- GROUP B: Player vs Opponent Historical Splits ----
            try:
                _pid_b = int(player_id)
                _opp_b = int(opponent_team_id) if opponent_team_id else None
                _pvo = _pre.get('player_vs_opponent', {}).get((_pid_b, _opp_b), {}) if _opp_b else {}
                stat_data['historical_avg_vs_opp']    = float(_pvo.get('historical_avg_vs_opp', 0.0))
                stat_data['historical_fg_pct_vs_opp'] = float(_pvo.get('historical_fg_pct_vs_opp', 0.45))
                stat_data['historical_ts_pct_vs_opp'] = float(_pvo.get('historical_ts_pct_vs_opp', 0.55))
                stat_data['historical_games_vs_opp']  = float(_pvo.get('historical_games_vs_opp', 0))
                stat_data['historical_min_vs_opp']    = float(_pvo.get('historical_min_vs_opp', 30.0))
            except Exception:
                stat_data.setdefault('historical_avg_vs_opp', 0.0)
                stat_data.setdefault('historical_fg_pct_vs_opp', 0.45)
                stat_data.setdefault('historical_ts_pct_vs_opp', 0.55)
                stat_data.setdefault('historical_games_vs_opp', 0)
                stat_data.setdefault('historical_min_vs_opp', 30.0)

            # ---- GROUP C: Team Rest Splits (opponent context when tired vs rested) ----
            try:
                _opp_c = int(opponent_team_id) if opponent_team_id else None
                _trs = _pre.get('team_rest_splits', {}).get(_opp_c, {}) if _opp_c else {}
                stat_data['opp_b2b_def_rating']        = float(_trs.get('opp_b2b_def_rating', 112.0))
                stat_data['opp_b2b_pace']              = float(_trs.get('opp_b2b_pace', 100.0))
                stat_data['opp_b2b_pts_allowed']       = float(_trs.get('opp_b2b_pts_allowed', 115.0))
                stat_data['opp_rested_def_rating']     = float(_trs.get('opp_rested_def_rating', 110.0))
                stat_data['opp_rested_pace']           = float(_trs.get('opp_rested_pace', 100.0))
                stat_data['opp_rest_def_rating_delta'] = float(_trs.get('opp_rest_def_rating_delta', 2.0))
            except Exception:
                stat_data.setdefault('opp_b2b_def_rating', 112.0)
                stat_data.setdefault('opp_b2b_pace', 100.0)
                stat_data.setdefault('opp_b2b_pts_allowed', 115.0)
                stat_data.setdefault('opp_rested_def_rating', 110.0)
                stat_data.setdefault('opp_rested_pace', 100.0)
                stat_data.setdefault('opp_rest_def_rating_delta', 2.0)

            # ---- GROUP D: Player Year-Over-Year Trajectory ----
            try:
                _pid_d = int(player_id)
                _yoy = _pre.get('player_yoy_stats', {}).get(_pid_d, {})
                stat_data['yoy_pts_change']    = float(_yoy.get('yoy_pts_change', 0.0))
                stat_data['yoy_ts_change']     = float(_yoy.get('yoy_ts_change', 0.0))
                stat_data['yoy_usage_change']  = float(_yoy.get('yoy_usage_change', 0.0))
                stat_data['seasons_in_league'] = float(_yoy.get('seasons_in_league', 5))
            except Exception:
                stat_data.setdefault('yoy_pts_change', 0.0)
                stat_data.setdefault('yoy_ts_change', 0.0)
                stat_data.setdefault('yoy_usage_change', 0.0)
                stat_data.setdefault('seasons_in_league', 5)

            # Injury trajectory (derived from game log stats, always computable)
            try:
                _game_log = stat_data.get('game_log', []) or []
                _games_since_return, _missed_before = _compute_injury_trajectory(_game_log)
                stat_data['games_since_return'] = float(_games_since_return)
                stat_data['missed_games_before_return'] = float(_missed_before)
            except Exception:
                stat_data.setdefault('games_since_return', 0.0)
                stat_data.setdefault('missed_games_before_return', 0.0)

            # Calendar position (always computable from date)
            try:
                _cal = _compute_calendar_features()
                stat_data.update(_cal)
            except Exception:
                stat_data.setdefault('days_into_season', 90.0)
                stat_data.setdefault('season_phase_numeric', 0.5)
                stat_data.setdefault('games_remaining_approx', 40.0)

            # Defender health (check if primary defender appeared in recent games)
            try:
                if _defs and _defs[0].get('player_id'):
                    _def_pid = int(_defs[0]['player_id'])
                    _def_active = _check_defender_active(_def_pid)
                    stat_data['primary_defender_active'] = 1.0 if _def_active else 0.0
                else:
                    stat_data.setdefault('primary_defender_active', 1.0)
                stat_data.setdefault('opp_lineup_changes_last5', 0.0)
            except Exception:
                stat_data.setdefault('primary_defender_active', 1.0)
                stat_data.setdefault('opp_lineup_changes_last5', 0.0)

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
    
    # ------------------------------------------------------------------
    # Incremental model integration
    # ------------------------------------------------------------------

    _DVP_POS_MAP = {
        # raw POSITION string from CommonPlayerInfo → DVP position key + G/F/C group
        # used to look up (opponent_team_id, dvp_pos) in PrecomputedStore
    }

    # class-level cache: (date_str) -> features dict — refreshed once per calendar day
    _ref_features_cache: dict = {}
    # opponent game log cache: (team_id, date_str) -> last_game_date or None
    _opp_gamelog_cache: dict = {}

    @staticmethod
    def _get_referee_features(game_id, precomputed):
        """
        Look up referee stats for today's game officials via ScoreboardV2.
        Result is cached once per calendar day (officials don't change intra-day).
        Falls back to league-average defaults if officials not known yet.
        """
        from datetime import date
        today_str = str(date.today())
        cached = BasketballBettingHelper._ref_features_cache.get(today_str)
        if cached is not None:
            return cached

        result = {'ref_foul_rate': 0.0, 'ref_home_bias': 0.5, 'ref_pace_tendency': 0.0}
        try:
            from nba_api.stats.endpoints import scoreboardv2
            import time as _time
            sb = scoreboardv2.ScoreboardV2()
            _time.sleep(0.5)
            dfs = sb.get_data_frames()
            officials_df = None
            for df in dfs:
                cols = [c.lower() for c in df.columns]
                if any('official' in c or 'first_name' in c for c in cols):
                    officials_df = df
                    break

            if officials_df is None or officials_df.empty:
                raise ValueError("no officials data")

            refs_store = precomputed.get('refs', {})
            foul_rates, home_biases, paces = [], [], []

            for _, row in officials_df.iterrows():
                name_parts = []
                for col in officials_df.columns:
                    if 'name' in col.lower() or 'first' in col.lower() or 'last' in col.lower():
                        name_parts.append(str(row[col]).strip())
                ref_name = ' '.join(p for p in name_parts if p and p != 'nan').lower()

                ref_data = refs_store.get(ref_name, {})
                if ref_data:
                    foul_rates.append(float(ref_data.get('foul_rate', 0.0)))
                    home_biases.append(float(ref_data.get('home_win_pct', 0.5)))
                    paces.append(float(ref_data.get('pace', 0.0)))

            if foul_rates:
                result = {
                    'ref_foul_rate':     float(sum(foul_rates) / len(foul_rates)),
                    'ref_home_bias':     float(sum(home_biases) / len(home_biases)),
                    'ref_pace_tendency': float(sum(paces) / len(paces)),
                }
        except Exception:
            pass

        BasketballBettingHelper._ref_features_cache[today_str] = result
        return result

    @staticmethod
    def _position_keys(raw_pos: str):
        """Map CommonPlayerInfo POSITION string to (dvp_pos, pos_group)."""
        p = raw_pos.upper()
        if 'C' in p and 'G' not in p:
            return 'C', 'C'
        if 'G' in p and 'F' not in p and 'C' not in p:
            return 'SG', 'G'
        if 'F' in p and 'C' not in p and 'G' not in p:
            return 'SF', 'F'
        if 'C' in p:
            return 'C', 'C'
        if 'G' in p:
            return 'SG', 'G'
        return 'SF', 'F'   # default forward

    def _build_incremental_features(
        self, prop_type, player_stats, player_context,
        team_context, opponent_context, opponent_team_id,
        is_home, stats, efficiency
    ):
        """
        Build the full ~262-key feature dict expected by IncrementalModelManager /
        ml_features.NUMERIC_FEATURE_KEYS.  Keys that can't be populated from
        currently-available data default to 0.0 — consistent with how the
        incremental model was trained (same zeros were used during training in
        update_incremental_models.py).
        """
        values    = player_stats.get('values', []) or []
        last5     = values[:5] if len(values) >= 5 else values
        recent_avg = float(player_stats.get('last5_avg', 0))
        season_avg = float(player_stats.get('avg',       0))
        stddev     = float(np.std(values)) if values else 0.0
        max_recent = float(max(last5))  if last5 else 0.0
        min_recent = float(min(last5))  if last5 else 0.0

        # ---- team / opponent context ----
        tc = team_context or {}
        oc = opponent_context or {}
        team_pace       = float(tc.get('pace', 100.0))
        opp_pace        = float(oc.get('pace', 100.0))
        team_off_rating = float(tc.get('offensive_rating', 110.0))
        team_def_rating = float(tc.get('defensive_rating', 110.0))
        opp_off_rating  = float(oc.get('offensive_rating', 110.0))
        opp_def_rating  = float(oc.get('defensive_rating', 110.0))
        rest_days       = int(tc.get('rest_days', 2))
        opp_injury_imp  = float(oc.get('injury_impact', 0.0))
        opp_key_out     = int((oc.get('injuries') or {}).get('key_players_out', 0))
        team_key_out    = int((tc.get('injuries') or {}).get('key_players_out', 0))

        # ---- player context ----
        pc = player_context or {}
        mh = pc.get('matchup_history') or {}
        vs_team_avg  = float(mh.get('avg_points',  recent_avg))
        matchup_games = int(mh.get('games_played', 0))
        position = str(pc.get('position', '') or '')
        dvp_pos, pos_group = self._position_keys(position)

        # ---- DVP (Defense vs Position) deltas ----
        pre = self._precomputed.refresh()
        dvp_map     = pre.get('dvp', {})
        dvp_avgs    = pre.get('dvp_pos_avgs', {})
        defenders_m = pre.get('defenders', {})

        dvp = dvp_map.get((int(opponent_team_id), dvp_pos), {})
        avg = dvp_avgs.get(dvp_pos, {})
        dvp_gp = int(dvp.get('gp', 0))
        dvp_deltas = {
            f'dvp_{k}_delta': float(dvp.get(k, 0.0)) - float(avg.get(k, 0.0))
            for k in ('pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov')
        }

        # ---- primary defender on opposing team ----
        defenders = defenders_m.get((int(opponent_team_id), pos_group), [])
        primary_def_score = float((defenders[0] if defenders else {}).get('score01', 0.0) or 0.0)

        # ---- efficiency from already-computed stats dict ----
        eff = efficiency or {}
        fg_pct_recent  = float(eff.get('recent_fg_pct', eff.get('fg_pct',  0.0)))
        ft_pct_recent  = float(eff.get('ft_pct',  0.75))
        fga_per_game   = float(sum(values) / max(len(values), 1)) if prop_type == 'points' else 0.0
        fg3_pct_recent = float(player_stats.get('fg3_pct', 0.0))  # populated for 3PT props
        avg_minutes    = float(eff.get('avg_minutes',    24.0))
        recent_minutes = float(eff.get('recent_minutes', 24.0))
        usage_rate     = float(eff.get('usage_rate',     18.0))

        # computed efficiency metrics (from efficiency block)
        total_pts = float(player_stats.get('avg', 0)) * len(values)
        pts_per_shot = total_pts / max(usage_rate * len(values), 1.0)

        # trend features
        trend_slope = float(player_stats.get('trend_slope', 0.0))
        prev5 = values[5:10] if len(values) >= 10 else values
        momentum_score = ((recent_avg - float(np.mean(prev5))) / max(float(np.mean(prev5)), 1.0)
                          if prev5 else 0.0)
        volatility_ratio = (stddev / max(season_avg, 1.0)) if season_avg > 0 else 0.0

        # games above season avg
        games_above_last5 = sum(1 for v in last5 if v > season_avg)

        # b2b
        b2b_flag = int(rest_days <= 1)

        # location splits
        home_avg = float(player_stats.get('home_avg', recent_avg))
        away_avg = float(player_stats.get('away_avg', recent_avg))

        feats = {
            # core rolling stats
            'recent_avg':     recent_avg,
            'season_avg':     season_avg,
            'stddev':         stddev,
            'games_played':   len(values),
            'max_recent':     max_recent,
            'min_recent':     min_recent,
            # schedule / minutes
            'mins_last5':     recent_minutes,
            'mins_season':    avg_minutes,
            'rest_days':      rest_days,
            'is_home_game':   (True if is_home else (False if is_home is False else None)),
            'recent_away_streak': 0,
            # team / opp context
            'team_pace':        team_pace,
            'opp_pace':         opp_pace,
            'team_off_rating':  team_off_rating,
            'team_def_rating':  team_def_rating,
            'opp_off_rating':   opp_off_rating,
            'opp_def_rating':   opp_def_rating,
            # injuries
            'team_key_players_out': team_key_out,
            'opp_key_players_out':  opp_key_out,
            'opp_injury_impact':    opp_injury_imp,
            # team style (no live data → 0.0 consistent with training)
            'team_pts_fb': 0.0, 'opp_pts_fb_allowed': 0.0,
            'team_pts_off_tov': 0.0, 'opp_pts_off_tov_allowed': 0.0,
            'opp_pts_paint': 0.0,
            'opp_fga': 0.0, 'opp_fg_pct': 0.47,
            'opp_fg3a': 0.0, 'opp_fg3_pct': 0.36,
            'opp_tov': 0.0, 'opp_stl': 0.0, 'opp_blk': 0.0,
            # league averages (NBA season norms)
            'lg_pts_fb': 12.0, 'lg_opp_pts_fb': 12.0,
            'lg_pts_off_tov': 16.0, 'lg_opp_pts_off_tov': 16.0,
            'lg_fga': 86.0, 'lg_fg_pct': 0.47,
            'lg_fg3a': 35.0, 'lg_tov': 14.0, 'lg_stl': 7.0,
            # matchup history
            'vs_team_avg':   vs_team_avg,
            'matchup_games': matchup_games,
            # DVP deltas — the core "defence vs position" signal
            'dvp_gp': dvp_gp,
            **dvp_deltas,
            # primary defender composite score
            'primary_defender_score01': primary_def_score,
            # shooting efficiency from game log
            'fg_pct_recent':  fg_pct_recent,
            'fg3_pct_recent': fg3_pct_recent,
            'ft_pct_recent':  ft_pct_recent,
            'fga_per_game':   fga_per_game,
            'fg3a_per_game':  0.0,   # populated when prop_type = three_pointers below
            'fta_per_game':   float(eff.get('usage_rate', 0.0)) * 0.15,  # rough FTA proxy
            'oreb_per_game':  0.0,
            'dreb_per_game':  0.0,
            'plus_minus_avg': 0.0,
            'fouls_per_game': 2.0,
            'win_rate_last10': float(tc.get('recent_form', {}).get('win_pct', 0.5)),
            # Tier 2: momentum
            'last_3_games_trend':            float(np.mean(values[:3]))  - season_avg if len(values) >= 3 else 0.0,
            'last_5_games_trend':            recent_avg - season_avg,
            'last_10_games_trend':           float(np.mean(values[:10])) - season_avg if len(values) >= 10 else 0.0,
            'games_above_season_avg_last5':  games_above_last5,
            # Tier 2: schedule
            'is_back_to_back':     b2b_flag,
            'days_since_last_game': rest_days + 1,
            'games_in_last_7_days': 2 if b2b_flag else 1,
            # Tier 2: advanced player metrics (0 → defaults; model learned to weight these low)
            'usage_rate': usage_rate / 100.0,  # normalise to 0-1
            'true_shooting_pct': float(eff.get('fg_pct', 0.55)),
            'effective_fg_pct':  float(eff.get('fg_pct', 0.50)),
            'assist_percentage': 0.0,
            'rebound_percentage': 0.0,
            'pie': 0.0,
            # Tier 2: shot location (populated if available, else 0.0)
            'rim_fga_per_game':       0.0,
            'paint_fga_per_game':     0.0,
            'mid_range_fga_per_game': 0.0,
            'corner_3_pct':           float(player_stats.get('corner3_pct',      0.37)),
            'above_break_3_pct':      float(player_stats.get('above_break3_pct', 0.35)),
            # Tier 2: clutch (0 → model uses other features)
            'clutch_pts_per_game':     0.0,
            'clutch_fg_pct':           0.0,
            'clutch_minutes_per_game': 0.0,
            # Tier 4: efficiency
            'points_per_shot':          pts_per_shot,
            'ast_to_tov_ratio':         float(player_stats.get('ast_tov_ratio', 1.0)),
            'reb_rate_per_36':          0.0,
            'scoring_efficiency_trend': 0.0,
            'usage_trend':              0.0,
            'minutes_volatility':       0.0,
            'blowout_game_pct':         0.0,
            'close_game_pct':           0.0,
            # Tier 4: opp defensive trends
            'opp_def_rating_home_away_split': 0.0,
            'opp_blocks_per_game_last5':      0.0,
            'opp_steals_per_game_last5':      0.0,
            # Tier 4: game context (unknown → 0)
            'days_rest_opponent': 2, 'opponent_back_to_back': 0,
            'playoff_implications': 0, 'rivalry_game': 0,
            'national_tv_game': 0, 'season_phase': 0,
            # Tier 4: teammate impact
            'primary_teammate_out': 0, 'secondary_teammate_out': 0,
            'new_teammate_games': 0, 'lineup_stability_score': 1.0, 'bench_strength': 0.0,
            # Tier 4: opponent-adjusted
            'pts_vs_top10_defenses':    season_avg,
            'pts_vs_bottom10_defenses': season_avg,
            'consistency_score':        max(0.0, 1.0 - volatility_ratio),
            'ceiling_game_frequency':   0.0,
            # Tier 4: advanced defensive (live → 0)
            'def_fg_pct_allowed': 0.0, 'def_rating_individual': 0.0,
            'deflections_per_game': 0.0, 'contested_shots_per_game': 0.0,
            # Tier 4: play type (live → 0)
            'pnr_ball_handler_pct': 0.0, 'pnr_roll_man_pct': 0.0,
            'isolation_pct': 0.0, 'spot_up_pct': 0.0,
            'post_up_pct': 0.0, 'transition_pct': 0.0,
            # Tier 5: streaks / game importance (0)
            'consecutive_over_games': 0, 'consecutive_under_games': 0,
            'hot_hand_indicator': 0.0, 'recent_variance_spike': 0.0,
            'playoff_seeding_impact': 0.5, 'tanking_indicator': 0.0,
            'must_win_situation': 0.0, 'games_back_from_playoff': 0.0,
            # Tier 5: rotation
            'fourth_quarter_usage_rate': 0.25 if avg_minutes > 30 else 0.18,
            'garbage_time_minutes_pct': 0.0,
            'typical_substitution_minute': min(48.0, avg_minutes + 3.0),
            'crunch_time_usage': 0.28 if avg_minutes > 28 else 0.15,
            # Tier 5: specific matchup (0)
            'career_vs_defender': 0.0, 'recent_vs_defender': 0.0, 'player_vs_arena': 0.0,
            # Tier 6: shot quality (0; model degrades gracefully)
            'avg_shot_distance': 0.0, 'contested_shot_pct': 0.5,
            'open_shot_pct': 0.3, 'wide_open_shot_pct': 0.2,
            'catch_and_shoot_pct': 0.3, 'pull_up_shot_pct': 0.3,
            'paint_touch_frequency': 0.0,
            'corner_three_pct':           float(player_stats.get('corner3_pct',      0.37)),
            'above_break_three_pct':      float(player_stats.get('above_break3_pct', 0.35)),
            'restricted_area_fg_pct':     float(eff.get('fg_pct', 0.55)),
            'mid_range_frequency': 0.0, 'shot_quality_vs_expected': 0.0,
            'avg_shot_clock_time': 12.0, 'late_clock_shot_frequency': 0.15,
            'early_clock_shot_frequency': 0.25,
            # Tier 6: touch/usage (0)
            'touches_per_game': float(fga_per_game + recent_avg * 0.3),
            'avg_dribbles_per_touch': 2.0, 'avg_seconds_per_touch': 3.0,
            'elbow_touches_per_game': 0.0, 'post_touches_per_game': 0.0,
            'paint_touches_per_game': 0.0, 'front_court_touches_per_game': 0.0,
            'time_of_possession_per_game': avg_minutes * 0.25,
            'touches_per_possession': 0.0, 'avg_points_per_touch': 0.0,
            # Tier 6: lineup (0)
            'net_rating_with_starters': 0.0,
            'usage_rate_with_star_out': usage_rate / 100.0 * 1.1,
            'minutes_with_starting_lineup_pct': 0.65 if avg_minutes > 25 else 0.35,
            'five_man_unit_net_rating': 0.0,
            'on_court_net_rating': 0.0, 'off_court_net_rating': 0.0,
            'on_off_differential': 0.0, 'lineups_played_count': 1.0,
            # Tier 3: travel / arena (0)
            'time_zone_change': 0.0, 'travel_distance': 0.0, 'coast_to_coast': 0.0,
            'arena_altitude': 0.0, 'arena_capacity': 0.0, 'home_court_advantage_rating': 0.0,
            # Tier 3: lineup on/off
            'on_court_plus_minus': 0.0, 'off_court_plus_minus': 0.0,
            'net_rating': 0.0, 'top_lineup_minutes_pct': 0.0,
            # Tier 3: vs-team history
            'vs_team_last_season_avg': vs_team_avg,
            'vs_team_home_away_split': home_avg - away_avg,
            'vs_team_win_pct': 0.5,
            # Tier 3: model calibration (0 → neutral)
            'model_accuracy_player': 0.0, 'avg_prediction_error_player': 0.0,
            'calibration_score_player': 0.0,
            # Tier 7: time-series
            'rolling_7day_avg': recent_avg, 'rolling_14day_avg': recent_avg,
            'rolling_30day_avg': season_avg,
            'ewm_alpha_0.3': recent_avg, 'ewm_alpha_0.5': recent_avg,
            'trend_slope_10games': trend_slope,
            'trend_slope_5games':  trend_slope,
            'volatility_ratio':    volatility_ratio,
            'momentum_score':      momentum_score,
            'games_above_season_avg_7day':  games_above_last5,
            'games_above_season_avg_14day': games_above_last5,
            # Tier 7: enhanced matchup
            'head_to_head_avg': vs_team_avg, 'head_to_head_games': matchup_games,
            'position_vs_position_dvp': float(dvp_deltas.get('dvp_pts_delta', 0.0)),
            'matchup_pace': (team_pace + opp_pace) / 2,
            'defender_switching_frequency': 0.0,
            'historical_game_script_avg': 0.0,
            # Tier 8: rest / form
            'rest_advantage': 0.0, 'rest_advantage_abs': 0.0, 'both_teams_rested': 0.0,
            'opp_def_rating_last5':  opp_def_rating,
            'opp_def_rating_last10': opp_def_rating,
            'opp_def_rating_trend':  0.0,
            'opp_pace_last5':   opp_pace,
            'opp_win_rate_last10': float(oc.get('recent_form', {}).get('win_pct', 0.5)),
            # Tier 8: player age/experience (0 → unknown)
            'player_age': 0.0, 'years_experience': 0.0, 'is_rookie': 0.0, 'is_veteran': 0.0,
            # Tier 8: game script
            'expected_game_script': 0.5, 'blowout_probability': 0.2, 'close_game_probability': 0.4,
            # Tier 8: quarter-specific (0)
            'first_quarter_avg': 0.0, 'fourth_quarter_avg': 0.0, 'clutch_performance_score': 0.0,
            # Tier 8: shot selection quality (0)
            'shot_selection_rating': 0.0, 'bad_shot_frequency': 0.0, 'shot_clock_management': 0.0,
            # Tier 8: team chemistry (0)
            'teammate_chemistry_score': 0.0, 'lineup_continuity': 0.0,
            'team_win_streak': 0.0, 'team_loss_streak': 0.0,
            # Tier 8: defender detail
            'primary_defender_rating': 0.0, 'primary_defender_age': 0.0,
            'defender_size_mismatch': 0.0, 'defender_recent_form': 0.0,
            # Tier 9: pace-adjusted (0)
            'pts_per_100': 0.0, 'ast_per_100': 0.0, 'reb_per_100': 0.0,
            'stl_per_100': 0.0, 'blk_per_100': 0.0, 'tov_per_100': 0.0,
            # Tier 9: FT / foul drawing
            'ft_rate': 0.0, 'fouls_drawn_per_game': 0.0,
            'ft_attempts_per_game': 0.0, 'and_one_frequency': 0.0, 'foul_drawing_ability': 0.0,
            # Tier 9: rebounding rates (0)
            'oreb_rate': 0.0, 'dreb_rate': 0.0, 'total_reb_rate': 0.0,
            'rebound_contested_pct': 0.0, 'rebound_positioning_score': 0.0,
            # Tier 9: paint scoring (0)
            'paint_pts_per_game': 0.0, 'paint_attempts_per_game': 0.0,
            'paint_fg_pct': float(eff.get('fg_pct', 0.55)),
            'paint_touch_to_points': 0.0, 'restricted_area_attempts': 0.0,
            # Tier 9: game situation (0)
            'performance_when_leading': season_avg, 'performance_when_trailing': season_avg,
            'performance_when_tied': season_avg, 'performance_in_overtime': season_avg,
            'performance_by_score_differential': season_avg,
            # Tier 10: minutes fatigue
            'minutes_last_3_games': recent_minutes * 3,
            'minutes_last_5_games': recent_minutes * 5,
            'minutes_last_7_games': avg_minutes   * 7,
            'avg_minutes_last_3':   recent_minutes,
            'minutes_fatigue_score': min(1.0, (recent_minutes * 5) / max(avg_minutes * 7, 1.0)),
            # Tier 10: player-level advanced (0)
            'player_off_rating': 0.0, 'player_def_rating': 0.0, 'player_pace': team_pace,
            'fta_rate_player': 0.0,
            'pct_fga_2pt': 0.65, 'pct_fga_3pt': 0.35,
            'pct_pts_in_paint': 0.3, 'pct_pts_off_tov': 0.1, 'pct_pts_fb': 0.1,
        }
        return feats

    def _get_player_team_id(self, player_id):
        try:
            player_info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            time.sleep(0.6)
            return int(player_info['TEAM_ID'].iloc[0])
        except Exception as e:
            print(f"couldn't get team id for player {player_id}: {e}")
            return None
    
