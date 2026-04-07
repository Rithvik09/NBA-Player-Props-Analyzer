from flask import Flask, render_template, request, jsonify
from .basketball_betting_helper import BasketballBettingHelper
from .data_collector import TrainingDataCollector
import threading
import time
import logging
from logging.handlers import RotatingFileHandler
import os
from nba_api.stats.static import players

app = Flask(__name__,
    static_url_path='',
    static_folder='../static',
    template_folder='../templates')

if not os.path.exists('logs'):
    os.mkdir('logs')

file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('app started')

betting_helper = BasketballBettingHelper()

def _startup_auto_grade():
    try:
        result = betting_helper.auto_grade_pending()
        app.logger.info(f"Startup auto-grade: {result}")
    except Exception as e:
        app.logger.error(f"Startup auto-grade failed: {e}")

threading.Thread(target=_startup_auto_grade, daemon=True).start()

# ── Retrain status tracker ────────────────────────────────────────────────────
_retrain_status = {'running': False, 'last_result': None}
_retrain_lock   = threading.Lock()

def _run_retrain_job(min_new_samples=50, num_players=100, num_seasons=3):
    """Runs in a background thread — grades pending preds first, then retrains."""
    _retrain_status['last_result'] = None
    try:
        grade_result = betting_helper.auto_grade_pending()
        app.logger.info(f"auto-grade before retrain: {grade_result}")

        result = betting_helper.retrain(
            min_new_samples=min_new_samples,
            num_players=num_players,
            num_seasons=num_seasons,
        )
        app.logger.info(f"Retrain job result: {result}")
        _retrain_status['last_result'] = result

        # recalibrate confidence thresholds from graded logs after retraining
        try:
            graded = betting_helper.get_confidence_calibration_data()
            if graded:
                betting_helper.ml_predictor.calibrate_confidence_thresholds(graded)
                app.logger.info(f"confidence thresholds recalibrated on {len(graded)} samples")
        except Exception as _ce:
            app.logger.warning(f"confidence calibration skipped: {_ce}")
    except Exception as e:
        app.logger.error(f"Retrain job error: {e}")
        _retrain_status['last_result'] = {'status': 'error', 'error': str(e)}
    finally:
        _retrain_status['running'] = False

def _nightly_scheduler():
    """Wakes every minute, fires the retrain job once per day around 4 AM."""
    from datetime import datetime as _dt
    triggered_today = None
    while True:
        try:
            now = _dt.now()
            today_str = now.strftime('%Y-%m-%d')
            if now.hour == 4 and triggered_today != today_str:
                triggered_today = today_str
                app.logger.info("nightly retrain triggered")
                with _retrain_lock:
                    if not _retrain_status['running']:
                        _retrain_status['running'] = True
                        threading.Thread(target=_run_retrain_job, daemon=True).start()
        except Exception as e:
            app.logger.error(f"Nightly scheduler error: {e}")
        time.sleep(60)

threading.Thread(target=_nightly_scheduler, daemon=True).start()


@app.route('/test_api')
def test_api():
    # quick sanity check that the NBA API is reachable
    try:
        all_players = players.get_players()
        active_players = [p for p in all_players if p['is_active']]
        return jsonify({
            'total_players': len(all_players),
            'active_players': len(active_players),
            'sample_player': active_players[0] if active_players else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search_players')
def search_players():
    try:
        query = request.args.get('q', '')
        
        if not query or len(query) < 2:
            return jsonify([])
            
        suggestions = betting_helper.get_player_suggestions(query)
        return jsonify(suggestions)
        
    except Exception as e:
        app.logger.error(f'Error searching players: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/player_game_info/<int:player_id>')
def player_game_info(player_id):
    """Returns today's opponent for a player, if there's a game. Used to auto-fill the dropdown."""
    try:
        from nba_api.stats.endpoints import CommonPlayerInfo, ScoreboardV2
        from datetime import datetime
        import time

        player_info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        time.sleep(0.6)
        team_id = int(player_info['TEAM_ID'].iloc[0])

        if team_id == 0:
            return jsonify({'team_id': None, 'opponent_team_id': None, 'is_home': None})

        today = datetime.now().strftime('%Y-%m-%d')
        games = ScoreboardV2(game_date=today).get_data_frames()[0]
        time.sleep(0.6)

        opponent_team_id = None
        is_home = None
        for _, game in games.iterrows():
            home = int(game['HOME_TEAM_ID'])
            visitor = int(game['VISITOR_TEAM_ID'])
            if home == team_id:
                opponent_team_id = visitor
                is_home = True
                break
            elif visitor == team_id:
                opponent_team_id = home
                is_home = False
                break

        return jsonify({
            'team_id': team_id,
            'opponent_team_id': opponent_team_id,
            'is_home': is_home
        })
    except Exception as e:
        app.logger.error(f'Error getting player game info: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/get_player_stats/<int:player_id>')
def get_player_stats(player_id):
    try:
        stats = betting_helper.get_player_stats(player_id)
        if stats:
            return jsonify(stats)
        else:
            return jsonify({'error': 'Unable to retrieve player stats'}), 404
            
    except Exception as e:
        app.logger.error(f'Error getting player stats: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_prop', methods=['POST'])
def analyze_prop():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        required_fields = ['player_id', 'prop_type', 'line', 'opponent_team_id']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields. Required: {required_fields}'}), 400

        player_id = data['player_id']
        prop_type = data['prop_type']
        line = float(data['line'])
        opponent_team_id = int(data['opponent_team_id'])
        # is_home can be None (auto), True, or False — but it might arrive as a string
        # bool('false') is True in Python, so compare explicitly
        is_home_raw = data.get('is_home', None)
        if is_home_raw is None or is_home_raw == '':
            is_home = None
        elif isinstance(is_home_raw, bool):
            is_home = is_home_raw
        else:
            is_home = str(is_home_raw).lower() == 'true'

        analysis = betting_helper.analyze_prop_bet(
            player_id=player_id,
            prop_type=prop_type,
            line=line,
            opponent_team_id=opponent_team_id,
            is_home=is_home
        )
        
        if analysis:
            return jsonify(analysis)
        else:
            return jsonify({'error': 'Unable to perform analysis', 'success': False}), 500
            
    except Exception as e:
        app.logger.error(f'Error analyzing prop: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

_training_status = {'running': False, 'last_result': None}
_training_lock   = threading.Lock()

@app.route('/train', methods=['POST'])
def train_models():
    """Kicks off a full retrain in the background. Returns 202 immediately."""
    with _training_lock:
        if _training_status['running']:
            return jsonify({'error': 'Training already in progress'}), 409
        _training_status['running'] = True  # pre-mark inside the lock

    data = request.get_json(silent=True) or {}
    num_players = int(data.get('num_players', 100))
    num_seasons = int(data.get('num_seasons', 3))

    def _run_training():
        _training_status['last_result'] = None
        try:
            collector = TrainingDataCollector()
            player_ids = collector.get_active_player_ids(n=num_players)
            seasons = collector._get_seasons(num_seasons=num_seasons)
            app.logger.info(f'training: {len(player_ids)} players, seasons={seasons}')

            training_data = collector.collect_bulk(player_ids, seasons=seasons)
            app.logger.info(f'training: got {len(training_data)} samples')

            if len(training_data) < 500:
                _training_status['last_result'] = {
                    'success': False,
                    'error': f'Only {len(training_data)} samples — need at least 500'
                }
                return

            metrics = betting_helper.ml_predictor.train(training_data)
            app.logger.info('training done, models saved')
            # save metrics to retrain_meta for the Accuracy tab
            try:
                conn = betting_helper.get_db()
                cur = conn.cursor()
                from datetime import datetime as _dt
                cur.execute('''
                    UPDATE retrain_meta
                    SET last_retrain_at = ?, samples_at_last_retrain = ?,
                        last_auc = ?, last_rmse = ?
                    WHERE id = 1
                ''', (_dt.now().isoformat(), len(training_data),
                      metrics.get('auc'), metrics.get('rmse')))
                conn.commit()
                conn.close()
            except Exception as _e:
                app.logger.warning(f'Could not update retrain_meta after /train: {_e}')
            # recalibrate confidence thresholds from graded logs after training
            try:
                graded = betting_helper.get_confidence_calibration_data()
                if graded:
                    betting_helper.ml_predictor.calibrate_confidence_thresholds(graded)
                    app.logger.info(f'confidence thresholds recalibrated on {len(graded)} samples')
            except Exception as _ce:
                app.logger.warning(f'confidence calibration skipped after /train: {_ce}')
            _training_status['last_result'] = {
                'success': True,
                'samples': len(training_data),
                'auc': metrics.get('auc'),
                'rmse': metrics.get('rmse'),
            }
        except Exception as e:
            app.logger.error(f'Training failed: {e}')
            _training_status['last_result'] = {'success': False, 'error': str(e)}
        finally:
            _training_status['running'] = False

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()
    return jsonify({'message': 'Training started', 'num_players': num_players, 'num_seasons': num_seasons}), 202


@app.route('/train/status', methods=['GET'])
def training_status():
    return jsonify({
        'running': _training_status['running'],
        'last_result': _training_status['last_result']
    })


@app.route('/logs/auto-grade', methods=['POST'])
def auto_grade():
    """Grades any ungraded predictions from previous days in the background."""
    def _run():
        result = betting_helper.auto_grade_pending()
        app.logger.info(f"auto-grade done: {result}")
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({'message': 'Auto-grading started in background'}), 202


@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        limit = int(request.args.get('limit', 50))
        prop_type = request.args.get('prop_type', None)
        player_id = request.args.get('player_id', None)
        graded_only = request.args.get('graded_only', 'false').lower() == 'true'
        logs = betting_helper.get_prediction_logs(limit, prop_type, player_id, graded_only)
        return jsonify(logs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/logs/<int:log_id>/result', methods=['POST'])
def update_result(log_id):
    try:
        data = request.get_json()
        if not data or 'actual_result' not in data:
            return jsonify({'error': 'actual_result required'}), 400
        actual_result = float(data['actual_result'])
        notes = data.get('notes', None)
        success, outcome = betting_helper.update_actual_result(log_id, actual_result, notes)
        if success:
            return jsonify({'success': True, 'outcome': outcome})
        return jsonify({'error': outcome}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/accuracy', methods=['GET'])
def accuracy_stats():
    try:
        stats = betting_helper.get_accuracy_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def trigger_retrain():
    """Manual trigger for the auto-grade + retrain cycle. Returns 202 immediately."""
    with _retrain_lock:
        if _retrain_status['running']:
            return jsonify({'error': 'Retrain already in progress'}), 409
        _retrain_status['running'] = True   # pre-mark inside the lock

    data = request.get_json(silent=True) or {}
    min_new_samples = int(data.get('min_new_samples', 10))  # lower bar when triggered manually
    num_players     = int(data.get('num_players', 100))
    num_seasons     = int(data.get('num_seasons', 3))

    threading.Thread(
        target=_run_retrain_job,
        args=(min_new_samples, num_players, num_seasons),
        daemon=True
    ).start()

    return jsonify({'message': 'Retrain started in background'}), 202


@app.route('/retrain/status', methods=['GET'])
def retrain_status():
    try:
        meta = betting_helper.get_retrain_meta()
        return jsonify({
            'running': _retrain_status['running'],
            'last_result': _retrain_status['last_result'],
            'meta': meta,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/bias', methods=['GET'])
def bias_report():
    try:
        report = betting_helper.get_bias_report()
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)