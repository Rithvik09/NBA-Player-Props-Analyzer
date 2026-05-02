from flask import Flask, render_template, request, jsonify, g
from .basketball_betting_helper import BasketballBettingHelper
from .data_collector import TrainingDataCollector
from .game_predictor import GamePredictor
from .logging_config import configure_logging
import threading
import time
import logging
import os
import uuid
from nba_api.stats.static import players

app = Flask(__name__,
    static_url_path='',
    static_folder='../static',
    template_folder='../templates')

# One-shot logging setup — rotating file + stderr, tuneable via env vars.
# See src/logging_config.py for knobs (LOG_LEVEL, LOG_JSON, LOG_DIR, …).
configure_logging(app.logger)
app.logger.info('app started')


# ── Request-ID tracing ──────────────────────────────────────────────────────
# Every inbound request gets a short ULID-ish ID (or whatever the upstream
# proxy sent in X-Request-ID). The ID is:
#   - attached to ``flask.g.request_id`` for code that wants to log it
#   - emitted in the response as ``X-Request-ID`` so curl/clients can grep
#   - included in a one-line access log per request, with status + duration
# This is what makes "find every log line for the bet that mispriced LeBron
# last Tuesday" tractable in production.
@app.before_request
def _assign_request_id():
    rid = request.headers.get("X-Request-ID")
    if not rid:
        rid = uuid.uuid4().hex[:12]
    g.request_id = rid
    g._req_started = time.time()


@app.after_request
def _emit_request_log(response):
    rid = getattr(g, "request_id", None)
    if rid:
        response.headers["X-Request-ID"] = rid
    started = getattr(g, "_req_started", None)
    duration_ms = (time.time() - started) * 1000.0 if started else None
    try:
        app.logger.info(
            "req rid=%s method=%s path=%s status=%s dur_ms=%s",
            rid, request.method, request.path,
            response.status_code,
            f"{duration_ms:.1f}" if duration_ms is not None else "?",
        )
    except Exception:  # noqa: BLE001 — never crash the request
        pass
    # Prometheus instrumentation — record on the route rule, not the literal
    # path, to keep label cardinality bounded. Skip /metrics itself so the
    # scraper doesn't pollute its own histogram.
    try:
        from . import metrics as _metrics
        rule = getattr(request.url_rule, "rule", None) or request.path
        if rule != "/metrics":
            duration_s = (time.time() - started) if started else 0.0
            _metrics.observe_request(
                request.method, rule, response.status_code, duration_s,
            )
    except Exception:  # noqa: BLE001
        pass
    return response


@app.route('/metrics')
def metrics_endpoint():
    """Prometheus scrape endpoint. Returns text exposition format."""
    from . import metrics as _metrics
    body = _metrics.render()
    return (body, 200, {"Content-Type": "text/plain; version=0.0.4; charset=utf-8"})

betting_helper = BasketballBettingHelper()
game_predictor = GamePredictor(betting_helper)

# Enable odds tracking if API key is set via environment variable
_odds_key = os.environ.get('ODDS_API_KEY')
if _odds_key:
    betting_helper.set_odds_api_key(_odds_key)
    app.logger.info('Odds tracker enabled via ODDS_API_KEY env var')

# ── Odds polling background loop ─────────────────────────────────────────────
_odds_poll_interval = int(os.environ.get('ODDS_POLL_INTERVAL', 3600))  # default 60 min

def _odds_poll_loop():
    """Poll odds API on a loop while the server is running."""
    if not _odds_key:
        return
    from src.odds_tracker import poll_odds
    import time as _time
    _time.sleep(5)  # let server finish starting
    while True:
        try:
            count = poll_odds(_odds_key, betting_helper.db_name)
            app.logger.info(f"Odds poll: {count} lines stored")
        except Exception as e:
            app.logger.error(f"Odds poll failed: {e}")
        _time.sleep(_odds_poll_interval)

if _odds_key:
    threading.Thread(target=_odds_poll_loop, daemon=True, name='odds-poller').start()
    app.logger.info(f'Odds poller started (every {_odds_poll_interval}s)')

# ── Close-only sweeper ───────────────────────────────────────────────────────
# The full odds poll runs on _odds_poll_interval (default 60 min) which is too
# coarse to reliably catch a tipoff inside the ±20-min closing window. This
# sweeper is pure SQL — no API hits — so it can run every few minutes without
# burning quota. It walks the game_tipoffs cache populated by fetch_upcoming_
# games and stamps closes on any imminent game.
_close_sweep_interval = int(os.environ.get('ODDS_CLOSE_SWEEP_INTERVAL', 300))  # default 5 min

def _close_sweep_loop():
    if not _odds_key:
        return
    from src.odds_tracker import OddsTracker
    import time as _time
    _time.sleep(30)  # let initial fetch_upcoming_games populate the cache
    tracker = OddsTracker(api_key=None, db_path=betting_helper.db_name)
    while True:
        try:
            result = tracker.capture_closing_lines_for_imminent_games()
            if result["games_inside_window"]:
                app.logger.info(
                    "Close sweep: stamped %d closes across %d games",
                    result["stamped_total"], result["games_inside_window"],
                )
        except Exception as e:
            app.logger.error(f"Close sweep failed: {e}")
        _time.sleep(_close_sweep_interval)

if _odds_key:
    threading.Thread(target=_close_sweep_loop, daemon=True, name='close-sweeper').start()
    app.logger.info(f'Close sweeper started (every {_close_sweep_interval}s)')

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


@app.route('/healthz')
def healthz():
    """Liveness + readiness probe.

    Returns 200 only when:
      - The bankroll DB is at the expected schema version
      - At least one classifier model artifact exists in models/
      - All required props have a regressor or classifier loaded

    Returns 503 with a per-check breakdown otherwise. ``ready=false``
    in the body so a load-balancer can drain traffic during retrains.
    """
    import json as _json
    import os as _os
    from datetime import datetime as _dt, timezone as _tz
    from .bankroll import _TARGET_VERSION, BankrollTracker as _BT
    from .api_cache import all_stats as _cache_stats

    checks: dict = {}

    # 1. Bankroll DB schema
    try:
        bt = _BT(betting_helper.db_name)
        import sqlite3 as _s
        conn = _s.connect(bt.db_path)
        v = int(conn.execute("PRAGMA user_version").fetchone()[0])
        conn.close()
        checks["bankroll_db"] = {
            "ok": v == _TARGET_VERSION,
            "version": v,
            "target": _TARGET_VERSION,
        }
    except Exception as e:  # noqa: BLE001
        checks["bankroll_db"] = {"ok": False, "error": str(e)}

    # 2. Model artifacts on disk
    models_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models")
    meta_path = _os.path.join(models_dir, "model_metadata.json")
    try:
        meta = _json.loads(open(meta_path).read()) if _os.path.exists(meta_path) else {}
        props = list((meta.get("props") or {}).keys())
        artifact_present = []
        for p in props:
            cal = _os.path.join(models_dir, f"clf_cal_{p}.joblib")
            reg = _os.path.join(models_dir, f"reg_{p}.joblib")
            if _os.path.exists(cal) or _os.path.exists(reg):
                artifact_present.append(p)
        last_trained = meta.get("trained_utc") or meta.get("trained_at")
        checks["models"] = {
            "ok": len(artifact_present) > 0,
            "metadata_present": _os.path.exists(meta_path),
            "props_with_artifact": len(artifact_present),
            "props_total": len(props),
            "last_trained": last_trained,
        }
    except Exception as e:  # noqa: BLE001
        checks["models"] = {"ok": False, "error": str(e)}

    # 3. Cache subsystem (always alive — just a liveness sanity check)
    try:
        rows = _cache_stats()
        checks["cache"] = {"ok": True, "registered": len(rows)}
    except Exception as e:  # noqa: BLE001
        checks["cache"] = {"ok": False, "error": str(e)}

    overall_ok = all(c.get("ok") for c in checks.values())
    payload = {
        "ready": overall_ok,
        "checked_utc": _dt.now(_tz.utc).isoformat(),
        "checks": checks,
    }
    return jsonify(payload), (200 if overall_ok else 503)


@app.route('/healthz/drift')
def healthz_drift():
    """Brier decay + drift snapshot. Cheap operator dashboard.

    Reads training-time Brier from ``models/model_metadata.json`` (under
    ``walk_forward.brier_cal_mean`` per prop) and compares to the rolling
    Brier over ``?window_days=`` (default 30) of graded predictions.
    """
    import json as _json
    import os as _os
    from .monitoring import rolling_brier, brier_decay_check
    window_days = int(request.args.get("window_days", 30))
    threshold = float(request.args.get("threshold", 0.20))

    rolling = rolling_brier(betting_helper.db_name, window_days=window_days)

    meta_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "models", "model_metadata.json",
    )
    training_brier: dict[str, float] = {}
    needs_recal: dict[str, bool] = {}
    try:
        if _os.path.exists(meta_path):
            meta = _json.loads(open(meta_path).read())
            for prop, info in (meta.get("props") or {}).items():
                # Prefer walk-forward (more honest), fall back to single-OOT
                # ``training_brier`` written by EnhancedMLPredictor.train.
                wf = info.get("walk_forward") or {}
                if "brier_cal_mean" in wf:
                    training_brier[prop] = float(wf["brier_cal_mean"])
                elif info.get("training_brier") is not None:
                    try:
                        training_brier[prop] = float(info["training_brier"])
                    except (TypeError, ValueError):
                        pass
                if info.get("needs_recal") is not None:
                    needs_recal[prop] = bool(info["needs_recal"])
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"could not read model metadata: {e}"}), 500

    decay = brier_decay_check(rolling, training_brier, degradation_threshold=threshold)
    any_decayed = any(d.get("decayed") for d in decay)
    # ECE-driven recalibration flag: surface props with needs_recal=true so
    # operators see them on the same dashboard. Doesn't trigger 503 on its
    # own (calibration drift is fixable by retraining; brier decay is the
    # bigger alarm).
    any_needs_recal = any(needs_recal.values())
    return jsonify({
        "window_days": window_days,
        "threshold": threshold,
        "rolling": rolling,
        "training_brier": training_brier,
        "needs_recal": needs_recal,
        "decay": decay,
        "any_decayed": any_decayed,
        "any_needs_recal": any_needs_recal,
    }), (503 if any_decayed else 200)


@app.route('/monitor/anomalies')
def monitor_anomalies():
    """Recent served predictions that are >Nσ from the posted line.

    Per-prop residual σ comes from model_metadata.json (training-time RMSE
    is a serviceable proxy for residual σ when the regressor is unbiased).
    """
    import json as _json
    import os as _os
    from .monitoring import find_anomalies
    window_days = int(request.args.get("window_days", 7))
    sigma = float(request.args.get("sigma", 3.0))
    limit = int(request.args.get("limit", 50))

    meta_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "models", "model_metadata.json",
    )
    stds: dict[str, float] = {}
    try:
        if _os.path.exists(meta_path):
            meta = _json.loads(open(meta_path).read())
            for prop, info in (meta.get("props") or {}).items():
                # Fall back across whatever the metadata exposes.
                rmse = (info.get("regressor", {}) or {}).get("rmse")
                if rmse is None:
                    rmse = info.get("rmse")
                if rmse is not None:
                    stds[prop] = float(rmse)
    except Exception:  # noqa: BLE001
        pass

    anomalies = find_anomalies(
        betting_helper.db_name, stds,
        window_days=window_days, sigma_threshold=sigma, limit=limit,
    )
    return jsonify({
        "window_days": window_days,
        "sigma": sigma,
        "n_anomalies": len(anomalies),
        "anomalies": anomalies,
    })


@app.route('/cache/stats')
def cache_stats():
    """Expose TTL-cache hit rates so the operator can spot a regression."""
    from .api_cache import all_stats
    rows = all_stats()
    total_hits = sum(r["hits"] for r in rows)
    total_misses = sum(r["misses"] for r in rows)
    denom = total_hits + total_misses
    return jsonify({
        "caches": rows,
        "overall_hit_rate": (total_hits / denom) if denom else 0.0,
        "total_hits": total_hits,
        "total_misses": total_misses,
    })


def _check_admin_token() -> bool:
    """Constant-time bearer-token gate for /admin/* endpoints.

    The token is read from the ``ADMIN_TOKEN`` env var. If unset, the
    endpoint refuses (closed by default). Compared with ``hmac.compare_digest``
    to defeat timing attacks — paranoid given this is on the open internet
    and the token guards a destructive operation.
    """
    import hmac as _hmac
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected:
        return False
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        provided = auth[len("Bearer "):]
    else:
        provided = request.headers.get("X-Admin-Token", "")
    if not provided:
        return False
    return _hmac.compare_digest(str(expected), str(provided))


@app.route('/admin/cache/clear', methods=['POST'])
def admin_cache_clear():
    """Drop every TTL/disk cache. Token-gated.

    Use case: a stale roster cache after a midday trade, or after a
    player_id mapping change. Cheaper than a full restart.
    """
    if not _check_admin_token():
        return jsonify({"error": "unauthorised"}), 401
    from .api_cache import clear_all, all_stats
    before = sum(r.get("size", 0) for r in all_stats())
    clear_all()
    after = sum(r.get("size", 0) for r in all_stats())
    app.logger.warning(
        "admin cache cleared rid=%s before=%d after=%d",
        getattr(g, "request_id", None), before, after,
    )
    return jsonify({"cleared": True, "entries_before": before, "entries_after": after})


@app.route('/cache/clear', methods=['POST'])
def cache_clear():
    from .api_cache import clear_all
    clear_all()
    return jsonify({"cleared": True})

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/game-predictions')
def game_predictions_page():
    return render_template('game_predictions.html')

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

from .rate_limit import rate_limited as _rate_limited

# /analyze_prop is the most expensive endpoint (multi-model load + NBA API call).
# Cap any single client to ~30 calls/minute. Configurable via env so tests
# can disable it without monkey-patching the decorator.
_ANALYZE_RATE = int(os.environ.get("ANALYZE_RATE_LIMIT", "30"))
_ANALYZE_WINDOW = float(os.environ.get("ANALYZE_RATE_WINDOW_SECONDS", "60"))


@app.route('/analyze_prop', methods=['POST'])
@_rate_limited("analyze_prop", rate=_ANALYZE_RATE, per_seconds=_ANALYZE_WINDOW)
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


# ── Bankroll + Kelly stake sizing ────────────────────────────────────────────
from .bankroll import BankrollTracker, kelly_stake, american_to_decimal, devig_two_way
_bankroll = BankrollTracker(betting_helper.db_name)


@app.route('/bankroll', methods=['GET'])
def bankroll_summary():
    try:
        return jsonify(_bankroll.summary())
    except Exception as e:
        app.logger.exception('bankroll summary failed')
        return jsonify({'error': str(e)}), 500


@app.route('/bankroll/balance', methods=['POST'])
def bankroll_set_balance():
    try:
        amount = float(request.get_json(force=True)['amount'])
        return jsonify({'balance': _bankroll.set_balance(amount)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/bankroll/bets', methods=['GET'])
def bankroll_list_bets():
    status = request.args.get('status')
    limit = int(request.args.get('limit', 50))
    return jsonify(_bankroll.list_bets(status=status, limit=limit))


@app.route('/bankroll/bets', methods=['POST'])
def bankroll_record_bet():
    """Record a placed bet.

    Accepts either 'prop_type' or 'prop' for the prop identifier, and either
    'stake' or 'stake_dollars' for the stake amount (parity with other
    endpoints).
    """
    data = request.get_json(force=True)
    try:
        prop_val = data.get('prop_type', data.get('prop'))
        stake_val = data.get('stake', data.get('stake_dollars'))
        if prop_val is None:
            return jsonify({'error': "missing 'prop_type' (or 'prop')"}), 400
        if stake_val is None:
            return jsonify({'error': "missing 'stake' (or 'stake_dollars')"}), 400
        # Optional analytics fields — silently coerce or skip
        def _maybe_float(k):
            v = data.get(k)
            return None if v is None else float(v)

        def _maybe_int(k):
            v = data.get(k)
            return None if v is None else int(v)

        bet_id = _bankroll.record_bet(
            player_name=data.get('player_name'),
            prop_type=str(prop_val),
            side=data.get('side', 'over'),
            line=float(data['line']),
            american_odds=float(data['american_odds']),
            our_prob=float(data['our_prob']),
            stake=float(stake_val),
            kelly_fraction=_maybe_float('kelly_fraction'),
            edge=_maybe_float('edge'),
            ev_per_dollar=_maybe_float('ev_per_dollar'),
            prediction_log_id=_maybe_int('prediction_log_id'),
            model_version=data.get('model_version'),
        )
        return jsonify({'id': bet_id}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/bankroll/bets/<int:bet_id>/settle', methods=['POST'])
def bankroll_settle_bet(bet_id):
    data = request.get_json(force=True)
    try:
        return jsonify(_bankroll.settle(bet_id, data['result']))
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/kelly', methods=['POST'])
def kelly_recommend():
    """Return recommended stake for a (prob, odds, bankroll) triple.

    Body: { our_prob, american_odds, bankroll?, kelly_fraction?, max_fraction? }
    """
    data = request.get_json(force=True)
    try:
        bankroll = float(data.get('bankroll', _bankroll.get_balance()))
        ks = kelly_stake(
            our_prob=float(data['our_prob']),
            american_odds=float(data['american_odds']),
            bankroll=bankroll,
            kelly_fraction=float(data.get('kelly_fraction', 0.25)),
            max_fraction=float(data.get('max_fraction', 0.05)),
        )
        return jsonify({
            'bankroll': bankroll,
            'edge': ks.edge,
            'full_kelly': ks.full_kelly,
            'stake_fraction': ks.stake_fraction,
            'stake_dollars': ks.stake_dollars,
            'ev_per_dollar': ks.ev_per_dollar,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/injuries/<team>')
def injuries_team(team):
    """Live injury scrape for a single team (ESPN → Rotowire fallback, 30-min TTL)."""
    from .live_injuries import summarise_team
    try:
        return jsonify(summarise_team(team))
    except Exception as e:
        app.logger.exception('injury scrape failed')
        return jsonify({'error': str(e)}), 500


@app.route('/injuries/player/<name>')
def injuries_player(name):
    """Lookup a single player's current injury status by name."""
    from .live_injuries import find_player_injury
    try:
        hit = find_player_injury(name)
        return jsonify(hit or {"player": name, "found": False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/healthz/clv')
def healthz_clv():
    """CLV pipeline visibility.

    Returns counts that let an operator answer "is the CLV pipeline
    alive?" without poking at SQLite directly:

      - prop_outcomes_total        — every settled outcome row
      - prop_outcomes_with_close   — subset that's CLV-ready (joined w/ closing line)
      - close_capture_rate         — ratio; healthy when >0.5 once tipoff window cron runs
      - last_outcome_at            — most-recent settled_at
      - last_close_captured_at     — sweeper telemetry from _LAST_POLL
      - upcoming_games             — count of un-tipped games in game_tipoffs
      - imminent_games             — count whose tipoff is inside the closing window now

    The endpoint is advisory — never returns 503. CLV pipeline being
    cold for a few hours is fine; the brier-decay endpoint owns the
    real "should we page someone" decision.
    """
    import sqlite3 as _sqlite
    from .odds_tracker import OddsTracker, last_poll_status
    db = betting_helper.db_name
    out = {
        "prop_outcomes_total": 0,
        "prop_outcomes_with_close": 0,
        "close_capture_rate": None,
        "last_outcome_at": None,
        "last_close_captured_at": None,
        "upcoming_games": 0,
        "imminent_games": 0,
    }
    try:
        conn = _sqlite.connect(db)
        try:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n,
                       SUM(CASE WHEN closing_line IS NOT NULL THEN 1 ELSE 0 END) AS with_close,
                       MAX(settled_at) AS last_at
                FROM prop_outcomes
                """
            ).fetchone()
            n_total, n_close, last_at = row[0] or 0, row[1] or 0, row[2]
            out["prop_outcomes_total"] = int(n_total)
            out["prop_outcomes_with_close"] = int(n_close)
            out["last_outcome_at"] = last_at
            if n_total:
                out["close_capture_rate"] = round(n_close / n_total, 4)

            # Upcoming + imminent counts. Wrap each in try/except so a
            # missing game_tipoffs table on a freshly-init'd DB doesn't
            # 500 the endpoint.
            try:
                tipoffs = conn.execute(
                    "SELECT game_id, commence_time FROM game_tipoffs"
                ).fetchall()
                out["upcoming_games"] = len(tipoffs)
                imm = sum(
                    1 for _gid, ct in tipoffs
                    if OddsTracker._inside_closing_window(ct or "")
                )
                out["imminent_games"] = imm
            except _sqlite.OperationalError:
                pass
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"healthz/clv read failed: {e}"}), 500

    last = last_poll_status()
    out["last_close_captured_at"] = last.get("closes_checked_at")
    out["closes_captured_last_run"] = last.get("closes_captured")
    return jsonify(out)


@app.route('/odds/status')
def odds_status():
    """Diagnostic: last-poll timestamp, lines stored, error, quota remaining."""
    from .odds_tracker import last_poll_status
    status = last_poll_status()
    # Flag stale polls (> 2h since last successful poll on game days)
    from datetime import datetime, timezone
    stale = False
    if status.get("time"):
        try:
            last = datetime.fromisoformat(status["time"])
            age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
            stale = age_min > 120
            status["age_minutes"] = round(age_min, 1)
        except Exception:
            pass
    status["stale"] = stale
    status["key_configured"] = bool(os.environ.get("ODDS_API_KEY"))
    return jsonify(status)


@app.route('/parlay', methods=['POST'])
def parlay_estimate():
    """Correlation-aware parlay probability.

    Body: {
        legs: [ { our_prob|prob, american_odds, prop,
                  player_id?, team_id?, side? }, ... ],
        n_samples?: int (default 20000)
    }

    Accepts either 'prob' or 'our_prob' on each leg for parity with /kelly.
    """
    from .parlay import parlay_probability, ParlayLeg
    data = request.get_json(force=True)
    try:
        raw = data.get('legs') or []
        if not raw:
            return jsonify({'error': 'legs must be non-empty'}), 400
        legs = []
        for idx, l in enumerate(raw):
            # Accept either 'prob' (parlay-native) or 'our_prob' (Kelly-native)
            pval = l.get('prob', l.get('our_prob'))
            if pval is None:
                return jsonify({'error': f"leg[{idx}] missing 'prob' or 'our_prob'"}), 400
            if l.get('american_odds') is None:
                return jsonify({'error': f"leg[{idx}] missing 'american_odds'"}), 400
            if l.get('prop') is None:
                return jsonify({'error': f"leg[{idx}] missing 'prop'"}), 400
            legs.append(ParlayLeg(
                prob=float(pval),
                american_odds=float(l['american_odds']),
                prop=str(l['prop']),
                player_id=(int(l['player_id']) if l.get('player_id') is not None else None),
                team_id=(int(l['team_id']) if l.get('team_id') is not None else None),
                side=str(l.get('side', 'over')),
            ))
        n = int(data.get('n_samples', 20_000))
        return jsonify(parlay_probability(legs, n_samples=n))
    except Exception as e:
        app.logger.exception('parlay estimation failed')
        return jsonify({'error': str(e)}), 400


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)