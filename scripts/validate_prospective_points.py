"""Collect paired predictions or settle them using free official NBA results.

Run with --collect for one fresh, bounded quote capture (at most once/day).
Without that flag, only grade/report; no Odds API credits are spent.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.prospective_points import digest, save_new, validate_quote, select_quotes, settlement, paired_report

OUT = Path('logs/prospective_points_v1')
MODEL_DIR = Path('logs/expanded_points_experiment_v4')
MODEL_PATHS = {'candidate': MODEL_DIR/'development_selected.joblib',
               'compact': MODEL_DIR/'compact_expanded.joblib'}
SOURCE_PATHS = ['src/audited_features.py', 'src/recovered_features.py',
                'src/historical_inputs.py', 'src/points_candidate.py', 'src/verified_training.py',
                'src/prospective_points.py', 'scripts/validate_prospective_points.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    expected = {'models': {k: {'path': str(v), 'sha256': digest(v)} for k, v in MODEL_PATHS.items()},
                'source_hashes': {p: digest(p) for p in SOURCE_PATHS},
                'feature_policy': 'point_in_time_v4',
                'primary': 'First fresh request-time quote per eligible player/game; lexicographically first available bookmaker. Same actual line for both frozen models.',
                'eligibility': 'At least 10 strictly earlier current-season appearances and prior20 mean MIN >=12, or original prior-season top50 cohort with at least10 appearances.',
                'warmup': 'All other nonempty histories saved separately; excluded from primary analysis.',
                'horizons': {'request_time': 'actual generation time', '60m': [50, 70], '15m': [10, 20]},
                'sampling': 'At most one metered snapshot per UTC day, original20-credit/month cap and50-credit account reserve. No guaranteed horizon coverage.',
                'evaluation': 'Paired Brier/logloss/AUC and pointsRMSE; game-cluster95% bootstrap CI; checkpoint100 eligible settled player-games per horizon. No retuning or promotion from interim results.',
                'voids': 'DNP excluded; pushes excluded from binary metrics but included in pointsRMSE; missing results remain unresolved.',
                'decision_rule': 'Research success requires candidate logloss improvement>=0.002 with game-cluster95% upper delta<0, nonworse Brier and RMSE, >=100 eligible settled player-games across>=20 games; subsequent fixed100-case checkpoints exploratory.',
                'deployment': 'Shadow validation only. Serving artifacts unchanged; no wagers.'}
    path = OUT/'protocol.json'
    if path.exists():
        frozen = json.loads(path.read_text())
        if {k: v for k, v in frozen.items() if k != 'frozen_at'} != expected:
            raise RuntimeError('Frozen model/source/protocol changed; start a separately versioned study')
        return frozen
    expected['frozen_at'] = now()
    save_new(path, expected)
    return expected


def collect(horizon):
    from src.free_market_capture import capture, normalize
    from src.current_rosters import load_snapshot, refresh_rosters
    from src.nba_officials import nba_json
    from src.historical_inputs import HistoricalInputs, load_season_logs
    from src.audited_runtime import attach_live_injuries
    from src.audited_features import build_audited_features, SPECS
    from scripts.train_models import _load_precomputed_player_data
    from scripts.build_expanded_points import eligible

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    day = datetime.now(timezone.utc).date().isoformat()
    marker = OUT/'attempts'/f'{day}.json'
    if marker.exists():
        return {'collection': 'already attempted today; no additional credits spent'}
    # All prerequisites before paid-per-credit capture; sources retained for replay.
    roster = load_snapshot() or refresh_rosters()
    season = roster['season']
    logs = load_season_logs(season, include_postseason=True)
    local_day = pd.Timestamp.now(tz='America/New_York').date()
    age = (pd.Timestamp(local_day)-pd.to_datetime(logs.GAME_DATE).max()).days
    if age < 0 or age > 2:
        raise ValueError('Current-season boxscore cache is stale; refresh free NBA data first')
    raw = _load_precomputed_player_data(os.environ['DATABASE_URL'])
    inputs = HistoricalInputs(season, raw, logs, audited=True)
    original = set(map(int, inputs.catalog(50).PLAYER_ID))
    board_raw = nba_json('scoreboardv3', GameDate=str(local_day), LeagueID='00')
    games = board_raw['scoreboard']['games']
    future = sorted([g for g in games if pd.Timestamp(g['gameTimeUTC']) > pd.Timestamp.now(tz='UTC')], key=lambda g: g['gameTimeUTC'])
    if not future:
        return {'collection': 'No remaining games today'}
    # Capture normally selects earliest future event. Window check before spending.
    minutes = (pd.Timestamp(future[0]['gameTimeUTC'])-pd.Timestamp.now(tz='UTC')).total_seconds()/60
    if horizon == '60m' and not 50 <= minutes <= 70 or horizon == '15m' and not 10 <= minutes <= 20:
        return {'collection': 'Not in requested horizon window; no credits spent'}
    if not os.environ.get('ODDS_API_KEY'):
        values = dict(line.split('=', 1) for line in Path('.env').read_text().splitlines()
                      if '=' in line and not line.startswith('#'))
        os.environ['ODDS_API_KEY'] = values['ODDS_API_KEY'].strip().strip('"').strip("'")
    key = os.environ['ODDS_API_KEY']
    run = OUT/'runs'/stamp
    run.mkdir(parents=True)
    save_new(run/'roster.json', roster)
    save_new(run/'scoreboard.json', board_raw)
    joblib.dump({'raw_precomputed': raw, 'player_logs': logs}, run/'inputs.joblib')
    save_new(marker, {'started_at': now(), 'run': str(run), 'horizon': horizon})
    snapshot = capture(key)
    save_new(run/'quote_snapshot.json', snapshot)
    canonical = normalize(snapshot, games, roster)
    save_new(run/'canonical.json', canonical)
    models = {k: joblib.load(v) for k, v in MODEL_PATHS.items()}
    games_by_id = {str(g['gameId']): g for g in games}
    roster_by_id = {int(p['nba_id']): p for p in roster['players'] if p.get('nba_id')}
    attach_live_injuries(inputs, pd.Timestamp(local_day))
    if hasattr(inputs, 'live_injury_context'):
        save_new(run/'injuries.json', inputs.live_injury_context)
    saved, rejected = [], []
    for q in select_quotes(canonical['props']):
        identity = f"{q['game_id']}_{q['player_id']}_{horizon}"
        dest = OUT/'forecasts'/f'{identity}.json'
        if dest.exists():
            continue
        try:
            validate_quote(q, now(), horizon)
            game = games_by_id[q['game_id']]
            pid = int(q['player_id'])
            team = int(roster_by_id[pid]['team_id'])
            home, away = int(game['homeTeam']['teamId']), int(game['awayTeam']['teamId'])
            if team not in (home, away):
                raise ValueError('Player team not in official game')
            prediction_day = pd.Timestamp(q['game_start']).tz_convert('America/New_York').tz_localize(None).normalize()
            history = inputs.player_log(pid)
            history = history[pd.to_datetime(history.GAME_DATE) < prediction_day].sort_values('GAME_DATE')
            if history.empty:
                raise ValueError('No prior current-season appearances')
            inputs.prediction_game_id = q['game_id']
            inputs.prediction_cutoff = pd.Timestamp.now(tz='UTC')
            inputs.recovery._cutoff_cache = {}
            inputs.recovery._injury_cache = {}
            features, gaps = build_audited_features(history, 'PTS', pid, team, away if team == home else home, prediction_day, team == home, inputs)
            for k in gaps:
                if SPECS[k].status == 'active':
                    features[k] = np.nan
            frame = pd.DataFrame([features])
            predictions = {name: {'points': float(m.predict(frame)[0]),
                                  'over_probability': float(m.predict_over(frame, [q['line']])[0])} for name, m in models.items()}
            generated = now()
            lead = validate_quote(q, generated, horizon)  # Check again AFTER feature/model work.
            if not all(np.isfinite(p['points']) and 0 <= p['over_probability'] <= 1 for p in predictions.values()):
                raise ValueError('Nonfinite prediction')
            record = {'id': identity, 'quote': q, 'generated_at': generated,
                      'feature_cutoff': inputs.prediction_cutoff.isoformat(), 'minutes_before_tipoff': lead,
                      'horizon': horizon, 'cohort': 'eligible' if eligible(history, pid in original) else 'warmup',
                      'prior_appearances': len(history), 'prior20_mean_minutes': float(history.tail(20).MIN.mean()),
                      'predictions': predictions, 'missing_features': sorted(gaps),
                      'features': {k: float(v) if np.isfinite(float(v)) else None for k, v in features.items()},
                      'protocol_sha256': digest(OUT/'protocol.json'), 'source_run': str(run),
                      'inputs_sha256': digest(run/'inputs.joblib')}
            save_new(dest, record)
            saved.append(identity)
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append({'id': identity, 'reason': str(exc)})
    result = {'saved': len(saved), 'ids': saved, 'rejected': rejected,
              'credits_used': snapshot['credits_used'], 'credits_remaining': snapshot['credits_remaining']}
    save_new(run/'collection.json', result)
    return result


def grade_and_report():
    from src.nba_officials import nba_json, game_summary
    forecasts = [json.loads(p.read_text()) for p in sorted((OUT/'forecasts').glob('*.json'))]
    existing = {p.stem: json.loads(p.read_text()) for p in (OUT/'settlements').glob('*.json')}
    temporary, failures, cache = {}, [], {}
    for f in forecasts:
        if f['id'] in existing or pd.Timestamp(f['quote']['game_start']) > pd.Timestamp.now(tz='UTC'):
            continue
        gid = f['quote']['game_id']
        try:
            if gid not in cache:
                summary = game_summary(gid)
                boxes = nba_json('boxscoretraditionalv3', GameID=gid, StartPeriod=0, EndPeriod=0, StartRange=0, EndRange=0, RangeType=0)['boxScoreTraditional'] if int(summary.get('gameStatus', 0)) == 3 else None
                cache[gid] = summary, boxes
                if boxes:
                    path = OUT/'results'/f'{gid}.json'
                    if not path.exists():
                        save_new(path, {'observed_at': now(), 'summary': summary, 'boxes': boxes})
            summary, boxes = cache[gid]
            if boxes is None:
                continue
            outcome = settlement(f, summary, boxes)
            outcome['observed_at'] = now()
            if outcome['status'] in ('settled', 'push', 'dnp'):
                save_new(OUT/'settlements'/f"{f['id']}.json", outcome)
                existing[f['id']] = outcome
            else:
                temporary[f['id']] = outcome
        except Exception as exc:
            failures.append({'game_id': gid, 'error_type': type(exc).__name__})
    report = paired_report(forecasts, {**existing, **temporary})
    report.update(updated_at=now(), collection_failures=failures,
                  cohort_counts={k: sum(f['cohort'] == k for f in forecasts) for k in ('eligible', 'warmup')},
                  protocol_sha256=digest(OUT/'protocol.json'))
    temp = OUT/'report.tmp'
    temp.write_text(json.dumps(report, indent=2, allow_nan=False))
    temp.replace(OUT/'report.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--horizon', choices=['request_time', '60m', '15m'], default='request_time')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        protocol()
        if args.collect:
            print(json.dumps(collect(args.horizon)), flush=True)
        print(json.dumps(grade_and_report(), indent=2), flush=True)


if __name__ == '__main__':
    main()
