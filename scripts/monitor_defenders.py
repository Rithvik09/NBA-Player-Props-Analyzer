"""Incremental free collection and grading around the unchanged frozen ranker.

Never reruns the opened holdout or changes training/qualification artifacts.
"""
from collections import deque
from datetime import datetime, timezone
import fcntl
import gzip
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.prospective_points import digest, save_new
from src.nba_officials import nba_json, game_summary
from src.defender_assignment import AssignmentState
from scripts.crawl_defender_matchups import validate
from scripts.qualify_defender_rules import summarize

ROOT = Path('logs/source_blocked/assignment_ranker')
MATCHUPS = Path('logs/source_blocked/current_matchups_holdout')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def verify_frozen():
    frozen = json.loads((ROOT/'frozen.json').read_text())
    assert digest(ROOT/'model.joblib') == frozen['model_sha256'], 'Frozen model drift'
    assert digest('src/defender_assignment.py') == frozen['source_sha256'], 'Frozen source drift'
    return frozen


def grade_one(forecast, rows, players):
    pid = int(forecast['player_id'])
    if pd.Timestamp(forecast['predicted_at']) >= pd.Timestamp(forecast['game_start']):
        return {'status': 'rejected_late'}
    own = players[players.PLAYER_ID == pid]
    if own.empty:
        return {'status': 'missing_player'}
    if float(own.MIN.iloc[0]) <= 0:
        return {'status': 'dnp'}
    actual = rows[rows.personIdOff == pid]
    if actual.empty:
        return {'status': 'missing_matchups'}
    if set(actual.defTeamId.astype(int)) != {int(forecast['opponent_id'])}:
        return {'status': 'rejected_identity'}
    totals = actual.groupby('personIdDef').partialPossessions.sum().sort_values(ascending=False)
    if len(totals) > 1 and totals.iloc[0] == totals.iloc[1]:
        return {'status': 'ambiguous_tie'}
    target = int(totals.index[0])
    return {'status': 'settled', 'target': target,
            'correct': target == int(forecast['defender_id']),
            'baseline_correct': target == int(forecast['baseline_defender_id'])}


def combined_stats(held, graded):
    def key(row):
        return str(row['game_id']), int(row['player_id'])
    old = {key(row): row for row in held}
    new = {key(row): row for row in graded if row['status'] == 'settled'}
    if len(old) != len(held) or len(new) != sum(row['status'] == 'settled' for row in graded):
        raise ValueError('Duplicate defender forecast identity')
    union = {**new, **old}
    stats = summarize(list(union.values()))
    passed = stats['n'] >= 100 and stats['accuracy'] >= .5 and stats['accuracy'] >= stats['minutes_baseline_accuracy']+.05
    return {'heldout': summarize(list(old.values())), 'prospective': summarize(list(new.values())),
            'deduplicated_union': stats, 'assignment_gates_passed': bool(passed),
            'seven_features_qualified': False,
            'note': 'Assignment gates alone do not certify seven-field coverage, dated training support or train/serve parity.'}


def refresh_boxes(run, season):
    from nba_api.stats.endpoints import leaguegamelog
    response = leaguegamelog.LeagueGameLog(player_or_team_abbreviation='P', season=season,
                                         season_type_all_star='Regular Season', timeout=15)
    save_new(run/'player_logs_raw.json', {'observed_at': timestamp(), 'payload': response.get_dict()})
    frame = response.get_data_frames()[0]
    frame['GAME_ID'] = frame.GAME_ID.astype(str).str.zfill(10)
    frame['GAME_DATE'] = pd.to_datetime(frame.GAME_DATE).dt.strftime('%Y-%m-%d')
    if frame.duplicated(['GAME_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate source boxscore identities')
    cache = Path('logs/points_pilot_cache')/f'players_{season}.json'
    old = pd.read_json(cache, dtype={'GAME_ID': str})
    save_new(run/'prior_player_cache.json', json.loads(cache.read_text()))
    completed, failures = [], []
    for gid, part in frame.groupby('GAME_ID'):
        if gid in set(old.GAME_ID):
            continue
        try:
            summary = game_summary(gid)
            save_new(run/f'{gid}_summary.json', {'observed_at': timestamp(), 'payload': summary})
            teams = {int(summary[s]['teamId']) for s in ('homeTeam', 'awayTeam')}
            if int(summary.get('gameStatus', 0)) != 3:
                continue
            if not gid.startswith('002'+season[2:4]) or set(part.TEAM_ID.astype(int)) != teams:
                raise ValueError('Wrong game/season/team identity')
            completed.append(part)
        except Exception as exc:
            failures.append({'game_id': gid, 'stage': 'boxes', 'error_type': type(exc).__name__})
    merged = pd.concat([old, *completed], ignore_index=True)
    merged['GAME_DATE'] = pd.to_datetime(merged.GAME_DATE).dt.strftime('%Y-%m-%d')
    if completed:
        tmp = cache.with_suffix('.tmp')
        tmp.write_text(merged.to_json(orient='records'))
        tmp.replace(cache)
    return merged, failures


def parse_matchup_response(payload, gid):
    """Archive the raw body before parsing; an unpublished table is not a loss."""
    from nba_api.stats.library.http import NBAStatsResponse
    body=payload['boxScoreMatchups']
    if str(body['gameId'])!=str(gid):raise ValueError('Wrong matchup response identity')
    if not body['homeTeam']['players'] and not body['awayTeam']['players']:return None
    sets=NBAStatsResponse(json.dumps(payload),200,'https://stats.nba.com/stats/boxscorematchupsv3').get_data_sets('boxscorematchupsv3')
    return pd.DataFrame(sets['PlayerStats']['data'],columns=sets['PlayerStats']['headers'])


def advance_history(state, later, matchup_dir):
    """Fail closed for forecasting while allowing collection/grading reports."""
    missing = sorted(str(gid) for gid in later.GAME_ID.unique()
                     if not (matchup_dir/f'{gid}.json.gz').exists())
    if missing:
        return missing
    for day, boxes in later.groupby('GAME_DATE', sort=True):
        rows = []
        for gid in boxes.GAME_ID.unique():
            with gzip.open(matchup_dir/f'{gid}.json.gz', 'rt') as handle:
                obj = json.load(handle)
            if obj['game_id'] != gid:
                raise ValueError('Wrong cached matchup game')
            rows.extend(obj['rows'])
        state.observe_matchups(pd.DataFrame(rows), day)
        state.observe_boxes(boxes, day)
    return []


def main():
    from src.current_rosters import load_snapshot, refresh_rosters
    with (ROOT/'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frozen = verify_frozen()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        run = ROOT/'monitor_runs'/stamp
        run.mkdir(parents=True)
        roster = load_snapshot() or refresh_rosters()
        save_new(run/'roster.json', roster)
        logs, failures = refresh_boxes(run, roster['season'])
        MATCHUPS.mkdir(exist_ok=True)
        for gid, players in logs.groupby('GAME_ID'):
            path = MATCHUPS/f'{gid}.json.gz'
            if path.exists():
                continue
            try:
                payload = nba_json('boxscorematchupsv3',GameID=gid)
                save_new(run/f'{gid}_matchups_raw.json', {'observed_at': timestamp(), 'payload': payload})
                frame = parse_matchup_response(payload,gid)
                if frame is None:
                    failures.append({'game_id':gid,'stage':'matchups','error_type':'SourceNotPublished'})
                    continue
                rows = validate(frame, players, gid)
                obj = {'game_id': gid, 'game_date': str(players.GAME_DATE.iloc[0])[:10],
                       'observed_at': timestamp(), 'source': 'NBA BoxScoreMatchupsV3', 'rows': rows}
                with gzip.open(path, 'xt') as handle:
                    json.dump(obj, handle)
            except Exception as exc:
                failures.append({'game_id': gid, 'stage': 'matchups', 'error_type': type(exc).__name__})

        forecasts = {}
        for path in sorted(ROOT.glob('shadow_*.json')):
            for row in json.loads(path.read_text()):
                if row['model_sha256'] != frozen['model_sha256'] or row['threshold'] != frozen['selection']['threshold']:
                    raise ValueError('Shadow forecast uses a different frozen model/cutoff')
                forecasts.setdefault((row['game_id'], row['player_id']), row)
        graded = []
        for (gid, pid), f in forecasts.items():
            outcome_path = ROOT/'prospective_settlements'/f'{gid}_{pid}.json'
            if outcome_path.exists():
                graded.append(json.loads(outcome_path.read_text()))
                continue
            path = MATCHUPS/f'{gid}.json.gz'
            result = {'status': 'pending'}
            if gid in set(logs.GAME_ID):
                result = {'status': 'missing_matchups'}
                if path.exists():
                    obj = json.load(gzip.open(path, 'rt'))
                    if obj['game_id'] != gid:
                        raise ValueError('Wrong cached matchup game')
                    result = grade_one(f, pd.DataFrame(obj['rows']), logs[logs.GAME_ID == gid])
            row = {**f, **result, 'graded_at': timestamp()}
            if result['status'] in ('settled', 'dnp'):
                save_new(outcome_path, row)
            graded.append(row)

        # Reconstruct from immutable Oct24 state; ingest each subsequent earlier
        # day once, matchups before boxes. Same-day results are never features.
        base = joblib.load(ROOT/'state.joblib')
        state = AssignmentState()
        state.boxes.update({k: deque(v, maxlen=10) for k, v in base['boxes'].items()})
        for key in ('latest', 'pairs', 'guards'):
            getattr(state, key).update(base[key])
        today = pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()
        later = logs[(pd.to_datetime(logs.GAME_DATE) > pd.Timestamp(base['through'])) & (pd.to_datetime(logs.GAME_DATE) < today)]
        missing_history = advance_history(state, later, MATCHUPS)
        board = (nba_json('scoreboardv3', GameDate=str(today.date()), LeagueID='00')
                 if not missing_history else {'scoreboard': {'games': []},
                     'forecast_blocked_missing_history': missing_history})
        save_new(run/'scoreboard.json', {'observed_at': timestamp(), 'payload': board})
        model = joblib.load(ROOT/'model.joblib')
        shadow, rejected = [], []
        for game in board['scoreboard']['games']:
            gid, start = str(game['gameId']), game['gameTimeUTC']
            if pd.Timestamp(start) <= pd.Timestamp.now(tz='UTC'):
                continue
            teams = [int(game[s]['teamId']) for s in ('homeTeam', 'awayTeam')]
            for player in roster['players']:
                pid, team = player.get('nba_id'), int(player['team_id'])
                if pid is None or team not in teams or (gid, pid) in forecasts:
                    continue
                opp = next(t for t in teams if t != team)
                ids, features = state.candidates(pid, opp, today)
                reason = 'insufficient_prior_history'
                if ids:
                    scores = model.predict_proba(np.asarray(features, dtype=np.float32))[:, 1]
                    index = int(np.argmax(scores))
                    reason = 'below_frozen_cutoff'
                    if scores[index] >= frozen['selection']['threshold']:
                        generated = timestamp()
                        if pd.Timestamp(generated) >= pd.Timestamp(start):
                            reason = 'tipoff_passed'
                        else:
                            shadow.append({'game_id': gid, 'player_id': int(pid), 'opponent_id': opp,
                                'defender_id': int(ids[index]), 'baseline_defender_id': int(ids[0]),
                                'predicted_at': generated, 'game_start': start,
                                'model_sha256': frozen['model_sha256'], 'source_sha256': frozen['source_sha256'],
                                'threshold': frozen['selection']['threshold'], 'score': float(scores[index]),
                                'status': 'shadow_only_unqualified', 'history_through': str(logs[pd.to_datetime(logs.GAME_DATE) < today].GAME_DATE.max()),
                                'roster_fetched_at': roster['fetched_at'], 'source_run': str(run)})
                            continue
                rejected.append({'game_id': gid, 'player_id': int(pid), 'reason': reason})
        if shadow:
            save_new(ROOT/f'shadow_{stamp}.json', shadow)
        save_new(run/'rejected_forecasts.json', rejected)
        held = json.loads((ROOT/'holdout.json').read_text())['predictions']
        report = combined_stats(held, graded)
        report.update(observed_at=timestamp(), new_shadow_forecasts=len(shadow),
                      forecast_blocked_missing_history=missing_history,
                      statuses={s: sum(r['status'] == s for r in graded) for s in sorted({r['status'] for r in graded})},
                      total_prospective_forecasts=len(graded)+len(shadow), failures=failures,
                      cached_games=int(logs.GAME_ID.nunique()), last_boxscore_date=str(logs.GAME_DATE.max()),
                      model_sha256=frozen['model_sha256'])
        save_new(run/'report.json', report)
        temp = ROOT/'monitor_report.tmp'
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(ROOT/'monitor_report.json')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
