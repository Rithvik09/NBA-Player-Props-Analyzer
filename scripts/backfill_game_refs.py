"""Resumable referee backfill. Unique per-game observations prevent double counting.

    python -m scripts.backfill_game_refs --seasons 2022-23,2023-24,2024-25,2025-26

Officials and advanced totals are cached independently. Team fouls come from one
league-wide game log per season. Incomplete games are retried and never defaulted.
Historical training tendencies use only games before the prediction's calendar day.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

from src import db
from src.nba_officials import nba_json
from src.precompute_jobs import ensure_tables
from src.referee_history import ensure_history, validate_game, save_game, rebuild_aggregates

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'logs/referee_cache'
REPORT = ROOT / 'logs/referee_backfill_coverage.json'


def cached_json(path, fetch):
    if path.exists():
        return json.loads(path.read_text())
    data = fetch()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, default=str))
    temporary.replace(path)
    return data


def league_games(season):
    from nba_api.stats.endpoints import leaguegamelog
    rows = cached_json(CACHE / f'games_{season}.json', lambda: leaguegamelog.LeagueGameLog(
        season=season,player_or_team_abbreviation='T',season_type_all_star='Regular Season',timeout=30
    ).get_data_frames()[0].to_dict('records'))
    games = {}
    for row in rows:
        gid = str(row['GAME_ID'])
        if gid.startswith('002'):
            games.setdefault(gid, []).append(row)
    if not games or any(len(rows) != 2 for rows in games.values()):
        raise ValueError('Incomplete league-wide game log')
    return games


def league_advanced(season):
    from nba_api.stats.endpoints import teamgamelogs
    rows = cached_json(CACHE / f'advanced_{season}.json', lambda: teamgamelogs.TeamGameLogs(
        season_nullable=season,season_type_nullable='Regular Season',
        measure_type_player_game_logs_nullable='Advanced',league_id_nullable='00',timeout=30
    ).get_data_frames()[0].to_dict('records'))
    games = {}
    for row in rows:
        games.setdefault(str(row['GAME_ID']), []).append({
            'teamId':int(row['TEAM_ID']), 'pace':float(row['PACE']),
            'offensiveRating':float(row['OFF_RATING']), 'defensiveRating':float(row['DEF_RATING'])})
    return games


def collect_game(gid, rows, advanced, delay):
    def fetch():
        time.sleep(delay)
        return nba_json('boxscoresummaryv3', GameID=gid)
    summary = cached_json(CACHE / f'{gid}_summary.json', fetch)['boxScoreSummary']
    return validate_game(gid, rows, summary, advanced)


def run(seasons, *, dsn=None, workers=3, delay=.35, max_games=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    report = {'started_at': time.time(), 'seasons': {}, 'complete': False}
    with db.connect(dsn) as conn:
        ensure_tables(conn)
        ensure_history(conn)
        conn.commit()
        for season in seasons:
            games = league_games(season)
            advanced = league_advanced(season)
            if set(games) - set(advanced):
                raise ValueError("Advanced game-log coverage incomplete")
            if max_games:
                games = dict(list(sorted(games.items()))[:max_games])
            existing = {r[0] for r in conn.execute('SELECT game_id FROM referee_game_stats WHERE season=%s', (season,))}
            conn.commit()
            pending = {gid: rows for gid, rows in games.items() if gid not in existing}
            state = {'expected_games':len(games), 'completed_games':len(set(games) & existing), 'failures': {}}
            report['seasons'][season] = state
            for attempt in range(3):
                failed = {}
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    tasks = {pool.submit(collect_game,gid,rows,advanced[gid],delay):gid for gid,rows in pending.items()}
                    for task in as_completed(tasks):
                        gid = tasks[task]
                        try:
                            game = task.result()
                            save_game(conn,season,game)
                            state['completed_games'] += 1
                        except Exception as exc:
                            failed[gid] = games[gid]
                            state['failures'][gid] = f'{type(exc).__name__}: {str(exc)[:160]}'
                        if (state['completed_games'] % 50 == 0) or len(tasks) < 10:
                            state['failures'] = {g: e for g,e in state['failures'].items() if g in failed}
                            REPORT.write_text(json.dumps(report,indent=2))
                            print(season, state['completed_games'], '/',state['expected_games'], 'retry',attempt, flush=True)
                pending = failed
                if not pending:
                    state['failures'] = {}
                    break
                time.sleep(2)
            state['complete'] = state['completed_games'] == state['expected_games']
            rebuild_aggregates(conn)
            REPORT.write_text(json.dumps(report,indent=2))
        report['complete'] = all(s['complete'] for s in report['seasons'].values())
        report['finished_at'] = time.time()
        report['referees'] = conn.execute('SELECT COUNT(*) FROM referee_stats').fetchone()[0]
        report['assignments'] = conn.execute('SELECT COUNT(*) FROM game_officials').fetchone()[0]
        report['historical_feature_rows'] = conn.execute('SELECT COUNT(*) FROM referee_game_features').fetchone()[0]
        REPORT.write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seasons',required=True)
    parser.add_argument('--db',default=None)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--sleep',type=float,default=.35)
    parser.add_argument('--max-games',type=int,default=None)
    args=parser.parse_args()
    report=run(args.seasons.split(','),dsn=args.db,workers=args.workers,delay=args.sleep,max_games=args.max_games)
    print(json.dumps(report,indent=2))
    return 0 if report['complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
