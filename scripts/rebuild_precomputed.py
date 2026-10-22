"""Resume the local historical crawl and verify its feature-source coverage.

Player tables retain season keys. Team/opponent tables remain snapshots and
are overwritten by each later season; this is not point-in-time training data.
Models are never trained or replaced by this command.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import db, precompute_jobs as jobs

# table, collector, writer, seasonal, required rows (0 means optional source)
PLAN = [
    ('dvp_by_position', 'scrape_dvp_by_position', 'upsert_dvp', False, 0),
    ('team_special_defenders', 'compute_special_defenders', 'upsert_defenders', False, 0),
    ('referee_stats', 'compute_referee_stats_from_history', 'upsert_ref_stats', False, 0),
    ('team_stats', 'compute_team_stats', 'upsert_team_stats', False, 30),
    ('dvp_rolling', 'compute_rolling_dvp', 'upsert_rolling_dvp', False, 0),
    ('player_advanced_stats', 'compute_advanced_player_stats', 'upsert_advanced_player_stats', True, 1),
    ('player_clutch_stats', 'compute_clutch_stats', 'upsert_clutch_stats', True, 1),
    ('player_hustle_stats', 'compute_hustle_stats', 'upsert_hustle_stats', True, 1),
    ('player_shot_profile', 'compute_shot_profile', 'upsert_shot_profile', True, 1),
    ('player_play_types', 'compute_synergy_play_types', 'upsert_play_types', True, 1),
    ('player_on_off', 'compute_on_off_ratings', 'upsert_on_off', True, 1),
    ('player_shot_zones', 'compute_shot_zone_breakdown', 'upsert_shot_zones', True, 1),
    ('player_quarter_splits', 'compute_quarter_splits', 'upsert_quarter_splits', True, 1),
    ('team_opp_shot_zones', 'compute_opp_shot_zones', 'upsert_opp_shot_zones', False, 30),
    ('team_synergy_defense', 'compute_synergy_team_defense', 'upsert_synergy_defense', False, 30),
    ('player_tracking_stats', 'compute_player_tracking', 'upsert_player_tracking', True, 1),
    ('team_standings', 'compute_team_standings', 'upsert_team_standings', False, 30),
    ('player_scoring_breakdown', 'compute_player_scoring_breakdown', 'upsert_player_scoring_breakdown', True, 1),
    ('player_vs_opponent', 'compute_player_vs_opponent', 'upsert_player_vs_opponent', False, 1),
    ('player_yoy_stats', 'compute_player_yoy', 'upsert_player_yoy', True, 1),
    ('team_rest_splits', 'compute_team_rest_splits', 'upsert_team_rest_splits', False, 30),
    ('team_home_away_splits', 'compute_team_home_away_splits', 'upsert_team_home_away_splits', False, 30),
    ('team_lineup_stats', 'compute_lineup_stats', 'upsert_lineup_stats', False, 30),
    ('team_injury_status', 'compute_injury_status', 'upsert_injury_status', False, 0),
]
REVISIONS = {'player_on_off': 2, 'player_play_types': 2, 'team_synergy_defense': 2, 'player_shot_zones': 2}

PLAYER_TABLES = tuple(table for table, _, _, seasonal, _ in PLAN if seasonal)


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)


def configure_http():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from nba_api.library.http import NBAHTTP
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(
        total=1, connect=1, read=0, status=1, backoff_factor=1., status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=('GET',))))
    NBAHTTP.set_session(session)


def write_rows(writer, rows, season):
    kwargs = {}
    params = inspect.signature(writer).parameters
    if 'updated_at' in params:
        kwargs['updated_at'] = int(time.time())
    if 'season' in params:
        kwargs['season'] = season
    with db.connect() as conn:
        writer(conn, rows, **kwargs)
        if writer is jobs.upsert_team_stats:
            jobs.upsert_team_foul_rates(conn, rows, updated_at=int(time.time()))


def run_stage(spec, season, stages, attempts):
    table, collector_name, writer_name, seasonal, required = spec
    if table == 'team_injury_status':
        return {'status': 'unavailable', 'rows': 0,
                'reason': 'Current injuries cannot be backfilled into historical seasons.'}
    previous = stages.get(table, {})
    if previous.get('status') == 'complete' and previous.get('revision') == REVISIONS.get(table):
        print(f'[rebuild] reuse {season} {table}: {previous["rows"]} rows', flush=True)
        return previous
    collector, writer = getattr(jobs, collector_name), getattr(jobs, writer_name)
    # A single-snapshot DB table may now contain a later season. Only
    # explicitly checkpointed rows can be reused as this season's partials.
    collected = {r['team_id']: r for r in previous.get('partial_rows', [])}
    error = None
    for attempt in range(1, attempts + 1):
        print(f'[rebuild] {season} {table} attempt {attempt}', flush=True)
        try:
            kwargs = {'season': season} if 'season' in inspect.signature(collector).parameters else {}
            if table == 'team_rest_splits' and collected:
                from nba_api.stats.static import teams
                kwargs['team_ids'] = [t['id'] for t in teams.get_teams() if t['id'] not in collected]
            rows = collector(**kwargs)
            if table == 'team_rest_splits':
                collected.update({r['team_id']: r for r in rows})
                rows = list(collected.values())
            if rows:
                write_rows(writer, rows, season)
            count = len(rows)
            if count >= required and count > 0:
                return {'status': 'complete', 'rows': count, 'attempts': attempt, 'revision': REVISIONS.get(table)}
            error = 'empty_or_partial_source'
            previous = {'status': 'incomplete', 'rows': count, 'attempts': attempt, 'error': error}
            # Optional sources can be unavailable; record it without fake rows.
            if required == 0:
                return previous
        except Exception as exc:
            error = type(exc).__name__
            print(f'[rebuild] {season} {table} failed: {error}', flush=True)
        time.sleep(2.)
    return {**previous, 'status': 'incomplete', 'error': error, 'attempts': attempts,
            **({'partial_rows': list(collected.values())} if collected else {})}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seasons', default='2022-23,2023-24,2024-25,2025-26')
    ap.add_argument('--report', default='logs/rebuild_coverage.json')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--attempts', type=int, default=3)
    args = ap.parse_args()
    configure_http()
    path = Path(args.report)
    report = json.loads(path.read_text()) if args.resume and path.exists() else {}
    with db.connect() as conn:
        jobs.ensure_tables(conn)
    seasons = list(map(str.strip, args.seasons.split(',')))
    repair_earlier = any(report.get(s, {}).get('status') != 'complete' or any(report.get(s, {}).get('stages', {}).get(t, {}).get('revision') != v for t, v in REVISIONS.items()) for s in seasons[:-1])
    for season in seasons:
        entry = report.setdefault(season, {'status': 'running', 'stages': {}})
        stages = entry.setdefault('stages', {})
        if season == seasons[-1] and repair_earlier and entry.get('status') == 'complete':
            # Repairing an older single-snapshot source overwrites the DB.
            # Restore the newest snapshot before reporting the run complete.
            for table, _, _, seasonal, _ in PLAN:
                if not seasonal:
                    stages.pop(table, None)
            entry['status'] = 'running'
        if entry.get('status') == 'complete' and all(stages.get(t, {}).get('revision') == version for t, version in REVISIONS.items()):
            print(f'[rebuild] previously verified {season}', flush=True)
            continue
        entry['status'] = 'running'
        # Existing seasonal rows are safe to reuse even without a checkpoint.
        if args.resume:
            with db.connect() as conn:
                for table in PLAYER_TABLES:
                    count = conn.execute(f'SELECT count(*) FROM {table} WHERE season = %s', (season,)).fetchone()[0]
                    if count and table not in stages:
                        stages[table] = {'status': 'complete', 'rows': count, 'reused_seasonal_rows': True}
        for spec in PLAN:
            table = spec[0]
            stages[table] = run_stage(spec, season, stages, max(1, args.attempts))
            save_report(path, report)
        with db.connect() as conn:
            entry['coverage'] = {table: conn.execute(f'SELECT count(*) FROM {table} WHERE season = %s', (season,)).fetchone()[0] for table in PLAYER_TABLES}
        missing = [table for table, _, _, _, required in PLAN if required and stages[table]['status'] != 'complete']
        entry['incomplete_required_sources'] = missing
        entry['unavailable_optional_sources'] = [t for t, _, _, _, required in PLAN if not required and stages[t]['status'] != 'complete']
        entry['status'] = 'incomplete' if missing else 'complete'
        save_report(path, report)
        print(f'[rebuild] {season}: {entry["status"]}; required gaps={missing}; coverage={entry["coverage"]}', flush=True)
    if any(entry['status'] != 'complete' for entry in report.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
