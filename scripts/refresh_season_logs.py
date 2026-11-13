"""Refresh one season's player game-log cache without quietly replacing good data.

There was no committed entry point for this. Every reference to
logs/points_pilot_cache/players_<season>.json reads it, and
crawl_postseason_features.py writes only the playoffs and play-in files, over a
hardcoded season tuple that stops before the live season. So the live season's
cache was produced ad hoc and had nothing to re-run.

Every check here guards one failure: a response that looks plausible and is
worse than what is already on disk. A truncated crawl, a renamed column, a
duplicated player-game, and a study pin voided in passing are each
indistinguishable from a good refresh once the bytes are written.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = 'logs/points_pilot_cache'
# input_hashes included deliberately: the expanded_points manifests pin their data
# inputs under that name, and nothing else in the repo checks that key.
PIN_KEYS = ('input_hashes', 'source_hashes', 'source_sha256')


def fetch_regular_season(season):
    """The live call. Imported lazily so this module is testable without nba_api."""
    from nba_api.stats.endpoints import leaguegamelog
    return leaguegamelog.LeagueGameLog(player_or_team_abbreviation='P', season=season,
                                       season_type_all_star='Regular Season',
                                       timeout=60).get_data_frames()[0]


def cache_path(season):
    return ROOT/CACHE/f'players_{season}.json'


def expected_columns(season):
    """The column order readers already consume: this season's file, else another's.

    Taken from disk rather than hardcoded, so the schema check cannot drift away
    from what the existing caches actually hold.
    """
    target = cache_path(season)
    if target.exists():
        return list(pd.read_json(target, dtype={'GAME_ID': str}).columns)
    others = sorted(p for p in (ROOT/CACHE).glob('players_*.json') if p != target)
    if not others:
        raise ValueError(f'No cache under {ROOT/CACHE} to take the schema from')
    return list(pd.read_json(others[-1], dtype={'GAME_ID': str}).columns)


def holders_of_a_valid_pin(path):
    """Studies whose recorded digest for this file still holds, so a write voids it."""
    from src.prospective_points import digest
    if not path.exists():
        return []
    current = digest(path)
    try:
        keys = {str(path), path.relative_to(ROOT).as_posix()}
    except ValueError:
        keys = {str(path)}
    holders = []
    for name in ('protocol.json', 'manifest.json'):
        for record in sorted((ROOT/'logs').glob('*/'+name)):
            try:
                body = json.loads(record.read_text())
            except json.JSONDecodeError:
                continue
            if not isinstance(body, dict):
                continue
            for key in PIN_KEYS:
                pins = body.get(key)
                if isinstance(pins, dict) and any(pins.get(k) == current for k in keys):
                    holders.append(f'{record.parent.name}/{record.name} [{key}]')
    return holders


def normalized(frame, columns):
    """One on-disk shape regardless of what the provider returns this time."""
    missing = [c for c in columns if c not in frame.columns]
    extra = [c for c in frame.columns if c not in columns]
    if missing or extra:
        raise ValueError(f'Schema changed; missing={missing} unexpected={extra}')
    frame = frame[columns].copy()
    frame['SEASON_ID'] = frame['SEASON_ID'].astype(str)
    frame['GAME_ID'] = frame['GAME_ID'].astype(str)
    frame['GAME_DATE'] = pd.to_datetime(frame['GAME_DATE']).dt.strftime('%Y-%m-%d')
    return frame


def refresh(season, fetch=fetch_regular_season, allow_shrink=False, break_pins=False):
    """Fetch one season's regular-season player logs and replace its cache."""
    target = cache_path(season)
    columns = expected_columns(season)
    fetched = fetch(season)
    if fetched is None or len(fetched) == 0:
        raise ValueError(f'Provider returned no rows for {season}; the cache is left alone')
    frame = normalized(fetched, columns)
    duplicated = frame.duplicated(['PLAYER_ID', 'GAME_ID'])
    if duplicated.any():
        # load_season_logs refuses duplicates, so writing them poisons every reader.
        sample = frame.loc[duplicated, ['PLAYER_ID', 'GAME_ID']].head(3).to_dict('records')
        raise ValueError(f'Duplicate player-game rows for {season}: {sample}')
    if target.exists():
        had = len(pd.read_json(target, dtype={'GAME_ID': str}))
        if len(frame) < had and not allow_shrink:
            raise ValueError(f'Refusing {len(frame)} rows where the cache holds {had}: fewer '
                             'player-games than on disk looks exactly like a truncated crawl. '
                             'Pass allow_shrink=True (--allow-shrink) if the loss is real.')
        holders = holders_of_a_valid_pin(target)
        if holders and not break_pins:
            raise ValueError(f'{target.name} is pinned, and the pin still holds, by: '
                             +', '.join(holders)+'. Refreshing it voids that evidence. Pass '
                             'break_pins=True (--break-pins) and close or re-pin those studies.')
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and moved, so an interrupted write cannot leave
    # a half-file where every reader expects a season of games.
    staging = target.with_suffix('.json.partial')
    staging.write_text(frame.to_json(orient='records'))
    staging.replace(target)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--season', required=True, help="e.g. 2026-27")
    p.add_argument('--allow-shrink', action='store_true', help='accept fewer rows than on disk')
    p.add_argument('--break-pins', action='store_true', help='refresh even though a study pin still holds')
    a = p.parse_args()
    before = cache_path(a.season)
    had = len(pd.read_json(before, dtype={'GAME_ID': str})) if before.exists() else 0
    out = refresh(a.season, allow_shrink=a.allow_shrink, break_pins=a.break_pins)
    now = pd.read_json(out, dtype={'GAME_ID': str})
    print(json.dumps({'season': a.season, 'path': str(out), 'rows_before': had,
                      'rows_after': len(now), 'games': int(now['GAME_ID'].nunique()),
                      'dates': [str(now['GAME_DATE'].min()), str(now['GAME_DATE'].max())]}, indent=2))


if __name__ == '__main__':
    main()
