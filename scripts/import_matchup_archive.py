"""Validate public Parquet matchups against local NBA boxscores before import.

Requires optional pyarrow. Downloads are deliberately separate from this tool.
Existing official observations are compared, never overwritten.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
from scripts.crawl_defender_matchups import validate, COLUMNS

MAPPING = dict(zip(
    ['person_id', 'matchups_person_id', 'team_id', 'partial_possessions',
     'switches_on', 'player_points', 'matchup_field_goals_made',
     'matchup_field_goals_attempted', 'matchup_minutes_sort'], COLUMNS))


def import_files(paths, cache=Path('logs/points_pilot_cache'),
                 output=Path('logs/source_blocked/matchups')):
    logs = pd.concat([pd.read_json(p, dtype={'GAME_ID': str})
                      for p in cache.glob('*.json')
                      if p.name.startswith(('players_', 'playoffs_', 'playin_'))])
    games = dict(tuple(logs.groupby('GAME_ID')))
    output.mkdir(parents=True, exist_ok=True)
    report = {'files': [], 'imported': 0, 'matched_existing': 0, 'rejected': []}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_parquet(path).rename(columns=MAPPING)
        frame['gameId'] = frame.game_id.map(lambda value: str(int(value)).zfill(10))
        report['files'].append({'path': str(path), 'sha256': digest, 'rows': len(frame),
                                'nonzero_switches': int((frame.switchesOn > 0).sum())})
        for gid, part in frame.groupby('gameId'):
            try:
                if gid not in games:
                    raise ValueError('No local NBA boxscore for identity validation')
                players = games[gid]
                rows = validate(part, players, gid)
                keys = [(r['personIdOff'], r['personIdDef']) for r in rows]
                if len(keys) != len(set(keys)):
                    raise ValueError('Duplicate matchup identities')
                target = output / (gid + '.json.gz')
                if target.exists():
                    with gzip.open(target, 'rt') as handle:
                        existing = json.load(handle)['rows']
                    order = lambda r: (r['personIdOff'], r['personIdDef'])
                    if sorted(existing, key=order) != sorted(rows, key=order):
                        raise ValueError('Archive differs from existing official observation')
                    report['matched_existing'] += 1
                    continue
                obj = {'game_id': gid, 'game_date': str(pd.Timestamp(players.GAME_DATE.iloc[0]).date()),
                       'source': 'NBA BoxScoreMatchupsV3 via cdechoch/nba-data-archive',
                       'archive_sha256': digest, 'rows': rows}
                temp = target.with_suffix('.tmp')
                with gzip.open(temp, 'wt') as handle:
                    json.dump(obj, handle)
                temp.replace(target)
                report['imported'] += 1
            except (ValueError, KeyError, TypeError) as exc:
                report['rejected'].append({'game_id': gid, 'reason': str(exc)})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    result = import_files(args.paths)
    Path('logs/source_blocked/archive_import.json').write_text(json.dumps(result, indent=2))
    print({k: v if k != 'rejected' else len(v) for k, v in result.items()})
