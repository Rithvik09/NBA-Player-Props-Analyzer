"""Refresh current rosters, injuries and officials without rerunning historical stats.

Run with DATABASE_URL set to persist team injury summaries as well as snapshots:
    python -m scripts.refresh_current_context
Current injury reports are deliberately excluded from historical training.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from src.current_rosters import refresh_rosters
from src.live_injuries import fetch_live_injuries
from src.live_officials import refresh_officials


def main():
    report = {'refreshed_at': datetime.now(timezone.utc).isoformat()}
    roster = refresh_rosters()
    report['rosters'] = {k: v for k, v in roster.items() if k != 'players'}
    report['rosters']['player_count'] = len(roster['players'])
    injuries = fetch_live_injuries()
    report['injuries'] = {'source': 'ESPN JSON with fallback', 'reported_teams': len(injuries),
                          'players': sum(map(len, injuries.values()))}
    root = Path(__file__).resolve().parents[1]
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs/current_injuries.json').write_text(json.dumps(injuries, indent=2))
    if os.environ.get('DATABASE_URL'):
        from src import db
        from src.precompute_jobs import compute_injury_status, upsert_injury_status, ensure_tables
        with db.connect() as conn:
            ensure_tables(conn)
            rows = compute_injury_status()
            if len(rows) != 30:
                raise ValueError('Injury refresh did not cover all 30 teams')
            upsert_injury_status(conn, rows, int(time.time()))
            report['injuries']['persisted_team_summaries'] = len(rows)
    officials = refresh_officials(days=3)
    report['officials'] = {'games_checked': len(officials['games']),
                           'games_with_crews': sum(bool(g['officials']) for g in officials['games']),
                           'games_without_crews': [g['espn_game_id'] for g in officials['games'] if not g['officials']],
                           'note': 'Crews are distinct from historical referee tendency statistics.'}
    (root / 'logs/current_context_refresh.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
