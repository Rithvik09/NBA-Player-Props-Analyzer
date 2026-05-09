"""Backfill the game_broadcasts table from nba_api ScoreboardV2.

Walks every NBA game day in the requested seasons and stores national-TV
+ home / away broadcaster info per game_id. Trainer reads this to set
the national_tv_game feature instead of the previous hardcoded 0.

Cost: ~1 ScoreboardV2 call per game day × ~180 days/season × 0.7s
≈ 2 min per season. Idempotent — running twice overwrites.

Usage
-----
    python -m scripts.backfill_game_broadcasts --seasons 2024-25
    python -m scripts.backfill_game_broadcasts --seasons 2023-24,2024-25
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.precompute_jobs import (  # noqa: E402
    compute_game_broadcasts_for_dates,
    upsert_game_broadcasts,
    ensure_tables,
)

log = logging.getLogger("backfill_game_broadcasts")


def _season_dates(season: str) -> list[str]:
    """Generate one date string per day from late October through mid June
    of the given season (covers regular season + playoffs).

    Off-days are fine — ScoreboardV2 just returns no games for those
    dates and we skip cleanly.
    """
    # Season string is "YYYY-YY", e.g. "2024-25" → starts Oct 2024.
    start_year = int(season.split("-")[0])
    start = date(start_year, 10, 15)
    end = date(start_year + 1, 6, 25)
    out = []
    cur = start
    while cur <= end:
        out.append(cur.isoformat())
        cur = cur + timedelta(days=1)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="basketball_data.db")
    ap.add_argument("--seasons", required=True,
                    help="Comma-separated season strings, e.g. 2023-24,2024-25")
    ap.add_argument("--sleep", type=float, default=0.7,
                    help="Sleep between scoreboard calls")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        log.error("DB not found: %s", args.db)
        return 1

    conn = sqlite3.connect(args.db)
    ensure_tables(conn)

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    log.info("seasons=%s sleep=%.2f", seasons, args.sleep)

    total = 0
    for s in seasons:
        dates = _season_dates(s)
        log.info("[%s] walking %d days", s, len(dates))
        t0 = time.time()
        rows = compute_game_broadcasts_for_dates(dates, sleep_between=args.sleep)
        log.info("[%s] %d game-broadcast rows in %.0fs",
                 s, len(rows), time.time() - t0)
        upsert_game_broadcasts(conn, rows)
        total += len(rows)

    natl_count = conn.execute(
        "SELECT COUNT(*) FROM game_broadcasts WHERE natl_tv IS NOT NULL"
    ).fetchone()[0]
    total_count = conn.execute("SELECT COUNT(*) FROM game_broadcasts").fetchone()[0]
    log.info("done. total broadcasts in table: %d (national-TV: %d)",
             total_count, natl_count)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
