"""Backfill the game_officials + referee_stats tables from NBA-API.

Walks every regular-season game of the requested seasons via
``BoxScoreSummaryV2`` (officials), ``BoxScoreTraditionalV2`` (fouls),
and ``BoxScoreAdvancedV3`` (pace), then aggregates per-ref stats and
writes the lookup table the trainer reads.

Why this script exists
----------------------
The previous data source (basketball-reference's /referees/ page)
turned out to be a directory only — names + first/last game date,
no foul-rate / pace / home-bias stats. Without a real data source,
the ref_* features were league-mean constants → zero importance in
trained models.

This script is the new data source. NBA-API's BoxScoreSummary
endpoint exposes per-game officials, and the per-team box scores
give us fouls + pace. Aggregating across enough games yields stable
per-ref stats.

Cost
----
3 API calls per game × ~1230 regular-season games per season ×
~0.7s sleep per call ≈ 75 min per season. Run once per off-season
plus an incremental top-up nightly.

Usage
-----
    # Backfill the current season:
    python -m scripts.backfill_game_refs --seasons 2024-25

    # Multiple seasons:
    python -m scripts.backfill_game_refs --seasons 2023-24,2024-25

    # Smoke test (10 games only):
    python -m scripts.backfill_game_refs --seasons 2024-25 --max-games 10

The script is idempotent — running it twice on the same season just
overwrites the same rows. ``referee_stats`` aggregates are
recomputed across ALL game_officials rows currently in the table,
so seasons accumulate.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.precompute_jobs import (  # noqa: E402
    compute_game_refs_for_season,
    upsert_game_officials,
    upsert_ref_stats,
)

log = logging.getLogger("backfill_game_refs")


def _aggregate_ref_stats_from_table(conn: sqlite3.Connection) -> list[dict]:
    """Re-aggregate per-ref stats from EVERY (game_id, ref) row currently
    in game_officials. We re-pull box-score totals on the fly here would
    be redundant; instead, the in-memory aggregates produced by the
    season walker have already been merged into referee_stats by the
    caller. This helper returns whatever is in referee_stats.
    """
    cur = conn.cursor()
    cur.execute("SELECT ref_name, games, foul_rate, home_win_pct, pace FROM referee_stats")
    return [
        {"ref_name": r[0], "games": r[1] or 0,
         "foul_rate": r[2] or 0.0, "home_win_pct": r[3] or 0.5,
         "pace": r[4] or 100.0}
        for r in cur.fetchall()
    ]


def _merge_ref_aggregates(
    existing: dict[str, dict],
    new: dict[str, dict],
) -> dict[str, dict]:
    """Sample-weighted merge of two ref aggregate dicts.

    For overlapping refs: new aggregate is the games-weighted mean of
    the old + new contributions. This lets a multi-season backfill
    (one season at a time) accumulate without throwing away the
    earlier season's signal.
    """
    out = dict(existing)
    for name, n in new.items():
        if name not in out:
            out[name] = dict(n)
            continue
        e = out[name]
        eg, ng = float(e.get("games", 0)), float(n.get("games", 0))
        total = eg + ng
        if total <= 0:
            continue
        out[name] = {
            "ref_name": name,
            "games": int(total),
            "foul_rate":    (eg * float(e.get("foul_rate", 0)) + ng * float(n.get("foul_rate", 0))) / total,
            "home_win_pct": (eg * float(e.get("home_win_pct", 0.5)) + ng * float(n.get("home_win_pct", 0.5))) / total,
            "pace":         (eg * float(e.get("pace", 100)) + ng * float(n.get("pace", 100))) / total,
        }
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="basketball_data.db")
    ap.add_argument("--seasons", required=True,
                    help="Comma-separated season strings, e.g. 2023-24,2024-25")
    ap.add_argument("--max-games", type=int, default=None,
                    help="Cap games per season (smoke testing)")
    ap.add_argument("--sleep", type=float, default=0.7,
                    help="Sleep between NBA-API calls (politeness)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        log.error("DB not found: %s", args.db)
        return 1

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    log.info("backfill_game_refs: seasons=%s max_games=%s sleep=%.2f",
             seasons, args.max_games, args.sleep)

    conn = sqlite3.connect(args.db)
    # Read existing aggregates so we accumulate across seasons.
    existing_rows = _aggregate_ref_stats_from_table(conn)
    existing_by_name: dict[str, dict] = {r["ref_name"]: r for r in existing_rows}
    log.info("starting from %d refs already aggregated", len(existing_by_name))

    total_game_rows = 0
    overall_aggs = existing_by_name
    for season in seasons:
        t0 = time.time()
        game_rows, season_aggs = compute_game_refs_for_season(
            season,
            max_games=args.max_games,
            sleep_between=args.sleep,
        )
        log.info("[%s] %d game-ref rows, %d refs aggregated in %.0fs",
                 season, len(game_rows), len(season_aggs), time.time() - t0)
        total_game_rows += len(game_rows)
        upsert_game_officials(conn, game_rows)
        overall_aggs = _merge_ref_aggregates(overall_aggs, season_aggs)

    # Write merged ref aggregates back
    upsert_ref_stats(conn, list(overall_aggs.values()), int(time.time()))
    log.info("upserted %d total ref aggregate rows", len(overall_aggs))

    # Sanity check: variance check
    foul_rates = [r.get("foul_rate", 0) for r in overall_aggs.values() if r.get("foul_rate", 0) > 0]
    if foul_rates:
        import statistics
        log.info("foul_rate range: [%.2f, %.2f] mean=%.2f stdev=%.2f",
                 min(foul_rates), max(foul_rates),
                 statistics.mean(foul_rates), statistics.stdev(foul_rates) if len(foul_rates) > 1 else 0.0)
    log.info("done. total game-ref rows ingested this run: %d", total_game_rows)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
