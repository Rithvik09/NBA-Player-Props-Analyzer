"""Backfill ``game_id`` + ``prop_outcomes`` rows for legacy predictions.

Why this script exists
----------------------
Before commit 215ea7c, ``auto_grade_pending`` graded predictions but did
not stash the NBA ``Game_ID`` on the prediction row, and never emitted a
``prop_outcomes`` row. So every row graded prior to that commit looks
like CLV-pipeline data EXCEPT it has no ``game_id`` (and therefore no
join key to ``prop_line_summary``) and no outcome row.

This is one-shot rescue work: read every graded ``prediction_logs`` row
that's missing ``game_id``, look up the player's game log for that
date, recover ``Game_ID``, write it back, and emit a ``prop_outcomes``
row. After this runs once, going forward ``auto_grade_pending`` keeps
both populated automatically.

Caveats
-------
* Closing lines won't be there for old games — we never captured them.
  The emitted ``prop_outcomes`` rows will have ``closing_line=NULL``
  and so won't appear in ``clv_training_rows()``. They're still useful
  as a clean (observed_line, actual_result) audit table.
* nba_api gamelog lookups are rate-limited. We cache by player×season
  and sleep 0.6s between fresh fetches, mirroring ``auto_grade_pending``.
* Idempotent: running this twice is fine. The second run finds nothing
  to do because ``game_id`` is already populated.

Usage
-----
    python -m scripts.backfill_prop_outcomes [--db basketball_data.db]
                                             [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.odds_tracker import OddsTracker  # noqa: E402

log = logging.getLogger("backfill_prop_outcomes")


def _season_for_date(d: datetime) -> str:
    """NBA season string like '2024-25' for a given date.

    Aug-Dec → starts the season for the trailing-year format.
    Jan-Jul → previous calendar year started the season.
    """
    if 1 <= d.month <= 7:
        return f"{d.year - 1}-{str(d.year)[2:]}"
    return f"{d.year}-{str(d.year + 1)[2:]}"


def _find_legacy_rows(db_path: str, limit: int | None = None) -> list[dict]:
    """Read graded predictions missing game_id."""
    conn = sqlite3.connect(db_path)
    try:
        sql = """
            SELECT id, player_id, player_name, prop_type, line, timestamp,
                   actual_result
            FROM prediction_logs
            WHERE actual_result IS NOT NULL
              AND (game_id IS NULL OR game_id = '')
              AND player_name IS NOT NULL
            ORDER BY timestamp ASC
        """
        if limit is not None and limit > 0:
            sql += f" LIMIT {int(limit)}"
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": r[0], "player_id": r[1], "player_name": r[2],
            "prop_type": r[3], "line": r[4], "timestamp": r[5],
            "actual_result": r[6],
        }
        for r in rows
    ]


def _resolve_game_id(
    *,
    player_id: int,
    pred_date: datetime,
    gamelog_cache: dict,
) -> str | None:
    """Look up the NBA Game_ID for ``player_id`` on ``pred_date``.

    Caches the full season df per (player_id, season) so we only hit the
    API once per player+season across the whole run. Tolerates +1 day
    drift for late-night games that log the next day (mirrors the
    behaviour in ``auto_grade_pending``).
    """
    # Local import so this script doesn't pay nba_api's import cost on
    # users who --dry-run or stop early.
    from nba_api.stats.endpoints import playergamelog

    season = _season_for_date(pred_date)
    cache_key = (player_id, season)
    if cache_key not in gamelog_cache:
        try:
            df = playergamelog.PlayerGameLog(
                player_id=player_id, season=season,
            ).get_data_frames()[0]
        except Exception as e:  # noqa: BLE001
            log.warning("gamelog fetch failed for player=%s season=%s: %s",
                        player_id, season, e)
            gamelog_cache[cache_key] = None
            return None
        time.sleep(0.6)  # API politeness, same cadence as auto_grade
        df["_date"] = pd.to_datetime(df["GAME_DATE"]).dt.date
        gamelog_cache[cache_key] = df

    df = gamelog_cache[cache_key]
    if df is None or df.empty:
        return None

    target = pred_date.date()
    row = df[df["_date"] == target]
    if row.empty:
        # +1 day tolerance for late-tipping games that log the next day
        from datetime import timedelta
        row = df[df["_date"] == target + timedelta(days=1)]
    if row.empty:
        return None
    try:
        return str(row.iloc[0]["Game_ID"])
    except Exception:
        return None


def backfill(db_path: str, *, limit: int | None = None,
             dry_run: bool = False) -> dict:
    """Walk legacy rows, stamp game_id, emit prop_outcomes rows.

    Returns counts: ``{candidates, stamped, outcomes_written, skipped, errors}``.
    Per-row failures are counted but don't abort the run.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    legacy = _find_legacy_rows(db_path, limit=limit)
    log.info("found %d legacy rows missing game_id", len(legacy))

    if not legacy:
        return {"candidates": 0, "stamped": 0, "outcomes_written": 0,
                "skipped": 0, "errors": 0}

    odds = OddsTracker(api_key=None, db_path=db_path)
    gamelog_cache: dict = {}
    stamped = outcomes_written = skipped = errors = 0

    for row in legacy:
        try:
            ts = row["timestamp"]
            try:
                pred_date = datetime.fromisoformat(ts)
            except ValueError:
                # Some legacy rows may have a date-only stamp
                pred_date = datetime.strptime(ts[:10], "%Y-%m-%d")

            game_id = _resolve_game_id(
                player_id=int(row["player_id"]),
                pred_date=pred_date,
                gamelog_cache=gamelog_cache,
            )
            if not game_id:
                skipped += 1
                continue

            if dry_run:
                log.info("[dry] log#%s player=%s prop=%s -> game_id=%s",
                         row["id"], row["player_name"], row["prop_type"], game_id)
                stamped += 1
                outcomes_written += 1
                continue

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE prediction_logs SET game_id = ? WHERE id = ?",
                    (game_id, row["id"]),
                )
                conn.commit()
            stamped += 1

            try:
                odds.record_outcome(
                    game_id=game_id,
                    player_name=row["player_name"],
                    prop_type=row["prop_type"],
                    actual_result=float(row["actual_result"]),
                    observed_line=float(row["line"]),
                    player_id=int(row["player_id"]) if row["player_id"] else None,
                )
                outcomes_written += 1
            except Exception as e:  # noqa: BLE001
                log.warning("record_outcome failed for log#%s: %s", row["id"], e)
                errors += 1

        except Exception as e:  # noqa: BLE001
            log.warning("backfill error on log#%s: %s", row.get("id"), e)
            errors += 1

    return {
        "candidates": len(legacy),
        "stamped": stamped,
        "outcomes_written": outcomes_written,
        "skipped": skipped,
        "errors": errors,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="basketball_data.db",
                    help="path to the SQLite DB used by the helper")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N legacy rows (for smoke testing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve game_ids but don't write to the DB")
    args = ap.parse_args()

    result = backfill(args.db, limit=args.limit, dry_run=args.dry_run)
    log.info("backfill result: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
