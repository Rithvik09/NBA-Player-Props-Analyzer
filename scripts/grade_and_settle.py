"""Auto-grade pending predictions and settle linked bankroll bets.

Designed to run on a cron, e.g. nightly at 04:00 ET. The script:

  1. Calls ``BasketballBettingHelper.auto_grade_pending()``, which fills in
     ``actual_result`` on every prediction_logs row whose game has finished.
  2. Walks every ``status='open'`` row in bankroll_bets that has a non-NULL
     ``prediction_log_id`` and looks up the matching (now-graded) prediction.
  3. Translates ``actual vs line`` into a win/loss/push outcome and calls
     ``BankrollTracker.settle()``, which credits the bankroll and updates ROI.

Bets without a ``prediction_log_id`` (manually placed) are left alone.

Usage
-----
    python -m scripts.grade_and_settle [--db basketball_data.db] [--dry-run]

Exit code is non-zero only on hard failure (DB unreachable). Per-bet errors
are counted and printed but do not abort the run.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

# Allow running as a module or a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.basketball_betting_helper import BasketballBettingHelper  # noqa: E402
from src.bankroll import BankrollTracker  # noqa: E402


_PUSH_TOLERANCE = 1e-9  # actual == line within float noise


def _decide_result(side: str, line: float, actual: float) -> str:
    """Translate side/line/actual into win | loss | push."""
    side = (side or "").lower().strip()
    if abs(actual - line) <= _PUSH_TOLERANCE:
        return "push"
    if side in ("over", "yes"):
        return "win" if actual > line else "loss"
    if side in ("under", "no"):
        return "win" if actual < line else "loss"
    raise ValueError(f"unknown side: {side!r}")


def settle_open_bets(db_path: str, *, dry_run: bool = False) -> dict:
    """Settle every open bet with a graded ``prediction_log_id``.

    Returns counts of {settled, skipped, errors}. Skipped means the linked
    prediction is still pending (actual_result is NULL).
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT b.id, b.side, b.line, b.prediction_log_id, p.actual_result
            FROM bankroll_bets b
            LEFT JOIN prediction_logs p ON p.id = b.prediction_log_id
            WHERE b.status = 'open'
              AND b.prediction_log_id IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()

    print(f"[settle] {len(rows)} open bets with linked predictions")

    tracker = BankrollTracker(db_path)
    settled = skipped = errors = 0

    for bet_id, side, line, log_id, actual in rows:
        if actual is None:
            skipped += 1
            continue
        try:
            result = _decide_result(side, float(line), float(actual))
            if dry_run:
                print(f"[settle] [dry] bet#{bet_id} log#{log_id} "
                      f"side={side} line={line} actual={actual} -> {result}")
            else:
                out = tracker.settle(int(bet_id), result)
                print(f"[settle] bet#{bet_id} -> {result} pnl={out['pnl']:.2f} "
                      f"new_balance={out['new_balance']:.2f}")
            settled += 1
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"[settle] bet#{bet_id} ERROR: {e}", file=sys.stderr)

    return {"settled": settled, "skipped": skipped, "errors": errors}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="basketball_data.db")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be settled but don't write to bankroll")
    ap.add_argument("--skip-grading", action="store_true",
                    help="don't call auto_grade_pending — just settle already-graded predictions")
    args = ap.parse_args()

    if not args.skip_grading:
        helper = BasketballBettingHelper(db_name=args.db)
        grade_result = helper.auto_grade_pending()
        print(f"[grade] result={grade_result}")

    settle_result = settle_open_bets(args.db, dry_run=args.dry_run)
    print(f"[settle] result={settle_result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
