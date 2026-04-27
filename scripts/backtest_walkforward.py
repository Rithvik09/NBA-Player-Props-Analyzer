"""Walk-forward backtest harness for graded predictions.

The classic mistake when validating a prop model is to compute Brier/log-loss
across all graded rows at once — that mixes training-set leakage with served
predictions. A walk-forward backtest instead replays history chronologically:
each bucket of ``--window-days`` days is evaluated using ONLY the predictions
that were live during that window, never re-scored with today's model.

Because we already log every served prediction in ``prediction_logs`` with its
served ``confidence`` and graded ``actual_result``, we don't need to retrain —
we just bin and aggregate.

Metrics per window:
  - n graded rows
  - hit rate at confidence >= --conf-threshold
  - Brier score (when probability is interpretable)
  - average ROI assuming flat -110 stake on every bet above threshold

Output is a CSV plus a printed summary so a human can spot regime changes.

Usage
-----
    python -m scripts.backtest_walkforward \
        --db basketball_data.db --window-days 14 --conf-threshold 0.6
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta


def _date_of(ts: str) -> date:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).date()


def _bucket(d: date, anchor: date, window_days: int) -> int:
    return (d - anchor).days // window_days


def _hit(actual: float, predicted: float, line: float, side_hint: str | None = None) -> int | None:
    """Return 1 if the served prediction's implied side won, 0 if it lost,
    None if exactly on the line (push) or missing data.

    Without an explicit side column, we infer side from predicted vs line:
    predicted > line => bet OVER. (Matches how the engine emits picks.)"""
    if actual is None or predicted is None or line is None:
        return None
    if abs(actual - line) < 1e-9:
        return None  # push
    if side_hint == "over" or (side_hint is None and predicted > line):
        return 1 if actual > line else 0
    if side_hint == "under" or (side_hint is None and predicted < line):
        return 1 if actual < line else 0
    return None


def _roi_at_dash110(hit: int) -> float:
    """Flat-stake ROI at -110: win pays +0.909, loss costs -1.0, push = 0."""
    if hit == 1:
        return 100.0 / 110.0
    return -1.0


def run(db_path: str, window_days: int = 14, conf_threshold: float = 0.6,
        out_csv: str | None = None) -> list[dict]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT timestamp, prop_type, line, predicted_value, confidence, actual_result
        FROM prediction_logs
        WHERE actual_result IS NOT NULL
          AND predicted_value IS NOT NULL
          AND line IS NOT NULL
        ORDER BY timestamp ASC
    """).fetchall()
    conn.close()

    if not rows:
        print("[backtest] no graded rows — populate prediction_logs first")
        return []

    anchor = _date_of(rows[0][0])
    bins: dict[int, list[tuple]] = defaultdict(list)
    for r in rows:
        try:
            d = _date_of(r[0])
        except Exception:
            continue
        bins[_bucket(d, anchor, window_days)].append(r)

    out: list[dict] = []
    for b in sorted(bins.keys()):
        win_start = anchor + timedelta(days=b * window_days)
        win_end = win_start + timedelta(days=window_days - 1)
        rs = bins[b]

        n = 0
        wins = 0
        roi_sum = 0.0
        brier_sum = 0.0
        brier_n = 0

        for ts, prop, line, pred, conf, actual in rs:
            try:
                conf = float(conf) if conf is not None else None
                pred = float(pred)
                line = float(line)
                actual = float(actual)
            except (TypeError, ValueError):
                continue
            if conf is None or conf < conf_threshold:
                continue
            h = _hit(actual, pred, line)
            if h is None:
                continue
            n += 1
            wins += h
            roi_sum += _roi_at_dash110(h)
            # Brier: treat conf as the served win probability
            if 0.0 <= conf <= 1.0:
                brier_sum += (conf - h) ** 2
                brier_n += 1

        rec = {
            "window_start": win_start.isoformat(),
            "window_end": win_end.isoformat(),
            "n": n,
            "hit_rate": (wins / n) if n else 0.0,
            "roi": (roi_sum / n) if n else 0.0,
            "brier": (brier_sum / brier_n) if brier_n else None,
        }
        out.append(rec)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"[backtest] wrote {out_csv}")

    print(f"[backtest] windows={len(out)}, threshold={conf_threshold}, span={window_days}d")
    if out:
        total_n = sum(r["n"] for r in out)
        if total_n:
            agg_roi = sum(r["roi"] * r["n"] for r in out) / total_n
            agg_hit = sum(r["hit_rate"] * r["n"] for r in out) / total_n
            print(f"[backtest] aggregate: n={total_n} hit_rate={agg_hit:.3f} roi={agg_roi:+.3f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="basketball_data.db")
    ap.add_argument("--window-days", type=int, default=14)
    ap.add_argument("--conf-threshold", type=float, default=0.60)
    ap.add_argument("--out", default="reports/backtest_walkforward.csv")
    args = ap.parse_args()
    run(args.db, args.window_days, args.conf_threshold, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
