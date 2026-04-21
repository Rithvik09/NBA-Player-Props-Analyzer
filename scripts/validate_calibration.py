"""Validate per-prop calibration by replaying graded prediction_logs.

Usage
-----
    python3 -m scripts.validate_calibration
    python3 -m scripts.validate_calibration --min-samples 30

Output
------
    logs/calibration/<UTC-stamp>/
        report.json        machine-readable summary
        report.txt         operator-friendly table
        reliability_*.png  per-prop reliability diagram (if matplotlib available)

What it measures
----------------
    * ``brier`` — mean squared error between predicted prob & outcome (lower better)
    * ``ece``   — Expected Calibration Error over 10 deciles (lower better)
    * ``log_loss``
    * Deviation alerts when |avg_pred − hit_rate| > 0.1

Caveat
------
Only replays against ``prediction_logs.actual_value``-graded rows. If you have
< 50 graded rows per prop the estimate is noisy — use the ``--min-samples``
gate to hide those props from the report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ensure repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    p_clipped = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected Calibration Error over equal-width bins."""
    n = len(p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket_info: list[dict] = []
    ece_total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        count = int(mask.sum())
        if count == 0:
            bucket_info.append({"lo": float(lo), "hi": float(hi), "n": 0})
            continue
        avg_pred = float(p[mask].mean())
        avg_true = float(y[mask].mean())
        gap = abs(avg_pred - avg_true)
        ece_total += (count / n) * gap
        bucket_info.append({
            "lo": float(lo), "hi": float(hi), "n": count,
            "avg_pred": avg_pred, "avg_true": avg_true, "gap": gap,
        })
    return float(ece_total), bucket_info


def _load_graded(db_path: str) -> dict[str, list[tuple[float, float, float]]]:
    """Return ``{prop_type: [(predicted_prob, line, actual_value), ...]}``."""
    if not os.path.exists(db_path):
        print(f"[err] db not found: {db_path}", file=sys.stderr)
        return {}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("""
            SELECT prop_type, over_probability, line, actual_result
            FROM prediction_logs
            WHERE actual_result IS NOT NULL
              AND over_probability IS NOT NULL
              AND line IS NOT NULL
        """).fetchall()
    finally:
        conn.close()

    out: dict[str, list[tuple[float, float, float]]] = {}
    for prop, prob, line, actual in rows:
        try:
            out.setdefault(prop, []).append((float(prob), float(line), float(actual)))
        except (TypeError, ValueError):
            continue
    return out


def _render_report(prop_stats: dict[str, dict], min_samples: int) -> str:
    lines: list[str] = []
    lines.append(f"{'prop':<22}{'n':>6}{'brier':>10}{'ece':>10}{'logloss':>10}{'avg_p':>10}{'hit_rate':>10}{'verdict':>12}")
    lines.append("-" * 90)
    for prop, s in sorted(prop_stats.items()):
        if s["n"] < min_samples:
            continue
        verdict = "OK"
        if s["ece"] > 0.10:
            verdict = "MISCAL"
        elif s["brier"] > 0.25:
            verdict = "LOW-INFO"
        lines.append(
            f"{prop:<22}{s['n']:>6}{s['brier']:>10.3f}{s['ece']:>10.3f}"
            f"{s['log_loss']:>10.3f}{s['avg_pred']:>10.3f}{s['hit_rate']:>10.3f}{verdict:>12}"
        )
    lines.append("")
    hidden = sum(1 for s in prop_stats.values() if s["n"] < min_samples)
    if hidden:
        lines.append(f"({hidden} props hidden — below min_samples={min_samples})")
    return "\n".join(lines)


def _plot_reliability(prop: str, buckets: list[dict], out_dir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    xs = [b["avg_pred"] for b in buckets if b.get("n")]
    ys = [b["avg_true"] for b in buckets if b.get("n")]
    ns = [b["n"] for b in buckets if b.get("n")]
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888", label="perfect")
    sizes = [10 + 5 * math.sqrt(n) for n in ns]
    ax.scatter(xs, ys, s=sizes, alpha=0.7)
    ax.plot(xs, ys, alpha=0.3)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical hit rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"reliability — {prop}")
    ax.grid(True, alpha=0.3)
    out_path = out_dir / f"reliability_{prop}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="basketball_data.db")
    parser.add_argument("--min-samples", type=int, default=30,
                        help="hide props with fewer than this many graded rows")
    parser.add_argument("--bins", type=int, default=10, help="ECE bin count")
    args = parser.parse_args()

    graded = _load_graded(args.db)
    if not graded:
        print("No graded prediction_logs rows — nothing to validate.", file=sys.stderr)
        print("Run the app for a while and grade predictions, then re-run.", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("logs") / "calibration" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    prop_stats: dict[str, dict] = {}
    all_buckets: dict[str, list[dict]] = {}
    for prop, rows in graded.items():
        if not rows:
            continue
        probs = np.array([r[0] for r in rows], dtype=float)
        lines_arr = np.array([r[1] for r in rows], dtype=float)
        actuals = np.array([r[2] for r in rows], dtype=float)
        y = (actuals > lines_arr).astype(float)
        ece, buckets = _ece(probs, y, bins=args.bins)
        prop_stats[prop] = {
            "n": len(rows),
            "brier": _brier(probs, y),
            "ece": ece,
            "log_loss": _log_loss(probs, y),
            "avg_pred": float(probs.mean()),
            "hit_rate": float(y.mean()),
        }
        all_buckets[prop] = buckets

    report_text = _render_report(prop_stats, args.min_samples)
    (out_dir / "report.txt").write_text(report_text + "\n")
    (out_dir / "report.json").write_text(json.dumps({
        "generated_utc": stamp,
        "per_prop": prop_stats,
        "buckets": all_buckets,
        "min_samples": args.min_samples,
    }, indent=2))

    plotted: list[Path] = []
    for prop, buckets in all_buckets.items():
        if prop_stats[prop]["n"] < args.min_samples:
            continue
        p = _plot_reliability(prop, buckets, out_dir)
        if p:
            plotted.append(p)

    print(report_text)
    print(f"\nwrote {out_dir}")
    if plotted:
        print(f"plotted {len(plotted)} reliability diagrams")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
