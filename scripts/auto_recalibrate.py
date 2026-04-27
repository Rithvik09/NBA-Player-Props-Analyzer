"""Auto-recalibrate served classifiers against recent graded predictions.

Daily/weekly job: pulls graded ``prediction_logs`` rows, computes Brier
score per prop on the served model, and refits an isotonic calibrator on
the live residuals when drift is detected. Writes a new
``clf_cal_<prop>.joblib`` only when the recalibrated model beats the
served model on Brier by ``--min-improvement`` (default 0.002).

Won't touch raw classifiers. Saves ``.joblib.bak`` of any swapped
artifact for rollback.

Usage
-----
    python -m scripts.auto_recalibrate \
        --db basketball_data.db --models-dir models --window-days 30

Cron-friendly. Exit 0 always — failures are logged, not raised, so a
schedule failure never blocks more critical jobs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Make `from src...` importable when invoked as `python scripts/auto_recalibrate.py`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_calibration import IsotonicCalibratedModel  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

log = logging.getLogger(__name__)


def _load_recent_graded(db_path: str, window_days: int) -> list[tuple]:
    if not os.path.exists(db_path):
        log.warning("[recal] db missing: %s", db_path)
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT prop_type, line, predicted_value, over_probability,
               actual_result
        FROM prediction_logs
        WHERE actual_result IS NOT NULL
          AND over_probability IS NOT NULL
          AND prop_type IS NOT NULL
          AND timestamp >= ?
    """, (cutoff,)).fetchall()
    conn.close()
    return rows


def _y_label(line: float, actual: float) -> int:
    return 1 if float(actual) > float(line) else 0


def _evaluate_brier(probs: np.ndarray, ys: np.ndarray) -> float:
    if len(ys) == 0:
        return float("nan")
    return float(brier_score_loss(ys, probs))


def recalibrate(db_path: str, models_dir: str, window_days: int,
                min_samples: int, min_improvement: float,
                dry_run: bool) -> dict:
    rows = _load_recent_graded(db_path, window_days)
    log.info("[recal] %d graded rows in last %d days", len(rows), window_days)

    # Bucket by prop
    by_prop: dict[str, list] = {}
    for prop, line, pred, over_prob, actual in rows:
        try:
            y = _y_label(float(line) if line is not None else 0.0, float(actual))
            p = float(over_prob)
            if not (0.0 <= p <= 1.0):
                continue
        except (TypeError, ValueError):
            continue
        by_prop.setdefault(str(prop).lower(), []).append((p, y))

    summary: dict[str, dict] = {}
    meta_path = Path(models_dir) / "model_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"props": {}}

    for prop, samples in by_prop.items():
        if len(samples) < min_samples:
            summary[prop] = {"status": "insufficient_samples",
                              "n": len(samples)}
            continue
        ps = np.array([s[0] for s in samples], dtype=float)
        ys = np.array([s[1] for s in samples], dtype=int)
        if ys.sum() == 0 or ys.sum() == len(ys):
            summary[prop] = {"status": "label_degenerate",
                              "n": len(samples)}
            continue

        served_brier = _evaluate_brier(ps, ys)

        # Fit isotonic calibrator on these live (p, y) pairs
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(ps, ys)
        ps_cal = iso.predict(ps)
        cal_brier = _evaluate_brier(ps_cal, ys)

        improvement = served_brier - cal_brier
        info = {"n": len(samples), "served_brier": round(served_brier, 6),
                "cal_brier": round(cal_brier, 6),
                "improvement": round(improvement, 6)}

        if improvement < min_improvement:
            info["status"] = "no_improvement"
            summary[prop] = info
            continue

        # Wrap served model + new isotonic, write to disk
        served_path = Path(models_dir) / f"clf_cal_{prop}.joblib"
        if not served_path.exists():
            info["status"] = "served_missing"
            summary[prop] = info
            continue

        if dry_run:
            info["status"] = "would_swap"
            summary[prop] = info
            continue

        try:
            served = joblib.load(served_path)
            base = getattr(served, "base_estimator", served)
            new_model = IsotonicCalibratedModel(base_estimator=base, calibrator=iso)
            shutil.copy(served_path, str(served_path) + ".bak")
            joblib.dump(new_model, served_path)
            info["status"] = "swapped"
            # Update metadata
            meta.setdefault("props", {}).setdefault(prop, {})
            meta["props"][prop]["last_auto_recal_utc"] = datetime.now(timezone.utc).isoformat()
            meta["props"][prop]["last_auto_recal_n"] = len(samples)
            meta["props"][prop]["last_auto_recal_brier_before"] = served_brier
            meta["props"][prop]["last_auto_recal_brier_after"] = cal_brier
        except Exception as e:  # noqa: BLE001
            log.warning("[recal] swap failed for %s: %s", prop, e)
            info["status"] = f"swap_failed: {e}"
        summary[prop] = info

    if not dry_run and meta_path.exists():
        meta_path.write_text(json.dumps(meta, indent=2))

    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="basketball_data.db")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--min-samples", type=int, default=50)
    ap.add_argument("--min-improvement", type=float, default=0.002)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        result = recalibrate(args.db, args.models_dir, args.window_days,
                              args.min_samples, args.min_improvement,
                              args.dry_run)
        print(json.dumps(result, indent=2))
    except Exception as e:  # noqa: BLE001
        log.exception("[recal] failed: %s", e)
    return 0  # never blow up scheduler


if __name__ == "__main__":
    sys.exit(main())
