"""Fit empirical same-player prop correlations from graded predictions.

Reads ``prediction_logs`` rows where ``actual_result`` is set, computes
the residual ``actual − predicted`` per row, then for each pair of
prop types observed on the same player on the same day calculates
Pearson correlation across the population of (player_id, date) bundles.

Output
------
``models/empirical_correlations.json`` — a dict keyed by
``"<scope>|<prop_a>|<prop_b>"`` (props canonical-sorted alphabetically),
mapping to ``{"r": float, "n": int}``.

Usage
-----
    python -m scripts.fit_parlay_correlations \
        --db basketball_data.db --min-samples 30

Only pairs with ``n >= --min-samples`` are written. The parlay engine
loads this file at import and overrides ``CORRELATION_PRIORS`` for any
pair found.

Same-team / opp-team scopes are not yet computed because prediction_logs
lacks team identifiers. Only ``same_player`` is fitted.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np


def _date_key(ts: str) -> str:
    """Coarsen timestamp to YYYY-MM-DD so same-game props bucket together."""
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return ts[:10]


def _canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def fit(db_path: str, min_samples: int = 30) -> dict:
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT timestamp, player_id, prop_type, predicted_value, actual_result
        FROM prediction_logs
        WHERE actual_result IS NOT NULL
          AND predicted_value IS NOT NULL
          AND player_id IS NOT NULL
          AND prop_type IS NOT NULL
    """).fetchall()
    conn.close()

    print(f"[fit] {len(rows)} graded rows loaded from {db_path}")

    # Bucket by (player, date) → {prop: residual}
    bundles: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    for ts, pid, prop, pred, actual in rows:
        if pred is None or actual is None:
            continue
        try:
            resid = float(actual) - float(pred)
        except (TypeError, ValueError):
            continue
        bundles[(int(pid), _date_key(ts))][str(prop).lower()] = resid

    # Pair up residuals by prop pair across bundles
    paired: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for _, props_map in bundles.items():
        keys = sorted(props_map.keys())
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                ca, cb = _canonical(a, b)
                paired[(ca, cb)].append((props_map[a], props_map[b]))

    out: dict[str, dict] = {}
    for (a, b), pairs in paired.items():
        if len(pairs) < min_samples:
            continue
        arr = np.array(pairs, dtype=float)
        x, y = arr[:, 0], arr[:, 1]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(r):
            continue
        # Clamp to [-0.95, 0.95] for numerical safety in the copula
        r = max(-0.95, min(0.95, r))
        out[f"same_player|{a}|{b}"] = {"r": round(r, 4), "n": int(len(pairs))}

    print(f"[fit] {len(out)} pairs passed min_samples={min_samples}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="basketball_data.db")
    ap.add_argument("--min-samples", type=int, default=30)
    ap.add_argument("--out", default="models/empirical_correlations.json")
    args = ap.parse_args()

    result = fit(args.db, min_samples=args.min_samples)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"fitted_utc": datetime.utcnow().isoformat(),
                   "min_samples": args.min_samples,
                   "pairs": result}, f, indent=2)
    print(f"[fit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
