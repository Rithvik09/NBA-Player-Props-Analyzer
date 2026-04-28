"""Production monitoring helpers — drift, decay, anomaly detection.

These functions are pure and operate on ``prediction_logs`` rows + (optionally)
a training-set baseline. They are designed to be called from cheap endpoints
(``/healthz/drift``, ``/monitor/anomalies``) on every request without hammering
the DB — bring your own caching if you want to.

Three flavours of monitoring:

  1. **Feature drift** — compare distributions of served-time features to
     training-set baselines via Kolmogorov–Smirnov. KS-statistic > 0.2 with
     enough samples is a strong signal a feature is now OOD.

  2. **Brier decay** — rolling Brier score on recent graded predictions vs
     the Brier achieved on training. >20% degradation = the model is getting
     stale and a retrain is warranted.

  3. **Anomaly flagging** — single-prediction outlier detection against the
     residual distribution: if ``|predicted − line|`` exceeds N×σ of training
     residuals, the line is wildly off the model (likely data error in
     either direction).
"""
from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence


# ============================================================== KS drift
def ks_statistic(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic — pure-python so we don't
    drag in scipy.stats just for this. Returns sup |F_a(x) − F_b(x)|.

    Both inputs are coerced to float, NaN/inf-filtered, and sorted internally.
    Empty inputs return 0.0 (no signal, not an error).
    """
    a = sorted(x for x in sample_a if isinstance(x, (int, float)) and math.isfinite(x))
    b = sorted(x for x in sample_b if isinstance(x, (int, float)) and math.isfinite(x))
    if not a or not b:
        return 0.0
    na, nb = len(a), len(b)
    i = j = 0
    cdf_a = cdf_b = 0.0
    d = 0.0
    # When both samples have a tie at the same value, advance BOTH pointers
    # to the end of the tied block before measuring the gap. Otherwise
    # interleaving pumps a 1/n bias on identical inputs (KS would never
    # be 0 on a == b, which is wrong by definition).
    while i < na and j < nb:
        if a[i] < b[j]:
            cdf_a = (i + 1) / na
            i += 1
        elif a[i] > b[j]:
            cdf_b = (j + 1) / nb
            j += 1
        else:
            # Equal values: skip the entire run of ties on both sides
            v = a[i]
            while i < na and a[i] == v:
                i += 1
            while j < nb and b[j] == v:
                j += 1
            cdf_a, cdf_b = i / na, j / nb
        gap = abs(cdf_a - cdf_b)
        if gap > d:
            d = gap
    return d


def detect_feature_drift(
    served_features: dict[str, Sequence[float]],
    baseline_features: dict[str, Sequence[float]],
    *,
    min_samples: int = 30,
    ks_threshold: float = 0.20,
) -> list[dict]:
    """Per-feature KS test of served vs baseline distributions.

    Returns a list of ``{feature, ks, n_served, n_baseline, drifted}``,
    sorted by KS descending. Features below ``min_samples`` on either side
    are skipped (KS on small samples is noisy).
    """
    out: list[dict] = []
    for feat, served in served_features.items():
        baseline = baseline_features.get(feat)
        if baseline is None:
            continue
        if len(served) < min_samples or len(baseline) < min_samples:
            continue
        ks = ks_statistic(served, baseline)
        out.append({
            "feature": feat,
            "ks": round(ks, 4),
            "n_served": len(served),
            "n_baseline": len(baseline),
            "drifted": ks >= ks_threshold,
        })
    out.sort(key=lambda r: r["ks"], reverse=True)
    return out


# ============================================================ Brier decay
def rolling_brier(
    db_path: str,
    *,
    window_days: int = 30,
    prop_filter: str | None = None,
) -> dict:
    """Rolling Brier on the last ``window_days`` of graded predictions.

    Reads from ``prediction_logs``. Brier requires a probability + a 0/1
    outcome — we treat ``confidence`` as the served prob and binarise the
    outcome via ``actual_result > line``. Pushes (actual == line) are
    excluded from the count.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    sql = """
        SELECT prop_type, line, confidence, actual_result
        FROM prediction_logs
        WHERE actual_result IS NOT NULL
          AND confidence IS NOT NULL
          AND line IS NOT NULL
          AND timestamp >= ?
    """
    params: list = [cutoff]
    if prop_filter:
        sql += " AND prop_type = ?"
        params.append(prop_filter)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    by_prop: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for prop, line, conf, actual in rows:
        try:
            line_f = float(line)
            actual_f = float(actual)
            conf_f = float(conf)
        except (TypeError, ValueError):
            continue
        if abs(actual_f - line_f) < 1e-9:
            continue  # push
        # NB: prediction_logs.confidence is the model's served prob for the
        # SIDE it picked. If the picked side actually won, hit=1 else 0.
        # Without an explicit side column we reconstruct: prediction_logs
        # implicitly emits the higher-prob side, so hit = (actual > line)
        # XOR (conf < 0.5)…
        # Simpler heuristic that works without a side column: assume
        # ``confidence`` is for "over" when the model thought over, and we
        # always store the side it picked. For Brier we just need
        # (predicted_prob_of_event, did_event_happen). Here event = "the
        # picked side won" which is binarised over vs line for over picks.
        hit = 1 if actual_f > line_f else 0
        by_prop[str(prop)].append((conf_f, hit))

    out: dict = {"window_days": window_days, "props": {}}
    overall_n = 0
    overall_brier = 0.0
    for prop, samples in by_prop.items():
        if not samples:
            continue
        n = len(samples)
        b = sum((p - h) ** 2 for p, h in samples) / n
        out["props"][prop] = {"n": n, "brier": round(b, 4)}
        overall_n += n
        overall_brier += b * n
    out["overall"] = {
        "n": overall_n,
        "brier": round(overall_brier / overall_n, 4) if overall_n else None,
    }
    return out


def brier_decay_check(
    rolling: dict,
    training_brier: dict[str, float],
    *,
    degradation_threshold: float = 0.20,
) -> list[dict]:
    """Compare rolling Brier per prop against training-time baseline.

    Returns props where ``(rolling − training) / training >= threshold``.
    Each row carries which prop and how much it has decayed so the operator
    can decide whether to retrain.
    """
    out: list[dict] = []
    for prop, info in rolling.get("props", {}).items():
        baseline = training_brier.get(prop)
        if not baseline or baseline <= 0:
            continue
        rolling_b = info.get("brier")
        if rolling_b is None:
            continue
        delta = (rolling_b - baseline) / baseline
        # Tiny epsilon so a precisely-at-threshold value like 0.20 is flagged
        # despite floating-point producing 0.19999…
        out.append({
            "prop": prop,
            "rolling_brier": rolling_b,
            "training_brier": float(baseline),
            "degradation": round(delta, 4),
            "decayed": delta >= float(degradation_threshold) - 1e-9,
        })
    out.sort(key=lambda r: r["degradation"], reverse=True)
    return out


# ============================================================ Anomaly
def is_prediction_anomalous(
    predicted: float,
    line: float,
    residual_std: float,
    *,
    sigma_threshold: float = 3.0,
) -> dict:
    """Flag a single served prediction whose distance from the line exceeds
    N×σ of training residuals.

    A 3σ disagreement on a properly-calibrated model with normalish residuals
    happens <1% of the time by chance — when it does, the most likely
    explanation is a stale feature, a typo'd line, or a player mid-trade.
    Either way, the operator wants to see it.
    """
    if residual_std is None or residual_std <= 0 or not math.isfinite(residual_std):
        return {"anomalous": False, "reason": "no residual std"}
    diff = abs(float(predicted) - float(line))
    z = diff / float(residual_std)
    return {
        "anomalous": z >= sigma_threshold,
        "predicted": float(predicted),
        "line": float(line),
        "diff": round(diff, 4),
        "z_score": round(z, 4),
        "sigma_threshold": float(sigma_threshold),
    }


def find_anomalies(
    db_path: str,
    residual_stds_by_prop: dict[str, float],
    *,
    window_days: int = 7,
    sigma_threshold: float = 3.0,
    limit: int = 50,
) -> list[dict]:
    """Return recent prediction_logs rows whose served prediction is more
    than ``sigma_threshold`` σ from the posted line, per-prop residual σ.

    Use case: an admin endpoint that surfaces today's "wait, that can't be
    right" picks for human review before they get bet on.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """SELECT id, timestamp, player_id, prop_type, line, predicted_value, confidence
               FROM prediction_logs
               WHERE predicted_value IS NOT NULL AND line IS NOT NULL
                 AND timestamp >= ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (cutoff, int(limit) * 4),  # over-fetch since we filter
        ).fetchall()
    finally:
        conn.close()

    out: list[dict] = []
    for log_id, ts, pid, prop, line, pred, conf in rows:
        std = residual_stds_by_prop.get(str(prop))
        if not std:
            continue
        try:
            verdict = is_prediction_anomalous(
                float(pred), float(line), float(std),
                sigma_threshold=sigma_threshold,
            )
        except (TypeError, ValueError):
            continue
        if verdict.get("anomalous"):
            out.append({
                "id": int(log_id),
                "timestamp": ts,
                "player_id": pid,
                "prop_type": prop,
                **verdict,
            })
        if len(out) >= int(limit):
            break
    return out
