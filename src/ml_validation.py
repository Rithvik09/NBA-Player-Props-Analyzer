"""Out-of-time validation helpers for the training pipeline.

Why OOT instead of random k-fold? Game data has *strong* temporal
structure: a player's role can shift mid-season, the league pace changes
year-over-year, and rule changes can shift category distributions.
Random splits leak future information into training; the model looks
better in cross-validation than it ever performs in production.

The fix is to **always** hold out the most-recent N days, train on
everything before, and report metrics on that holdout. The numbers will
look worse than k-fold — that's the *point*. Production performance is
in the OOT number, not the k-fold one.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np


def parse_iso(ts: str | datetime) -> datetime:
    """Coerce ISO string or datetime to tz-aware UTC datetime."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    s = str(ts).replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        d = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def out_of_time_split(
    timestamps: Iterable,
    *,
    holdout_days: int = 14,
    reference_time: datetime | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, oot_idx)`` boolean masks.

    The OOT set is every row with ``ts >= reference_time - holdout_days``.
    Reference defaults to ``max(ts)`` so the function works on dataframes
    that don't span up to "now" (e.g. an offline backtest).
    """
    parsed = np.array([parse_iso(t) for t in timestamps])
    if parsed.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)
    ref = reference_time or parsed.max()
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    cutoff = ref - timedelta(days=int(holdout_days))
    oot_mask = parsed >= cutoff
    train_mask = ~oot_mask
    return train_mask, oot_mask


def oot_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    line: Iterable[float] | None = None,
    proba: Iterable[float] | None = None,
) -> dict:
    """Compute OOT metrics from arrays.

    - ``rmse`` and ``mae`` always.
    - ``hit_rate`` and ``brier`` if ``line`` is provided (binarised at the line).
    - ``brier_calibrated`` if ``proba`` is provided alongside ``line``.

    Returns a flat dict with whatever's computable; missing inputs just
    skip the relevant metric.
    """
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    if y_true.size == 0:
        return {"n": 0}
    out: dict = {
        "n": int(y_true.size),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }
    if line is not None:
        line_arr = np.asarray(list(line), dtype=float)
        # Hit-rate of the obvious "predict over iff predicted > line" rule
        hits = ((y_true > line_arr) == (y_pred > line_arr)).astype(float)
        out["hit_rate"] = float(np.mean(hits))
        if proba is not None:
            p = np.asarray(list(proba), dtype=float)
            actual = (y_true > line_arr).astype(float)
            out["brier"] = float(np.mean((p - actual) ** 2))
    return out
