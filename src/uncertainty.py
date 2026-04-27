"""Prediction-interval helpers for the regression head.

Two complementary tools:

1. ``residual_quantile_interval`` — non-parametric, conformal-style PI built
   from a held-out residual distribution. Distribution-free, calibrated,
   doesn't assume normality, and survives heavy tails (e.g. blowouts where
   minutes drop and the prediction overshoots by 15+ pts).

2. ``bootstrap_pi`` — block-bootstrap of residuals to give an empirical
   sampling distribution around a point prediction. Useful when you want a
   smoother PI than raw quantiles or want median + percentile bands.

Both are pure functions of (prediction, residual_array) so they can be applied
post-hoc to any regressor without retraining.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def residual_quantile_interval(
    prediction: float,
    residuals: Iterable[float],
    alpha: float = 0.10,
) -> tuple[float, float]:
    """Conformal-style PI: ``[ŷ + Q_{α/2}(r), ŷ + Q_{1-α/2}(r)]``.

    Where ``r = y_true − y_pred`` on a held-out validation set. With
    ``alpha=0.10`` this returns a 90% PI. Empty/short residual arrays
    fall back to a ±2σ-style heuristic so the helper is total.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    arr = np.asarray(list(residuals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        # Too few residuals to estimate quantiles — use a wide nominal band.
        spread = float(np.std(arr)) if arr.size else 5.0
        return float(prediction - 2 * spread), float(prediction + 2 * spread)
    lo_q = float(np.quantile(arr, alpha / 2.0))
    hi_q = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return float(prediction + lo_q), float(prediction + hi_q)


def bootstrap_pi(
    prediction: float,
    residuals: Iterable[float],
    alpha: float = 0.10,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Resample-with-replacement bootstrap PI.

    Returns ``{"point", "lower", "upper", "median", "samples"}``. Use this
    when downstream code wants the full sample (e.g. to compute P(over))
    rather than just the bounds.
    """
    rng = rng or np.random.default_rng(42)
    arr = np.asarray(list(residuals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        # No data — return prediction as both bounds (no information).
        return {
            "point": float(prediction),
            "lower": float(prediction),
            "upper": float(prediction),
            "median": float(prediction),
            "samples": np.array([prediction] * n_boot),
        }
    draws = rng.choice(arr, size=int(n_boot), replace=True)
    samples = float(prediction) + draws
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    med = float(np.median(samples))
    return {
        "point": float(prediction),
        "lower": lo,
        "upper": hi,
        "median": med,
        "samples": samples,
    }


def prob_over_from_samples(samples: np.ndarray, line: float) -> float:
    """Frequency-of-over from a bootstrap sample. Useful for converting a PI
    sample to a calibrated win probability for a specific line."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return 0.5
    return float(np.mean(arr > float(line)))
