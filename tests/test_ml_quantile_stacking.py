"""Tests for ml_quantile, ml_validation, ml_stacking helpers."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_quantile import QuantileEnsemble  # noqa: E402
from src.ml_stacking import StackingBlender  # noqa: E402
from src.ml_validation import (  # noqa: E402
    oot_metrics,
    out_of_time_split,
    parse_iso,
)


# ---------------------------------------------------------------- quantile
def _toy_regression():
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(400, 3))
    # y = linear in features + heteroskedastic noise
    y = 3 * X[:, 0] + 2 * X[:, 1] + rng.normal(0, 1.5, size=400)
    return X, y


def test_quantile_ensemble_orders_predictions():
    X, y = _toy_regression()
    qe = QuantileEnsemble(quantile_low=0.10, quantile_mid=0.50, quantile_high=0.90,
                          max_iter=80)
    qe.fit(X, y)
    out = qe.predict_intervals(X)
    # Repaired output must satisfy lo <= mid <= hi everywhere
    assert (out["lower"] <= out["median"] + 1e-9).all()
    assert (out["median"] <= out["upper"] + 1e-9).all()


def test_quantile_ensemble_calibration_approx():
    """80% PI should bracket roughly 80% of held-out points."""
    rng = np.random.default_rng(11)
    X = rng.normal(0, 1, size=(800, 3))
    y = 3 * X[:, 0] + rng.normal(0, 1.5, size=800)
    train, test = X[:600], X[600:]
    y_train, y_test = y[:600], y[600:]
    qe = QuantileEnsemble(max_iter=80).fit(train, y_train)
    out = qe.predict_intervals(test)
    cover = ((y_test >= out["lower"]) & (y_test <= out["upper"])).mean()
    # Expect ~80% but allow slack for finite sample + boosting bias
    assert 0.70 <= cover <= 0.95


def test_quantile_ensemble_prob_over_monotone():
    X, y = _toy_regression()
    qe = QuantileEnsemble(max_iter=80).fit(X, y)
    # P(y > line) must be monotonically decreasing in line for a fixed row
    x = X[:5]
    p_lo = qe.prob_over(x, line=-2.0)
    p_hi = qe.prob_over(x, line=+2.0)
    assert (p_lo >= p_hi - 1e-9).all()


def test_quantile_ensemble_rejects_bad_quantiles():
    with pytest.raises(ValueError):
        QuantileEnsemble(quantile_low=0.5, quantile_mid=0.5, quantile_high=0.5)


# --------------------------------------------------------------- OOT split
def test_oot_split_holds_out_recent_days():
    base = datetime(2025, 4, 1, tzinfo=timezone.utc)
    timestamps = [base - timedelta(days=d) for d in range(30)]  # 30 days back
    train, oot = out_of_time_split(timestamps, holdout_days=7)
    assert oot.sum() == 8  # day 0..7 inclusive
    assert train.sum() == 22


def test_oot_split_uses_max_when_no_reference():
    # Mix of recent + ancient — ref_time should be the max
    timestamps = ["2025-04-01T00:00:00Z", "2024-01-01T00:00:00Z"]
    train, oot = out_of_time_split(timestamps, holdout_days=7)
    # Most-recent 7 days from 2025-04-01 covers the 2025-04-01 row only
    assert oot.sum() == 1
    assert train.sum() == 1


def test_oot_metrics_with_full_inputs():
    y = np.array([20.0, 30.0, 25.0, 28.0])
    yhat = np.array([22.0, 28.0, 26.0, 29.0])
    line = np.array([25.0, 25.0, 25.0, 25.0])
    proba = np.array([0.4, 0.7, 0.6, 0.65])
    out = oot_metrics(y, yhat, line=line, proba=proba)
    assert out["n"] == 4
    assert "rmse" in out and "mae" in out
    assert "hit_rate" in out and "brier" in out


def test_oot_metrics_minimal_inputs():
    out = oot_metrics([20, 30], [22, 28])
    assert out["n"] == 2
    assert "rmse" in out and "hit_rate" not in out


def test_parse_iso_handles_z_suffix():
    d = parse_iso("2025-04-01T00:00:00Z")
    assert d.tzinfo is not None


# ------------------------------------------------------------- stacking
def test_stacking_blender_fits_and_outputs_unit_interval():
    rng = np.random.default_rng(2)
    n = 400
    base = rng.uniform(0, 1, size=(n, 2))  # two base models
    meta = rng.normal(0, 1, size=(n, 1))   # one meta feature
    # Generate labels that depend on a logistic combination
    z = 2.0 * base[:, 0] + 0.5 * base[:, 1] + 0.3 * meta[:, 0]
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(0, 1, size=n) < p).astype(int)

    sb = StackingBlender(base_names=["hgb", "xgb"], meta_names=["line_dist"])
    sb.fit(base, y, meta=meta)
    pred = sb.predict_proba(base, meta=meta)
    assert pred.shape == (n,)
    assert (pred >= 0).all() and (pred <= 1).all()


def test_stacking_blender_learns_to_favour_signal():
    rng = np.random.default_rng(3)
    n = 600
    # base[:, 0] is informative, base[:, 1] is noise
    base = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    y = (base[:, 0] > 0.5).astype(int)
    sb = StackingBlender(base_names=["good", "noise"]).fit(base, y)
    coefs = sb.coefficients
    assert abs(coefs["good"]) > abs(coefs["noise"])


def test_stacking_blender_rejects_single_class_fit():
    sb = StackingBlender()
    with pytest.raises(ValueError):
        sb.fit(np.array([[0.5], [0.6]]), [0, 0])


def test_stacking_blender_predict_without_fit_errors():
    sb = StackingBlender()
    with pytest.raises(RuntimeError):
        sb.predict_proba(np.array([[0.5]]))


def test_stacking_blender_meta_dim_mismatch_errors():
    sb = StackingBlender().fit(
        np.array([[0.4], [0.6], [0.5], [0.7]]),
        np.array([0, 1, 0, 1]),
    )
    with pytest.raises(ValueError):
        sb.predict_proba(np.array([[0.5]]), meta=np.array([[1.0], [2.0]]))
