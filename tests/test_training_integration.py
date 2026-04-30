"""End-to-end tests for the upgraded training pipeline.

These exercise EnhancedMLPredictor.train on synthetic data so we know:
  - OOT split engages when timestamps are present
  - Random-split fallback works when they aren't
  - Recency weights + sample_weight propagate through every fit
  - QuantileEnsemble replaces the regressor (predict() returns median)
  - StackedCalibratedClassifier wraps base_a + base_b + blender
  - model_metadata.json gets training_brier + reliability written
  - The downstream sklearn-style API (predict_proba shape (n,2)) is preserved
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_quantile import QuantileEnsemble  # noqa: E402
from src.ml_stacking import StackedCalibratedClassifier, StackingBlender  # noqa: E402


# ─────────────────────────────────────────────────────────────────── helpers
def _synth_training_data(n=400, with_timestamps=True, seed=0):
    """Synthesize NBA-shaped training rows: dict per sample."""
    rng = np.random.default_rng(seed)
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    samples = []
    for i in range(n):
        # Two informative features + a noise feature
        recent_avg = float(rng.normal(20, 5))
        season_avg = float(recent_avg + rng.normal(0, 2))
        noise = float(rng.normal(0, 1))
        line = float(20.0 + rng.normal(0, 1))
        # Stat scales with recent_avg + a bit of noise
        result = float(recent_avg + rng.normal(0, 4))
        sample = {
            "features": {
                "recent_avg": recent_avg,
                "season_avg": season_avg,
                "noise": noise,
                "is_home": float(i % 2),
            },
            "result": result,
            "line": line,
            "prop_type": "points",
        }
        if with_timestamps:
            sample["timestamp"] = (base_date + timedelta(days=i)).isoformat()
        samples.append(sample)
    return samples


@pytest.fixture
def predictor(tmp_path, monkeypatch):
    """Fresh EnhancedMLPredictor pointed at a tmp model dir."""
    monkeypatch.setenv("RECENCY_HALFLIFE_DAYS", "180")
    monkeypatch.setenv("OOT_HOLDOUT_DAYS", "30")
    from src.models import EnhancedMLPredictor
    return EnhancedMLPredictor(model_dir=str(tmp_path))


# ─────────────────────────────────────────────────────────── helper unit tests
def test_recency_weights_decay_with_age(predictor):
    base = datetime(2025, 4, 1, tzinfo=timezone.utc)
    timestamps = [(base - timedelta(days=d)).isoformat() for d in (0, 180, 360)]
    w = predictor._compute_recency_weights(timestamps, halflife_days=180)
    # Most recent = 1.0, halflife = 0.5, two halflives = 0.25
    assert abs(w[0] - 1.0) < 1e-6
    assert abs(w[1] - 0.5) < 1e-6
    assert abs(w[2] - 0.25) < 1e-6


def test_recency_weights_uniform_when_timestamps_missing(predictor):
    w = predictor._compute_recency_weights([None, None, None])
    assert (w == 1.0).all()


def test_temporal_split_uses_oot_when_timestamps_present(predictor):
    base = datetime(2025, 4, 1, tzinfo=timezone.utc)
    # 30 days of data — with 30-day holdout, train should be the older half
    timestamps = [(base - timedelta(days=d)).isoformat() for d in range(60)]
    train_idx, test_idx, mode = predictor._temporal_split_indices(
        timestamps, n=60, holdout_days=30
    )
    assert mode == "oot"
    assert len(train_idx) > 0 and len(test_idx) > 0


def test_temporal_split_falls_back_to_random_without_timestamps(predictor):
    train_idx, test_idx, mode = predictor._temporal_split_indices(
        [None] * 100, n=100, holdout_days=30
    )
    assert mode == "random"
    # Roughly 80/20
    assert 70 <= len(train_idx) <= 90


def test_reliability_bins_well_calibrated_returns_close_pred_actual():
    # Predict 0.7 for every row, with 70% actually positive — perfectly cal'd
    probs = np.full(100, 0.7)
    y = np.array([1] * 70 + [0] * 30)
    from src.models import EnhancedMLPredictor as P
    bins = P._reliability_bins(probs, y, n_bins=10)
    # All probs land in the [0.7, 0.8) bin
    assert len(bins) == 1
    bin_07 = bins[0]
    assert abs(bin_07["mean_pred"] - 0.7) < 1e-9
    assert abs(bin_07["mean_actual"] - 0.7) < 1e-9


def test_ece_zero_when_perfectly_calibrated():
    from src.models import EnhancedMLPredictor as P
    probs = np.full(100, 0.7)
    y = np.array([1] * 70 + [0] * 30)
    bins = P._reliability_bins(probs, y, n_bins=10)
    ece = P._expected_calibration_error(bins)
    assert ece is not None
    assert ece < 1e-9


def test_ece_flags_misscalibrated_model():
    """Predict 0.9, but only 50% actually win — gap of 0.4 → ECE = 0.4."""
    from src.models import EnhancedMLPredictor as P
    probs = np.full(100, 0.9)
    y = np.array([1] * 50 + [0] * 50)
    bins = P._reliability_bins(probs, y, n_bins=10)
    ece = P._expected_calibration_error(bins)
    assert ece is not None
    assert abs(ece - 0.4) < 1e-9


def test_train_writes_global_ece_and_recal_flag(predictor, tmp_path):
    data = _synth_training_data(n=600, with_timestamps=True)
    predictor.train(data)
    metadata = json.loads((tmp_path / "model_metadata.json").read_text())
    assert "ece" in metadata["global"]
    assert "needs_recal" in metadata["global"]
    assert isinstance(metadata["global"]["needs_recal"], bool)


# ─────────────────────────────────────────────────────────── full-pipeline tests
def test_train_with_timestamps_produces_artefacts(predictor, tmp_path):
    data = _synth_training_data(n=600, with_timestamps=True)
    out = predictor.train(data)

    # Every advertised artefact should be on disk
    for fname in ("classification_model.joblib", "regression_model.joblib",
                  "scaler.joblib", "model_metadata.json"):
        assert (tmp_path / fname).exists(), f"missing {fname}"

    # Return contract — new fields appear and old fields still appear
    assert out["model_version"] is not None
    assert out["split_mode"] in ("oot", "random")
    assert "training_brier" in out
    assert "pi80_coverage" in out


def test_train_uses_oot_split_when_timestamps_present(predictor):
    data = _synth_training_data(n=600, with_timestamps=True)
    out = predictor.train(data)
    assert out["split_mode"] == "oot"


def test_train_falls_back_to_random_without_timestamps(predictor):
    data = _synth_training_data(n=400, with_timestamps=False)
    out = predictor.train(data)
    assert out["split_mode"] == "random"


def test_train_writes_per_prop_brier_and_reliability(predictor, tmp_path):
    data = _synth_training_data(n=600, with_timestamps=True)
    predictor.train(data)
    metadata = json.loads((tmp_path / "model_metadata.json").read_text())
    assert "global" in metadata
    assert "training_brier" in metadata["global"]
    assert metadata["props"]["points"]["training_brier"] is not None
    assert isinstance(metadata["props"]["points"]["reliability"], list)


def test_train_resulting_classifier_returns_n_by_2_probas(predictor):
    """Critical: downstream inference does predict_proba(X)[:, 1]. The new
    StackedCalibratedClassifier must preserve that shape."""
    data = _synth_training_data(n=600, with_timestamps=True)
    predictor.train(data)
    # Build a small inference batch with the same features
    Xq = pd.DataFrame([{
        "recent_avg": 22.0, "season_avg": 21.0, "noise": 0.0, "is_home": 1.0,
    } for _ in range(5)])
    scaled = predictor.scaler.transform(Xq)
    probs = predictor.classification_model.predict_proba(scaled)
    assert probs.shape == (5, 2)
    assert (probs >= 0).all() and (probs <= 1).all()
    # rows sum to ~1
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_train_resulting_regressor_predict_returns_median(predictor):
    """QuantileEnsemble.predict must return the median (q_0.5)."""
    data = _synth_training_data(n=600, with_timestamps=True)
    predictor.train(data)
    Xq = pd.DataFrame([{
        "recent_avg": 22.0, "season_avg": 21.0, "noise": 0.0, "is_home": 1.0,
    } for _ in range(3)])
    scaled = predictor.scaler.transform(Xq)
    pt_pred = predictor.regression_model.predict(scaled)
    intervals = predictor.regression_model.predict_intervals(scaled)
    # Point prediction must equal the median forecast
    assert np.allclose(pt_pred, intervals["median"])
    # And should fall inside the [lower, upper] band
    assert (pt_pred >= intervals["lower"] - 1e-9).all()
    assert (pt_pred <= intervals["upper"] + 1e-9).all()


def test_stacked_classifier_inherits_feature_names(predictor):
    """For column alignment in inference, feature_names_in_ must be present."""
    data = _synth_training_data(n=600, with_timestamps=True)
    predictor.train(data)
    clf = predictor.classification_model
    # Either the wrapper exposes feature_names_in_ directly, or the inner cal_a does
    names = getattr(clf, "feature_names_in_", None)
    if names is None and hasattr(clf, "base_a"):
        inner = getattr(clf.base_a, "estimator", getattr(clf.base_a, "base_estimator", None))
        names = getattr(inner, "feature_names_in_", None) or getattr(clf.base_a, "feature_names_in_", None)
    assert names is not None
    assert len(names) == 4  # the four synth features


def test_monotonic_cst_helper_returns_signed_array():
    from src.models import EnhancedMLPredictor as P
    rules = {"recent_avg": 1, "line": -1, "noise": 0}
    feat = ["recent_avg", "noise", "line", "unrelated"]
    arr = P._build_monotonic_cst(feat, rules)
    assert arr is not None
    assert list(arr) == [1, 0, -1, 0]


def test_monotonic_cst_helper_returns_none_when_all_zero():
    from src.models import EnhancedMLPredictor as P
    arr = P._build_monotonic_cst(["a", "b"], {"c": 1})
    assert arr is None


def test_train_attaches_calibration_warning_to_predictions(predictor):
    """predict_prop must surface model_calibration_warning on every response."""
    data = _synth_training_data(n=600, with_timestamps=True)
    predictor.train(data)
    feat = {"recent_avg": 22.0, "season_avg": 21.0, "noise": 0.0, "is_home": 1.0}
    out = predictor.predict(feat, line=20.0, prop_type="points")
    assert "model_calibration_warning" in out
    assert isinstance(out["model_calibration_warning"], bool)


def test_train_recency_weighting_can_be_disabled_via_env(predictor, monkeypatch):
    """Setting RECENCY_HALFLIFE_DAYS=0 should produce uniform weights."""
    monkeypatch.setenv("RECENCY_HALFLIFE_DAYS", "0")
    base = datetime(2025, 4, 1, tzinfo=timezone.utc)
    timestamps = [(base - timedelta(days=d)).isoformat() for d in range(10)]
    w = predictor._compute_recency_weights(timestamps)
    assert (w == 1.0).all()
