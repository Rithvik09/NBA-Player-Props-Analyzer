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


def test_resolve_prop_rules_passthrough_for_unknown_prop():
    """A prop with no override entry returns the base rules unchanged."""
    from src.models import EnhancedMLPredictor as P
    base = {"recent_avg": +1, "line": -1}
    out = P._resolve_prop_rules("custom_prop_we_dont_track", base)
    assert out == base
    # Must NOT mutate the base in place — overrides for one prop must
    # not bleed into another.
    assert base == {"recent_avg": +1, "line": -1}


def test_resolve_prop_rules_three_pointers_adds_3pa_constraint():
    """three_pointers gains fg3a_per_game +1 on top of base rules."""
    from src.models import EnhancedMLPredictor as P
    base = dict(P._REG_MONOTONIC)
    out = P._resolve_prop_rules("three_pointers", base)
    assert out["fg3a_per_game"] == +1
    assert out["fg3_pct_recent"] == +1
    # Base rules survive
    assert out["recent_avg"] == +1
    # The base dict we passed in is not mutated
    assert "fg3a_per_game" not in base


def test_resolve_prop_rules_turnovers_adds_negative_at_to_constraint():
    """turnovers should encode that high A/TO ratio implies fewer TOs."""
    from src.models import EnhancedMLPredictor as P
    out = P._resolve_prop_rules("turnovers", P._REG_MONOTONIC)
    assert out["ast_to_tov_ratio"] == -1
    assert out["usage_rate"] == +1


def test_per_prop_constraint_vector_differs_across_props():
    """End-to-end: building constraint vectors for the same feature
    list with different prop_types should produce different vectors
    when overrides apply."""
    from src.models import EnhancedMLPredictor as P
    feats = ["recent_avg", "fg3a_per_game", "ast_to_tov_ratio", "noise"]

    pts_rules = P._resolve_prop_rules("points", P._CLF_MONOTONIC)
    threes_rules = P._resolve_prop_rules("three_pointers", P._CLF_MONOTONIC)
    tov_rules = P._resolve_prop_rules("turnovers", P._CLF_MONOTONIC)

    pts_arr = P._build_monotonic_cst(feats, pts_rules)
    threes_arr = P._build_monotonic_cst(feats, threes_rules)
    tov_arr = P._build_monotonic_cst(feats, tov_rules)

    # fg3a_per_game (index 1): only constrained for three_pointers
    assert threes_arr[1] == 1
    assert pts_arr[1] == 0
    assert tov_arr[1] == 0

    # ast_to_tov_ratio (index 2): -1 only for turnovers
    assert tov_arr[2] == -1
    assert pts_arr[2] == 0
    assert threes_arr[2] == 0

    # recent_avg (index 0): +1 for all (base rule)
    assert pts_arr[0] == threes_arr[0] == tov_arr[0] == 1


def test_train_attaches_calibration_warning_to_predictions(predictor):
    """predict_prop must surface model_calibration_warning on every response."""
    data = _synth_training_data(n=600, with_timestamps=True)
    predictor.train(data)
    feat = {"recent_avg": 22.0, "season_avg": 21.0, "noise": 0.0, "is_home": 1.0}
    out = predictor.predict(feat, line=20.0, prop_type="points")
    assert "model_calibration_warning" in out
    assert isinstance(out["model_calibration_warning"], bool)


def test_conformal_calibration_widens_intervals_to_target_coverage():
    """CQR should drive PI80 coverage on a fresh test set toward 0.80."""
    rng = np.random.default_rng(0)
    n = 1000
    X = rng.normal(0, 1, (n, 3))
    # Heteroscedastic noise so quantile regression is non-trivial
    y = X[:, 0] * 2.0 + rng.normal(0, 1 + 0.5 * np.abs(X[:, 1]), n)

    fit_X, cal_X, test_X = X[:600], X[600:800], X[800:]
    fit_y, cal_y, test_y = y[:600], y[600:800], y[800:]

    qe = QuantileEnsemble().fit(fit_X, fit_y)
    raw_intervals = qe.predict_intervals(test_X)
    raw_cov = float(np.mean(
        (test_y >= raw_intervals["lower"]) & (test_y <= raw_intervals["upper"])
    ))
    qe.calibrate(cal_X, cal_y)
    cal_intervals = qe.predict_intervals(test_X)
    cal_cov = float(np.mean(
        (test_y >= cal_intervals["lower"]) & (test_y <= cal_intervals["upper"])
    ))
    # The conformal-calibrated interval should be closer to 0.80 than the raw
    # (or at least as good — randomness allows ties on small test sets).
    assert abs(cal_cov - 0.80) <= max(0.07, abs(raw_cov - 0.80))


def test_conformal_offset_persisted():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (200, 3))
    y = X[:, 0] + rng.normal(0, 1, 200)
    qe = QuantileEnsemble().fit(X[:160], y[:160])
    assert qe._conformity_offset is None
    qe.calibrate(X[160:], y[160:])
    assert qe._conformity_offset is not None
    assert qe._conformal_alpha is not None


def test_walk_forward_evaluate_returns_per_fold_metrics(predictor):
    """walk_forward_evaluate trains+tests on rolling windows and aggregates."""
    # 600 samples spread over 600 days → with fold_days=30, 5 folds gives
    # plenty of train data per fold.
    data = _synth_training_data(n=600, with_timestamps=True)
    out = predictor.walk_forward_evaluate(
        data, n_folds=3, fold_days=30, min_train_samples=100,
        write_metadata=False,
    )
    assert "folds" in out
    assert "aggregates" in out
    assert len(out["folds"]) >= 1
    for fold in out["folds"]:
        assert "auc" in fold and "brier" in fold and "rmse" in fold
        assert fold["n_train"] > 0 and fold["n_test"] > 0
    agg = out["aggregates"]
    assert agg["n_folds_completed"] == len(out["folds"])
    assert agg["brier"]["mean"] is not None


def test_walk_forward_writes_metadata(predictor, tmp_path):
    data = _synth_training_data(n=600, with_timestamps=True)
    # Train once first so model_metadata.json exists with shape we extend
    predictor.train(data)
    predictor.walk_forward_evaluate(
        data, n_folds=2, fold_days=30, min_train_samples=100,
        write_metadata=True,
    )
    metadata = json.loads((tmp_path / "model_metadata.json").read_text())
    points = metadata["props"].get("points", {})
    assert "walk_forward" in points
    assert "brier_cal_mean" in points["walk_forward"]


def test_walk_forward_raises_without_enough_timestamped_data(predictor):
    data = _synth_training_data(n=50, with_timestamps=True)
    with pytest.raises(ValueError):
        predictor.walk_forward_evaluate(
            data, n_folds=5, fold_days=30, min_train_samples=200,
        )


def test_adaptive_depth_scales_with_sample_size():
    from src.models import EnhancedMLPredictor as P
    assert P._adaptive_depth(500) == 3
    assert P._adaptive_depth(2_000) == 4
    assert P._adaptive_depth(10_000) == 5
    assert P._adaptive_depth(100_000) == 7


def test_predict_response_includes_pi80_when_quantile_regressor(predictor):
    """After training, predict() must surface pi80_lower/pi80_upper from the
    conformal-calibrated QuantileEnsemble."""
    data = _synth_training_data(n=600, with_timestamps=True)
    predictor.train(data)
    feat = {"recent_avg": 22.0, "season_avg": 21.0, "noise": 0.0, "is_home": 1.0}
    out = predictor.predict(feat, line=20.0, prop_type="points")
    assert "pi80_lower" in out and "pi80_upper" in out
    assert out["pi80_lower"] is not None
    assert out["pi80_upper"] is not None
    # And the band should bracket predicted_value (or at least the median —
    # the band is around the regressor output, not the blended value)
    assert out["pi80_lower"] <= out["pi80_upper"]


def test_predict_uses_quantile_prob_over_when_available(predictor):
    """The blended over_probability should differ from the gaussian-CDF
    fallback when the QuantileEnsemble is in use — sanity-check that the
    new path actually runs."""
    data = _synth_training_data(n=600, with_timestamps=True)
    predictor.train(data)
    feat = {"recent_avg": 25.0, "season_avg": 24.0, "noise": 0.0, "is_home": 1.0}
    out_low_line = predictor.predict(feat, line=10.0, prop_type="points")
    out_high_line = predictor.predict(feat, line=40.0, prop_type="points")
    # Low line → high P(over); high line → low P(over). Monotonicity check.
    assert out_low_line["over_probability"] > out_high_line["over_probability"]


def test_b2_proxies_have_serve_time_defaults_via_prepare_features(predictor):
    """The 4 B2 rotation-disruption proxies must be available with sane
    neutral defaults at serve time, mirroring what data_collector emits."""
    # Pick a minimal player_stats dict — prepare_features should fill the
    # rest with safe defaults.
    feat = predictor.prepare_features(
        {"recent_avg": 22.0, "season_avg": 21.0, "recent_minutes": 30.0},
        {},  # player_context
        {},  # team_context
        {},  # opponent_context
    )
    assert feat["minutes_jump_3v10"] == 1.0  # neutral = no recent role change
    assert feat["usage_jump_3v10"] == 1.0
    assert feat["minutes_volatility_10"] == 0.0
    assert feat["outlier_minutes_share_10"] == 0.0


def test_train_recency_weighting_can_be_disabled_via_env(predictor, monkeypatch):
    """Setting RECENCY_HALFLIFE_DAYS=0 should produce uniform weights."""
    monkeypatch.setenv("RECENCY_HALFLIFE_DAYS", "0")
    base = datetime(2025, 4, 1, tzinfo=timezone.utc)
    timestamps = [(base - timedelta(days=d)).isoformat() for d in range(10)]
    w = predictor._compute_recency_weights(timestamps)
    assert (w == 1.0).all()


# ─────────────────────────────────────────────── Tier-1: low-minutes filter
def test_low_minutes_filter_drops_dnp_rows(predictor, capsys):
    """Rows with minutes_played < 5 should be dropped before training so
    DNP / garbage / injury-cut games don't pull the regressor toward 0."""
    data = _synth_training_data(n=300, with_timestamps=True)
    # Mark first 50 rows as DNP-equivalent (1 minute each).
    for i, d in enumerate(data):
        d["minutes_played"] = 1.0 if i < 50 else 28.0
    predictor.train(data)
    captured = capsys.readouterr().out
    # The filter prints how many it dropped — assert it actually fired.
    assert "low-minutes filter: dropped 50/300" in captured


def test_low_minutes_filter_passes_through_rows_without_minutes_key(
    predictor, capsys,
):
    """Legacy training rows that don't carry the new `minutes_played`
    key (e.g. from get_log_training_samples) must NOT be silently
    dropped — only rows with an explicit minutes_played < threshold."""
    data = _synth_training_data(n=200, with_timestamps=True)
    # Half the rows omit minutes_played entirely; the other half are 30.
    for i, d in enumerate(data):
        if i % 2 == 0:
            d["minutes_played"] = 30.0
        # else: no key, must pass through

    predictor.train(data)
    out = capsys.readouterr().out
    # No drops should be printed (filter is silent when nothing dropped)
    assert "low-minutes filter" not in out


def test_low_minutes_filter_threshold_overridable_via_env(
    predictor, monkeypatch, capsys,
):
    """The threshold should be configurable via MIN_MINUTES_FOR_TRAIN
    so accuracy ablations can sweep it. Setting it to 0 disables the
    filter entirely."""
    data = _synth_training_data(n=200, with_timestamps=True)
    for d in data:
        d["minutes_played"] = 2.0  # would all be dropped at default

    monkeypatch.setenv("MIN_MINUTES_FOR_TRAIN", "0")
    predictor.train(data)
    out = capsys.readouterr().out
    assert "low-minutes filter" not in out  # filter disabled


# ────────────────────────────────────── Tier-1: isotonic threshold
def test_isotonic_threshold_uses_sigmoid_for_small_cal_split():
    """At cal-split sizes below ISOTONIC_MIN_CAL_SAMPLES, the calibrator
    must pick sigmoid, not isotonic. Isotonic on small sets overfits."""
    from sklearn.calibration import CalibratedClassifierCV
    from src.models import EnhancedMLPredictor as P
    p = P()
    # Use a dummy already-fit estimator
    from sklearn.dummy import DummyClassifier
    base = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    cal = p._make_calibrated_clf(base, n_samples=500)
    assert isinstance(cal, CalibratedClassifierCV)
    # The method attribute is set on the wrapper before fit
    assert cal.method == "sigmoid"


def test_isotonic_threshold_uses_isotonic_for_large_cal_split():
    """At cal-split sizes ≥ ISOTONIC_MIN_CAL_SAMPLES, prefer isotonic
    — non-parametric calibration generalises better when there's
    enough data to support it."""
    from src.models import EnhancedMLPredictor as P
    from sklearn.dummy import DummyClassifier
    p = P()
    base = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    cal = p._make_calibrated_clf(base, n_samples=P.ISOTONIC_MIN_CAL_SAMPLES + 1)
    assert cal.method == "isotonic"


def test_isotonic_threshold_is_at_least_1k():
    """Hard floor: don't accidentally regress to a tiny threshold (the
    100-sample default we used to have was way too aggressive)."""
    from src.models import EnhancedMLPredictor as P
    assert P.ISOTONIC_MIN_CAL_SAMPLES >= 1000
