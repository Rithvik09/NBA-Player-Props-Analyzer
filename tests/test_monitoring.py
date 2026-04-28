"""Tests for src/monitoring.py — drift, decay, anomaly helpers."""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring import (  # noqa: E402
    brier_decay_check,
    detect_feature_drift,
    find_anomalies,
    is_prediction_anomalous,
    ks_statistic,
    rolling_brier,
)


# ---------------------------------------------------------------- KS
def test_ks_identical_samples_zero():
    a = list(range(100))
    assert ks_statistic(a, a) == 0.0


def test_ks_disjoint_samples_one():
    a = [0.0] * 50
    b = [10.0] * 50
    assert ks_statistic(a, b) == 1.0


def test_ks_partial_overlap_in_unit_range():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500).tolist()
    b = rng.normal(0.5, 1, 500).tolist()
    d = ks_statistic(a, b)
    # Mean shift of 0.5σ should give KS roughly in 0.15-0.30
    assert 0.10 < d < 0.40


def test_ks_empty_returns_zero():
    assert ks_statistic([], [1, 2, 3]) == 0.0
    assert ks_statistic([1, 2], []) == 0.0


def test_ks_filters_nan_inf():
    a = [1.0, 2.0, float("nan"), 3.0, float("inf")]
    b = [1.0, 2.0, 3.0]
    assert ks_statistic(a, b) == 0.0  # after filter both = [1,2,3]


# -------------------------------------------------------- detect_feature_drift
def test_detect_drift_flags_shifted_feature():
    rng = np.random.default_rng(1)
    served = {
        "minutes": rng.normal(0, 1, 200).tolist(),  # same distribution
        "fg_pct": rng.normal(2.0, 1, 200).tolist(),  # SHIFTED
    }
    baseline = {
        "minutes": rng.normal(0, 1, 200).tolist(),
        "fg_pct": rng.normal(0.0, 1, 200).tolist(),
    }
    out = detect_feature_drift(served, baseline, ks_threshold=0.20)
    assert len(out) == 2
    fg = next(r for r in out if r["feature"] == "fg_pct")
    mins = next(r for r in out if r["feature"] == "minutes")
    assert fg["drifted"] is True
    assert mins["drifted"] is False
    # Sorted by KS desc
    assert out[0]["feature"] == "fg_pct"


def test_detect_drift_skips_below_min_samples():
    served = {"x": [1, 2, 3]}
    baseline = {"x": [1, 2, 3] * 100}
    out = detect_feature_drift(served, baseline, min_samples=30)
    assert out == []


# -------------------------------------------------------- rolling_brier
def _seed_prediction_logs(db_path: str, rows):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, player_id INTEGER, prop_type TEXT,
            line REAL, predicted_value REAL, confidence REAL,
            actual_result REAL
        )
    """)
    conn.executemany(
        "INSERT INTO prediction_logs (timestamp, player_id, prop_type, "
        "line, predicted_value, confidence, actual_result) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_rolling_brier_computes_per_prop(tmp_path):
    db = str(tmp_path / "t.db")
    now = datetime.now(timezone.utc)
    rows = [
        # All within window (default 30 days)
        ((now - timedelta(days=2)).isoformat(), 1, "points", 25.5, 28, 0.7, 30),  # over win, p=0.7 → (0.7-1)^2=0.09
        ((now - timedelta(days=3)).isoformat(), 1, "points", 25.5, 28, 0.7, 20),  # over loss → (0.7-0)^2=0.49
        # Out of window — must be excluded
        ((now - timedelta(days=60)).isoformat(), 1, "points", 25.5, 28, 0.7, 30),
    ]
    _seed_prediction_logs(db, rows)
    out = rolling_brier(db, window_days=30)
    assert "points" in out["props"]
    n = out["props"]["points"]["n"]
    b = out["props"]["points"]["brier"]
    assert n == 2
    assert abs(b - (0.09 + 0.49) / 2) < 1e-3


def test_rolling_brier_excludes_pushes(tmp_path):
    db = str(tmp_path / "t.db")
    now = datetime.now(timezone.utc)
    rows = [
        ((now - timedelta(days=1)).isoformat(), 1, "points", 25.0, 28, 0.7, 25.0),  # push
        ((now - timedelta(days=1)).isoformat(), 1, "points", 25.5, 28, 0.7, 30),    # win
    ]
    _seed_prediction_logs(db, rows)
    out = rolling_brier(db)
    assert out["props"]["points"]["n"] == 1


# -------------------------------------------------------- brier_decay_check
def test_brier_decay_flags_when_over_threshold():
    rolling = {"props": {
        "points":   {"n": 100, "brier": 0.30},  # +20% vs 0.25 → flagged
        "rebounds": {"n": 100, "brier": 0.22},  # -12% (improved!) → not flagged
    }}
    training = {"points": 0.25, "rebounds": 0.25}
    out = brier_decay_check(rolling, training, degradation_threshold=0.20)
    by_prop = {r["prop"]: r for r in out}
    assert by_prop["points"]["decayed"] is True
    assert by_prop["rebounds"]["decayed"] is False
    # Sorted by degradation desc
    assert out[0]["prop"] == "points"


def test_brier_decay_skips_props_without_training_baseline():
    rolling = {"props": {"unknown_prop": {"n": 50, "brier": 0.5}}}
    out = brier_decay_check(rolling, {})
    assert out == []


# ---------------------------------------------------------- anomaly
def test_is_prediction_anomalous_flags_4sigma():
    out = is_prediction_anomalous(predicted=40.0, line=25.0, residual_std=3.0,
                                   sigma_threshold=3.0)
    assert out["anomalous"] is True
    assert out["z_score"] == 5.0


def test_is_prediction_anomalous_passes_within_3sigma():
    out = is_prediction_anomalous(predicted=27.0, line=25.0, residual_std=3.0,
                                   sigma_threshold=3.0)
    assert out["anomalous"] is False


def test_is_prediction_anomalous_handles_missing_std():
    out = is_prediction_anomalous(40, 25, residual_std=0.0)
    assert out["anomalous"] is False


def test_find_anomalies_filters_by_sigma(tmp_path):
    db = str(tmp_path / "t.db")
    now = datetime.now(timezone.utc)
    rows = [
        ((now - timedelta(days=1)).isoformat(), 1, "points", 25.5, 50, 0.9, None),  # 24.5 over → anomaly
        ((now - timedelta(days=1)).isoformat(), 2, "points", 25.5, 27, 0.6, None),  # within range
    ]
    _seed_prediction_logs(db, rows)
    out = find_anomalies(db, {"points": 3.0}, window_days=7, sigma_threshold=3.0)
    assert len(out) == 1
    assert out[0]["player_id"] == 1
