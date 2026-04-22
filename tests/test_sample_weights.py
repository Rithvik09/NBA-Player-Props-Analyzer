"""compute_sample_weights: recency × minutes, tz handling, floor."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from scripts.train_models import compute_sample_weights


@dataclass
class _Ex:
    game_date: pd.Timestamp
    minutes: float = 30.0


def test_tz_naive_game_dates_accepted():
    """Regression: tz-naive game dates must not TypeError against today."""
    exs = [_Ex(game_date=pd.to_datetime("2024-01-15"))]
    w = compute_sample_weights(exs, recency_half_life_days=365.0,
                               minutes_weight=False, min_weight=0.1,
                               today=pd.Timestamp("2025-01-15"))
    assert w.shape == (1,)
    # 365 days old, half-life 365 -> 2^-1 = 0.5
    assert 0.49 < w[0] < 0.51


def test_tz_aware_today_is_stripped():
    """If caller passes tz-aware today, we coerce rather than crash."""
    exs = [_Ex(game_date=pd.to_datetime("2024-01-15"))]
    aware = pd.Timestamp("2025-01-15", tz="UTC")
    w = compute_sample_weights(exs, recency_half_life_days=365.0,
                               minutes_weight=False, min_weight=0.1,
                               today=aware)
    assert 0.49 < w[0] < 0.51


def test_recency_disabled_when_none():
    exs = [_Ex(game_date=pd.to_datetime("2000-01-01"))]  # ancient
    w = compute_sample_weights(exs, recency_half_life_days=None,
                               minutes_weight=False, min_weight=0.1,
                               today=pd.Timestamp("2025-01-15"))
    assert w[0] == pytest.approx(1.0)


def test_min_weight_floor_applied():
    exs = [_Ex(game_date=pd.to_datetime("1990-01-01"))]  # ~35 years old
    w = compute_sample_weights(exs, recency_half_life_days=365.0,
                               minutes_weight=False, min_weight=0.1,
                               today=pd.Timestamp("2025-01-15"))
    # ancient -> collapses to floor
    assert w[0] == pytest.approx(0.1)


def test_minutes_weight_scales_and_clips():
    exs = [
        _Ex(game_date=pd.to_datetime("2025-01-15"), minutes=30.0),  # median
        _Ex(game_date=pd.to_datetime("2025-01-15"), minutes=5.0),   # below clip
        _Ex(game_date=pd.to_datetime("2025-01-15"), minutes=60.0),  # above clip
    ]
    w = compute_sample_weights(exs, recency_half_life_days=None,
                               minutes_weight=True, min_weight=0.01,
                               today=pd.Timestamp("2025-01-15"))
    # median minutes = 30.0 → baseline=1.0
    assert w[0] == pytest.approx(1.0)
    # 5/30 = 0.167 → clipped to 0.4
    assert w[1] == pytest.approx(0.4)
    # 60/30 = 2.0 → clipped to 1.5
    assert w[2] == pytest.approx(1.5)


def test_zero_minutes_noops_minutes_scale():
    exs = [_Ex(game_date=pd.to_datetime("2025-01-15"), minutes=0.0)]
    w = compute_sample_weights(exs, recency_half_life_days=None,
                               minutes_weight=True, min_weight=0.01,
                               today=pd.Timestamp("2025-01-15"))
    assert w[0] == pytest.approx(1.0)  # minutes==0 leaves weight untouched
