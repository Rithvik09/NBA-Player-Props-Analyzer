"""Tests for the pre-game data freshness detector."""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_freshness import assess_freshness, downweight_confidence  # noqa: E402


_TODAY = date(2025, 4, 1)


def test_fresh_within_4_days():
    out = assess_freshness(date(2025, 3, 30), today=_TODAY)
    assert out["status"] == "fresh"
    assert out["days_since"] == 2
    assert out["confidence_multiplier"] == 1.0


def test_stale_between_5_and_13_days():
    out = assess_freshness(date(2025, 3, 25), today=_TODAY)
    assert out["status"] == "stale"
    assert out["days_since"] == 7
    assert 0.7 < out["confidence_multiplier"] < 1.0
    assert "unreliable" in (out["warning"] or "")


def test_very_stale_14_plus_days():
    out = assess_freshness(date(2025, 3, 1), today=_TODAY)
    assert out["status"] == "very_stale"
    assert out["confidence_multiplier"] < 0.7


def test_unknown_when_no_date():
    out = assess_freshness(None, today=_TODAY)
    assert out["status"] == "unknown"
    assert out["confidence_multiplier"] == 1.0


def test_iso_string_input():
    out = assess_freshness("2025-03-30T20:30:00", today=_TODAY)
    assert out["status"] == "fresh"


def test_iso_string_with_z_suffix():
    out = assess_freshness("2025-03-25T00:00:00Z", today=_TODAY)
    assert out["status"] == "stale"


def test_garbage_string_yields_unknown():
    out = assess_freshness("not-a-date", today=_TODAY)
    assert out["status"] == "unknown"


def test_downweight_confidence_applies_multiplier():
    fresh_out = {"confidence_multiplier": 1.0}
    stale_out = {"confidence_multiplier": 0.85}
    very_stale = {"confidence_multiplier": 0.65}
    assert downweight_confidence(0.80, fresh_out) == 0.80
    assert downweight_confidence(0.80, stale_out) == 0.80 * 0.85
    assert abs(downweight_confidence(0.80, very_stale) - 0.80 * 0.65) < 1e-9


def test_downweight_confidence_clamps_to_unit_interval():
    assert downweight_confidence(1.5, {"confidence_multiplier": 1.0}) == 1.0
    assert downweight_confidence(-0.5, {"confidence_multiplier": 1.0}) == 0.0
