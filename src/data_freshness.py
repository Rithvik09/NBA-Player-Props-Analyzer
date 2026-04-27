"""Stale-data detector for analyze_prop_bet.

The recent-form features (last5_avg, last10_avg, trend_slope, b2b_flag…) all
assume the player has a normal NBA cadence — roughly one game every 1-3 days.
When that cadence breaks, the model is silently extrapolating from very old
state:

  - returning from a multi-week injury → "recent form" is pre-injury
  - all-star break / off-season tail → ditto, plus possible role change
  - rest day clusters at the playoff start → less severe, but still a flag

This module classifies the freshness of a player's data into one of:

  - ``fresh``       — last game within 4 days
  - ``stale``       — 5-13 days since last game, lower confidence
  - ``very_stale``  — 14+ days, treat predictions as a guess
  - ``unknown``     — no game-date metadata available

The output is a small dict that ``analyze_prop_bet`` attaches to its response
so the UI / engine can downweight or veto. We intentionally do NOT silently
modify the prediction itself — the model wasn't trained with a freshness
flag, so applying one at inference would distort calibration. We just flag.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional


_FRESH_DAYS = 4
_STALE_DAYS = 13


def _coerce_date(value) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return datetime.strptime(value[:10], "%Y-%m-%d").date()
            except ValueError:
                return None
    return None


def assess_freshness(
    last_game_date,
    today: Optional[date] = None,
) -> dict:
    """Return a freshness verdict.

    Output keys:
      - ``status``        — "fresh" | "stale" | "very_stale" | "unknown"
      - ``days_since``    — int days since last game (None if unknown)
      - ``confidence_multiplier`` — suggested multiplier on ``confidence``
                                    (1.0 fresh, 0.85 stale, 0.65 very stale,
                                    1.0 unknown so we don't over-penalise)
      - ``warning``       — human-readable string for the UI
    """
    today = today or datetime.now(timezone.utc).date()
    last = _coerce_date(last_game_date)

    if last is None:
        return {
            "status": "unknown",
            "days_since": None,
            "confidence_multiplier": 1.0,
            "warning": None,
        }

    days = (today - last).days

    if days <= _FRESH_DAYS:
        return {
            "status": "fresh",
            "days_since": int(days),
            "confidence_multiplier": 1.0,
            "warning": None,
        }
    if days <= _STALE_DAYS:
        return {
            "status": "stale",
            "days_since": int(days),
            "confidence_multiplier": 0.85,
            "warning": f"Player has not played in {days} days — recent-form features may be unreliable.",
        }
    return {
        "status": "very_stale",
        "days_since": int(days),
        "confidence_multiplier": 0.65,
        "warning": f"Player last played {days} days ago — predictions are highly uncertain (likely returning from injury / break).",
    }


def downweight_confidence(
    confidence: float,
    freshness: dict,
) -> float:
    """Apply the freshness multiplier to a confidence score."""
    try:
        m = float(freshness.get("confidence_multiplier", 1.0))
    except (TypeError, ValueError):
        m = 1.0
    return max(0.0, min(1.0, float(confidence) * m))
