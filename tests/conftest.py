"""Shared test fixtures.

These smoke tests MUST NOT hit live nba_api endpoints — network flakiness
would make CI red for the wrong reason. We patch every nba_api.endpoints
constructor with canned fixtures. ``NBA_CACHE_DISABLE=1`` is set so no
stale entries leak between tests.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("NBA_CACHE_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def scoreboard_df():
    """Two-game scoreboard: BOS (home) vs LAL (away), playoff game."""
    return pd.DataFrame([
        {
            "GAME_ID": "0042400101",
            "HOME_TEAM_ID": 1610612738,  # BOS
            "VISITOR_TEAM_ID": 1610612747,  # LAL
            "GAME_DATE_EST": "2026-04-21T00:00:00",
        },
        {
            "GAME_ID": "0042400102",
            "HOME_TEAM_ID": 1610612744,  # GSW
            "VISITOR_TEAM_ID": 1610612743,  # DEN
            "GAME_DATE_EST": "2026-04-21T00:00:00",
        },
    ])


@pytest.fixture
def empty_playoff_games_df():
    """No prior playoff games head-to-head."""
    return pd.DataFrame(columns=["GAME_ID", "GAME_DATE", "WL", "TEAM_ID", "MATCHUP"])


@pytest.fixture
def three_win_playoff_games_df():
    """Team won 3 prior games — next game is elimination."""
    today = pd.Timestamp("2026-04-21")
    rows = []
    for i in range(3):
        rows.append({
            "GAME_ID": f"00424001{i:02d}",
            "GAME_DATE": (today - pd.Timedelta(days=(3 - i) * 2)).strftime("%Y-%m-%d"),
            "WL": "W",
            "TEAM_ID": 1610612738,
            "MATCHUP": "BOS vs. LAL",
        })
    return pd.DataFrame(rows)
