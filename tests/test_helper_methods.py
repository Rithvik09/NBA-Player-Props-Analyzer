"""Smoke tests for BasketballBettingHelper internal helpers.

We only exercise pieces that are pure + don't need a loaded ML model.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


def _make_helper():
    from src.basketball_betting_helper import BasketballBettingHelper
    # Constructor pulls models from disk — skip by avoiding model-hungry paths
    helper = BasketballBettingHelper.__new__(BasketballBettingHelper)
    helper.db_name = ":memory:"
    return helper


def test_detect_game_type_playoffs(scoreboard_df):
    helper = _make_helper()
    with patch("src.api_cache.fetch_scoreboard_v2", return_value=scoreboard_df):
        result = helper._detect_game_type(1610612738, 1610612747)
    assert result is not None
    assert result["game_type"] == "playoffs"
    assert result["is_playoff"] is True
    assert result["game_id"] == "0042400101"


def test_detect_game_type_no_match(scoreboard_df):
    helper = _make_helper()
    with patch("src.api_cache.fetch_scoreboard_v2", return_value=scoreboard_df):
        # Teams not facing each other today
        result = helper._detect_game_type(1610612738, 1610612743)
    assert result is None


def test_detect_home_away_home(scoreboard_df):
    helper = _make_helper()
    with patch("src.api_cache.fetch_scoreboard_v2", return_value=scoreboard_df):
        assert helper._detect_home_away(1610612738, 1610612747) is True
        assert helper._detect_home_away(1610612747, 1610612738) is False


def test_compute_series_state_empty(empty_playoff_games_df):
    helper = _make_helper()
    with patch("src.api_cache.fetch_playoff_games", return_value=empty_playoff_games_df):
        state = helper._compute_series_state(1610612738, 1610612747, season="2025-26")
    assert state["series_game_num"] == 0.0
    assert state["team_series_wins_in"] == 0.0
    assert state["opp_series_wins_in"] == 0.0
    assert state["is_elimination_game"] == 0.0


def test_compute_series_state_elimination(three_win_playoff_games_df):
    helper = _make_helper()
    with patch("src.api_cache.fetch_playoff_games", return_value=three_win_playoff_games_df):
        state = helper._compute_series_state(1610612738, 1610612747, season="2025-26")
    assert state["series_game_num"] == 4.0
    assert state["team_series_wins_in"] == 3.0
    assert state["opp_series_wins_in"] == 0.0
    assert state["is_elimination_game"] == 1.0
