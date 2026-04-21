"""Injury-severity unit tests (no network)."""
from __future__ import annotations

from unittest.mock import patch

from src.live_injuries import status_severity, find_player_injury, summarise_team


def test_severity_out():
    assert status_severity("Out") == 1.0


def test_severity_doubtful():
    assert status_severity("Doubtful") == 0.85


def test_severity_questionable_capital():
    assert status_severity("QUESTIONABLE") == 0.50


def test_severity_unknown_string():
    assert status_severity("totally novel status") == 0.0


FAKE = {
    "Boston Celtics": [
        {"player": "Kristaps Porziņģis", "status": "Out",
         "injury": "calf", "return": "TBD", "severity": 1.0},
        {"player": "Jayson Tatum", "status": "Questionable",
         "injury": "ankle", "return": "game-time", "severity": 0.5},
    ],
    "Los Angeles Lakers": [
        {"player": "LeBron James", "status": "Probable",
         "injury": "groin", "return": "plays", "severity": 0.15},
    ],
}


def test_find_player_exact():
    with patch("src.live_injuries.fetch_live_injuries", return_value=FAKE):
        hit = find_player_injury("Jayson Tatum")
    assert hit is not None
    assert hit["status"] == "Questionable"


def test_find_player_partial():
    with patch("src.live_injuries.fetch_live_injuries", return_value=FAKE):
        hit = find_player_injury("Porzingis")
    assert hit is not None
    assert hit["severity"] == 1.0


def test_find_player_missing():
    with patch("src.live_injuries.fetch_live_injuries", return_value=FAKE):
        assert find_player_injury("Nobody McPlayer") is None


def test_summarise_team_celtics():
    with patch("src.live_injuries.fetch_live_injuries", return_value=FAKE):
        s = summarise_team("celtics")
    assert s["count"] == 2
    assert s["out_count"] == 1
    assert s["questionable_count"] == 1
    assert s["max_severity"] == 1.0


def test_summarise_team_unknown():
    with patch("src.live_injuries.fetch_live_injuries", return_value=FAKE):
        s = summarise_team("warriors")
    assert s["count"] == 0
    assert s["max_severity"] == 0.0
