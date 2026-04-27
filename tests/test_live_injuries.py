"""Injury-severity unit tests (no network)."""
from __future__ import annotations

from unittest.mock import patch

from src.live_injuries import (
    status_severity, find_player_injury, summarise_team,
    team_severity_features, matchup_injury_context, INJURY_FEATURE_KEYS,
)


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


# ---------------------------------------------------------------------------
# team_severity_features / matchup_injury_context
# ---------------------------------------------------------------------------


def test_team_severity_features_none_id():
    out = team_severity_features(None)
    for k in ("team_severity_max", "team_out_count",
              "team_questionable_count", "team_total_injuries"):
        assert k in out
    assert out["team_severity_max"] == 0.0
    assert out["team_total_injuries"] == 0


def test_team_severity_features_lookup_fail():
    # Force nba_api lookup to return None — should yield zero dict
    with patch("nba_api.stats.static.teams.find_team_name_by_id",
               return_value=None):
        out = team_severity_features(1610612738)
    assert out["team_severity_max"] == 0.0
    assert out["team_out_count"] == 0


def test_team_severity_features_valid():
    fake_info = {"id": 1610612738, "full_name": "Boston Celtics",
                 "abbreviation": "BOS", "nickname": "Celtics"}
    with patch("nba_api.stats.static.teams.find_team_name_by_id",
               return_value=fake_info), \
         patch("src.live_injuries.fetch_live_injuries", return_value=FAKE):
        out = team_severity_features(1610612738, prefix="team_")
    assert out["team_total_injuries"] == 2
    assert out["team_out_count"] == 1
    assert out["team_questionable_count"] == 1
    assert out["team_severity_max"] == 1.0


def test_team_severity_features_prefix():
    fake_info = {"full_name": "Los Angeles Lakers"}
    with patch("nba_api.stats.static.teams.find_team_name_by_id",
               return_value=fake_info), \
         patch("src.live_injuries.fetch_live_injuries", return_value=FAKE):
        out = team_severity_features(1610612747, prefix="opp_")
    assert "opp_severity_max" in out
    assert "opp_total_injuries" in out
    assert out["opp_total_injuries"] == 1


def test_matchup_injury_context_merges_team_and_opp():
    fake_a = {"full_name": "Boston Celtics"}
    fake_b = {"full_name": "Los Angeles Lakers"}

    def _lookup(tid):
        return fake_a if int(tid) == 1 else fake_b

    with patch("nba_api.stats.static.teams.find_team_name_by_id",
               side_effect=_lookup), \
         patch("src.live_injuries.fetch_live_injuries", return_value=FAKE):
        out = matchup_injury_context(1, 2)
    # both halves present
    assert out["team_total_injuries"] == 2
    assert out["opp_total_injuries"] == 1
    assert out["team_severity_max"] == 1.0
    assert out["opp_severity_max"] == 0.15


def test_matchup_injury_context_handles_none():
    out = matchup_injury_context(None, None)
    assert out["team_total_injuries"] == 0
    assert out["opp_total_injuries"] == 0


def test_injury_feature_keys_constant():
    # Ensure the canonical key list matches what team_severity_features emits
    out = team_severity_features(None)
    for k in INJURY_FEATURE_KEYS:
        # team_<x> form
        full_key = k if k.startswith("team_") else f"team_{k}"
        # The constant lists "team_..." names already
        assert full_key in out
