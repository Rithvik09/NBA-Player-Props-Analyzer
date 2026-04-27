"""HTTP contract test for /analyze_prop.

Heavy lifting in ``BasketballBettingHelper.analyze_prop_bet`` is mocked —
this only verifies the route-level contract (validation, status codes,
payload shape).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BANKROLL_DB", str(tmp_path / "bankroll.db"))
    from src import app as _app
    return _app.app.test_client()


def test_analyze_prop_missing_payload(client):
    r = client.post("/analyze_prop", json={})
    assert r.status_code == 400
    assert "Missing" in r.get_json().get("error", "") or \
           "data" in r.get_json().get("error", "").lower()


def test_analyze_prop_missing_required_field(client):
    # Missing 'opponent_team_id'
    r = client.post("/analyze_prop", json={
        "player_id": 2544, "prop_type": "points", "line": 27.5,
    })
    assert r.status_code == 400
    assert "Missing required fields" in r.get_json()["error"]


def test_analyze_prop_happy_path_mocked(client):
    fake = {
        "player_id": 2544,
        "prop_type": "points",
        "line": 27.5,
        "ml_prediction": 28.4,
        "over_probability": 0.58,
        "recommendation": "OVER",
        "confidence": 0.62,
    }
    with patch("src.app.betting_helper.analyze_prop_bet", return_value=fake):
        r = client.post("/analyze_prop", json={
            "player_id": 2544, "prop_type": "points", "line": 27.5,
            "opponent_team_id": 1610612738,
        })
    assert r.status_code == 200
    j = r.get_json()
    assert j["recommendation"] == "OVER"
    assert "over_probability" in j


def test_analyze_prop_helper_returns_none(client):
    with patch("src.app.betting_helper.analyze_prop_bet", return_value=None):
        r = client.post("/analyze_prop", json={
            "player_id": 1, "prop_type": "points", "line": 10,
            "opponent_team_id": 2,
        })
    assert r.status_code == 500
    assert r.get_json()["success"] is False


def test_analyze_prop_is_home_string_false(client):
    """``is_home='false'`` (string) must be coerced to bool False, not True."""
    captured = {}

    def fake(*args, **kwargs):
        captured["is_home"] = kwargs.get("is_home")
        return {"recommendation": "PASS", "over_probability": 0.5,
                "ml_prediction": 0, "line": 1, "prop_type": "points"}

    with patch("src.app.betting_helper.analyze_prop_bet", side_effect=fake):
        r = client.post("/analyze_prop", json={
            "player_id": 1, "prop_type": "points", "line": 10,
            "opponent_team_id": 2, "is_home": "false",
        })
    assert r.status_code == 200
    assert captured["is_home"] is False


def test_analyze_prop_is_home_omitted_means_none(client):
    captured = {}

    def fake(*args, **kwargs):
        captured["is_home"] = kwargs.get("is_home")
        return {"recommendation": "PASS"}

    with patch("src.app.betting_helper.analyze_prop_bet", side_effect=fake):
        r = client.post("/analyze_prop", json={
            "player_id": 1, "prop_type": "points", "line": 10,
            "opponent_team_id": 2,
        })
    assert r.status_code == 200
    assert captured["is_home"] is None
