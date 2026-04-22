"""HTTP contract tests for the new endpoints.

Uses Flask test client so no real NBA/odds API is hit. Network-touching
endpoints (/injuries/*) are mocked.
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point bankroll SQLite at a throwaway path per-test
    monkeypatch.setenv("BANKROLL_DB", str(tmp_path / "bankroll.db"))
    # Avoid any inadvertent network calls from scrapers during import
    from src import app as _app
    return _app.app.test_client()


def test_cache_stats(client):
    r = client.get("/cache/stats")
    assert r.status_code == 200
    j = r.get_json()
    assert "caches" in j
    assert "total_hits" in j


def test_kelly_accepts_our_prob(client):
    r = client.post("/kelly", json={"our_prob": 0.58, "american_odds": -110,
                                     "bankroll": 1000})
    assert r.status_code == 200
    j = r.get_json()
    assert j["stake_dollars"] > 0
    assert 0 < j["stake_fraction"] <= 0.05  # respects max_fraction cap


def test_kelly_zero_stake_when_no_edge(client):
    # 0.48 < break-even at -110 (0.5238), so zero stake expected
    r = client.post("/kelly", json={"our_prob": 0.48, "american_odds": -110,
                                     "bankroll": 1000})
    assert r.status_code == 200
    assert r.get_json()["stake_dollars"] == 0.0


def test_parlay_accepts_prob_or_our_prob(client):
    body = {"legs": [
        {"prop": "points", "american_odds": -110, "prob": 0.55},
        {"prop": "assists", "american_odds": -110, "prob": 0.55},
    ], "n_samples": 1000}
    r = client.post("/parlay", json=body)
    assert r.status_code == 200
    j = r.get_json()
    assert 0 < j["naive_prob"] < 1
    assert 0 < j["correlated_prob"] < 1

    # alias
    body2 = {"legs": [
        {"prop": "points", "american_odds": -110, "our_prob": 0.55},
        {"prop": "assists", "american_odds": -110, "our_prob": 0.55},
    ], "n_samples": 1000}
    r2 = client.post("/parlay", json=body2)
    assert r2.status_code == 200


def test_parlay_missing_prob_rejected(client):
    r = client.post("/parlay", json={"legs": [
        {"prop": "points", "american_odds": -110}  # no prob
    ]})
    assert r.status_code == 400
    assert "prob" in r.get_json()["error"]


def test_parlay_empty_legs_rejected(client):
    r = client.post("/parlay", json={"legs": []})
    assert r.status_code == 400


def test_odds_status(client):
    r = client.get("/odds/status")
    assert r.status_code == 200
    j = r.get_json()
    assert isinstance(j, dict)


def test_bankroll_summary(client):
    r = client.get("/bankroll")
    assert r.status_code == 200
    j = r.get_json()
    for k in ("balance", "open_bets", "wins", "losses", "pushes", "roi"):
        assert k in j


def test_bankroll_set_balance(client):
    r = client.post("/bankroll/balance", json={"amount": 2500.0})
    assert r.status_code == 200
    assert r.get_json()["balance"] == 2500.0


def test_bankroll_record_bet_accepts_prop_alias(client):
    # 'prop' alias + 'stake_dollars' alias should both work
    r = client.post("/bankroll/bets", json={
        "player_name": "Jayson Tatum",
        "prop": "points",
        "line": 27.5,
        "side": "over",
        "our_prob": 0.58,
        "american_odds": -110,
        "stake_dollars": 25,
    })
    assert r.status_code == 201, r.get_json()
    assert "id" in r.get_json()


def test_bankroll_record_bet_missing_prop(client):
    r = client.post("/bankroll/bets", json={
        "line": 27.5, "our_prob": 0.58, "american_odds": -110, "stake": 25,
    })
    assert r.status_code == 400


def test_bankroll_settle_roundtrip(client):
    # place -> settle -> summary reflects result
    r = client.post("/bankroll/bets", json={
        "player_name": "Jayson Tatum", "prop": "points", "line": 27.5,
        "side": "over", "our_prob": 0.58, "american_odds": -110,
        "stake": 50,
    })
    bet_id = r.get_json()["id"]
    r2 = client.post(f"/bankroll/bets/{bet_id}/settle", json={"result": "win"})
    assert r2.status_code == 200
    summary = client.get("/bankroll").get_json()
    assert summary["wins"] >= 1


def test_injuries_endpoint_mocked(client):
    fake = {
        "Boston Celtics": [
            {"player": "Jayson Tatum", "status": "Questionable",
             "injury": "ankle", "return": "game-time", "severity": 0.5}
        ]
    }
    with patch("src.live_injuries.fetch_live_injuries", return_value=fake):
        r = client.get("/injuries/celtics")
    assert r.status_code == 200
    j = r.get_json()
    assert j["count"] == 1


def test_injuries_player_endpoint_mocked(client):
    fake = {
        "Boston Celtics": [
            {"player": "Jayson Tatum", "status": "Out",
             "injury": "calf", "return": "TBD", "severity": 1.0}
        ]
    }
    with patch("src.live_injuries.fetch_live_injuries", return_value=fake):
        r = client.get("/injuries/player/Jayson%20Tatum")
    assert r.status_code == 200
    assert r.get_json()["status"] == "Out"
