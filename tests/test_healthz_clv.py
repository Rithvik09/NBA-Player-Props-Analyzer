"""Contract tests for /healthz/clv.

The endpoint must respond with a stable shape so an operator dashboard
can poll it without conditional logic. We swap the betting_helper's
``db_name`` to a tmp file pre-populated with whatever shape the test
needs, then GET the endpoint and assert.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.odds_tracker import OddsTracker  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BANKROLL_DB", str(tmp_path / "bankroll.db"))
    from src import app as _app
    db = str(tmp_path / "clv.db")
    # OddsTracker init creates the prop_outcomes + game_tipoffs tables
    OddsTracker(api_key=None, db_path=db)
    monkeypatch.setattr(_app.betting_helper, "db_name", db)
    return _app.app.test_client(), db


def test_healthz_clv_empty_db_returns_zero_counts(client):
    cli, _db = client
    r = cli.get("/healthz/clv")
    assert r.status_code == 200
    j = r.get_json()
    assert j["prop_outcomes_total"] == 0
    assert j["prop_outcomes_with_close"] == 0
    assert j["close_capture_rate"] is None
    assert j["last_outcome_at"] is None
    assert j["upcoming_games"] == 0
    assert j["imminent_games"] == 0


def test_healthz_clv_counts_outcomes_and_close_rate(client):
    cli, db = client
    conn = sqlite3.connect(db)
    # 3 outcomes, 2 with closing line → rate = 0.6667
    rows = [
        ("g1", "A", "points", 24.5, 25.0, 27.0, "2026-05-01T22:00:00+00:00"),
        ("g2", "B", "rebounds", 8.5, None, 9.0, "2026-05-01T23:00:00+00:00"),
        ("g3", "C", "assists", 6.5, 7.0, 5.0, "2026-05-02T01:00:00+00:00"),
    ]
    for gid, name, ptype, obs, close_l, actual, settled in rows:
        conn.execute(
            """
            INSERT INTO prop_outcomes
            (game_id, player_name, prop_type, observed_line,
             closing_line, actual_result, settled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (gid, name, ptype, obs, close_l, actual, settled),
        )
    conn.commit()
    conn.close()

    j = cli.get("/healthz/clv").get_json()
    assert j["prop_outcomes_total"] == 3
    assert j["prop_outcomes_with_close"] == 2
    assert abs(j["close_capture_rate"] - 0.6667) < 1e-3
    assert j["last_outcome_at"] == "2026-05-02T01:00:00+00:00"


def test_healthz_clv_counts_upcoming_and_imminent(client):
    cli, db = client
    near = datetime.now(timezone.utc) + timedelta(minutes=10)
    far = datetime.now(timezone.utc) + timedelta(hours=4)
    conn = sqlite3.connect(db)
    for gid, ct in (
        ("g_near", near.strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("g_far", far.strftime("%Y-%m-%dT%H:%M:%SZ")),
    ):
        conn.execute(
            """INSERT INTO game_tipoffs (game_id, commence_time, last_seen)
               VALUES (?, ?, ?)""",
            (gid, ct, datetime.now(timezone.utc).isoformat()),
        )
    conn.commit()
    conn.close()

    j = cli.get("/healthz/clv").get_json()
    assert j["upcoming_games"] == 2
    assert j["imminent_games"] == 1


def test_healthz_clv_never_returns_503(client):
    """Even with everything broken, the endpoint must stay informational
    so it doesn't trigger ops alarms while CLV is still warming up."""
    cli, _db = client
    r = cli.get("/healthz/clv")
    assert r.status_code != 503
