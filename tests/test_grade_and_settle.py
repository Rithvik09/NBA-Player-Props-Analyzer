"""Tests for the auto-grade-and-settle cron script."""
from __future__ import annotations

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.grade_and_settle import _decide_result, settle_open_bets  # noqa: E402
from src.bankroll import BankrollTracker  # noqa: E402


def test_decide_result_over_win():
    assert _decide_result("over", 25.5, 30) == "win"


def test_decide_result_over_loss():
    assert _decide_result("over", 25.5, 20) == "loss"


def test_decide_result_under_win():
    assert _decide_result("under", 25.5, 20) == "win"


def test_decide_result_under_loss():
    assert _decide_result("under", 25.5, 30) == "loss"


def test_decide_result_exact_line_pushes():
    assert _decide_result("over", 25.0, 25.0) == "push"
    assert _decide_result("under", 25.0, 25.0) == "push"


def test_decide_result_rejects_unknown_side():
    with pytest.raises(ValueError):
        _decide_result("sideways", 10.0, 12.0)


def test_settle_open_bets_settles_only_graded(tmp_path):
    db = str(tmp_path / "test.db")
    # Build a minimal prediction_logs table (mirroring the schema we depend on)
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actual_result REAL,
            timestamp TEXT
        )
    """)
    cur = conn.execute(
        "INSERT INTO prediction_logs (actual_result, timestamp) VALUES (?, ?)",
        (32.0, "2025-04-01T00:00:00+00:00"),
    )
    graded_log_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO prediction_logs (actual_result, timestamp) VALUES (NULL, ?)",
        ("2025-04-01T00:00:00+00:00",),
    )
    pending_log_id = cur.lastrowid
    conn.commit()
    conn.close()

    # Set up bankroll on the same DB and place two bets
    tracker = BankrollTracker(db)
    tracker.set_balance(1000.0)

    bet_graded = tracker.record_bet(
        player_name="Tatum", prop_type="points", side="over", line=27.5,
        american_odds=-110, our_prob=0.6, stake=50.0,
        prediction_log_id=graded_log_id,
    )
    bet_pending = tracker.record_bet(
        player_name="Brown", prop_type="points", side="over", line=22.5,
        american_odds=-110, our_prob=0.55, stake=50.0,
        prediction_log_id=pending_log_id,
    )

    result = settle_open_bets(db)
    assert result["settled"] == 1, result
    assert result["skipped"] == 1, result
    assert result["errors"] == 0, result

    # graded bet should now be settled with a win (actual 32 > line 27.5)
    bets = tracker.list_bets()
    by_id = {b["id"]: b for b in bets}
    assert by_id[bet_graded]["status"] == "settled"
    assert by_id[bet_graded]["result"] == "win"
    assert by_id[bet_pending]["status"] == "open"


def test_settle_open_bets_dry_run_does_not_mutate(tmp_path):
    db = str(tmp_path / "test.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actual_result REAL,
            timestamp TEXT
        )
    """)
    cur = conn.execute(
        "INSERT INTO prediction_logs (actual_result, timestamp) VALUES (?, ?)",
        (32.0, "2025-04-01T00:00:00+00:00"),
    )
    log_id = cur.lastrowid
    conn.commit()
    conn.close()

    tracker = BankrollTracker(db)
    tracker.set_balance(1000.0)
    bet_id = tracker.record_bet(
        player_name="Tatum", prop_type="points", side="over", line=27.5,
        american_odds=-110, our_prob=0.6, stake=50.0,
        prediction_log_id=log_id,
    )

    settle_open_bets(db, dry_run=True)
    bets = tracker.list_bets()
    assert {b["id"]: b for b in bets}[bet_id]["status"] == "open"
