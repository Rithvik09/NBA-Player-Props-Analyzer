"""Integration test for auto_grade_pending → record_outcome wiring.

Why this exists: B3's record_outcome / clv_training_rows are useless if
nothing calls them. This test patches the nba_api gamelog endpoint, runs
auto_grade_pending against a synthetic prediction_logs row, and confirms
that:

  1. The graded result lands on prediction_logs.actual_result.
  2. The Game_ID from the gamelog row is stamped onto prediction_logs.game_id.
  3. A prop_outcomes row appears with the closing_line we'd already stamped
     on prop_line_summary, joined by (game_id, player_name, prop_type).

If any link in that chain breaks (column missing, OddsTracker init fails,
record_outcome silently swallowed) the CLV pipeline produces no usable
training data — so this test guards the whole flow end-to-end.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.basketball_betting_helper import BasketballBettingHelper  # noqa: E402
from src.odds_tracker import OddsTracker  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _seed_pending_prediction(
    db_path: str,
    *,
    player_id: int,
    player_name: str,
    prop_type: str,
    line: float,
    timestamp: str,
) -> int:
    """Insert a single ungraded row into prediction_logs and return its id."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            """
            INSERT INTO prediction_logs
            (timestamp, player_id, player_name, prop_type, line,
             is_home, recommendation)
            VALUES (?, ?, ?, ?, ?, 1, 'OVER')
            """,
            (timestamp, player_id, player_name, prop_type, line),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _make_fake_gamelog(*, game_date: str, game_id: str, pts: int) -> pd.DataFrame:
    """Build a tiny PlayerGameLog-shaped DataFrame the grader can join on."""
    return pd.DataFrame([
        {
            "GAME_DATE": game_date,
            "Game_ID": game_id,
            "PTS": pts,
            "AST": 5,
            "REB": 4,
            "STL": 1,
            "BLK": 0,
            "TOV": 2,
            "FG3M": 2,
        }
    ])


class _FakePlayerGameLog:
    """Stand-in for nba_api.stats.endpoints.playergamelog.PlayerGameLog
    that ignores its constructor args and returns a pre-baked df."""

    df: pd.DataFrame = pd.DataFrame()

    def __init__(self, *args, **kwargs):
        pass

    def get_data_frames(self):  # noqa: D401 — match the real interface
        return [self.df]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

@pytest.fixture
def helper(tmp_path):
    db = str(tmp_path / "auto_grade.db")
    h = BasketballBettingHelper(db_name=db)
    return h


def test_auto_grade_records_outcome_with_closing_line(helper, monkeypatch):
    """Grading a pending prediction should write a prop_outcomes row that
    joins with the closing_line previously stamped on prop_line_summary."""
    db = helper.db_name
    yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
    timestamp = f"{yesterday}T20:00:00"

    log_id = _seed_pending_prediction(
        db,
        player_id=2544,
        player_name="LeBron James",
        prop_type="points",
        line=24.5,
        timestamp=timestamp,
    )

    # Pre-stamp a closing line on prop_line_summary so record_outcome has
    # something to join against. We need the same game_id the fake gamelog
    # will return.
    game_id = "0022300456"
    tracker = OddsTracker(api_key=None, db_path=db)
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO prop_line_summary
        (game_id, player_name, prop_type, opening_line, current_line,
         line_movement, opening_over_price, current_over_price,
         price_movement, num_snapshots, first_seen, last_updated,
         consensus_line, line_std, sharp_action_score,
         closing_line, closing_line_at)
        VALUES (?, ?, ?, 24.0, 25.0, 1.0, -110, -110, 0, 1, ?, ?, 25.0, 0, 0,
                25.5, ?)
        """,
        (game_id, "LeBron James", "points",
         f"{yesterday} 18:00:00", f"{yesterday} 19:55:00",
         f"{yesterday} 19:55:00"),
    )
    conn.commit()
    conn.close()

    # Patch the nba_api endpoint used inside auto_grade_pending. Player
    # actually scored 31 → over the 24.5 line.
    _FakePlayerGameLog.df = _make_fake_gamelog(
        game_date=yesterday, game_id=game_id, pts=31,
    )
    monkeypatch.setattr(
        "src.basketball_betting_helper.playergamelog.PlayerGameLog",
        _FakePlayerGameLog,
    )

    out = helper.auto_grade_pending()
    assert out["graded"] == 1, out
    assert out["errors"] == 0, out

    # 1) prediction_logs row was graded AND game_id was stamped.
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT actual_result, actual_outcome, game_id FROM prediction_logs WHERE id = ?",
        (log_id,),
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(31.0)
    assert row[1] == "OVER"
    assert row[2] == game_id

    # 2) prop_outcomes row exists with the closing line carried over from
    # prop_line_summary, plus the observed line we predicted against.
    rows = tracker.clv_training_rows(prop_type="points")
    assert len(rows) == 1, rows
    r = rows[0]
    assert r["game_id"] == game_id
    assert r["player_name"] == "LeBron James"
    assert r["observed_line"] == pytest.approx(24.5)
    assert r["closing_line"] == pytest.approx(25.5)
    assert r["actual_result"] == pytest.approx(31.0)
    assert r["hit"] == 1
    # Drift: closing 25.5 vs observed 24.5 → +1.0 (line moved against us
    # if we bet OVER 24.5, since closer is now 25.5)
    assert r["line_to_close_drift"] == pytest.approx(1.0)


def test_auto_grade_records_outcome_without_closing_line(helper, monkeypatch):
    """If no prop_line_summary row exists, grading still produces a
    prop_outcomes row — just with closing_line NULL. clv_training_rows
    filters those out, but the actual_result is preserved for later
    backfill if a closing line ever shows up."""
    db = helper.db_name
    yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
    timestamp = f"{yesterday}T20:00:00"

    log_id = _seed_pending_prediction(
        db,
        player_id=201939,
        player_name="Stephen Curry",
        prop_type="three_pointers",
        line=4.5,
        timestamp=timestamp,
    )

    game_id = "0022300999"
    _FakePlayerGameLog.df = _make_fake_gamelog(
        game_date=yesterday, game_id=game_id, pts=24,
    )
    # Override FG3M for this test
    _FakePlayerGameLog.df.loc[0, "FG3M"] = 6

    monkeypatch.setattr(
        "src.basketball_betting_helper.playergamelog.PlayerGameLog",
        _FakePlayerGameLog,
    )

    out = helper.auto_grade_pending()
    assert out["graded"] == 1

    conn = sqlite3.connect(db)
    out_row = conn.execute(
        """
        SELECT closing_line, actual_result FROM prop_outcomes
        WHERE game_id = ? AND player_name = ? AND prop_type = ?
        """,
        (game_id, "Stephen Curry", "three_pointers"),
    ).fetchone()
    pred_row = conn.execute(
        "SELECT game_id FROM prediction_logs WHERE id = ?", (log_id,),
    ).fetchone()
    conn.close()

    assert out_row is not None
    assert out_row[0] is None              # closing_line NULL — never captured
    assert out_row[1] == pytest.approx(6.0)
    assert pred_row[0] == game_id          # game_id still stamped on the prediction


def test_auto_grade_skips_when_gamelog_has_no_match(helper, monkeypatch):
    """If the player's gamelog has no row matching the prediction date,
    we skip — no grading, no outcome, no error."""
    db = helper.db_name
    yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
    timestamp = f"{yesterday}T20:00:00"

    _seed_pending_prediction(
        db,
        player_id=999,
        player_name="No Match",
        prop_type="points",
        line=20.5,
        timestamp=timestamp,
    )

    # Empty gamelog → nothing to grade
    _FakePlayerGameLog.df = pd.DataFrame(columns=[
        "GAME_DATE", "Game_ID", "PTS", "AST", "REB",
        "STL", "BLK", "TOV", "FG3M",
    ])
    monkeypatch.setattr(
        "src.basketball_betting_helper.playergamelog.PlayerGameLog",
        _FakePlayerGameLog,
    )

    out = helper.auto_grade_pending()
    assert out["graded"] == 0
    assert out["skipped"] == 1
    assert out["errors"] == 0

    # No prop_outcomes row should have been created
    tracker = OddsTracker(api_key=None, db_path=db)
    assert tracker.clv_training_rows() == []
