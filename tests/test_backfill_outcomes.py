"""Tests for scripts/backfill_prop_outcomes.

Exercises the rescue path that stamps ``game_id`` on legacy
prediction_logs rows and emits prop_outcomes. nba_api is monkeypatched
— we never hit the network.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backfill_prop_outcomes import (  # noqa: E402
    _season_for_date,
    backfill,
)
from src.odds_tracker import OddsTracker  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers + fakes
# --------------------------------------------------------------------------- #

def _bootstrap_db(tmp_path) -> str:
    """Create an empty DB with the prediction_logs schema this script
    expects. We don't drag in BasketballBettingHelper just for the
    table — tests read/write directly.
    """
    db = str(tmp_path / "back.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            player_id INTEGER,
            player_name TEXT,
            prop_type TEXT NOT NULL,
            line REAL NOT NULL,
            actual_result REAL,
            game_id TEXT
        )
    """)
    conn.commit()
    conn.close()
    # Initialising OddsTracker creates the prop_line_summary +
    # prop_outcomes tables as a side-effect.
    OddsTracker(api_key=None, db_path=db)
    return db


def _seed_graded(db: str, *, player_id: int, player_name: str,
                 prop_type: str, line: float, actual: float,
                 timestamp: str) -> int:
    conn = sqlite3.connect(db)
    cur = conn.execute(
        """
        INSERT INTO prediction_logs
        (timestamp, player_id, player_name, prop_type, line, actual_result)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (timestamp, player_id, player_name, prop_type, line, actual),
    )
    conn.commit()
    log_id = int(cur.lastrowid)
    conn.close()
    return log_id


class _FakeGameLog:
    """Stand-in for nba_api.stats.endpoints.playergamelog.PlayerGameLog."""
    df = pd.DataFrame()

    def __init__(self, *args, **kwargs):
        pass

    def get_data_frames(self):  # noqa: D401
        return [self.df]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_season_for_date_winter():
    assert _season_for_date(datetime(2024, 12, 1)) == "2024-25"


def test_season_for_date_summer():
    # July → still belongs to the season that started the prior calendar year
    assert _season_for_date(datetime(2025, 5, 15)) == "2024-25"


def test_backfill_stamps_game_id_and_writes_outcome(tmp_path, monkeypatch):
    db = _bootstrap_db(tmp_path)
    pred_date = (datetime.now() - timedelta(days=2)).date()
    log_id = _seed_graded(
        db,
        player_id=2544, player_name="LeBron James",
        prop_type="points", line=24.5, actual=31.0,
        timestamp=f"{pred_date.isoformat()}T20:00:00",
    )

    _FakeGameLog.df = pd.DataFrame([{
        "GAME_DATE": pred_date.isoformat(),
        "Game_ID": "0022300456",
        "PTS": 31,
    }])
    monkeypatch.setattr(
        "nba_api.stats.endpoints.playergamelog.PlayerGameLog",
        _FakeGameLog,
    )

    result = backfill(db)
    assert result["candidates"] == 1
    assert result["stamped"] == 1
    assert result["outcomes_written"] == 1
    assert result["errors"] == 0

    conn = sqlite3.connect(db)
    pred_row = conn.execute(
        "SELECT game_id FROM prediction_logs WHERE id = ?", (log_id,),
    ).fetchone()
    out_row = conn.execute(
        "SELECT observed_line, actual_result, closing_line FROM prop_outcomes "
        "WHERE game_id = ? AND player_name = ? AND prop_type = ?",
        ("0022300456", "LeBron James", "points"),
    ).fetchone()
    conn.close()

    assert pred_row[0] == "0022300456"
    assert out_row is not None
    assert out_row[0] == pytest.approx(24.5)
    assert out_row[1] == pytest.approx(31.0)
    # No closing line for legacy rows — that's the documented limitation.
    assert out_row[2] is None


def test_backfill_idempotent(tmp_path, monkeypatch):
    """Running the backfill twice should be a no-op the second time
    (since the first run populated game_id on every candidate row)."""
    db = _bootstrap_db(tmp_path)
    pred_date = (datetime.now() - timedelta(days=3)).date()
    _seed_graded(
        db,
        player_id=201939, player_name="Stephen Curry",
        prop_type="three_pointers", line=4.5, actual=6.0,
        timestamp=f"{pred_date.isoformat()}T20:00:00",
    )

    _FakeGameLog.df = pd.DataFrame([{
        "GAME_DATE": pred_date.isoformat(),
        "Game_ID": "0022300999",
        "FG3M": 6,
    }])
    monkeypatch.setattr(
        "nba_api.stats.endpoints.playergamelog.PlayerGameLog",
        _FakeGameLog,
    )

    first = backfill(db)
    second = backfill(db)

    assert first["stamped"] == 1
    assert second["candidates"] == 0
    assert second["stamped"] == 0


def test_backfill_skips_when_no_gamelog_match(tmp_path, monkeypatch):
    """If the player's gamelog has no row matching the prediction date,
    the row is skipped — not stamped, not errored."""
    db = _bootstrap_db(tmp_path)
    pred_date = (datetime.now() - timedelta(days=2)).date()
    log_id = _seed_graded(
        db,
        player_id=999, player_name="No Match",
        prop_type="points", line=20.5, actual=18.0,
        timestamp=f"{pred_date.isoformat()}T20:00:00",
    )

    _FakeGameLog.df = pd.DataFrame(columns=["GAME_DATE", "Game_ID"])
    monkeypatch.setattr(
        "nba_api.stats.endpoints.playergamelog.PlayerGameLog",
        _FakeGameLog,
    )

    result = backfill(db)
    assert result["skipped"] == 1
    assert result["stamped"] == 0
    assert result["errors"] == 0

    conn = sqlite3.connect(db)
    game_id = conn.execute(
        "SELECT game_id FROM prediction_logs WHERE id = ?", (log_id,),
    ).fetchone()[0]
    conn.close()
    assert game_id is None


def test_backfill_dry_run_does_not_mutate(tmp_path, monkeypatch):
    db = _bootstrap_db(tmp_path)
    pred_date = (datetime.now() - timedelta(days=2)).date()
    log_id = _seed_graded(
        db,
        player_id=2544, player_name="LeBron James",
        prop_type="points", line=24.5, actual=31.0,
        timestamp=f"{pred_date.isoformat()}T20:00:00",
    )

    _FakeGameLog.df = pd.DataFrame([{
        "GAME_DATE": pred_date.isoformat(),
        "Game_ID": "0022300456", "PTS": 31,
    }])
    monkeypatch.setattr(
        "nba_api.stats.endpoints.playergamelog.PlayerGameLog",
        _FakeGameLog,
    )

    result = backfill(db, dry_run=True)
    assert result["stamped"] == 1  # counts what *would* have been stamped
    # ...but the actual DB is untouched
    conn = sqlite3.connect(db)
    pred_row = conn.execute(
        "SELECT game_id FROM prediction_logs WHERE id = ?", (log_id,),
    ).fetchone()
    out_count = conn.execute("SELECT COUNT(*) FROM prop_outcomes").fetchone()[0]
    conn.close()
    assert pred_row[0] is None
    assert out_count == 0


def test_backfill_skips_ungraded_rows(tmp_path, monkeypatch):
    """Rows without actual_result aren't candidates — they'll get the
    full grade-and-record treatment when auto_grade_pending runs next."""
    db = _bootstrap_db(tmp_path)
    pred_date = (datetime.now() - timedelta(days=2)).date()
    # Insert one ungraded (actual_result NULL) and one graded row
    conn = sqlite3.connect(db)
    conn.execute(
        """INSERT INTO prediction_logs (timestamp, player_id, player_name,
           prop_type, line, actual_result) VALUES (?, ?, ?, ?, ?, NULL)""",
        (f"{pred_date.isoformat()}T20:00:00", 1, "Ungraded", "points", 20.5),
    )
    conn.commit()
    conn.close()
    _seed_graded(
        db, player_id=2, player_name="Graded", prop_type="points",
        line=22.5, actual=25.0,
        timestamp=f"{pred_date.isoformat()}T20:00:00",
    )

    _FakeGameLog.df = pd.DataFrame([{
        "GAME_DATE": pred_date.isoformat(),
        "Game_ID": "0022300111", "PTS": 25,
    }])
    monkeypatch.setattr(
        "nba_api.stats.endpoints.playergamelog.PlayerGameLog",
        _FakeGameLog,
    )

    result = backfill(db)
    assert result["candidates"] == 1  # only the graded row qualifies
    assert result["stamped"] == 1
