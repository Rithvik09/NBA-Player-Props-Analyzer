"""B3: closing-line capture + prop_outcomes (CLV groundwork).

These tests exercise the data plumbing only — they never hit the Odds API.
We pre-seed prop_line_history with synthetic snapshots, then call the
capture / record / query methods to confirm the join wiring is right.

Why this matters: once enough rows accumulate, training can use
``clv_training_rows`` to weight or filter on closing-line drift. If the
schema or the join is wrong, we won't notice until weeks of polling have
already been wasted on un-joinable rows.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.odds_tracker import OddsTracker  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

FAKE_KEY = "test-key-not-used"


def _make_tracker(tmp_path) -> OddsTracker:
    db = str(tmp_path / "odds.db")
    return OddsTracker(api_key=FAKE_KEY, db_path=db)


def _seed_history(
    db_path: str,
    *,
    game_id: str,
    player_name: str,
    prop_type: str,
    snapshots: list[tuple[str, float, str]],
) -> None:
    """Insert (snapshot_time, line, bookmaker) rows directly into history.

    We bypass _store_snapshot because we want full control over the
    timestamps for the closing-line window test.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for ts, line, bk in snapshots:
        c.execute(
            """
            INSERT OR IGNORE INTO prop_line_history
            (game_id, player_name, prop_type, bookmaker, line,
             over_price, under_price, snapshot_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (game_id, player_name, prop_type, bk, line, -110, -110, ts),
        )
    conn.commit()
    conn.close()


def _seed_summary_row(
    db_path: str,
    *,
    game_id: str,
    player_name: str,
    prop_type: str,
    current_line: float,
) -> None:
    """Insert a minimal prop_line_summary row so capture_closing_lines has
    a target to UPDATE."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        """
        INSERT OR REPLACE INTO prop_line_summary
        (game_id, player_name, prop_type, opening_line, current_line,
         line_movement, opening_over_price, current_over_price,
         price_movement, num_snapshots, first_seen, last_updated,
         consensus_line, line_std, sharp_action_score)
        VALUES (?, ?, ?, ?, ?, 0, -110, -110, 0, 1, ?, ?, ?, 0, 0)
        """,
        (game_id, player_name, prop_type, current_line, current_line,
         now, now, current_line),
    )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Schema migration
# --------------------------------------------------------------------------- #

def test_init_db_adds_closing_line_columns_idempotently(tmp_path):
    db = str(tmp_path / "odds.db")
    OddsTracker(api_key=FAKE_KEY, db_path=db)
    # Re-init must not error and must not duplicate columns.
    OddsTracker(api_key=FAKE_KEY, db_path=db)

    conn = sqlite3.connect(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(prop_line_summary)")}
    conn.close()

    assert "closing_line" in cols
    assert "closing_line_at" in cols


def test_init_db_creates_prop_outcomes_table(tmp_path):
    db = str(tmp_path / "odds.db")
    OddsTracker(api_key=FAKE_KEY, db_path=db)

    conn = sqlite3.connect(db)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    cols = {row[1] for row in conn.execute("PRAGMA table_info(prop_outcomes)")}
    conn.close()

    assert "prop_outcomes" in tables
    for expected in ("game_id", "player_name", "prop_type",
                     "observed_line", "closing_line", "actual_result",
                     "settled_at"):
        assert expected in cols


# --------------------------------------------------------------------------- #
# capture_closing_lines
# --------------------------------------------------------------------------- #

def test_capture_closing_lines_stamps_latest_pre_tipoff_consensus(tmp_path):
    """The closing line should be the average across bookmakers at the
    most recent snapshot strictly before tipoff and inside the window."""
    tracker = _make_tracker(tmp_path)

    tipoff = datetime(2026, 4, 30, 19, 0, 0)  # 7:00pm
    tipoff_iso = tipoff.strftime("%Y-%m-%d %H:%M:%S")
    # Snapshots at -30m (out of 15-min window), -10m (in window), -5m (in
    # window, latest). Two bookmakers at the latest snapshot — we want
    # the consensus to be their average.
    snaps = [
        ((tipoff - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"), 25.0, "fanduel"),
        ((tipoff - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S"), 25.5, "fanduel"),
        ((tipoff - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"), 26.0, "fanduel"),
        ((tipoff - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"), 26.5, "draftkings"),
    ]
    _seed_history(
        tracker.db_path,
        game_id="G1", player_name="Tatum", prop_type="points",
        snapshots=snaps,
    )
    _seed_summary_row(
        tracker.db_path,
        game_id="G1", player_name="Tatum", prop_type="points",
        current_line=25.0,
    )

    stamped = tracker.capture_closing_lines("G1", tipoff_iso, window_minutes=15)
    assert stamped == 1

    conn = sqlite3.connect(tracker.db_path)
    row = conn.execute(
        """
        SELECT closing_line, closing_line_at FROM prop_line_summary
        WHERE game_id = ? AND player_name = ? AND prop_type = ?
        """,
        ("G1", "Tatum", "points"),
    ).fetchone()
    conn.close()

    closing_line, closing_at = row
    # (26.0 + 26.5) / 2 = 26.25
    assert closing_line == pytest.approx(26.25, abs=1e-6)
    # The snapshot we used must be the latest pre-tipoff one
    assert closing_at == (tipoff - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")


def test_capture_closing_lines_skips_post_tipoff_snapshots(tmp_path):
    """Snapshots taken after tipoff must NOT be used as the closing line —
    we'd be cheating with future info. The latest pre-tipoff one wins."""
    tracker = _make_tracker(tmp_path)

    tipoff = datetime(2026, 4, 30, 19, 0, 0)
    tipoff_iso = tipoff.strftime("%Y-%m-%d %H:%M:%S")
    snaps = [
        ((tipoff - timedelta(minutes=8)).strftime("%Y-%m-%d %H:%M:%S"), 20.5, "fd"),
        # +5 min — in-game line, must be ignored
        ((tipoff + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"), 22.0, "fd"),
    ]
    _seed_history(
        tracker.db_path,
        game_id="G2", player_name="Brown", prop_type="rebounds",
        snapshots=snaps,
    )
    _seed_summary_row(
        tracker.db_path,
        game_id="G2", player_name="Brown", prop_type="rebounds",
        current_line=20.0,
    )

    tracker.capture_closing_lines("G2", tipoff_iso, window_minutes=15)

    conn = sqlite3.connect(tracker.db_path)
    closing_line = conn.execute(
        """
        SELECT closing_line FROM prop_line_summary
        WHERE game_id = ? AND player_name = ? AND prop_type = ?
        """,
        ("G2", "Brown", "rebounds"),
    ).fetchone()[0]
    conn.close()

    assert closing_line == pytest.approx(20.5, abs=1e-6)


def test_capture_closing_lines_skips_outside_window(tmp_path):
    """A snapshot 60 minutes before tip with a 15-minute window must be
    ignored. With nothing inside the window, no row is stamped."""
    tracker = _make_tracker(tmp_path)

    tipoff = datetime(2026, 4, 30, 19, 0, 0)
    tipoff_iso = tipoff.strftime("%Y-%m-%d %H:%M:%S")
    snaps = [
        ((tipoff - timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S"), 8.5, "fd"),
    ]
    _seed_history(
        tracker.db_path,
        game_id="G3", player_name="Holiday", prop_type="assists",
        snapshots=snaps,
    )
    _seed_summary_row(
        tracker.db_path,
        game_id="G3", player_name="Holiday", prop_type="assists",
        current_line=8.0,
    )

    stamped = tracker.capture_closing_lines("G3", tipoff_iso, window_minutes=15)
    assert stamped == 0

    conn = sqlite3.connect(tracker.db_path)
    closing_line = conn.execute(
        """
        SELECT closing_line FROM prop_line_summary
        WHERE game_id = ? AND player_name = ? AND prop_type = ?
        """,
        ("G3", "Holiday", "assists"),
    ).fetchone()[0]
    conn.close()
    assert closing_line is None


# --------------------------------------------------------------------------- #
# record_outcome
# --------------------------------------------------------------------------- #

def test_record_outcome_joins_closing_line_from_summary(tmp_path):
    """``record_outcome`` should pull the already-captured closing_line
    from prop_line_summary and stamp it onto prop_outcomes."""
    tracker = _make_tracker(tmp_path)

    _seed_summary_row(
        tracker.db_path,
        game_id="G4", player_name="Doncic", prop_type="points",
        current_line=29.5,
    )
    # Pretend we already ran capture_closing_lines for this row
    conn = sqlite3.connect(tracker.db_path)
    conn.execute(
        """
        UPDATE prop_line_summary SET closing_line = ?, closing_line_at = ?
        WHERE game_id = ? AND player_name = ? AND prop_type = ?
        """,
        (30.5, "2026-04-30 18:55:00", "G4", "Doncic", "points"),
    )
    conn.commit()
    conn.close()

    tracker.record_outcome(
        game_id="G4", player_name="Doncic", prop_type="points",
        actual_result=33.0, observed_line=29.5, player_id=12345,
    )

    conn = sqlite3.connect(tracker.db_path)
    row = conn.execute(
        """
        SELECT player_id, observed_line, closing_line, actual_result
        FROM prop_outcomes
        WHERE game_id = ? AND player_name = ? AND prop_type = ?
        """,
        ("G4", "Doncic", "points"),
    ).fetchone()
    conn.close()

    assert row[0] == 12345
    assert row[1] == pytest.approx(29.5)
    assert row[2] == pytest.approx(30.5)
    assert row[3] == pytest.approx(33.0)


def test_record_outcome_handles_missing_closing_line(tmp_path):
    """If capture_closing_lines never ran (e.g. odds polling broke that
    day), record_outcome must still write the row with closing_line NULL
    so we don't silently drop the actual stat result."""
    tracker = _make_tracker(tmp_path)

    _seed_summary_row(
        tracker.db_path,
        game_id="G5", player_name="Curry", prop_type="three_pointers",
        current_line=4.5,
    )
    # Note: do NOT set closing_line on the summary row.

    tracker.record_outcome(
        game_id="G5", player_name="Curry", prop_type="three_pointers",
        actual_result=6.0, observed_line=4.5,
    )

    conn = sqlite3.connect(tracker.db_path)
    row = conn.execute(
        """
        SELECT closing_line, actual_result FROM prop_outcomes
        WHERE game_id = ? AND player_name = ? AND prop_type = ?
        """,
        ("G5", "Curry", "three_pointers"),
    ).fetchone()
    conn.close()

    assert row[0] is None
    assert row[1] == pytest.approx(6.0)


def test_record_outcome_replaces_on_conflict(tmp_path):
    """Re-recording the same (game, player, prop) should overwrite — e.g.
    the settlement job re-runs after an upstream stat-correction."""
    tracker = _make_tracker(tmp_path)
    _seed_summary_row(
        tracker.db_path,
        game_id="G6", player_name="Booker", prop_type="points",
        current_line=27.5,
    )

    tracker.record_outcome("G6", "Booker", "points",
                           actual_result=20.0, observed_line=27.5)
    tracker.record_outcome("G6", "Booker", "points",
                           actual_result=22.0, observed_line=27.5)

    conn = sqlite3.connect(tracker.db_path)
    rows = conn.execute(
        "SELECT actual_result FROM prop_outcomes WHERE game_id = ?", ("G6",)
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(22.0)


# --------------------------------------------------------------------------- #
# clv_training_rows
# --------------------------------------------------------------------------- #

def test_clv_training_rows_computes_drift_and_hit(tmp_path):
    tracker = _make_tracker(tmp_path)

    # Row 1 — line drifted from 25 (we bet) to 27 (close); player went 30
    _seed_summary_row(
        tracker.db_path,
        game_id="GA", player_name="LeBron", prop_type="points",
        current_line=27.0,
    )
    conn = sqlite3.connect(tracker.db_path)
    conn.execute(
        "UPDATE prop_line_summary SET closing_line=27.0 WHERE game_id='GA'"
    )
    conn.commit()
    conn.close()
    tracker.record_outcome("GA", "LeBron", "points",
                           actual_result=30.0, observed_line=25.0)

    # Row 2 — line stayed at 8.5; player went 7 (under)
    _seed_summary_row(
        tracker.db_path,
        game_id="GB", player_name="Embiid", prop_type="rebounds",
        current_line=8.5,
    )
    conn = sqlite3.connect(tracker.db_path)
    conn.execute(
        "UPDATE prop_line_summary SET closing_line=8.5 WHERE game_id='GB'"
    )
    conn.commit()
    conn.close()
    tracker.record_outcome("GB", "Embiid", "rebounds",
                           actual_result=7.0, observed_line=8.5)

    rows = tracker.clv_training_rows()
    by_game = {r["game_id"]: r for r in rows}

    assert set(by_game) == {"GA", "GB"}
    a = by_game["GA"]
    assert a["line_to_close_drift"] == pytest.approx(2.0)  # 27 - 25
    assert a["hit"] == 1
    b = by_game["GB"]
    assert b["line_to_close_drift"] == pytest.approx(0.0)
    assert b["hit"] == 0


def test_clv_training_rows_excludes_unsettled(tmp_path):
    """Rows missing closing_line must be excluded — they're not yet
    usable as CLV training signal."""
    tracker = _make_tracker(tmp_path)

    _seed_summary_row(
        tracker.db_path,
        game_id="GC", player_name="Mitchell", prop_type="points",
        current_line=24.5,
    )
    # Closing line never captured.
    tracker.record_outcome("GC", "Mitchell", "points",
                           actual_result=27.0, observed_line=24.5)

    assert tracker.clv_training_rows() == []


# --------------------------------------------------------------------------- #
# In-loop closing-line capture (snapshot_all_games + _inside_closing_window)
# --------------------------------------------------------------------------- #

def test_inside_closing_window_true_within_minutes(tmp_path):
    """A tipoff that's a few minutes from now must register as inside
    the window."""
    near = datetime.now(timezone.utc) + timedelta(minutes=5)
    near_iso = near.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert OddsTracker._inside_closing_window(near_iso) is True


def test_inside_closing_window_false_far_off(tmp_path):
    """A tipoff hours away must not register."""
    far = datetime.now(timezone.utc) + timedelta(hours=4)
    far_iso = far.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert OddsTracker._inside_closing_window(far_iso) is False


def test_inside_closing_window_false_in_the_past(tmp_path):
    """A tipoff well in the past (game already played out) must not
    register either — we don't want to retroactively stamp closing
    lines on stale games."""
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    old_iso = old.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert OddsTracker._inside_closing_window(old_iso) is False


def test_inside_closing_window_returns_false_on_garbage_input(tmp_path):
    """Malformed timestamps must NOT spuriously trigger capture for
    every game — return False on parse failure."""
    assert OddsTracker._inside_closing_window("") is False
    assert OddsTracker._inside_closing_window("not-a-date") is False
    assert OddsTracker._inside_closing_window(None) is False  # type: ignore[arg-type]


def test_snapshot_all_games_captures_closing_for_imminent_tipoff(tmp_path, monkeypatch):
    """End-to-end: a game whose tipoff is inside the window should have
    its closing_line stamped on prop_line_summary as a side-effect of
    snapshot_all_games. A game tipping off in 4 hours should NOT be
    stamped (still mid-day pricing)."""
    tracker = _make_tracker(tmp_path)

    # Two games: one imminent, one hours away.
    near_tipoff = datetime.now(timezone.utc) + timedelta(minutes=8)
    far_tipoff = datetime.now(timezone.utc) + timedelta(hours=4)

    fake_games = [
        {
            "id": "evt_near",
            "home_team": "BOS", "away_team": "LAL",
            "commence_time": near_tipoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        {
            "id": "evt_far",
            "home_team": "DEN", "away_team": "GSW",
            "commence_time": far_tipoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    ]

    def fake_fetch_props(event_id):
        # Two-bookmaker line for one player.
        return [
            {"player_name": "Tatum", "prop_type": "points", "bookmaker": "fd",
             "line": 27.0, "over_price": -110, "under_price": -110},
            {"player_name": "Tatum", "prop_type": "points", "bookmaker": "dk",
             "line": 28.0, "over_price": -115, "under_price": -105},
        ]

    monkeypatch.setattr(tracker, "fetch_upcoming_games", lambda: fake_games)
    monkeypatch.setattr(tracker, "fetch_player_props", fake_fetch_props)

    # Snapshot uses time.sleep(1) between games — patch it out so the
    # test stays under a second.
    monkeypatch.setattr("src.odds_tracker.time.sleep", lambda *a, **kw: None)

    total = tracker.snapshot_all_games()
    assert total == 4  # 2 props × 2 games

    conn = sqlite3.connect(tracker.db_path)
    near_close = conn.execute(
        "SELECT closing_line FROM prop_line_summary WHERE game_id = 'evt_near'"
    ).fetchone()
    far_close = conn.execute(
        "SELECT closing_line FROM prop_line_summary WHERE game_id = 'evt_far'"
    ).fetchone()
    conn.close()

    # Near game: closing line is the consensus across bookmakers
    # (27.0 + 28.0) / 2 = 27.5
    assert near_close[0] == pytest.approx(27.5, abs=1e-6)
    # Far game: never within window, closing_line stays NULL
    assert far_close[0] is None


def test_clv_training_rows_filters_by_prop_type(tmp_path):
    tracker = _make_tracker(tmp_path)

    for gid, ptype, line, actual in (
        ("GD", "points", 25.0, 30.0),
        ("GE", "rebounds", 8.0, 9.0),
    ):
        _seed_summary_row(
            tracker.db_path, game_id=gid, player_name="X",
            prop_type=ptype, current_line=line,
        )
        conn = sqlite3.connect(tracker.db_path)
        conn.execute(
            f"UPDATE prop_line_summary SET closing_line=? WHERE game_id=?",
            (line, gid),
        )
        conn.commit()
        conn.close()
        tracker.record_outcome(gid, "X", ptype,
                               actual_result=actual, observed_line=line)

    only_pts = tracker.clv_training_rows(prop_type="points")
    assert len(only_pts) == 1
    assert only_pts[0]["prop_type"] == "points"
