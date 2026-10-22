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
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.odds_tracker import OddsTracker  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

FAKE_KEY = "test-key-not-used"


def _make_tracker(db_url) -> OddsTracker:
    return OddsTracker(api_key=FAKE_KEY, db_path=db_url)


def _seed_history(
    conn,
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
    c = conn.cursor()
    for ts, line, bk in snapshots:
        c.execute(
            """
            INSERT INTO prop_line_history
            (game_id, player_name, prop_type, bookmaker, line,
             over_price, under_price, snapshot_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (game_id, player_name, prop_type, bk, line, -110, -110, ts),
        )


def _seed_summary_row(
    conn,
    *,
    game_id: str,
    player_name: str,
    prop_type: str,
    current_line: float,
) -> None:
    """Insert a minimal prop_line_summary row so capture_closing_lines has
    a target to UPDATE."""
    c = conn.cursor()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        """
        INSERT INTO prop_line_summary
        (game_id, player_name, prop_type, opening_line, current_line,
         line_movement, opening_over_price, current_over_price,
         price_movement, num_snapshots, first_seen, last_updated,
         consensus_line, line_std, sharp_action_score)
        VALUES (%s, %s, %s, %s, %s, 0, -110, -110, 0, 1, %s, %s, %s, 0, 0)
        ON CONFLICT (game_id, player_name, prop_type) DO UPDATE SET
            opening_line  = excluded.opening_line,
            current_line  = excluded.current_line,
            first_seen    = excluded.first_seen,
            last_updated  = excluded.last_updated,
            consensus_line = excluded.consensus_line
        """,
        (game_id, player_name, prop_type, current_line, current_line,
         now, now, current_line),
    )


# --------------------------------------------------------------------------- #
# Schema migration
# --------------------------------------------------------------------------- #

def test_init_db_adds_closing_line_columns_idempotently(db_conn, db_url):
    from src import db as _db
    OddsTracker(api_key=FAKE_KEY, db_path=db_url)
    # Re-init must not error and must not duplicate columns.
    OddsTracker(api_key=FAKE_KEY, db_path=db_url)

    cols = _db.table_columns(db_conn, "prop_line_summary")

    assert "closing_line" in cols
    assert "closing_line_at" in cols


def test_init_db_creates_prop_outcomes_table(db_conn, db_url):
    from src import db as _db
    OddsTracker(api_key=FAKE_KEY, db_path=db_url)

    tables = {row[0] for row in db_conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = current_schema()"
    )}
    cols = _db.table_columns(db_conn, "prop_outcomes")

    assert "prop_outcomes" in tables
    for expected in ("game_id", "player_name", "prop_type",
                     "observed_line", "closing_line", "actual_result",
                     "settled_at"):
        assert expected in cols


# --------------------------------------------------------------------------- #
# capture_closing_lines
# --------------------------------------------------------------------------- #

def test_capture_closing_lines_stamps_latest_pre_tipoff_consensus(db_conn, db_url):
    """The closing line should be the average across bookmakers at the
    most recent snapshot strictly before tipoff and inside the window."""
    tracker = _make_tracker(db_url)

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
        db_conn,
        game_id="G1", player_name="Tatum", prop_type="points",
        snapshots=snaps,
    )
    _seed_summary_row(
        db_conn,
        game_id="G1", player_name="Tatum", prop_type="points",
        current_line=25.0,
    )

    stamped = tracker.capture_closing_lines("G1", tipoff_iso, window_minutes=15)
    assert stamped == 1

    row = db_conn.execute(
        """
        SELECT closing_line, closing_line_at FROM prop_line_summary
        WHERE game_id = %s AND player_name = %s AND prop_type = %s
        """,
        ("G1", "Tatum", "points"),
    ).fetchone()

    closing_line, closing_at = row
    # (26.0 + 26.5) / 2 = 26.25
    assert closing_line == pytest.approx(26.25, abs=1e-6)
    # The snapshot we used must be the latest pre-tipoff one
    assert closing_at == (tipoff - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")


def test_capture_closing_lines_skips_post_tipoff_snapshots(db_conn, db_url):
    """Snapshots taken after tipoff must NOT be used as the closing line —
    we'd be cheating with future info. The latest pre-tipoff one wins."""
    tracker = _make_tracker(db_url)

    tipoff = datetime(2026, 4, 30, 19, 0, 0)
    tipoff_iso = tipoff.strftime("%Y-%m-%d %H:%M:%S")
    snaps = [
        ((tipoff - timedelta(minutes=8)).strftime("%Y-%m-%d %H:%M:%S"), 20.5, "fd"),
        # +5 min — in-game line, must be ignored
        ((tipoff + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"), 22.0, "fd"),
    ]
    _seed_history(
        db_conn,
        game_id="G2", player_name="Brown", prop_type="rebounds",
        snapshots=snaps,
    )
    _seed_summary_row(
        db_conn,
        game_id="G2", player_name="Brown", prop_type="rebounds",
        current_line=20.0,
    )

    tracker.capture_closing_lines("G2", tipoff_iso, window_minutes=15)

    closing_line = db_conn.execute(
        """
        SELECT closing_line FROM prop_line_summary
        WHERE game_id = %s AND player_name = %s AND prop_type = %s
        """,
        ("G2", "Brown", "rebounds"),
    ).fetchone()[0]

    assert closing_line == pytest.approx(20.5, abs=1e-6)


def test_capture_closing_lines_skips_outside_window(db_conn, db_url):
    """A snapshot 60 minutes before tip with a 15-minute window must be
    ignored. With nothing inside the window, no row is stamped."""
    tracker = _make_tracker(db_url)

    tipoff = datetime(2026, 4, 30, 19, 0, 0)
    tipoff_iso = tipoff.strftime("%Y-%m-%d %H:%M:%S")
    snaps = [
        ((tipoff - timedelta(minutes=60)).strftime("%Y-%m-%d %H:%M:%S"), 8.5, "fd"),
    ]
    _seed_history(
        db_conn,
        game_id="G3", player_name="Holiday", prop_type="assists",
        snapshots=snaps,
    )
    _seed_summary_row(
        db_conn,
        game_id="G3", player_name="Holiday", prop_type="assists",
        current_line=8.0,
    )

    stamped = tracker.capture_closing_lines("G3", tipoff_iso, window_minutes=15)
    assert stamped == 0

    closing_line = db_conn.execute(
        """
        SELECT closing_line FROM prop_line_summary
        WHERE game_id = %s AND player_name = %s AND prop_type = %s
        """,
        ("G3", "Holiday", "assists"),
    ).fetchone()[0]
    assert closing_line is None


# --------------------------------------------------------------------------- #
# record_outcome
# --------------------------------------------------------------------------- #

def test_record_outcome_joins_closing_line_from_summary(db_conn, db_url):
    """``record_outcome`` should pull the already-captured closing_line
    from prop_line_summary and stamp it onto prop_outcomes."""
    tracker = _make_tracker(db_url)

    _seed_summary_row(
        db_conn,
        game_id="G4", player_name="Doncic", prop_type="points",
        current_line=29.5,
    )
    # Pretend we already ran capture_closing_lines for this row
    db_conn.execute(
        """
        UPDATE prop_line_summary SET closing_line = %s, closing_line_at = %s
        WHERE game_id = %s AND player_name = %s AND prop_type = %s
        """,
        (30.5, "2026-04-30 18:55:00", "G4", "Doncic", "points"),
    )

    tracker.record_outcome(
        game_id="G4", player_name="Doncic", prop_type="points",
        actual_result=33.0, observed_line=29.5, player_id=12345,
    )

    row = db_conn.execute(
        """
        SELECT player_id, observed_line, closing_line, actual_result
        FROM prop_outcomes
        WHERE game_id = %s AND player_name = %s AND prop_type = %s
        """,
        ("G4", "Doncic", "points"),
    ).fetchone()

    assert row[0] == 12345
    assert row[1] == pytest.approx(29.5)
    assert row[2] == pytest.approx(30.5)
    assert row[3] == pytest.approx(33.0)


def test_record_outcome_handles_missing_closing_line(db_conn, db_url):
    """If capture_closing_lines never ran (e.g. odds polling broke that
    day), record_outcome must still write the row with closing_line NULL
    so we don't silently drop the actual stat result."""
    tracker = _make_tracker(db_url)

    _seed_summary_row(
        db_conn,
        game_id="G5", player_name="Curry", prop_type="three_pointers",
        current_line=4.5,
    )
    # Note: do NOT set closing_line on the summary row.

    tracker.record_outcome(
        game_id="G5", player_name="Curry", prop_type="three_pointers",
        actual_result=6.0, observed_line=4.5,
    )

    row = db_conn.execute(
        """
        SELECT closing_line, actual_result FROM prop_outcomes
        WHERE game_id = %s AND player_name = %s AND prop_type = %s
        """,
        ("G5", "Curry", "three_pointers"),
    ).fetchone()

    assert row[0] is None
    assert row[1] == pytest.approx(6.0)


def test_record_outcome_replaces_on_conflict(db_conn, db_url):
    """Re-recording the same (game, player, prop) should overwrite — e.g.
    the settlement job re-runs after an upstream stat-correction."""
    tracker = _make_tracker(db_url)
    _seed_summary_row(
        db_conn,
        game_id="G6", player_name="Booker", prop_type="points",
        current_line=27.5,
    )

    tracker.record_outcome("G6", "Booker", "points",
                           actual_result=20.0, observed_line=27.5)
    tracker.record_outcome("G6", "Booker", "points",
                           actual_result=22.0, observed_line=27.5)

    rows = db_conn.execute(
        "SELECT actual_result FROM prop_outcomes WHERE game_id = %s", ("G6",)
    ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(22.0)


# --------------------------------------------------------------------------- #
# clv_training_rows
# --------------------------------------------------------------------------- #

def test_clv_training_rows_computes_drift_and_hit(db_conn, db_url):
    tracker = _make_tracker(db_url)

    # Row 1 — line drifted from 25 (we bet) to 27 (close); player went 30
    _seed_summary_row(
        db_conn,
        game_id="GA", player_name="LeBron", prop_type="points",
        current_line=27.0,
    )
    db_conn.execute(
        "UPDATE prop_line_summary SET closing_line=27.0 WHERE game_id='GA'"
    )
    tracker.record_outcome("GA", "LeBron", "points",
                           actual_result=30.0, observed_line=25.0)

    # Row 2 — line stayed at 8.5; player went 7 (under)
    _seed_summary_row(
        db_conn,
        game_id="GB", player_name="Embiid", prop_type="rebounds",
        current_line=8.5,
    )
    db_conn.execute(
        "UPDATE prop_line_summary SET closing_line=8.5 WHERE game_id='GB'"
    )
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


def test_clv_training_rows_excludes_unsettled(db_conn, db_url):
    """Rows missing closing_line must be excluded — they're not yet
    usable as CLV training signal."""
    tracker = _make_tracker(db_url)

    _seed_summary_row(
        db_conn,
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

def test_inside_closing_window_true_within_minutes(db_conn, db_url):
    """A tipoff that's a few minutes from now must register as inside
    the window."""
    near = datetime.now(timezone.utc) + timedelta(minutes=5)
    near_iso = near.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert OddsTracker._inside_closing_window(near_iso) is True


def test_inside_closing_window_false_far_off(db_conn, db_url):
    """A tipoff hours away must not register."""
    far = datetime.now(timezone.utc) + timedelta(hours=4)
    far_iso = far.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert OddsTracker._inside_closing_window(far_iso) is False


def test_inside_closing_window_false_in_the_past(db_conn, db_url):
    """A tipoff well in the past (game already played out) must not
    register either — we don't want to retroactively stamp closing
    lines on stale games."""
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    old_iso = old.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert OddsTracker._inside_closing_window(old_iso) is False


def test_inside_closing_window_returns_false_on_garbage_input(db_conn, db_url):
    """Malformed timestamps must NOT spuriously trigger capture for
    every game — return False on parse failure."""
    assert OddsTracker._inside_closing_window("") is False
    assert OddsTracker._inside_closing_window("not-a-date") is False
    assert OddsTracker._inside_closing_window(None) is False  # type: ignore[arg-type]


def test_snapshot_all_games_captures_closing_for_imminent_tipoff(db_conn, db_url, monkeypatch):
    """End-to-end: a game whose tipoff is inside the window should have
    its closing_line stamped on prop_line_summary as a side-effect of
    snapshot_all_games. A game tipping off in 4 hours should NOT be
    stamped (still mid-day pricing)."""
    tracker = _make_tracker(db_url)

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

    near_close = db_conn.execute(
        "SELECT closing_line FROM prop_line_summary WHERE game_id = 'evt_near'"
    ).fetchone()
    far_close = db_conn.execute(
        "SELECT closing_line FROM prop_line_summary WHERE game_id = 'evt_far'"
    ).fetchone()

    # Near game: closing line is the consensus across bookmakers
    # (27.0 + 28.0) / 2 = 27.5
    assert near_close[0] == pytest.approx(27.5, abs=1e-6)
    # Far game: never within window, closing_line stays NULL
    assert far_close[0] is None


# --------------------------------------------------------------------------- #
# capture_closing_lines_for_imminent_games (close-only sweeper)
# --------------------------------------------------------------------------- #

def _seed_tipoff(conn, *, game_id: str, commence_time: str,
                 home: str = "BOS", away: str = "LAL") -> None:
    """Insert a row into game_tipoffs directly. The sweeper reads from
    this table — in production it's populated by fetch_upcoming_games."""
    conn.execute(
        """
        INSERT INTO game_tipoffs (game_id, commence_time, home_team, away_team, last_seen)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (game_id, commence_time, home, away, datetime.now(timezone.utc).isoformat()),
    )


def test_imminent_sweeper_stamps_only_imminent_games(db_conn, db_url):
    tracker = _make_tracker(db_url)

    near = datetime.now(timezone.utc) + timedelta(minutes=8)
    far = datetime.now(timezone.utc) + timedelta(hours=4)

    _seed_tipoff(db_conn, game_id="g_near",
                 commence_time=near.strftime("%Y-%m-%dT%H:%M:%SZ"))
    _seed_tipoff(db_conn, game_id="g_far",
                 commence_time=far.strftime("%Y-%m-%dT%H:%M:%SZ"))

    # Seed a snapshot inside the window for the near game and a row in
    # prop_line_summary so capture_closing_lines has somewhere to UPDATE.
    near_snap = (near - timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M:%S")
    _seed_history(
        db_conn,
        game_id="g_near", player_name="Tatum", prop_type="points",
        snapshots=[(near_snap, 27.5, "fanduel"),
                   (near_snap, 28.0, "draftkings")],
    )
    _seed_summary_row(
        db_conn,
        game_id="g_near", player_name="Tatum", prop_type="points",
        current_line=27.0,
    )

    result = tracker.capture_closing_lines_for_imminent_games()
    assert result["checked"] == 2
    assert result["games_inside_window"] == 1
    assert result["skipped_no_window"] == 1
    assert result["stamped_total"] == 1

    closing = db_conn.execute(
        "SELECT closing_line FROM prop_line_summary WHERE game_id = 'g_near'"
    ).fetchone()[0]
    # (27.5 + 28.0) / 2 = 27.75
    assert closing == pytest.approx(27.75, abs=1e-6)


def test_imminent_sweeper_drops_old_tipoffs_from_scan(db_conn, db_url):
    """A tipoff from yesterday should never even reach capture_closing_lines;
    the SELECT filter cuts anything > 6h in the past out of the result."""
    tracker = _make_tracker(db_url)

    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    _seed_tipoff(db_conn, game_id="g_old",
                 commence_time=yesterday.strftime("%Y-%m-%dT%H:%M:%SZ"))

    result = tracker.capture_closing_lines_for_imminent_games()
    assert result["checked"] == 0
    assert result["games_inside_window"] == 0


def test_imminent_sweeper_updates_last_poll_telemetry(db_conn, db_url):
    """After the sweeper runs, _LAST_POLL must record closes_checked_at
    so /healthz/clv can show liveness even on no-game days."""
    from src.odds_tracker import last_poll_status
    tracker = _make_tracker(db_url)
    tracker.capture_closing_lines_for_imminent_games()
    status = last_poll_status()
    assert status["closes_checked_at"] is not None


def test_fetch_upcoming_games_persists_tipoffs(db_conn, db_url, monkeypatch):
    """fetch_upcoming_games must side-effect into game_tipoffs so the
    sweeper has data to walk without re-hitting the API."""
    tracker = _make_tracker(db_url)
    fake = [
        {"id": "g1", "home_team": "BOS", "away_team": "LAL",
         "commence_time": "2026-05-15T19:00:00Z"},
        {"id": "g2", "home_team": "GSW", "away_team": "DEN",
         "commence_time": "2026-05-15T22:00:00Z"},
    ]
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: [
        {"id": g["id"], "home_team": g["home_team"],
         "away_team": g["away_team"], "commence_time": g["commence_time"]}
        for g in fake
    ])
    games = tracker.fetch_upcoming_games()
    assert len(games) == 2

    rows = db_conn.execute(
        "SELECT game_id, commence_time FROM game_tipoffs ORDER BY game_id"
    ).fetchall()
    assert rows == [
        ("g1", "2026-05-15T19:00:00Z"),
        ("g2", "2026-05-15T22:00:00Z"),
    ]


def test_fetch_upcoming_games_upserts_changed_tipoffs(db_conn, db_url, monkeypatch):
    """If the API returns a revised commence_time for a known game (game
    postponed), the tipoff cache must reflect the new value, not stay
    stale on the original."""
    tracker = _make_tracker(db_url)

    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: [
        {"id": "g1", "home_team": "BOS", "away_team": "LAL",
         "commence_time": "2026-05-15T19:00:00Z"},
    ])
    tracker.fetch_upcoming_games()

    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: [
        {"id": "g1", "home_team": "BOS", "away_team": "LAL",
         "commence_time": "2026-05-16T19:00:00Z"},  # postponed by a day
    ])
    tracker.fetch_upcoming_games()

    ct = db_conn.execute(
        "SELECT commence_time FROM game_tipoffs WHERE game_id = 'g1'"
    ).fetchone()[0]
    assert ct == "2026-05-16T19:00:00Z"


def test_clv_training_rows_filters_by_prop_type(db_conn, db_url):
    tracker = _make_tracker(db_url)

    for gid, ptype, line, actual in (
        ("GD", "points", 25.0, 30.0),
        ("GE", "rebounds", 8.0, 9.0),
    ):
        _seed_summary_row(
            db_conn, game_id=gid, player_name="X",
            prop_type=ptype, current_line=line,
        )
        db_conn.execute(
            f"UPDATE prop_line_summary SET closing_line=%s WHERE game_id=%s",
            (line, gid),
        )
        tracker.record_outcome(gid, "X", ptype,
                               actual_result=actual, observed_line=line)

    only_pts = tracker.clv_training_rows(prop_type="points")
    assert len(only_pts) == 1
    assert only_pts[0]["prop_type"] == "points"
