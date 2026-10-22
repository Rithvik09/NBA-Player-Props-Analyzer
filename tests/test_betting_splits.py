"""Public/sharp betting-splits ingestion from Action Network.

The source is an undocumented internal endpoint: free, keyless, and able
to change shape — or start demanding auth — without notice. The failure
that matters is not a crash, it is a quiet one: storing zero rows looks
exactly like a night with no games. So the canary tests below are as
load-bearing as the parsing ones.

Nothing here touches the network. The fixture in ``tests/fixtures`` is a
real capture of the live payload taken on 2026-10-21, trimmed to three
games but with every outcome dict left verbatim, so it carries the
awkward parts of the real feed:

* book 30 quotes odds but no ``bet_info`` at all — normal, not an error;
* book 49 appears on some games and not others;
* an in-progress game (295292) repeats every outcome with
  ``is_live: true``, which is why ``is_live`` is in the UNIQUE tuple;
* a ``core_bet_type_6_team_score`` (team totals) market we do not store;
* moneyline outcomes whose ``value`` is 0 — no line, not a line of zero;
* total outcomes with no ``team_id`` key.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import betting_splits as _bs  # noqa: E402
from src.betting_splits import BettingSplitsTracker  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / \
    "action_network_publicbetting_nba.json"

SPLITS_ENV = (
    "BETTING_SPLITS_POLL_INTERVAL",
    "BETTING_SPLITS_USER_AGENT",
)


@pytest.fixture(autouse=True)
def clean_splits_env(monkeypatch):
    """Start every test from the documented defaults."""
    for name in SPLITS_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def payload() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def frozen_now(monkeypatch) -> str:
    """Pin the snapshot clock.

    ``snapshot_time`` is second-resolution wall clock and sits in the
    UNIQUE tuple, so idempotency would otherwise depend on how fast the
    test ran rather than on the constraint.
    """
    return _freeze(monkeypatch, datetime(2026, 10, 21, 22, 40, 0, tzinfo=timezone.utc))


def _freeze(monkeypatch, at: datetime) -> str:
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return at if tz is not None else at.replace(tzinfo=None)

    monkeypatch.setattr("src.betting_splits.datetime", _Frozen)
    return at.strftime("%Y-%m-%d %H:%M:%S")


def _tracker(db_url, payload) -> BettingSplitsTracker:
    """Tracker whose HTTP layer hands back the captured fixture."""
    tracker = BettingSplitsTracker(db_path=db_url)
    tracker.fetch = lambda: payload  # type: ignore[method-assign]
    return tracker


def _splits(conn, **where) -> list[tuple]:
    sql = """
        SELECT event_id, book_id, market, side, team_id, line, odds,
               tickets_pct, money_pct, is_live
        FROM betting_splits_history
    """
    params: list = []
    if where:
        sql += " WHERE " + " AND ".join(f"{k} = %s" for k in where)
        params = list(where.values())
    sql += " ORDER BY event_id, book_id, market, side, is_live"
    return conn.execute(sql, params).fetchall()


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def test_init_db_creates_betting_splits_history(db_conn, db_url):
    from src import db as _db
    BettingSplitsTracker(db_path=db_url)
    # Re-init must be a no-op, like the other history tables.
    BettingSplitsTracker(db_path=db_url)

    assert _db.table_columns(db_conn, "betting_splits_history") == {
        "id", "event_id", "book_id", "market", "side", "team_id", "line",
        "odds", "tickets_pct", "money_pct", "is_live", "home_team_abbr",
        "away_team_abbr", "start_time", "start_date_utc", "snapshot_time",
    }


def test_unique_tuple_is_what_makes_repolling_idempotent(db_conn, db_url):
    """Every column in the tuple is NOT NULL — this is PostgreSQL 14, so
    a NULL in the tuple would make duplicates pass straight through."""
    BettingSplitsTracker(db_path=db_url)

    uniques = db_conn.execute(
        """
        SELECT a.attname, a.attnotnull
        FROM pg_constraint c
        JOIN pg_attribute a ON a.attrelid = c.conrelid
                           AND a.attnum = ANY(c.conkey)
        WHERE c.contype = 'u'
          AND c.conrelid = to_regclass(current_schema()
                                       || '.betting_splits_history')
        """
    ).fetchall()
    assert {r[0] for r in uniques} == {
        "event_id", "book_id", "market", "side", "is_live", "snapshot_time",
    }
    assert all(r[1] for r in uniques)


# --------------------------------------------------------------------------- #
# Parsing the real payload
# --------------------------------------------------------------------------- #

def test_capture_stores_one_row_per_outcome_carrying_bet_info(db_conn, db_url, payload):
    """116 of the fixture's 134 game-market outcomes carry ``bet_info``.

    The other 18 are book 30's, which quotes prices and no splits.
    """
    stored = _tracker(db_url, payload).capture()

    assert stored == 116
    assert len(_splits(db_conn)) == 116


def test_a_book_with_no_bet_info_is_skipped_not_raised_on(db_conn, db_url, payload):
    """Book 30 prices all three markets and reports no splits at all.

    Dropping its rows is the point: this table means "an observed
    split", so a row with no percentages in it would be noise.
    """
    _tracker(db_url, payload).capture()

    assert _splits(db_conn, book_id=30) == []
    assert {r[1] for r in _splits(db_conn)} == {15, 49, 68, 69, 71, 75}


def test_values_land_exactly_as_the_feed_reported_them(db_conn, db_url, payload):
    """WAS/MIL on book 15 — every market, both sides, verbatim."""
    _tracker(db_url, payload).capture()

    assert _splits(db_conn, event_id=295286, book_id=15) == [
        # event_id, book, market, side, team_id, line, odds, tickets%, money%, live
        (295286, 15, "moneyline", "away", 172, None, 145.0, 11.0, 43.0, False),
        (295286, 15, "moneyline", "home", 157, None, -168.0, 89.0, 57.0, False),
        (295286, 15, "spread", "away", 172, 4.5, -113.0, 46.0, 48.0, False),
        (295286, 15, "spread", "home", 157, -4.5, -106.0, 54.0, 52.0, False),
        (295286, 15, "total", "over", None, 238.5, -107.0, 56.0, 57.0, False),
        (295286, 15, "total", "under", None, 238.5, -112.0, 44.0, 43.0, False),
    ]


def test_a_moneyline_has_no_line_rather_than_a_line_of_zero(db_conn, db_url, payload):
    """The feed sends ``value: 0`` on moneylines. Zero is not a number
    anyone bet to — ``game_line_history`` NULLs the same case on h2h."""
    _tracker(db_url, payload).capture()

    lines = db_conn.execute(
        "SELECT DISTINCT line FROM betting_splits_history WHERE market = 'moneyline'"
    ).fetchall()
    assert lines == [(None,)]


def test_markets_outside_the_three_we_model_are_ignored(db_conn, db_url, payload):
    """The feed also carries ``core_bet_type_6_team_score`` (team
    totals) with its own splits. Out of scope, and silently storing it
    under a market name nobody reads would be worse than dropping it."""
    _tracker(db_url, payload).capture()

    markets = db_conn.execute(
        "SELECT DISTINCT market FROM betting_splits_history ORDER BY market"
    ).fetchall()
    assert markets == [("moneyline",), ("spread",), ("total",)]


def test_live_and_pregame_quotes_are_kept_apart(db_conn, db_url, payload):
    """295292 is in progress, so the feed repeats every outcome with
    ``is_live: true`` and a wildly different price. Without ``is_live``
    in the UNIQUE tuple one of the two would be silently dropped."""
    _tracker(db_url, payload).capture()

    rows = _splits(db_conn, event_id=295292, book_id=15, market="moneyline",
                   side="home")
    assert [r[9] for r in rows] == [False, True]
    pregame, live = rows
    assert pregame[6] == -440.0
    assert live[6] == -12004.0


# --------------------------------------------------------------------------- #
# The natural key that makes a later join possible
# --------------------------------------------------------------------------- #

def test_each_row_carries_the_matchup_and_utc_start_date(db_conn, db_url, payload):
    """Action Network's ``event_id`` is its own id — not the NBA
    ``game_id``, not the Odds API event id. The abbreviations and the
    UTC start date are the only handle a later join has.

    The ``teams`` array is not ordered home-first (295286 lists the home
    team first, 295287 does not), so home/away come from the explicit
    ``home_team_id`` / ``away_team_id``.
    """
    _tracker(db_url, payload).capture()

    rows = db_conn.execute(
        """
        SELECT DISTINCT event_id, home_team_abbr, away_team_abbr,
                        start_time, start_date_utc
        FROM betting_splits_history ORDER BY event_id
        """
    ).fetchall()
    assert rows == [
        (295286, "WAS", "MIL", "2026-10-21T23:00:00.000Z", "2026-10-21"),
        (295290, "MEM", "UTA", "2026-10-22T00:00:00.000Z", "2026-10-22"),
        (295292, "HOU", "DAL", "2026-10-22T00:30:00.000Z", "2026-10-22"),
    ]


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #

def test_recapturing_the_same_minute_does_not_duplicate(db_conn, db_url, payload, frozen_now):
    tracker = _tracker(db_url, payload)
    tracker.capture()
    first = _splits(db_conn)
    tracker.capture()

    assert _splits(db_conn) == first
    assert len(first) == 116
    assert db_conn.execute(
        "SELECT DISTINCT snapshot_time FROM betting_splits_history"
    ).fetchall() == [(frozen_now,)]


def test_a_later_snapshot_appends_rather_than_overwrites(db_conn, db_url, payload, monkeypatch):
    """Movement in the splits is the whole point — the 20-minute poll
    has to accumulate history, not keep one current row per outcome."""
    tracker = _tracker(db_url, payload)
    _freeze(monkeypatch, datetime(2026, 10, 21, 22, 40, 0, tzinfo=timezone.utc))
    tracker.capture()
    _freeze(monkeypatch, datetime(2026, 10, 21, 23, 0, 0, tzinfo=timezone.utc))
    tracker.capture()

    assert db_conn.execute(
        "SELECT COUNT(*) FROM betting_splits_history"
    ).fetchone()[0] == 232
    assert db_conn.execute(
        "SELECT COUNT(DISTINCT snapshot_time) FROM betting_splits_history"
    ).fetchone()[0] == 2


# --------------------------------------------------------------------------- #
# The canary
#
# This endpoint is undocumented. The way it will break is by returning
# 200 with a shape we no longer understand, and the result of that is an
# empty table — which is also what a night with no NBA looks like.
# --------------------------------------------------------------------------- #

def test_a_payload_with_no_games_logs_an_error(db_conn, db_url, caplog):
    tracker = _tracker(db_url, {"league": {"name": "nba"}, "games": []})

    with caplog.at_level(logging.DEBUG, logger="src.betting_splits"):
        assert tracker.capture() == 0

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "0 games" in errors[0].getMessage()


def test_games_with_no_bet_info_anywhere_logs_an_error(db_conn, db_url, payload, caplog):
    """The shape that actually scares us: the board is there, the
    splits have quietly gone. Stripped ``bet_info`` stands in for the
    endpoint deciding splits are a paid feature now."""
    for game in payload["games"]:
        for book in game["markets"].values():
            for outcomes in book["event"].values():
                for outcome in outcomes:
                    outcome.pop("bet_info", None)
    tracker = _tracker(db_url, payload)

    with caplog.at_level(logging.DEBUG, logger="src.betting_splits"):
        assert tracker.capture() == 0

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert "3 games" in message
    assert "bet_info" in message


def test_a_healthy_capture_logs_no_error(db_conn, db_url, payload, caplog):
    with caplog.at_level(logging.DEBUG, logger="src.betting_splits"):
        _tracker(db_url, payload).capture()

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_a_failed_fetch_logs_an_error_and_stores_nothing(db_conn, db_url, caplog):
    """A 403 the day they add auth must not pass for a quiet night."""
    tracker = BettingSplitsTracker(db_path=db_url)
    tracker.fetch = lambda: None  # type: ignore[method-assign]

    with caplog.at_level(logging.DEBUG, logger="src.betting_splits"):
        assert tracker.capture() == 0

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert _splits(db_conn) == []


# --------------------------------------------------------------------------- #
# The HTTP call itself
# --------------------------------------------------------------------------- #

def test_fetch_asks_action_network_with_an_honest_user_agent(db_url, monkeypatch):
    calls = []

    class _Resp:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"games": []}

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return _Resp()

    monkeypatch.setattr("src.betting_splits.requests.get", fake_get)
    BettingSplitsTracker(db_path=db_url).fetch()

    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == _bs.PUBLIC_BETTING_URL
    assert "the-odds-api.com" not in url
    agent = kwargs["headers"]["User-Agent"]
    assert agent == _bs.DEFAULT_USER_AGENT
    # An honest agent says who we are and how to reach us — not a
    # browser string pretending to be someone else.
    assert "Mozilla" not in agent


def test_user_agent_is_overridable(db_url, monkeypatch):
    monkeypatch.setenv("BETTING_SPLITS_USER_AGENT", "someone-else/2.0")
    assert _bs.user_agent() == "someone-else/2.0"


def test_a_transport_error_returns_none_rather_than_raising(db_url, monkeypatch, caplog):
    import requests

    def boom(url, **kwargs):
        raise requests.exceptions.ConnectionError("no route to host")

    monkeypatch.setattr("src.betting_splits.requests.get", boom)

    with caplog.at_level(logging.DEBUG, logger="src.betting_splits"):
        assert BettingSplitsTracker(db_path=db_url).fetch() is None

    assert [r for r in caplog.records if r.levelno >= logging.ERROR]


# --------------------------------------------------------------------------- #
# Cadence
# --------------------------------------------------------------------------- #

def test_default_cadence_is_twenty_minutes():
    """Free source, so this is courtesy rather than budget. Twenty
    minutes is frequent enough to see a line move and slow enough that
    nobody notices us."""
    assert _bs.DEFAULT_POLL_INTERVAL_SECONDS == 1200
    assert _bs.poll_interval_seconds() == 1200


def test_cadence_is_env_tunable(monkeypatch):
    monkeypatch.setenv("BETTING_SPLITS_POLL_INTERVAL", "900")
    assert _bs.poll_interval_seconds() == 900


def test_a_junk_cadence_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("BETTING_SPLITS_POLL_INTERVAL", "every-so-often")
    assert _bs.poll_interval_seconds() == 1200


# --------------------------------------------------------------------------- #
# Where the commit lands
# --------------------------------------------------------------------------- #

@pytest.fixture
def real_conn(pg_dsn):
    """A genuine connection in its own throwaway schema.

    ``db_conn`` neuters ``commit()`` so tests can roll back, which makes
    it exactly the wrong harness for asking "did this write actually
    land?" — that distinction has hidden a real transaction bug in this
    codebase before. The DSN pins ``search_path`` so the connections
    ``capture`` opens for itself land in the same throwaway schema.
    """
    import psycopg

    schema = "betting_splits_real_test"
    dsn = f"{pg_dsn}?options=-csearch_path%3D{schema}"
    admin = psycopg.connect(pg_dsn, autocommit=True)
    admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    admin.execute(f"CREATE SCHEMA {schema}")
    conn = psycopg.connect(dsn)
    try:
        yield conn, dsn
    finally:
        conn.close()
        admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        admin.close()


def test_capture_commits_before_it_returns(real_conn, payload):
    """The poller is a long-lived loop; rows have to be durable the
    moment ``capture`` returns, not whenever the process next exits."""
    conn, dsn = real_conn
    tracker = BettingSplitsTracker(db_path=dsn)
    tracker.fetch = lambda: payload  # type: ignore[method-assign]

    assert tracker.capture() == 116

    # A separate connection sees committed rows and nothing else.
    assert conn.execute(
        "SELECT COUNT(*) FROM betting_splits_history"
    ).fetchone()[0] == 116
