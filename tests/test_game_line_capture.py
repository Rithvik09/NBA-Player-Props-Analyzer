"""Game-line (moneyline / spread / total) history capture.

``fetch_upcoming_games`` already asks The Odds API for the game markets
and then threw the whole odds payload away, keeping only the matchup.
That movement cannot be bought back later — there is no free historical
feed — so every day without capture is history lost.

What it may not do is cost the quota it hasn't got. The API bills
markets × regions per call: ``h2h,spreads,totals`` across ``us,eu``
(needed for Pinnacle, which is not in ``us``) is 6 credits against a
500/month free tier. So capture is opt-in per call and scheduled twice a
game day — an "open" and a "close" — while the hourly prop poll keeps
the cheap 2-credit request it always made. These tests pin all of that:
the rows that land, the labels they carry, the request parameters each
path sends, the quota floor, and the fact that a storage failure can
never cost the caller its games.

Nothing here hits the Odds API: ``_get`` is patched with canned payloads
shaped the way the real endpoint responds.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import odds_tracker as _ot  # noqa: E402
from src.odds_tracker import OddsTracker  # noqa: E402


FAKE_KEY = "test-key-not-used"

GAME_LINE_ENV = (
    "ODDS_REGIONS",
    "GAME_LINE_SNAPSHOTS",
    "GAME_LINE_OPEN_HOUR",
    "ODDS_QUOTA_FLOOR",
)


def _make_tracker(db_url) -> OddsTracker:
    return OddsTracker(api_key=FAKE_KEY, db_path=db_url)


# A single event as the Odds API returns it for
# ``markets=h2h,spreads,totals``: two bookmakers, all three markets,
# both signs of ``point`` on the spread, Over/Under on the total, and no
# ``point`` at all on the moneyline.
EVENT_PAYLOAD = [
    {
        "id": "evt1",
        "sport_key": "basketball_nba",
        "commence_time": "2026-05-15T19:00:00Z",
        "home_team": "Boston Celtics",
        "away_team": "Los Angeles Lakers",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "2026-05-15T17:02:11Z",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Boston Celtics", "price": -180},
                        {"name": "Los Angeles Lakers", "price": 155},
                    ]},
                    {"key": "spreads", "outcomes": [
                        {"name": "Boston Celtics", "price": -110, "point": -4.5},
                        {"name": "Los Angeles Lakers", "price": -110, "point": 4.5},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "price": -105, "point": 224.5},
                        {"name": "Under", "price": -115, "point": 224.5},
                    ]},
                ],
            },
            {
                # Pinnacle only shows up once regions includes "eu" —
                # the sharp anchor the analysis layer is being built for.
                "key": "pinnacle",
                "title": "Pinnacle",
                "last_update": "2026-05-15T17:01:48Z",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Boston Celtics", "price": -175},
                        {"name": "Los Angeles Lakers", "price": 148},
                    ]},
                    {"key": "spreads", "outcomes": [
                        {"name": "Boston Celtics", "price": -108, "point": -4.0},
                        {"name": "Los Angeles Lakers", "price": -112, "point": 4.0},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "price": -110, "point": 225.0},
                        {"name": "Under", "price": -110, "point": 225.0},
                    ]},
                ],
            },
        ],
    },
]


@pytest.fixture(autouse=True)
def clean_capture_env(monkeypatch):
    """Start every test from the documented defaults.

    ``_LAST_POLL`` is module state shared by the whole suite and the
    quota guard reads it, so a value left behind by one test would
    silently decide another.
    """
    for name in GAME_LINE_ENV:
        monkeypatch.delenv(name, raising=False)
    for key in ("quota_remaining", "game_lines_captured", "game_lines_captured_at"):
        monkeypatch.setitem(_ot._LAST_POLL, key, None)


@pytest.fixture
def frozen_now(monkeypatch):
    """Pin the snapshot clock.

    ``snapshot_time`` is second-resolution wall clock, so two captures
    in one test would usually — but not always — share a timestamp. The
    idempotency check is about the UNIQUE tuple, not about how fast the
    test runs, so the clock is held still rather than raced.
    """
    return _freeze(monkeypatch, datetime(2026, 5, 15, 18, 0, 0, tzinfo=timezone.utc))


def _freeze(monkeypatch, at: datetime) -> str:
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return at if tz is not None else at.replace(tzinfo=None)

    monkeypatch.setattr("src.odds_tracker.datetime", _Frozen)
    return at.strftime("%Y-%m-%d %H:%M:%S")


def _game_lines(conn) -> list[tuple]:
    return conn.execute(
        """
        SELECT game_id, bookmaker, market, outcome_name, point, price, snapshot_label
        FROM game_line_history
        ORDER BY bookmaker, market, outcome_name
        """
    ).fetchall()


def _seed_tipoff(conn, *, game_id: str, commence_time: str) -> None:
    conn.execute(
        """
        INSERT INTO game_tipoffs (game_id, commence_time, home_team, away_team, last_seen)
        VALUES (%s, %s, 'BOS', 'LAL', %s)
        ON CONFLICT (game_id) DO UPDATE SET commence_time = excluded.commence_time
        """,
        (game_id, commence_time, datetime.now(timezone.utc).isoformat()),
    )


def _iso_z(when: datetime) -> str:
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def test_init_db_creates_game_line_history(db_conn, db_url):
    from src import db as _db
    OddsTracker(api_key=FAKE_KEY, db_path=db_url)
    # Re-init must be a no-op, like the other odds tables.
    OddsTracker(api_key=FAKE_KEY, db_path=db_url)

    cols = _db.table_columns(db_conn, "game_line_history")
    assert cols == {
        "id", "game_id", "bookmaker", "market", "outcome_name",
        "point", "price", "snapshot_label", "snapshot_time",
    }

    # The UNIQUE tuple is what makes re-polling idempotent; pin it.
    uniques = db_conn.execute(
        """
        SELECT a.attname
        FROM pg_constraint c
        JOIN pg_attribute a ON a.attrelid = c.conrelid
                           AND a.attnum = ANY(c.conkey)
        WHERE c.contype = 'u'
          AND c.conrelid = to_regclass(current_schema() || '.game_line_history')
        """
    ).fetchall()
    assert {r[0] for r in uniques} == {
        "game_id", "bookmaker", "market", "outcome_name", "snapshot_time",
    }


# --------------------------------------------------------------------------- #
# What each path asks the API for — this is the quota budget
# --------------------------------------------------------------------------- #

def test_plain_fetch_is_unchanged_and_captures_nothing(db_conn, db_url, monkeypatch):
    """The hourly prop poll must keep its 2-credit request.

    Widening it to h2h across us,eu would triple the cost of a call that
    happens every hour — the whole reason capture is scheduled instead.
    """
    tracker = _make_tracker(db_url)
    calls = []

    def fake_get(url, params=None):
        calls.append((url, params))
        return EVENT_PAYLOAD

    monkeypatch.setattr(tracker, "_get", fake_get)
    tracker.fetch_upcoming_games()

    assert len(calls) == 1
    assert calls[0][1]["regions"] == "us"
    assert calls[0][1]["markets"] == "spreads,totals"
    assert _game_lines(db_conn) == []


def test_capture_widens_the_same_single_call(db_conn, db_url, monkeypatch):
    """Capture rides the request we already make — never a second one."""
    tracker = _make_tracker(db_url)
    calls = []

    def fake_get(url, params=None):
        calls.append((url, params))
        return EVENT_PAYLOAD

    monkeypatch.setattr(tracker, "_get", fake_get)
    tracker.fetch_upcoming_games(capture="open")

    assert len(calls) == 1
    _url, params = calls[0]
    assert set(params["markets"].split(",")) == {"h2h", "spreads", "totals"}
    assert params["regions"] == "us,eu"  # "eu" is where Pinnacle lives


def test_odds_regions_env_overrides_capture_regions(db_conn, db_url, monkeypatch):
    """Moving to a paid tier should be config, not code."""
    monkeypatch.setenv("ODDS_REGIONS", "us,eu,uk")
    tracker = _make_tracker(db_url)
    seen = {}

    def fake_get(url, params=None):
        seen.update(params)
        return []

    monkeypatch.setattr(tracker, "_get", fake_get)
    tracker.fetch_upcoming_games(capture="open")

    assert seen["regions"] == "us,eu,uk"


def test_snapshot_all_games_does_not_capture_game_lines(db_conn, db_url, monkeypatch):
    """The hourly poll loop must not trigger capture.

    Six credits an hour exhausts the 500/month tier in under a week.
    """
    tracker = _make_tracker(db_url)

    def fake_get(url, params=None):
        if "/events/" in url:
            return {}
        return EVENT_PAYLOAD

    monkeypatch.setattr(tracker, "_get", fake_get)
    monkeypatch.setattr("src.odds_tracker.time.sleep", lambda *a, **kw: None)

    tracker.snapshot_all_games()

    assert _game_lines(db_conn) == []


# --------------------------------------------------------------------------- #
# The rows that land
# --------------------------------------------------------------------------- #

def test_capture_stores_every_outcome(db_conn, db_url, monkeypatch, frozen_now):
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)

    tracker.fetch_upcoming_games(capture="open")

    assert _game_lines(db_conn) == [
        ("evt1", "draftkings", "h2h", "Boston Celtics", None, -180.0, "open"),
        ("evt1", "draftkings", "h2h", "Los Angeles Lakers", None, 155.0, "open"),
        ("evt1", "draftkings", "spreads", "Boston Celtics", -4.5, -110.0, "open"),
        ("evt1", "draftkings", "spreads", "Los Angeles Lakers", 4.5, -110.0, "open"),
        ("evt1", "draftkings", "totals", "Over", 224.5, -105.0, "open"),
        ("evt1", "draftkings", "totals", "Under", 224.5, -115.0, "open"),
        ("evt1", "pinnacle", "h2h", "Boston Celtics", None, -175.0, "open"),
        ("evt1", "pinnacle", "h2h", "Los Angeles Lakers", None, 148.0, "open"),
        ("evt1", "pinnacle", "spreads", "Boston Celtics", -4.0, -108.0, "open"),
        ("evt1", "pinnacle", "spreads", "Los Angeles Lakers", 4.0, -112.0, "open"),
        ("evt1", "pinnacle", "totals", "Over", 225.0, -110.0, "open"),
        ("evt1", "pinnacle", "totals", "Under", 225.0, -110.0, "open"),
    ]

    snapshots = db_conn.execute(
        "SELECT DISTINCT snapshot_time FROM game_line_history"
    ).fetchall()
    assert snapshots == [(frozen_now,)]


def test_recapturing_the_same_snapshot_does_not_duplicate(db_conn, db_url, monkeypatch, frozen_now):
    """Two captures inside one snapshot_time collapse to one set of rows."""
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)

    tracker.fetch_upcoming_games(capture="open")
    first = _game_lines(db_conn)
    tracker.fetch_upcoming_games(capture="open")

    assert _game_lines(db_conn) == first
    assert len(first) == 12


def test_the_close_snapshot_appends_rather_than_overwrites(db_conn, db_url, monkeypatch):
    """Open → close is the pair the CLV analysis needs; the close must
    be new history, not an update of the open."""
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)

    _freeze(monkeypatch, datetime(2026, 5, 15, 14, 0, 0, tzinfo=timezone.utc))
    tracker.fetch_upcoming_games(capture="open")
    _freeze(monkeypatch, datetime(2026, 5, 15, 18, 40, 0, tzinfo=timezone.utc))
    tracker.fetch_upcoming_games(capture="close")

    counts = db_conn.execute(
        "SELECT snapshot_label, COUNT(*) FROM game_line_history "
        "GROUP BY snapshot_label ORDER BY snapshot_label"
    ).fetchall()
    assert counts == [("close", 12), ("open", 12)]


def test_capture_records_telemetry(db_conn, db_url, monkeypatch, frozen_now):
    """A capture that silently stopped looks exactly like a quiet day,
    so /odds/status has to be able to tell them apart."""
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)

    tracker.fetch_upcoming_games(capture="open")

    status = _ot.last_poll_status()
    assert status["game_lines_captured"] == 12
    assert status["game_lines_captured_at"] is not None


# --------------------------------------------------------------------------- #
# The fetch must survive capture
# --------------------------------------------------------------------------- #

def test_fetch_upcoming_games_keeps_its_return_shape(db_conn, db_url, monkeypatch):
    """Callers get the same four-key dicts they always did — the odds
    payload is persisted, not returned."""
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)

    expected = [{
        "id": "evt1",
        "home_team": "Boston Celtics",
        "away_team": "Los Angeles Lakers",
        "commence_time": "2026-05-15T19:00:00Z",
    }]
    assert tracker.fetch_upcoming_games() == expected
    assert tracker.fetch_upcoming_games(capture="open") == expected


def test_fetch_upcoming_games_returns_games_when_storage_raises(db_conn, db_url, monkeypatch, caplog):
    """A broken capture is a lost snapshot, not a broken poll."""
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)

    def boom(*a, **kw):
        raise RuntimeError("no disk")

    monkeypatch.setattr(tracker, "_store_game_lines", boom)

    with caplog.at_level("WARNING"):
        games = tracker.fetch_upcoming_games(capture="open")

    assert len(games) == 1
    assert games[0]["id"] == "evt1"
    assert "no disk" in caplog.text


def test_partial_events_do_not_raise(db_conn, db_url, monkeypatch, frozen_now):
    """The feed is not always complete: an event can arrive with no
    bookmakers, a market with no outcomes, or an outcome whose ``point``
    is absent. None of those may cost us the good rows."""
    tracker = _make_tracker(db_url)
    payload = [
        {
            "id": "no_books",
            "home_team": "BOS", "away_team": "LAL",
            "commence_time": "2026-05-15T19:00:00Z",
        },
        {
            "id": "thin",
            "home_team": "DEN", "away_team": "GSW",
            "commence_time": "2026-05-15T22:00:00Z",
            "bookmakers": [
                {"key": "betmgm", "markets": [
                    {"key": "spreads"},                      # no outcomes key
                    {"key": "totals", "outcomes": []},       # empty outcomes
                    {"key": "h2h", "outcomes": [             # no point — h2h never has one
                        {"name": "Denver Nuggets", "price": -140},
                        {"name": "Golden State Warriors"},   # no price either
                    ]},
                ]},
            ],
        },
    ]
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: payload)

    games = tracker.fetch_upcoming_games(capture="adhoc")

    assert [g["id"] for g in games] == ["no_books", "thin"]
    assert _game_lines(db_conn) == [
        ("thin", "betmgm", "h2h", "Denver Nuggets", None, -140.0, "adhoc"),
    ]


# --------------------------------------------------------------------------- #
# Quota guard
# --------------------------------------------------------------------------- #

def test_capture_refused_below_the_quota_floor(db_conn, db_url, monkeypatch, caplog):
    tracker = _make_tracker(db_url)

    def forbidden(*a, **kw):
        raise AssertionError("quota guard must stop the call before the API")

    monkeypatch.setattr(tracker, "_get", forbidden)
    monkeypatch.setitem(_ot._LAST_POLL, "quota_remaining", 12)

    with caplog.at_level("ERROR"):
        assert tracker.capture_game_lines("open") is None

    assert "12 credits remaining" in caplog.text
    assert _game_lines(db_conn) == []


def test_capture_proceeds_above_the_quota_floor(db_conn, db_url, monkeypatch):
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)
    monkeypatch.setitem(_ot._LAST_POLL, "quota_remaining", 497)

    assert tracker.capture_game_lines("open") == 1
    assert len(_game_lines(db_conn)) == 12


def test_capture_proceeds_when_quota_is_not_yet_known(db_conn, db_url, monkeypatch):
    """No call has returned a header yet — that first call is how we
    learn, so it must not be blocked."""
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)

    assert tracker.capture_game_lines("adhoc") == 1


def test_quota_floor_is_configurable(db_conn, db_url, monkeypatch):
    monkeypatch.setenv("ODDS_QUOTA_FLOOR", "600")
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)
    monkeypatch.setitem(_ot._LAST_POLL, "quota_remaining", 497)

    assert tracker.capture_game_lines("open") is None


# --------------------------------------------------------------------------- #
# Scheduling: two snapshots a game day
# --------------------------------------------------------------------------- #

def test_open_snapshot_fires_once_per_day(db_conn, db_url, monkeypatch):
    monkeypatch.setenv("GAME_LINE_SNAPSHOTS", "open")
    monkeypatch.setenv("GAME_LINE_OPEN_HOUR", "0")  # always past the open hour
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)
    _seed_tipoff(db_conn, game_id="later_today",
                 commence_time=_iso_z(datetime.now(timezone.utc) + timedelta(hours=6)))

    assert tracker.maybe_capture_game_lines() == {"captured": ["open"], "skipped_quota": False}
    taken = len(_game_lines(db_conn))
    assert taken == 12

    # The sweeper calls this every few minutes — it must not re-spend.
    assert tracker.maybe_capture_game_lines() == {"captured": [], "skipped_quota": False}
    assert len(_game_lines(db_conn)) == taken


def test_an_empty_capture_is_not_retried_every_tick(db_conn, db_url, monkeypatch):
    """A capture that comes back empty (API error, board not up yet)
    writes no rows, so the history-table cooldown cannot hold it back.
    Without a separate attempt cooldown the few-minute sweeper would
    re-spend six credits a tick and empty the month by evening.
    """
    monkeypatch.setenv("GAME_LINE_SNAPSHOTS", "open")
    monkeypatch.setenv("GAME_LINE_OPEN_HOUR", "0")
    tracker = _make_tracker(db_url)
    calls = []
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: calls.append(1) or [])
    _seed_tipoff(db_conn, game_id="later_today",
                 commence_time=_iso_z(datetime.now(timezone.utc) + timedelta(hours=6)))

    tracker.maybe_capture_game_lines()
    assert len(calls) == 1

    tracker.maybe_capture_game_lines()
    tracker.maybe_capture_game_lines()
    assert len(calls) == 1


def test_open_snapshot_waits_for_the_open_hour(db_conn, db_url, monkeypatch):
    monkeypatch.setenv("GAME_LINE_SNAPSHOTS", "open")
    monkeypatch.setenv("GAME_LINE_OPEN_HOUR", "25")  # unreachable local hour
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)
    _seed_tipoff(db_conn, game_id="later_today",
                 commence_time=_iso_z(datetime.now(timezone.utc) + timedelta(hours=6)))

    assert tracker.maybe_capture_game_lines()["captured"] == []
    assert _game_lines(db_conn) == []


def test_open_snapshot_skipped_on_days_with_no_games(db_conn, db_url, monkeypatch):
    """An empty board costs the same six credits as a full one."""
    monkeypatch.setenv("GAME_LINE_SNAPSHOTS", "open")
    monkeypatch.setenv("GAME_LINE_OPEN_HOUR", "0")
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)
    _seed_tipoff(db_conn, game_id="yesterday",
                 commence_time=_iso_z(datetime.now(timezone.utc) - timedelta(days=1)))

    assert tracker.maybe_capture_game_lines()["captured"] == []
    assert _game_lines(db_conn) == []


def test_close_snapshot_fires_for_an_imminent_tipoff(db_conn, db_url, monkeypatch):
    """Reuses the close sweeper's notion of imminent: a tipoff inside
    the existing closing window, read from the game_tipoffs cache."""
    monkeypatch.setenv("GAME_LINE_SNAPSHOTS", "close")
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)
    _seed_tipoff(db_conn, game_id="soon",
                 commence_time=_iso_z(datetime.now(timezone.utc) + timedelta(minutes=8)))

    assert tracker.maybe_capture_game_lines()["captured"] == ["close"]
    labels = {r[6] for r in _game_lines(db_conn)}
    assert labels == {"close"}


def test_close_snapshot_does_not_fire_hours_before_tipoff(db_conn, db_url, monkeypatch):
    monkeypatch.setenv("GAME_LINE_SNAPSHOTS", "close")
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)
    _seed_tipoff(db_conn, game_id="tonight",
                 commence_time=_iso_z(datetime.now(timezone.utc) + timedelta(hours=4)))

    assert tracker.maybe_capture_game_lines()["captured"] == []
    assert _game_lines(db_conn) == []


def test_game_line_snapshots_env_can_disable_capture(db_conn, db_url, monkeypatch):
    monkeypatch.setenv("GAME_LINE_SNAPSHOTS", "")
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)
    _seed_tipoff(db_conn, game_id="soon",
                 commence_time=_iso_z(datetime.now(timezone.utc) + timedelta(minutes=8)))

    assert tracker.maybe_capture_game_lines() == {"captured": [], "skipped_quota": False}
    assert _game_lines(db_conn) == []


def test_scheduler_reports_a_quota_skip(db_conn, db_url, monkeypatch):
    monkeypatch.setenv("GAME_LINE_SNAPSHOTS", "close")
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)
    monkeypatch.setitem(_ot._LAST_POLL, "quota_remaining", 3)
    _seed_tipoff(db_conn, game_id="soon",
                 commence_time=_iso_z(datetime.now(timezone.utc) + timedelta(minutes=8)))

    assert tracker.maybe_capture_game_lines() == {"captured": [], "skipped_quota": True}
    assert _game_lines(db_conn) == []


def test_unknown_snapshot_label_is_warned_about(db_conn, db_url, monkeypatch, caplog):
    monkeypatch.setenv("GAME_LINE_SNAPSHOTS", "opne")  # typo
    tracker = _make_tracker(db_url)
    monkeypatch.setattr(tracker, "_get", lambda *a, **kw: EVENT_PAYLOAD)

    with caplog.at_level("WARNING"):
        assert tracker.maybe_capture_game_lines()["captured"] == []

    assert "opne" in caplog.text
