"""Shared test fixtures.

These smoke tests MUST NOT hit live nba_api endpoints — network flakiness
would make CI red for the wrong reason. We patch every nba_api.endpoints
constructor with canned fixtures. ``NBA_CACHE_DISABLE=1`` is set so no
stale entries leak between tests.

The Postgres fixtures below give every DB-touching test a connection that
is rolled back on teardown, so tests stay isolated without the cost of
creating and dropping a database per test.
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg
import pytest

os.environ.setdefault("NBA_CACHE_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import db as _db  # noqa: E402


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

def pytest_sessionstart(session):
    """Start every run from an empty schema.

    Most DB work in the suite happens on the rollback fixture below, but a
    few modules (``src.app`` and the Flask endpoint tests behind it) open
    their own connections at import time and commit. Without a reset those
    rows survive into the next run and make later assertions depend on run
    order.

    Guarded on the database name so a stray ``DATABASE_URL`` pointing at a
    real database cannot be dropped by running the tests.
    """
    dsn = os.environ.get(_db.DSN_ENV_VAR, "").strip()
    if not dsn:
        return
    with psycopg.connect(dsn, autocommit=True) as conn:
        name = conn.execute("SELECT current_database()").fetchone()[0]
        if "test" not in name:
            raise RuntimeError(
                f"refusing to reset database {name!r}: the test suite drops "
                f"and recreates its schema, so {_db.DSN_ENV_VAR} must point "
                f"at a database with 'test' in its name."
            )
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """DSN for the test database.

    Deliberately fails rather than skipping: the project is PostgreSQL-only,
    so "no database configured" is a broken setup, not an optional extra.
    Skipping here would let CI go green having exercised none of the ~110
    tests that touch the database.
    """
    dsn = os.environ.get(_db.DSN_ENV_VAR, "").strip()
    if not dsn:
        raise RuntimeError(
            f"{_db.DSN_ENV_VAR} is not set. The test suite needs a PostgreSQL "
            f"database with 'test' in its name, e.g. "
            f"DATABASE_URL=postgresql:///nba_props_test pytest"
        )
    return dsn


@pytest.fixture(scope="session")
def _pg_session_conn(pg_dsn):
    conn = psycopg.connect(pg_dsn)
    yield conn
    conn.close()


class _SharedConn:
    """Non-closing proxy over the session connection.

    Production code opens and closes its own connections freely. Handing
    every ``db.connect()`` the same underlying connection — with
    ``commit``/``close`` neutered and ``rollback`` scoped to the test's
    savepoint — keeps all of that work inside one transaction the fixture
    can throw away afterwards.

    ``autocommit`` is honoured rather than ignored. Several readers run a
    long series of independent probes and swallow "relation does not
    exist" per probe; on a real autocommit connection a failed statement
    leaves the next one usable, so the proxy reproduces that with a
    per-statement savepoint. Without it those modules would behave
    differently under test than in production.
    """

    _stmt_sp = 0

    def __init__(self, conn, savepoint: str, autocommit: bool = False,
                 row_factory=None):
        self._conn = conn
        self._savepoint = savepoint
        self._autocommit = autocommit
        # Each proxy pins its own row factory. psycopg keeps it on the
        # connection, so without this the dict_row that production code
        # asks for would leak into whatever the test queries next.
        self._row_factory = row_factory or psycopg.rows.tuple_row

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Deliberately neither commits nor closes: the fixture owns both.
        return False

    def commit(self):
        """No-op — the test's outer transaction must stay rollback-able."""

    def close(self):
        """No-op — the session connection outlives every test."""

    def rollback(self):
        self._conn.execute(f"ROLLBACK TO SAVEPOINT {self._savepoint}")

    def cursor(self, *a, **kw):
        kw.setdefault("row_factory", self._row_factory)
        cur = self._conn.cursor(*a, **kw)
        return _SharedCursor(self, cur) if self._autocommit else cur

    def execute(self, *a, **kw):
        self._conn.row_factory = self._row_factory
        if not self._autocommit:
            return self._conn.execute(*a, **kw)
        return self._isolated(self._conn.execute, *a, **kw)

    def _isolated(self, fn, *a, **kw):
        type(self)._stmt_sp += 1
        sp = f"stmt_sp_{type(self)._stmt_sp}"
        self._conn.execute(f"SAVEPOINT {sp}")
        try:
            out = fn(*a, **kw)
        except Exception:
            self._conn.execute(f"ROLLBACK TO SAVEPOINT {sp}")
            raise
        self._conn.execute(f"RELEASE SAVEPOINT {sp}")
        return out


class _SharedCursor:
    """Cursor whose ``execute`` is statement-isolated (autocommit proxy)."""

    def __init__(self, owner: "_SharedConn", cur):
        self._owner = owner
        self._cur = cur

    def __getattr__(self, name):
        return getattr(self._cur, name)

    def __iter__(self):
        return iter(self._cur)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._cur.close()
        return False

    def execute(self, *a, **kw):
        self._owner._isolated(self._cur.execute, *a, **kw)
        return self


_SCHEMA_SEQ = itertools.count()


@pytest.fixture
def db_conn(_pg_session_conn, monkeypatch):
    """A Postgres connection whose writes are discarded after the test.

    Two layers of isolation:

    * a private schema on the ``search_path``, so the test starts with no
      tables at all and cannot see — or be seen by — the ``public`` tables
      that ``src.app`` creates and commits at import time;
    * one transaction for the whole test, rolled back on teardown. Both
      ``CREATE SCHEMA`` and DDL are transactional in PostgreSQL, so the
      rollback takes the schema with it.

    Also patches :func:`src.db.connect` so production code under test
    joins that same transaction instead of opening its own connection.
    """
    conn = _pg_session_conn
    conn.rollback()  # start from a clean slate even if a prior test errored
    schema = f"test_{next(_SCHEMA_SEQ)}"
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}")
    conn.execute("SAVEPOINT test_sp")
    shared = _SharedConn(conn, "test_sp")

    def _fake_connect(dsn=None, *, row_factory=None, autocommit=False):
        return _SharedConn(conn, "test_sp", autocommit=autocommit,
                           row_factory=row_factory)

    monkeypatch.setattr(_db, "connect", _fake_connect)
    try:
        yield shared
    finally:
        conn.rollback()
        conn.row_factory = psycopg.rows.tuple_row
        conn.execute("SET search_path TO public")


@pytest.fixture
def db_url(db_conn) -> str:
    """DSN-shaped value to hand to code that still takes a ``db_path``.

    The value is never dialled — :func:`src.db.connect` is patched by
    ``db_conn`` — but it has to survive ``resolve_dsn``'s file-path guard.
    """
    return "postgresql:///nba_props_test"


@pytest.fixture
def scoreboard_df():
    """Two-game scoreboard: BOS (home) vs LAL (away), playoff game."""
    return pd.DataFrame([
        {
            "GAME_ID": "0042400101",
            "HOME_TEAM_ID": 1610612738,  # BOS
            "VISITOR_TEAM_ID": 1610612747,  # LAL
            "GAME_DATE_EST": "2026-04-21T00:00:00",
        },
        {
            "GAME_ID": "0042400102",
            "HOME_TEAM_ID": 1610612744,  # GSW
            "VISITOR_TEAM_ID": 1610612743,  # DEN
            "GAME_DATE_EST": "2026-04-21T00:00:00",
        },
    ])


@pytest.fixture
def empty_playoff_games_df():
    """No prior playoff games head-to-head."""
    return pd.DataFrame(columns=["GAME_ID", "GAME_DATE", "WL", "TEAM_ID", "MATCHUP"])


@pytest.fixture
def three_win_playoff_games_df():
    """Team won 3 prior games — next game is elimination."""
    today = pd.Timestamp("2026-04-21")
    rows = []
    for i in range(3):
        rows.append({
            "GAME_ID": f"00424001{i:02d}",
            "GAME_DATE": (today - pd.Timedelta(days=(3 - i) * 2)).strftime("%Y-%m-%d"),
            "WL": "W",
            "TEAM_ID": 1610612738,
            "MATCHUP": "BOS vs. LAL",
        })
    return pd.DataFrame(rows)
