"""Tests for src/db.py — the single PostgreSQL connection factory.

Every module used to call ``sqlite3.connect(<file>)`` directly. The port
funnels all of them through ``db.connect(dsn)`` so there is exactly one
place that knows how to reach the database.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import db  # noqa: E402


def test_connect_uses_database_url_when_dsn_omitted(pg_dsn):
    with db.connect() as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_connect_accepts_explicit_dsn(pg_dsn):
    with db.connect(pg_dsn) as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_connect_treats_empty_dsn_as_unset(pg_dsn):
    """``--db ''`` and ``--db None`` both mean "fall back to DATABASE_URL"."""
    with db.connect("") as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_connect_raises_when_no_dsn_available(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        db.connect()


@pytest.mark.parametrize("path", [
    "basketball_data.db",
    "/var/data/basketball_data.db",
    "bankroll.sqlite",
    "bankroll.sqlite3",
])
def test_connect_rejects_a_sqlite_file_path(pg_dsn, path):
    """Guard against a stale ``--db basketball_data.db`` reaching psycopg."""
    with pytest.raises(ValueError, match="SQLite file path"):
        db.connect(path)


@pytest.mark.parametrize("dsn", [
    "postgresql://user@host/weird.db",
    "postgres://user@host/weird.db",
    "host=localhost dbname=weird.db",
])
def test_file_path_guard_does_not_fire_on_real_dsns(dsn):
    """A URL or keyword/value DSN may legitimately end in ``.db``."""
    assert db.resolve_dsn(dsn) == dsn


def test_row_factory_gives_mapping_rows(pg_dsn):
    """``sqlite3.Row`` call sites need key access and ``dict(row)``."""
    with db.connect(row_factory=db.dict_row) as conn:
        row = conn.execute("SELECT 1 AS a, 2 AS b").fetchone()
        assert row["a"] == 1
        assert dict(row) == {"a": 1, "b": 2}


def test_default_rows_are_tuples(pg_dsn):
    """Unmarked call sites unpack positionally, exactly as sqlite3 did."""
    with db.connect() as conn:
        assert conn.execute("SELECT 1, 2").fetchone() == (1, 2)
