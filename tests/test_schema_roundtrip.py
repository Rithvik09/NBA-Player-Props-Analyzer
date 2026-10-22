"""Round-trip tests: schema → insert → read back, values AND types.

SQLite's dynamic typing let a column declared ``INTEGER`` happily hold a
float or a string. PostgreSQL does not. These tests pin, per table group,
that the values the production upserts actually write survive a round
trip with the Python type the readers expect.

They also pin the two widenings the port makes deliberately:
NBA player/team ids are ``BIGINT`` (an int4 column would be a silent time
bomb), and every ``REAL`` is ``DOUBLE PRECISION`` (SQLite's ``REAL`` is
8-byte; PostgreSQL's ``REAL`` is 4-byte and would quietly lose precision).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A value past int4's 2147483647 ceiling. Nothing in today's NBA feed is
# this large, but the columns are declared BIGINT so that the day one is,
# it inserts instead of raising.
BIG_ID = 9_000_000_000


def _coltypes(conn, table: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = %s AND table_schema = current_schema()",
        (table,),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


# --------------------------------------------------------------- precompute
@pytest.fixture
def real_conn(pg_dsn):
    """A genuine psycopg connection in its own throwaway schema.

    The ``db_conn`` fixture neuters ``commit()`` so tests can roll back.
    That is exactly the wrong harness for code whose correctness depends
    on where its commits land — it hid a transaction bug in
    ``ensure_tables`` once already. Anything that commits mid-routine gets
    tested here instead.
    """
    import psycopg
    conn = psycopg.connect(pg_dsn)
    conn.execute("CREATE SCHEMA IF NOT EXISTS real_conn_test")
    conn.execute("SET search_path TO real_conn_test")
    conn.commit()
    yield conn
    conn.rollback()
    conn.execute("SET search_path TO public")
    conn.execute("DROP SCHEMA IF EXISTS real_conn_test CASCADE")
    conn.commit()
    conn.close()


def test_ensure_tables_leaves_connection_usable(real_conn):
    """``ensure_tables`` must not strand the caller in an aborted transaction.

    Its per-season migration commits once per table. Wrapping those calls
    in a savepoint looked right but was not: a COMMIT discards the
    savepoint, so the following RELEASE raised and poisoned the
    transaction for every later statement.
    """
    from src import precompute_jobs
    precompute_jobs.ensure_tables(real_conn)
    # The real assertion: the connection still works afterwards.
    assert real_conn.execute("SELECT 1").fetchone()[0] == 1
    assert "season" in _coltypes(real_conn, "player_advanced_stats")


def test_ensure_tables_is_idempotent(real_conn):
    """A second pass must be a no-op, not a re-migration or an error."""
    from src import precompute_jobs
    precompute_jobs.ensure_tables(real_conn)
    precompute_jobs.ensure_tables(real_conn)
    assert real_conn.execute("SELECT 1").fetchone()[0] == 1
    assert "season" in _coltypes(real_conn, "player_clutch_stats")


@pytest.fixture
def precompute_schema(db_conn):
    from src import precompute_jobs
    precompute_jobs.ensure_tables(db_conn)
    return db_conn


def test_precompute_ids_are_bigint(precompute_schema):
    """player_id / team_id must not be int4-narrow."""
    conn = precompute_schema
    for table, col in [
        ("player_advanced_stats", "player_id"),
        ("player_vs_opponent", "player_id"),
        ("player_vs_opponent", "opponent_team_id"),
        ("team_stats", "team_id"),
        ("dvp_by_position", "team_id"),
        ("team_special_defenders", "player_id"),
        ("game_team_stats", "team_id"),
        ("game_officials", "ref_id"),
    ]:
        assert _coltypes(conn, table)[col] == "bigint", f"{table}.{col}"


def test_precompute_reals_are_double_precision(precompute_schema):
    """SQLite REAL is 8-byte; PostgreSQL REAL is 4-byte and would lose bits."""
    conn = precompute_schema
    types = _coltypes(conn, "team_stats")
    assert types["pts_fb"] == "double precision"
    assert types["opp_fg_pct"] == "double precision"


def test_dvp_by_position_roundtrip(precompute_schema):
    conn = precompute_schema
    from src.precompute_jobs import upsert_dvp
    upsert_dvp(conn, {(1610612738, "PG"): {
        "gp": 50, "pts": 24.5, "reb": 4.25, "ast": 6.125, "fg3m": 2.5,
        "stl": 1.5, "blk": 0.5, "tov": 2.25, "fd_pts": 38.75,
    }}, updated_at=1700000000)
    row = conn.execute(
        "SELECT team_id, position, gp, pts, fd_pts, source, updated_at "
        "FROM dvp_by_position"
    ).fetchone()
    assert row == (1610612738, "PG", 50, 24.5, 38.75, "fantasypros", 1700000000)
    assert isinstance(row[0], int) and isinstance(row[3], float)


def test_dvp_by_position_upsert_replaces_not_duplicates(precompute_schema):
    """INSERT OR REPLACE on a full column list is an upsert on the PK."""
    conn = precompute_schema
    from src.precompute_jobs import upsert_dvp
    key = (1610612738, "PG")
    upsert_dvp(conn, {key: {"pts": 1.0}}, updated_at=1)
    upsert_dvp(conn, {key: {"pts": 99.0}}, updated_at=2)
    rows = conn.execute("SELECT pts, updated_at FROM dvp_by_position").fetchall()
    assert rows == [(99.0, 2)]


def test_dvp_rolling_quotes_reserved_window_column(precompute_schema):
    """``window`` is a reserved word in PostgreSQL but not in SQLite."""
    conn = precompute_schema
    from src.precompute_jobs import upsert_rolling_dvp
    upsert_rolling_dvp(conn, [{"team_id": 1610612738, "window": 5, "pts": 3.5}],
                       updated_at=7)
    row = conn.execute(
        'SELECT team_id, "window", pts FROM dvp_rolling'
    ).fetchone()
    assert row == (1610612738, 5, 3.5)


def test_player_tables_are_per_season_after_ensure_tables(precompute_schema):
    """ensure_tables runs the per-(player, season) migration on 11 tables."""
    conn = precompute_schema
    from src.precompute_jobs import PER_PLAYER_TABLES_TO_MIGRATE
    for tbl in PER_PLAYER_TABLES_TO_MIGRATE:
        assert "season" in _coltypes(conn, tbl), tbl


def test_advanced_player_stats_roundtrip(precompute_schema):
    conn = precompute_schema
    from src.precompute_jobs import upsert_advanced_player_stats
    upsert_advanced_player_stats(
        conn, [{"player_id": BIG_ID, "usg_pct": 0.3125, "ts_pct": 0.6}],
        updated_at=1700000000, season="2024-25",
    )
    row = conn.execute(
        "SELECT player_id, season, usg_pct, ts_pct FROM player_advanced_stats"
    ).fetchone()
    assert row == (BIG_ID, "2024-25", 0.3125, 0.6)


def test_team_stats_roundtrip_full_column_upsert(precompute_schema):
    conn = precompute_schema
    from src.precompute_jobs import upsert_team_stats
    upsert_team_stats(conn, [{"team_id": 1610612747, "pts_fb": 14.5}], updated_at=11)
    upsert_team_stats(conn, [{"team_id": 1610612747, "pts_fb": 17.25}], updated_at=12)
    rows = conn.execute("SELECT team_id, pts_fb, updated_at FROM team_stats").fetchall()
    assert rows == [(1610612747, 17.25, 12)]


# ------------------------------------------------------------- odds_tracker
def test_odds_tracker_schema_roundtrip(db_conn, db_url):
    from src.odds_tracker import OddsTracker
    tracker = OddsTracker(api_key=None, db_path=db_url)

    # id must be an identity column, not a plain integer we have to supply.
    db_conn.execute(
        "INSERT INTO prop_line_history "
        "(game_id, player_id, player_name, prop_type, bookmaker, line, "
        " over_price, under_price, snapshot_time) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        ("g1", BIG_ID, "A Player", "points", "fanduel", 25.5, -110.0, -110.0,
         "2026-01-01 00:00:00"),
    )
    row = db_conn.execute(
        "SELECT id, player_id, line, over_price FROM prop_line_history"
    ).fetchone()
    assert isinstance(row[0], int) and row[0] >= 1
    assert row[1] == BIG_ID
    assert row[2] == 25.5 and row[3] == -110.0
    assert tracker is not None


def test_prop_outcomes_upsert_is_full_column(db_conn, db_url):
    from src.odds_tracker import OddsTracker
    tracker = OddsTracker(api_key=None, db_path=db_url)
    tracker.record_outcome("g1", "A Player", "points", actual_result=30.0,
                           observed_line=25.5, player_id=BIG_ID)
    tracker.record_outcome("g1", "A Player", "points", actual_result=31.0,
                           observed_line=26.5, player_id=BIG_ID)
    rows = db_conn.execute(
        "SELECT player_id, observed_line, actual_result FROM prop_outcomes"
    ).fetchall()
    assert rows == [(BIG_ID, 26.5, 31.0)]


# ----------------------------------------------------------------- bankroll
def test_bankroll_schema_roundtrip(db_conn, db_url):
    from src.bankroll import BankrollTracker
    bk = BankrollTracker(db_url)
    bk.set_balance(1234.5)
    assert bk.get_balance() == 1234.5
    types = _coltypes(db_conn, "bankroll_bets")
    assert types["id"] == "bigint"
    assert types["stake"] == "double precision"


def test_bankroll_peak_balance_uses_greatest(db_conn, db_url):
    """SQLite's 2-arg MAX() is PostgreSQL's GREATEST()."""
    from src.bankroll import BankrollTracker
    bk = BankrollTracker(db_url)
    bk.set_balance(1000.0)
    bk.set_balance(2000.0)
    bk.set_balance(500.0)
    peak = db_conn.execute(
        "SELECT peak_balance FROM bankroll_state WHERE id = 1"
    ).fetchone()[0]
    assert peak == 2000.0


def test_bankroll_migrations_are_idempotent(db_conn, db_url):
    """Re-running _ensure_schema must not abort on duplicate columns."""
    from src.bankroll import BankrollTracker
    BankrollTracker(db_url)
    BankrollTracker(db_url)  # second pass hits every ALTER again
    cols = _coltypes(db_conn, "bankroll_bets")
    assert "clv" in cols and "model_version" in cols


# ------------------------------------------------------- betting helper core
def test_helper_core_schema_roundtrip(db_conn, db_url):
    from src.basketball_betting_helper import BasketballBettingHelper
    BasketballBettingHelper(db_name=db_url)
    assert _coltypes(db_conn, "players")["id"] == "bigint"
    assert _coltypes(db_conn, "game_logs")["player_id"] == "bigint"
    assert _coltypes(db_conn, "prediction_logs")["player_id"] == "bigint"
    # retrain_meta is seeded by an INSERT ... ON CONFLICT DO NOTHING
    assert db_conn.execute(
        "SELECT samples_at_last_retrain FROM retrain_meta WHERE id = 1"
    ).fetchone() == (0,)


def test_game_logs_id_is_identity(db_conn, db_url):
    from src.basketball_betting_helper import BasketballBettingHelper
    BasketballBettingHelper(db_name=db_url)
    db_conn.execute("INSERT INTO players (id, full_name) VALUES (%s, %s)",
                    (BIG_ID, "A Player"))
    db_conn.execute(
        "INSERT INTO game_logs (player_id, game_date, pts) VALUES (%s,%s,%s)",
        (BIG_ID, "2026-01-01", 30),
    )
    row = db_conn.execute("SELECT id, player_id, pts FROM game_logs").fetchone()
    assert isinstance(row[0], int) and row[0] >= 1
    assert row[1] == BIG_ID and row[2] == 30


def test_game_logs_foreign_key_is_now_enforced(db_conn, db_url):
    """Behaviour change the port makes unavoidable, pinned deliberately.

    ``game_logs`` has always declared ``FOREIGN KEY (player_id)
    REFERENCES players (id)``, but the helper's connection never set
    ``PRAGMA foreign_keys = ON``, so SQLite accepted orphan rows.
    PostgreSQL enforces foreign keys unconditionally. No production code
    writes ``game_logs`` today, so nothing breaks — but a future writer
    has to insert the player first.
    """
    import psycopg
    from src.basketball_betting_helper import BasketballBettingHelper
    BasketballBettingHelper(db_name=db_url)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        db_conn.execute(
            "INSERT INTO game_logs (player_id, game_date, pts) VALUES (%s,%s,%s)",
            (BIG_ID, "2026-01-01", 30),
        )


from tests.test_precompute_seasons import UPSERTS


@pytest.mark.parametrize('upsert', UPSERTS, ids=lambda fn: fn.__name__)
def test_player_upsert_preserves_two_historical_seasons(real_conn, upsert):
    """Exercise real PostgreSQL constraints and the production insert SQL."""
    import inspect
    import re
    from src import precompute_jobs

    precompute_jobs.ensure_tables(real_conn)
    table = re.search(r'INSERT INTO (\w+)', inspect.getsource(upsert)).group(1)
    for season in ('2022-23', '2023-24', '2022-23'):
        upsert(real_conn, [{'player_id': 2544}], updated_at=123, season=season)
    rows = real_conn.execute(
        f'SELECT season FROM {table} WHERE player_id = %s ORDER BY season',
        (2544,),
    ).fetchall()
    assert rows == [('2022-23',), ('2023-24',)]
