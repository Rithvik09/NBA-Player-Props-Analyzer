"""The per-(player, season) upserts must actually land rows.

``ensure_tables`` migrates eleven player tables to a composite
``(player_id, season)`` primary key with ``season TEXT NOT NULL``. Ten of the
eleven upserts never listed ``season`` in their INSERT column list, so every
row violated the NOT NULL constraint, and the broad ``except Exception`` in
:func:`update_precomputed` turned that into a one-line print. The job reported
success having written nothing, and ten feature tables stayed empty.

So these tests assert on rows *landing*, never on "no exception raised" — the
latter would have passed against the bug, because the exception was swallowed
one frame up.
"""
from __future__ import annotations

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One representative column per table, with two distinct values so the
# idempotency test can tell an update from a no-op. The payload key and the
# column name are the same for all eleven tables.
#   (upsert function, table, column, first value, second value)
PER_SEASON_UPSERTS = [
    ("upsert_advanced_player_stats",   "player_advanced_stats",    "usg_pct",            0.3125,  0.4375),
    ("upsert_clutch_stats",            "player_clutch_stats",      "clutch_pts_pg",      4.5,     6.25),
    ("upsert_hustle_stats",            "player_hustle_stats",      "contested_shots_pg", 7.5,     9.25),
    ("upsert_shot_profile",            "player_shot_profile",      "open_shot_fg_pct",   0.5625,  0.625),
    ("upsert_play_types",              "player_play_types",        "iso_poss_pct",       0.125,   0.1875),
    ("upsert_on_off",                  "player_on_off",            "on_court_net_rating", 8.5,   -3.25),
    ("upsert_shot_zones",              "player_shot_zones",        "rim_fga_pct",        0.375,   0.4375),
    ("upsert_quarter_splits",          "player_quarter_splits",    "q1_avg",             6.75,    8.5),
    ("upsert_player_tracking",         "player_tracking_stats",    "avg_speed",          4.25,    4.75),
    ("upsert_player_scoring_breakdown", "player_scoring_breakdown", "pct_pts_3pt",       0.28125, 0.34375),
    ("upsert_player_yoy",              "player_yoy_stats",         "yoy_pts_change",     2.5,    -1.75),
]

_IDS = [spec[1] for spec in PER_SEASON_UPSERTS]

# Past int4's 2147483647 ceiling, matching tests/test_schema_roundtrip.py —
# player_id is BIGINT and must stay that way.
BIG_ID = 9_000_000_000


def _upsert(name):
    from src import precompute_jobs
    return getattr(precompute_jobs, name)


def test_every_migrated_table_is_covered_here():
    """A new per-season table must not slip in without these three tests."""
    from src.precompute_jobs import PER_PLAYER_TABLES_TO_MIGRATE
    assert sorted(_IDS) == sorted(PER_PLAYER_TABLES_TO_MIGRATE)


@pytest.mark.parametrize("fn_name,table,column,value,_other", PER_SEASON_UPSERTS, ids=_IDS)
def test_upsert_lands_a_row_tagged_with_its_season(
    precompute_schema, fn_name, table, column, value, _other
):
    """The row has to be *there* afterwards, with the season it was given."""
    conn = precompute_schema
    _upsert(fn_name)(
        conn, [{"player_id": BIG_ID, column: value}],
        updated_at=1700000000, season="2023-24",
    )
    rows = conn.execute(
        f"SELECT player_id, season, {column}, updated_at FROM {table}"
    ).fetchall()
    assert rows == [(BIG_ID, "2023-24", value, 1700000000)]


@pytest.mark.parametrize("fn_name,table,column,value,other", PER_SEASON_UPSERTS, ids=_IDS)
def test_upsert_is_idempotent_within_a_season(
    precompute_schema, fn_name, table, column, value, other
):
    """Second write to the same (player_id, season) updates, not duplicates."""
    conn = precompute_schema
    fn = _upsert(fn_name)
    fn(conn, [{"player_id": BIG_ID, column: value}], updated_at=1, season="2023-24")
    fn(conn, [{"player_id": BIG_ID, column: other}], updated_at=2, season="2023-24")
    rows = conn.execute(
        f"SELECT {column}, updated_at FROM {table}"
    ).fetchall()
    assert rows == [(other, 2)]


@pytest.mark.parametrize("fn_name,table,column,value,other", PER_SEASON_UPSERTS, ids=_IDS)
def test_two_seasons_for_one_player_coexist(
    precompute_schema, fn_name, table, column, value, other
):
    """The whole point of the composite key: one player, two season rows.

    If ``season`` were dropped from the INSERT (or hardcoded to the current
    season), a backfill of an older year would overwrite the newer row
    instead of sitting beside it.
    """
    conn = precompute_schema
    fn = _upsert(fn_name)
    fn(conn, [{"player_id": BIG_ID, column: value}], updated_at=1, season="2023-24")
    fn(conn, [{"player_id": BIG_ID, column: other}], updated_at=2, season="2024-25")
    rows = conn.execute(
        f"SELECT season, {column} FROM {table} ORDER BY season"
    ).fetchall()
    assert rows == [("2023-24", value), ("2024-25", other)]


# --------------------------------------------------------- failure reporting
class _StubCursor:
    def execute(self, *args, **kwargs):
        return self


class _StubConn:
    """Just enough connection for ``update_precomputed`` with jobs stubbed out.

    Every compute/scrape call returns nothing, so the upserts never execute a
    statement. All this has to do is count the rollbacks.
    """

    def __init__(self):
        self.rollbacks = 0

    def cursor(self, *args, **kwargs):
        return _StubCursor()

    def execute(self, *args, **kwargs):
        return _StubCursor()

    def commit(self):
        pass

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


def _stub(value):
    return lambda *args, **kwargs: value


@pytest.fixture
def offline_precompute(monkeypatch):
    """``update_precomputed`` with every NBA call and the database stubbed.

    Returns the stub connection so a test can inspect what the run did to it.
    """
    from src import precompute_jobs
    for name in dir(precompute_jobs):
        if not name.startswith(("compute_", "scrape_")):
            continue
        if name == "compute_current_season":
            continue  # still needed for real: it supplies the default season
        # scrape_dvp_by_position is the one job keyed by (team, position).
        empty = {} if name == "scrape_dvp_by_position" else []
        monkeypatch.setattr(precompute_jobs, name, _stub(empty))
    monkeypatch.setattr(precompute_jobs, "ensure_tables", _stub(None))
    conn = _StubConn()
    monkeypatch.setattr(precompute_jobs.db, "connect", _stub(conn))
    return conn


def test_clean_run_reports_no_failed_jobs(offline_precompute):
    from src.precompute_jobs import update_precomputed
    summary = update_precomputed("postgresql:///nba_props_test", season="2024-25")
    assert summary["failed_jobs"] == []
    assert summary["failed_job_count"] == 0


def test_write_failure_is_surfaced_not_swallowed(offline_precompute, monkeypatch, caplog):
    """A dead upsert must show up in the summary and in the log at ERROR.

    This is the regression guard for the original bug: ten upserts raised on
    every single row for months, and the only trace was a stdout line nobody
    read. A summary that lists the failed jobs is what makes the next one
    noticeable.
    """
    from src import precompute_jobs

    def _boom(*args, **kwargs):
        raise RuntimeError("no season column")

    monkeypatch.setattr(precompute_jobs, "upsert_clutch_stats", _boom)
    with caplog.at_level(logging.ERROR, logger="src.precompute_jobs"):
        summary = precompute_jobs.update_precomputed(
            "postgresql:///nba_props_test", season="2024-25"
        )

    assert summary["failed_jobs"] == ["clutch stats"]
    assert summary["failed_job_count"] == 1
    assert any(
        rec.levelno == logging.ERROR and "no season column" in rec.getMessage()
        for rec in caplog.records
    )


def test_one_failure_does_not_cascade_into_the_later_jobs(offline_precompute, monkeypatch):
    """A failed statement aborts the whole PostgreSQL transaction.

    Without a rollback on the way out of the handler every *later* job fails
    too, so "which job broke" is unanswerable from the summary. Same reasoning
    as the per-table rollback in ``ensure_tables``' migration loop.
    """
    from src import precompute_jobs

    def _boom(*args, **kwargs):
        raise RuntimeError("no season column")

    monkeypatch.setattr(precompute_jobs, "upsert_clutch_stats", _boom)
    summary = precompute_jobs.update_precomputed(
        "postgresql:///nba_props_test", season="2024-25"
    )
    assert summary["failed_jobs"] == ["clutch stats"]
    assert offline_precompute.rollbacks == 1
