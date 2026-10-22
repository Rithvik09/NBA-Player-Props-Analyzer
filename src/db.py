"""Single PostgreSQL connection factory.

Every module used to open its own SQLite connection to a local file. The
Postgres port funnels all of them through :func:`connect` so there is
exactly one place that knows how to reach the database.

Connection string resolution, in order:

1. the ``dsn`` argument, when it is a non-empty string;
2. the ``DATABASE_URL`` environment variable.

Callers that still thread a ``db_path`` / ``db_name`` string around (the
``--db`` CLI flag, ``BasketballBettingHelper(db_name=...)``, …) now pass a
DSN through that same parameter. Passing ``None`` — the new default
everywhere — means "use ``DATABASE_URL``".
"""
from __future__ import annotations

import os
import re

import psycopg
from psycopg.rows import dict_row  # noqa: F401  (re-exported for call sites)

__all__ = [
    "connect",
    "resolve_dsn",
    "dict_row",
    "DSN_ENV_VAR",
    "table_columns",
    "column_info",
]

DSN_ENV_VAR = "DATABASE_URL"

# A leftover ``--db basketball_data.db`` would otherwise reach libpq, which
# reads a bare word as a *database name* and fails with a confusing
# "database ... does not exist". Fail loudly on anything file-shaped instead.
#
# Only bare strings are checked. A real DSN is either a URL (``postgres://``,
# ``postgresql://``) or libpq keyword/value form (``host=... dbname=...``), and
# either of those may legitimately end in ``.db`` — a Neon database happens to
# be named that, say — so neither is treated as a file path.
_DB_FILE_RE = re.compile(r"\.(db|sqlite\d*)$", re.IGNORECASE)


def _looks_like_a_file_path(candidate: str) -> bool:
    if "://" in candidate or "=" in candidate:
        return False
    return bool(_DB_FILE_RE.search(candidate))


def resolve_dsn(dsn: str | None = None) -> str:
    """Return the connection string to use, or raise if there is none."""
    candidate = (dsn or "").strip() or os.environ.get(DSN_ENV_VAR, "").strip()
    if not candidate:
        raise RuntimeError(
            f"No PostgreSQL connection string. Pass a DSN explicitly or set "
            f"{DSN_ENV_VAR} (e.g. postgresql://user@localhost/nba_props)."
        )
    if _looks_like_a_file_path(candidate):
        raise ValueError(
            f"{candidate!r} looks like a SQLite file path, not a PostgreSQL "
            f"DSN. This project moved to PostgreSQL — set {DSN_ENV_VAR} or "
            f"pass --db postgresql://..."
        )
    return candidate


def connect(dsn: str | None = None, *, row_factory=None, autocommit: bool = False):
    """Open a PostgreSQL connection.

    ``row_factory`` defaults to psycopg's tuple rows, which is what the
    old SQLite default produced. Call sites that used to set a
    ``Row``-style factory pass ``row_factory=dict_row``.
    """
    kwargs = {"autocommit": autocommit}
    if row_factory is not None:
        kwargs["row_factory"] = row_factory
    return psycopg.connect(resolve_dsn(dsn), **kwargs)


# ---------------------------------------------------------------------------
# Introspection
#
# Replaces ``PRAGMA table_info(<t>)``, which several modules used to probe
# for optional columns before SELECTing them.
# ---------------------------------------------------------------------------

def table_columns(conn, table: str) -> set[str]:
    """Column names of ``table``, or an empty set if it does not exist."""
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = %s",
        (table,),
    ).fetchall()
    return {r[0] for r in rows}


def column_info(conn, table: str) -> list[tuple[str, str, bool, str | None]]:
    """``(name, type, not_null, default)`` per column, in ordinal order.

    ``type`` is the fully-formatted SQL type (``character varying(20)``,
    not ``character varying``) so callers can paste it straight into DDL.
    """
    rows = conn.execute(
        """
        SELECT column_name,
               format_type(a.atttypid, a.atttypmod),
               attnotnull,
               pg_get_expr(d.adbin, d.adrelid)
        FROM information_schema.columns c
        JOIN pg_attribute a
          ON a.attrelid = to_regclass(quote_ident(c.table_schema) || '.'
                                      || quote_ident(c.table_name))
         AND a.attname = c.column_name
        LEFT JOIN pg_attrdef d
          ON d.adrelid = a.attrelid AND d.adnum = a.attnum
        WHERE c.table_schema = current_schema() AND c.table_name = %s
        ORDER BY c.ordinal_position
        """,
        (table,),
    ).fetchall()
    return [(r[0], r[1], bool(r[2]), r[3]) for r in rows]
