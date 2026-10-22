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
from urllib.parse import unquote

import psycopg
from psycopg.rows import dict_row  # noqa: F401  (re-exported for call sites)

__all__ = [
    "connect",
    "resolve_dsn",
    "scrub_credentials",
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
        # The candidate is deliberately not quoted back. The guard skips
        # anything holding "://" or "=", but a near-miss typo like
        # "postgres:/user:pw@host/x.db" — one slash short of a URL —
        # reaches here carrying a password, and this message ends up in
        # logs and HTTP bodies.
        raise ValueError(
            f"The connection string looks like a SQLite file path, not a "
            f"PostgreSQL DSN (it is not echoed here: it may carry a "
            f"password). This project moved to PostgreSQL — set "
            f"{DSN_ENV_VAR} or pass --db postgresql://..."
        )
    return candidate


# ---------------------------------------------------------------------------
# Credential scrubbing
#
# libpq quotes the offending token back on a parse error — a password
# holding a bare "%" produces ``invalid percent-encoded token: "<pw>"`` —
# and names the connection target when a connection fails. Those strings
# used to travel out of :func:`connect` into ``jsonify({'error': str(e)})``
# on routes that need no authentication, so they are scrubbed here, at the
# one place that knows the DSN.
# ---------------------------------------------------------------------------

_REDACTED = "[redacted]"

# ``scheme://user:pass@`` anywhere in a message. Matches the whole
# userinfo segment rather than just the password: the username is half a
# credential on its own.
_URL_CREDENTIALS_RE = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.\-]*://)[^\s/?#@]*@")

# ``password=secret`` / ``password = 'se cret'`` in libpq keyword/value form.
_KEYWORD_PASSWORD_RE = re.compile(
    r"(?i)\bpassword\s*=\s*('(?:[^'\\]|\\.)*'|\S+)"
)

_URL_USERINFO_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://([^/?#@]*)@")


def _secret_tokens(dsn: str | None) -> list[str]:
    """Every substring of ``dsn`` that must not survive into a message.

    Longest first, so replacing the password does not leave the rest of
    the DSN — which contains it — stranded in the output.
    """
    candidate = (dsn or "").strip() or os.environ.get(DSN_ENV_VAR, "").strip()
    if not candidate:
        return []
    tokens = {candidate}
    match = _URL_USERINFO_RE.match(candidate)
    if match:
        userinfo = match.group(1)
        tokens.add(userinfo)
        if ":" in userinfo:
            password = userinfo.split(":", 1)[1]
            # libpq percent-decodes the userinfo, so a message can quote
            # either spelling depending on how far the connection got.
            tokens.update({password, unquote(password)})
    for match in _KEYWORD_PASSWORD_RE.finditer(candidate):
        value = match.group(1)
        tokens.update({value, value.strip("'")})
    return sorted((t for t in tokens if t), key=len, reverse=True)


def scrub_credentials(text: str, dsn: str | None = None) -> str:
    """Remove DSN credentials from ``text``.

    Two passes. The first strips literal pieces of ``dsn`` — the whole
    string, its ``user:pass`` segment, the password both raw and
    percent-decoded — which catches the tokens libpq quotes back without
    any surrounding syntax. ``dsn`` defaults to ``DATABASE_URL`` so
    call sites that only have a message (the odds poller's last-error
    field, say) still get the configured credentials removed.

    The second is syntactic and catches credentials this process never
    resolved: any ``scheme://user:pass@`` and any ``password=...``.
    """
    out = str(text)
    for token in _secret_tokens(dsn):
        out = out.replace(token, _REDACTED)
    out = _URL_CREDENTIALS_RE.sub(rf"\g<1>{_REDACTED}@", out)
    out = _KEYWORD_PASSWORD_RE.sub(f"password={_REDACTED}", out)
    return out


def connect(dsn: str | None = None, *, row_factory=None, autocommit: bool = False):
    """Open a PostgreSQL connection.

    ``row_factory`` defaults to psycopg's tuple rows, which is what the
    old SQLite default produced. Call sites that used to set a
    ``Row``-style factory pass ``row_factory=dict_row``.

    A failure is re-raised as the same psycopg class with a scrubbed
    message, so no caller can print the password by printing the error.
    """
    kwargs = {"autocommit": autocommit}
    if row_factory is not None:
        kwargs["row_factory"] = row_factory
    resolved = resolve_dsn(dsn)
    try:
        return psycopg.connect(resolved, **kwargs)
    except psycopg.Error as exc:
        # ``from None``: the original holds the DSN in its own ``args``
        # and (for connection errors) a PGconn that can be asked for the
        # password, so chaining it would hand both back to the caller.
        message = scrub_credentials(str(exc), resolved)
        try:
            scrubbed = type(exc)(message)
        except Exception:  # noqa: BLE001 — exotic subclass, still must not leak
            scrubbed = psycopg.Error(message)
        raise scrubbed from None


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
