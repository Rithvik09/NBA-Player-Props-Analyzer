"""The database password must never leave the process.

libpq quotes the offending token back at you: a DSN whose password
contains a ``%`` fails with ``invalid percent-encoded token: "<password>"``,
and an unreachable host names the host and the user. Those strings used to
travel straight out of ``db.connect()`` into ``jsonify({'error': str(e)})``
on routes that need no authentication at all.

So the scrub happens at the source — in :func:`src.db.connect` — and the
health endpoints that an anonymous caller can reach return a fixed string
with the detail going to the log instead.

Every test here uses a sentinel password and asserts the sentinel appears
in *nothing* the caller can see: the exception, its repr, its args, the
formatted traceback, the JSON body, and the log records.
"""
from __future__ import annotations

import logging
import os
import sys
import traceback

import psycopg
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import db  # noqa: E402

SENTINEL = "Sw0rdf1shTunaMackerel"

# Three ways the same password reaches libpq, and the three failure modes
# that quote it back. The ``%`` one is a *parse* error raised before any
# socket is opened; the other two are connection failures against a port
# nothing listens on.
DSN_SHAPES = {
    "url": f"postgresql://leakuser:{SENTINEL}@127.0.0.1:1/nba_props",
    "url_percent": f"postgresql://leakuser:{SENTINEL}%ZZ@127.0.0.1:1/nba_props",
    "keyword_value": (
        f"host=127.0.0.1 port=1 user=leakuser password={SENTINEL} "
        f"dbname=nba_props"
    ),
}


def _exception_surface(exc: BaseException) -> str:
    """Everything a caller could print about ``exc``, concatenated."""
    return "\n".join([
        str(exc),
        repr(exc),
        repr(exc.args),
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    ])


@pytest.mark.parametrize("shape", sorted(DSN_SHAPES))
def test_connect_failure_never_quotes_the_password(shape):
    with pytest.raises(psycopg.Error) as excinfo:
        db.connect(DSN_SHAPES[shape])
    assert SENTINEL not in _exception_surface(excinfo.value)


@pytest.mark.parametrize("shape", sorted(DSN_SHAPES))
def test_connect_failure_drops_the_whole_dsn(shape):
    """Not just the password — the DSN itself must not be echoed."""
    dsn = DSN_SHAPES[shape]
    with pytest.raises(psycopg.Error) as excinfo:
        db.connect(dsn)
    surface = _exception_surface(excinfo.value)
    assert dsn not in surface
    assert "leakuser" not in surface


@pytest.mark.parametrize("shape", sorted(DSN_SHAPES))
def test_connect_failure_has_no_chained_original(shape):
    """``raise ... from None``: the original exception carried the DSN in
    its own args, and a chained traceback would print all of it."""
    with pytest.raises(psycopg.Error) as excinfo:
        db.connect(DSN_SHAPES[shape])
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


def test_connect_keeps_the_exception_class():
    """Call sites branch on ``psycopg.OperationalError`` — keep that usable."""
    with pytest.raises(psycopg.OperationalError):
        db.connect(DSN_SHAPES["url"])


def test_connect_keeps_the_diagnostic_signal():
    """Scrubbing must not reduce the error to a shrug.

    Host and port are not secrets and are the first thing an operator
    needs, so they stay.
    """
    with pytest.raises(psycopg.Error) as excinfo:
        db.connect(DSN_SHAPES["url"])
    assert "127.0.0.1" in str(excinfo.value)


def test_connect_does_not_log_the_password(caplog):
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(psycopg.Error):
            db.connect(DSN_SHAPES["url"])
    assert SENTINEL not in caplog.text


def test_successful_connect_is_untouched(pg_dsn):
    """The wrapper must not change the happy path."""
    with db.connect(pg_dsn) as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# scrub_credentials — the shared helper the health endpoints also use
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    f"invalid percent-encoded token: \"{SENTINEL}%ZZ\"",
    f"connection failed for postgresql://u:{SENTINEL}@h:5432/db",
    f"could not connect: host=h user=u password={SENTINEL} dbname=d",
    f"could not connect: password = '{SENTINEL}' dbname=d",
    f"missing \"=\" after \"{SENTINEL}\" in connection info string",
])
def test_scrub_credentials_removes_every_shape(text, monkeypatch):
    monkeypatch.setenv(db.DSN_ENV_VAR, DSN_SHAPES["url"])
    assert SENTINEL not in db.scrub_credentials(text)


def test_scrub_credentials_takes_an_explicit_dsn(monkeypatch):
    monkeypatch.delenv(db.DSN_ENV_VAR, raising=False)
    text = f"nope: {SENTINEL}"
    assert SENTINEL in db.scrub_credentials(text)  # unknown token, nothing to key on
    assert SENTINEL not in db.scrub_credentials(text, DSN_SHAPES["url"])


def test_scrub_credentials_survives_an_unset_env(monkeypatch):
    monkeypatch.delenv(db.DSN_ENV_VAR, raising=False)
    assert db.scrub_credentials("plain message") == "plain message"


def test_scrub_credentials_leaves_innocent_text_alone(monkeypatch):
    monkeypatch.setenv(db.DSN_ENV_VAR, DSN_SHAPES["url"])
    assert db.scrub_credentials("relation \"bets\" does not exist") == (
        "relation \"bets\" does not exist"
    )


# ---------------------------------------------------------------------------
# resolve_dsn must not echo the candidate either (finding 2)
# ---------------------------------------------------------------------------

def test_resolve_dsn_file_path_guard_does_not_echo_the_candidate():
    """``postgres:/user:pw@host/x.db`` — one slash short of a URL, so the
    ``://`` escape hatch misses it and the guard fires on a string that
    carries a password."""
    near_miss = f"postgres:/leakuser:{SENTINEL}@host/nba_props.db"
    with pytest.raises(ValueError) as excinfo:
        db.resolve_dsn(near_miss)
    surface = _exception_surface(excinfo.value)
    assert SENTINEL not in surface
    assert near_miss not in surface
    # Still has to say what is wrong.
    assert "SQLite file path" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The HTTP boundary: /healthz, /healthz/clv and /odds/status are all
# unauthenticated, and all three used to hand back str(e) verbatim.
# ---------------------------------------------------------------------------

@pytest.fixture
def poisoned_connect(monkeypatch):
    """Make every ``db.connect()`` dial a DSN carrying the sentinel.

    Deliberately runs the *real* connect so the endpoints see the error
    libpq actually produces rather than a hand-written stand-in.
    """
    real_connect = db.connect

    def _poisoned(dsn=None, **kwargs):
        return real_connect(DSN_SHAPES["url"], **kwargs)

    monkeypatch.setattr(db, "connect", _poisoned)
    return _poisoned


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("BANKROLL_DB", str(tmp_path / "bankroll.db"))
    from src import app as _app
    return _app.app.test_client()


def test_healthz_does_not_return_the_password(client, poisoned_connect, caplog):
    with caplog.at_level(logging.DEBUG):
        r = client.get("/healthz")
    assert r.status_code == 503
    body = r.get_data(as_text=True)
    assert SENTINEL not in body
    assert SENTINEL not in caplog.text


def test_healthz_returns_a_fixed_string_for_the_db_check(client, poisoned_connect):
    r = client.get("/healthz")
    check = r.get_json()["checks"]["bankroll_db"]
    assert check["ok"] is False
    assert check["error"] == "db unavailable"


def test_healthz_logs_the_detail_server_side(client, poisoned_connect, caplog):
    """The operator still needs to know *why* — just not the caller."""
    with caplog.at_level(logging.DEBUG):
        client.get("/healthz")
    assert "127.0.0.1" in caplog.text


def test_healthz_clv_does_not_return_the_password(client, poisoned_connect, caplog):
    with caplog.at_level(logging.DEBUG):
        r = client.get("/healthz/clv")
    assert r.status_code == 500
    assert SENTINEL not in r.get_data(as_text=True)
    assert SENTINEL not in caplog.text


def test_healthz_clv_returns_a_fixed_string(client, poisoned_connect):
    r = client.get("/healthz/clv")
    assert r.get_json() == {"error": "db unavailable"}


def test_healthz_clv_logs_the_detail_server_side(client, poisoned_connect, caplog):
    with caplog.at_level(logging.DEBUG):
        client.get("/healthz/clv")
    assert "127.0.0.1" in caplog.text


@pytest.fixture
def last_poll_reset():
    """``_LAST_POLL`` is a module global — put it back for the next test."""
    from src import odds_tracker as ot
    before = dict(ot._LAST_POLL)
    yield ot
    ot._LAST_POLL.clear()
    ot._LAST_POLL.update(before)


def test_last_poll_error_is_scrubbed(last_poll_reset, monkeypatch):
    ot = last_poll_reset

    class _Boom:
        def __init__(self, *args, **kwargs):
            pass

        def snapshot_all_games(self):
            raise RuntimeError(
                f"connection failed for postgresql://u:{SENTINEL}@h:5432/db"
            )

    monkeypatch.setattr(ot, "OddsTracker", _Boom)
    assert ot.poll_odds("dummy-key", "postgresql:///nba_props_test") == 0
    assert SENTINEL not in (ot.last_poll_status()["error"] or "")


def test_odds_status_does_not_return_the_password(client, last_poll_reset,
                                                  monkeypatch):
    ot = last_poll_reset

    class _Boom:
        def __init__(self, *args, **kwargs):
            pass

        def snapshot_all_games(self):
            raise RuntimeError(
                f"could not connect: host=h user=u password={SENTINEL} dbname=d"
            )

    monkeypatch.setattr(ot, "OddsTracker", _Boom)
    ot.poll_odds("dummy-key", "postgresql:///nba_props_test")
    r = client.get("/odds/status")
    assert r.status_code == 200
    assert SENTINEL not in r.get_data(as_text=True)
