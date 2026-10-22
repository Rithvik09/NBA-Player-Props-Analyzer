"""The suite's schema reset must be unable to reach a real database.

``pytest_sessionstart`` runs ``DROP SCHEMA public CASCADE``. It used to
read ``DATABASE_URL`` — the *same* variable the application uses — and
accept any database whose name contained the substring "test", which
``nba_props_latest``, ``contest_db`` and ``attestation`` all do. It also
fired on ``pytest --collect-only``, which IDEs run on their own.

So the reset now needs a separate opt-in variable, a word-boundary name
match, and a local host, and it stands down during collection.
"""
from __future__ import annotations

import pytest

from tests import conftest as ct


# ---------------------------------------------------------------------------
# Which variable the reset reads
# ---------------------------------------------------------------------------

def test_reset_ignores_database_url(monkeypatch):
    """The application's own variable must not arm the drop."""
    monkeypatch.setenv("DATABASE_URL", "postgresql:///nba_props")
    monkeypatch.delenv(ct.TEST_DSN_ENV_VAR, raising=False)
    assert ct.reset_target_dsn() is None


def test_reset_reads_its_own_variable(monkeypatch):
    monkeypatch.setenv(ct.TEST_DSN_ENV_VAR, "postgresql:///nba_props_test")
    monkeypatch.setenv("DATABASE_URL", "postgresql:///nba_props")
    assert ct.reset_target_dsn() == "postgresql:///nba_props_test"


def test_nothing_is_dialled_without_the_test_variable(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql:///nba_props")
    monkeypatch.delenv(ct.TEST_DSN_ENV_VAR, raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError("the reset opened a connection")

    monkeypatch.setattr(ct.psycopg, "connect", _boom)
    ct.pytest_sessionstart(_FakeSession(collectonly=False))


class _FakeSession:
    def __init__(self, collectonly: bool):
        self.config = type(
            "_Config", (), {"option": type("_Opt", (), {"collectonly": collectonly})()}
        )()


def test_collect_only_does_not_reset(monkeypatch):
    """IDEs run --collect-only on open; that must not drop anything."""
    monkeypatch.setenv(ct.TEST_DSN_ENV_VAR, "postgresql:///nba_props_test")

    def _boom(*args, **kwargs):
        raise AssertionError("the reset opened a connection during collection")

    monkeypatch.setattr(ct.psycopg, "connect", _boom)
    ct.pytest_sessionstart(_FakeSession(collectonly=True))


# ---------------------------------------------------------------------------
# Which database names count as a test database
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "test",
    "nba_props_test",
    "test_nba_props",
    "nba_test_props",
])
def test_names_that_are_test_databases(name):
    assert ct.is_test_database_name(name) is True


@pytest.mark.parametrize("name", [
    "nba_props",
    "nba_props_latest",   # the substring check let this one through
    "contest_db",
    "attestation",
    "testing",
    "protest",
])
def test_names_that_are_not_test_databases(name):
    assert ct.is_test_database_name(name) is False


# ---------------------------------------------------------------------------
# Which hosts the reset may drop a schema on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dsn", [
    "postgresql:///nba_props_test",                       # unix socket
    "postgresql://localhost/nba_props_test",
    "postgresql://127.0.0.1:5432/nba_props_test",
    "postgresql://[::1]:5432/nba_props_test",
    "postgresql://postgres:postgres@postgres:5432/nba_props_test",  # CI service
    "host=/tmp dbname=nba_props_test",
    "host=localhost dbname=nba_props_test",
])
def test_local_reset_targets_are_allowed(dsn):
    ct.assert_local_reset_target(dsn)


@pytest.mark.parametrize("dsn", [
    "postgresql://u:p@ep-cool-sun-123.us-east-2.aws.neon.tech/nba_props_test",
    "postgresql://u:p@db.example.com/nba_props_test",
    "host=10.0.0.4 dbname=nba_props_test",
])
def test_remote_reset_targets_are_refused(dsn):
    with pytest.raises(RuntimeError, match="refusing to reset"):
        ct.assert_local_reset_target(dsn)


def test_refusal_does_not_echo_the_dsn():
    """This message lands in CI logs — it must not carry the password."""
    dsn = "postgresql://u:Sw0rdf1shTunaMackerel@db.example.com/nba_props_test"
    with pytest.raises(RuntimeError) as excinfo:
        ct.assert_local_reset_target(dsn)
    assert "Sw0rdf1shTunaMackerel" not in str(excinfo.value)
    assert "db.example.com" in str(excinfo.value)  # the host is the useful part


@pytest.mark.parametrize("dsn", [
    "postgresql://u:Sw0rdf1shTunaMackerel%ZZ@localhost/nba_props_test",
    "host=localhost Sw0rdf1shTunaMackerel",
])
def test_a_malformed_dsn_is_refused_without_echoing_it(dsn):
    """psycopg's own parse error quotes the offending token, which on a
    typo'd URL is the password."""
    with pytest.raises(RuntimeError, match="refusing to reset") as excinfo:
        ct.assert_local_reset_target(dsn)
    assert "Sw0rdf1shTunaMackerel" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# The application under test must follow TEST_DATABASE_URL
# ---------------------------------------------------------------------------

def test_test_dsn_is_propagated_to_database_url(monkeypatch):
    """Production code reads DATABASE_URL. Leaving the two pointing at
    different databases is exactly how a test run writes to a real one."""
    monkeypatch.setenv(ct.TEST_DSN_ENV_VAR, "postgresql:///nba_props_test")
    monkeypatch.setenv("DATABASE_URL", "postgresql:///nba_props")
    ct.propagate_test_dsn()
    import os
    assert os.environ["DATABASE_URL"] == "postgresql:///nba_props_test"


def test_propagation_leaves_database_url_alone_when_unset(monkeypatch):
    monkeypatch.delenv(ct.TEST_DSN_ENV_VAR, raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql:///nba_props_test")
    ct.propagate_test_dsn()
    import os
    assert os.environ["DATABASE_URL"] == "postgresql:///nba_props_test"
