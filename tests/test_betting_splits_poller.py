"""The betting-splits poller must start without an Odds API key.

Action Network's public-betting feed is free and keyless. Hanging it off
``ODDS_API_KEY`` — the way the prop poller and the close sweeper are,
because they spend a paid quota — would mean a deployment with no Odds
API account silently collects no splits either.

Like ``test_prop_poll_gate``, these assert on whether the thread is
*constructed* rather than on ``threading.enumerate()``: the threads are
daemons that outlive the test that starts them, so enumerating would see
leftovers from earlier tests. Patching the constructor also keeps the
suite off the network.
"""
from __future__ import annotations

import importlib
import logging
import sys

import pytest


@pytest.fixture
def import_app(monkeypatch):
    """Re-import ``src.app`` under a given env, recording started threads."""
    started: list[str] = []

    def _factory(**env):
        for key, value in env.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)

        real_thread = __import__("threading").Thread

        class _RecordingThread(real_thread):  # type: ignore[misc,valid-type]
            def start(self):  # noqa: D102
                started.append(self.name)
                # Deliberately not calling super().start(): these tests are
                # about whether a loop WOULD run, and running one would
                # reach for the network.

        monkeypatch.setattr("threading.Thread", _RecordingThread)
        sys.modules.pop("src.app", None)
        return importlib.import_module("src.app"), started

    yield _factory
    sys.modules.pop("src.app", None)


def test_splits_poller_starts_without_an_odds_api_key(import_app):
    """The whole point: this source needs no key, so no key gates it."""
    mod, started = import_app(ODDS_API_KEY=None, BETTING_SPLITS_POLL="1",
                              AUTO_JOBS="0")
    assert "betting-splits" in started
    assert "close-sweeper" not in started  # that one does need the key


def test_splits_poller_is_off_by_default(import_app):
    """Off unless asked for, like its sibling ODDS_PROP_POLL.

    It used to default on, which left ``tests/conftest.py`` forcing it to
    "0" as the only thing standing between the suite and the live Action
    Network endpoint — a guard any entry point that skips that conftest
    would walk straight past.
    """
    mod, started = import_app(ODDS_API_KEY=None, BETTING_SPLITS_POLL=None,
                              AUTO_JOBS="0")
    assert mod._splits_poll_enabled is False
    assert "betting-splits" not in started


def test_disabled_splits_poller_says_so(import_app, caplog):
    """An empty betting_splits_history must not be a silent mystery."""
    with caplog.at_level(logging.INFO):
        import_app(ODDS_API_KEY=None, BETTING_SPLITS_POLL="0", AUTO_JOBS="0")
    assert "BETTING_SPLITS_POLL" in caplog.text


def test_splits_poller_can_be_switched_off(import_app):
    mod, started = import_app(ODDS_API_KEY=None, BETTING_SPLITS_POLL="0",
                              AUTO_JOBS="0")
    assert mod._splits_poll_enabled is False
    assert "betting-splits" not in started


def test_splits_poller_does_not_re_enable_the_prop_poller(import_app):
    """ODDS_PROP_POLL stays off; nothing here may spend Odds API credits."""
    mod, started = import_app(ODDS_API_KEY="dummy-key", ODDS_PROP_POLL=None,
                              BETTING_SPLITS_POLL="1", AUTO_JOBS="0")
    assert mod._odds_prop_poll_enabled is False
    assert "odds-poller" not in started
    assert "betting-splits" in started


def test_splits_cadence_comes_from_the_env(import_app):
    mod, _ = import_app(ODDS_API_KEY=None, BETTING_SPLITS_POLL="1",
                        BETTING_SPLITS_POLL_INTERVAL="900", AUTO_JOBS="0")
    assert mod._splits_poll_interval == 900
