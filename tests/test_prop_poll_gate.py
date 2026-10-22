"""The prop odds poller must stay off unless explicitly enabled.

One ``snapshot_all_games`` cycle costs 2 credits for the games call plus 12
per event (12 prop markets x 1 region). Measured live on 2026-10-21 the board
was 43 events, so 2 + 43*12 = 518 credits against a 500/month free tier: the
first cycle exhausts the quota and dies partway through the slate. That is
also why the 11 market features in ml_features.py have always been zeros.

These tests pin the gate so it cannot be re-enabled by accident.

They assert on whether the poller thread is *constructed*, not on
``threading.enumerate()``. Enumerating is unreliable here: the threads are
daemons that outlive the test that started them, so a later test would see a
leftover poller from an earlier one. Patching the constructor also keeps the
suite from starting real pollers that would reach for the network.
"""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def import_app(monkeypatch):
    """Re-import ``src.app`` under a given env, recording started threads.

    The poller starts at module scope, so the flag is only read at import
    time; the module has to be dropped and re-imported to observe a
    different setting.
    """
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
                # about whether a poller WOULD run, and actually running one
                # would hit the network on a 5-second delay.

        monkeypatch.setattr("threading.Thread", _RecordingThread)
        sys.modules.pop("src.app", None)
        return importlib.import_module("src.app"), started

    yield _factory
    sys.modules.pop("src.app", None)


def test_poller_does_not_start_without_the_flag(import_app):
    """A key alone must not be enough — that is the 518-credit mistake."""
    mod, started = import_app(ODDS_API_KEY="dummy-key", ODDS_PROP_POLL=None, AUTO_JOBS="0")
    assert mod._odds_prop_poll_enabled is False
    assert "odds-poller" not in started


def test_poller_stays_off_when_flag_is_falsey(import_app):
    mod, started = import_app(ODDS_API_KEY="dummy-key", ODDS_PROP_POLL="0", AUTO_JOBS="0")
    assert mod._odds_prop_poll_enabled is False
    assert "odds-poller" not in started


def test_explicit_opt_in_does_start_the_poller(import_app):
    """The escape hatch has to actually work on a paid plan."""
    mod, started = import_app(ODDS_API_KEY="dummy-key", ODDS_PROP_POLL="1", AUTO_JOBS="0")
    assert mod._odds_prop_poll_enabled is True
    assert "odds-poller" in started


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " 1 "])
def test_flag_accepts_the_usual_truthy_spellings(import_app, value):
    mod, _ = import_app(ODDS_API_KEY="dummy-key", ODDS_PROP_POLL=value, AUTO_JOBS="0")
    assert mod._odds_prop_poll_enabled is True


@pytest.mark.parametrize("value", ["0", "", "no", "off", "False"])
def test_flag_rejects_everything_else(import_app, value):
    mod, started = import_app(ODDS_API_KEY="dummy-key", ODDS_PROP_POLL=value, AUTO_JOBS="0")
    assert mod._odds_prop_poll_enabled is False
    assert "odds-poller" not in started


def test_no_key_means_no_poller_even_when_enabled(import_app):
    """The key check gates the thread independently of the new flag."""
    mod, started = import_app(ODDS_API_KEY=None, ODDS_PROP_POLL="1", AUTO_JOBS="0")
    assert mod._odds_prop_poll_enabled is True
    assert "odds-poller" not in started


def test_game_line_capture_is_not_affected_by_the_gate(import_app):
    """Capture rides the close sweeper, which must still start."""
    mod, started = import_app(ODDS_API_KEY="dummy-key", ODDS_PROP_POLL="0", AUTO_JOBS="0")
    assert mod._odds_prop_poll_enabled is False
    assert "close-sweeper" in started
