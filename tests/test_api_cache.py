"""TTL cache unit tests."""
from __future__ import annotations

import os
import time

import pytest

# Re-enable caching just for this module (conftest sets DISABLE=1 globally).
os.environ["NBA_CACHE_DISABLE"] = "0"
# Force a fresh import so the module re-reads the env var.
import importlib  # noqa: E402
import src.api_cache  # noqa: E402
importlib.reload(src.api_cache)
from src.api_cache import ttl_cache, all_stats, clear_all  # noqa: E402


def test_second_call_hits_cache():
    calls = {"n": 0}

    @ttl_cache(ttl_seconds=60, name="t1")
    def f(x):
        calls["n"] += 1
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert calls["n"] == 1


def test_different_args_miss():
    calls = {"n": 0}

    @ttl_cache(ttl_seconds=60, name="t2")
    def f(x, y=0):
        calls["n"] += 1
        return x + y

    f(1)
    f(1, y=1)
    f(1)
    assert calls["n"] == 2


def test_ttl_expiry():
    calls = {"n": 0}

    @ttl_cache(ttl_seconds=0.05, name="t3")
    def f():
        calls["n"] += 1
        return 42

    f()
    f()
    time.sleep(0.07)
    f()
    assert calls["n"] == 2


def test_none_return_not_cached():
    calls = {"n": 0}

    @ttl_cache(ttl_seconds=60, name="t4")
    def f():
        calls["n"] += 1
        return None

    f()
    f()
    f()
    assert calls["n"] == 3


def test_clear_all_resets():
    @ttl_cache(ttl_seconds=60, name="t5")
    def f(x):
        return x

    f(1)
    f(2)
    stats_before = [s for s in all_stats() if s["name"] == "t5"][0]
    assert stats_before["size"] == 2
    clear_all()
    stats_after = [s for s in all_stats() if s["name"] == "t5"][0]
    assert stats_after["size"] == 0
