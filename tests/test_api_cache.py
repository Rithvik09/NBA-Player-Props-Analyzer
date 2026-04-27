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


def test_stampede_protection_serialises_misses():
    """Concurrent threads asking for same uncached key fire fn once."""
    import threading
    calls = {"n": 0}
    counter_lock = threading.Lock()

    @ttl_cache(ttl_seconds=60, name="t_stampede")
    def slow(x):
        # Hold long enough for siblings to queue up on the per-key lock
        with counter_lock:
            calls["n"] += 1
        time.sleep(0.15)
        return x * 10

    results = [None] * 5

    def worker(i):
        results[i] = slow(7)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
        # Tiny stagger so they all start before the first finishes
        time.sleep(0.005)
    for t in threads:
        t.join(timeout=5.0)

    assert all(r == 70 for r in results)
    # Only ONE thread should have actually executed slow's body.
    assert calls["n"] == 1
    stats = [s for s in all_stats() if s["name"] == "t_stampede"][0]
    # The other 4 should have been blocked by the per-key lock.
    assert stats["stampede_blocks"] == 4


def test_disk_cache_persists_across_instances(tmp_path, monkeypatch):
    """A disk-backed cache should survive a fresh import."""
    monkeypatch.setenv("NBA_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("NBA_CACHE_DISABLE", "0")
    import importlib
    import src.api_cache as ac
    importlib.reload(ac)

    calls = {"n": 0}

    @ac.ttl_cache(ttl_seconds=3600, name="t_disk", disk=True)
    def f(x):
        calls["n"] += 1
        return x + 100

    assert f(5) == 105
    assert f(5) == 105
    assert calls["n"] == 1

    # Re-import module to simulate process restart
    importlib.reload(ac)
    calls2 = {"n": 0}

    @ac.ttl_cache(ttl_seconds=3600, name="t_disk", disk=True)
    def f2(x):
        calls2["n"] += 1
        return x + 100

    assert f2(5) == 105  # served from disk snapshot, no fn call
    assert calls2["n"] == 0
