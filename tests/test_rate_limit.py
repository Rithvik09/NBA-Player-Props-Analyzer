"""Tests for the token-bucket rate limiter + the /analyze_prop integration."""
from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rate_limit import TokenBucketLimiter, limiter as global_limiter  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    global_limiter.reset()
    yield
    global_limiter.reset()


def test_token_bucket_allows_up_to_capacity():
    lim = TokenBucketLimiter()
    for _ in range(5):
        ok, _ = lim.allow("b", "k", rate=5, per_seconds=10)
        assert ok
    ok, retry = lim.allow("b", "k", rate=5, per_seconds=10)
    assert not ok
    assert retry > 0


def test_token_bucket_refills_over_time():
    lim = TokenBucketLimiter()
    # Drain
    for _ in range(3):
        lim.allow("b", "k", rate=3, per_seconds=1)
    ok, _ = lim.allow("b", "k", rate=3, per_seconds=1)
    assert not ok
    # 0.4s ⇒ ~1.2 tokens regenerate (rate=3/sec)
    time.sleep(0.4)
    ok, _ = lim.allow("b", "k", rate=3, per_seconds=1)
    assert ok


def test_token_bucket_keys_are_independent():
    lim = TokenBucketLimiter()
    for _ in range(2):
        lim.allow("b", "alice", rate=2, per_seconds=10)
    ok_a, _ = lim.allow("b", "alice", rate=2, per_seconds=10)
    ok_b, _ = lim.allow("b", "bob", rate=2, per_seconds=10)
    assert not ok_a
    assert ok_b


def test_token_bucket_zero_rate_is_disabled():
    lim = TokenBucketLimiter()
    # rate=0 is the "off switch" — never blocks
    for _ in range(100):
        ok, _ = lim.allow("b", "k", rate=0, per_seconds=10)
        assert ok


def test_analyze_prop_returns_429_after_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("BANKROLL_DB", str(tmp_path / "bankroll.db"))
    # Re-import app fresh so our env-driven rate is picked up. We can't easily
    # change it after the fact (decorator captured at import), so instead we
    # exhaust the bucket using the configured default.
    from src import app as _app
    client = _app.app.test_client()

    # Drain whatever the configured rate is (env default = 30, but tests may
    # have set a lower value). We do this by hammering until we get a 429.
    saw_429 = False
    saw_non_429 = False
    for _ in range(80):
        r = client.post("/analyze_prop", json={})  # missing fields → 400 quickly
        if r.status_code == 429:
            saw_429 = True
            assert r.headers.get("Retry-After")
            assert r.get_json()["error"] == "rate limit exceeded"
            break
        else:
            saw_non_429 = True
    assert saw_429, "limiter did not engage within 80 calls"
    assert saw_non_429, "limiter blocked the very first call (suspect)"


def test_429_payload_includes_retry_seconds(monkeypatch, tmp_path):
    monkeypatch.setenv("BANKROLL_DB", str(tmp_path / "bankroll.db"))
    from src import app as _app
    client = _app.app.test_client()
    last_429 = None
    for _ in range(80):
        r = client.post("/analyze_prop", json={})
        if r.status_code == 429:
            last_429 = r
            break
    assert last_429 is not None
    body = last_429.get_json()
    assert "retry_after_seconds" in body
    assert body["bucket"] == "analyze_prop"
