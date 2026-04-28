"""In-process token-bucket rate limiter.

Why not Flask-Limiter?
----------------------
Flask-Limiter pulls in ``limits`` and (for distributed setups) Redis. We're
running a single gunicorn process at the moment and the only endpoint that
needs throttling is ``/analyze_prop`` — which is expensive because it loads
five models and hits the NBA API. A 30-line bucket implementation suffices.

If we ever scale to multiple workers and need consistent global limits,
swap in Flask-Limiter with a Redis storage URI; the decorator interface
here is intentionally close enough that the call sites won't change.

Algorithm
---------
Token bucket per (key, route): each call to ``allow()`` refills based on
elapsed time, then consumes one token if any are available. ``per_seconds``
is the refill window — ``rate`` tokens become available over that window.

Concurrency: one mutex around the whole map; per-key rates are low enough
(< 100/s aggregate) that a single lock is fine. If profiling ever shows it
contended, partition by ``hash(key) % N`` into N locks.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Callable

from flask import jsonify, request


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    """One instance can host many independent (route, key) buckets."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, str], _Bucket] = {}

    def allow(
        self,
        bucket_id: str,
        key: str,
        *,
        rate: int,
        per_seconds: float,
    ) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_seconds)``.

        ``retry_after_seconds`` is 0 when allowed; when denied it's how long
        until at least one token is available again.
        """
        if rate <= 0 or per_seconds <= 0:
            return True, 0.0  # disabled
        now = time.time()
        full_capacity = float(rate)
        refill_rate = rate / float(per_seconds)  # tokens per second
        bkey = (bucket_id, key)
        with self._lock:
            b = self._buckets.get(bkey)
            if b is None:
                b = _Bucket(tokens=full_capacity, last_refill=now)
                self._buckets[bkey] = b
            elapsed = max(0.0, now - b.last_refill)
            b.tokens = min(full_capacity, b.tokens + elapsed * refill_rate)
            b.last_refill = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True, 0.0
            # Time until one token regenerates
            need = 1.0 - b.tokens
            retry = need / refill_rate
            return False, retry

    def reset(self) -> None:
        """Test helper — drop all buckets."""
        with self._lock:
            self._buckets.clear()


# Module-level singleton — endpoints share state, tests can reset it
limiter = TokenBucketLimiter()


def _client_key() -> str:
    """Identify the caller. Prefer explicit X-API-Key header, fall back to
    proxy-aware IP. (request.remote_addr is the LB; X-Forwarded-For has the
    real client when set by a trusted proxy.)"""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        # Don't leak the key into the bucket id; just use a stable hash
        return f"key:{abs(hash(api_key)) % (10 ** 12)}"
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return f"ip:{xff.split(',')[0].strip()}"
    return f"ip:{request.remote_addr or 'unknown'}"


def rate_limited(
    bucket_id: str,
    *,
    rate: int,
    per_seconds: float,
    key_fn: Callable[[], str] | None = None,
) -> Callable:
    """Flask view decorator. Returns 429 with ``Retry-After`` if the caller
    exceeds the bucket.

    Configuration is per-decorator at import time (rate, per_seconds), so the
    limit is part of the route definition — easy to grep, easy to review.
    """

    def deco(view: Callable) -> Callable:
        @wraps(view)
        def wrapped(*args, **kwargs):
            key = (key_fn or _client_key)()
            allowed, retry = limiter.allow(
                bucket_id, key, rate=rate, per_seconds=per_seconds,
            )
            if not allowed:
                resp = jsonify({
                    "error": "rate limit exceeded",
                    "bucket": bucket_id,
                    "retry_after_seconds": round(retry, 2),
                })
                resp.status_code = 429
                # Standard header — ceil to whole seconds for HTTP/1.1 spec
                resp.headers["Retry-After"] = str(max(1, int(retry + 0.999)))
                return resp
            return view(*args, **kwargs)
        return wrapped
    return deco
