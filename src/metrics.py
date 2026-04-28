"""Prometheus-format metrics — pure-python so we don't drag in
``prometheus_client`` for what is, in practice, three counters and a histogram.

Exposition format reference:
  https://prometheus.io/docs/instrumenting/exposition_formats/

What we track:
  - ``http_requests_total{method,path,status}`` — request counter
  - ``http_request_duration_seconds_bucket{...}`` — histogram of durations
  - ``app_info{version,build}`` — single 1-valued gauge for service info

The path label is the *route rule* (``/player_game_info/<int:player_id>``)
not the literal URL — otherwise label cardinality explodes the moment a real
user hits a unique player. ``flask.request.url_rule.rule`` does this for free.

All state lives in module-level dicts protected by a lock. Counters are
monotonic; on process restart they reset (Prometheus handles that case via
``rate()`` which is robust to counter resets).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Iterable

# Histogram buckets in seconds — chosen to cover "fast cache hit" through
# "the model warm-up call took its sweet time". Last bucket is +Inf via the
# ``_count`` field (every observation lands in at least one bucket because we
# always include +Inf implicitly).
_DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

_lock = threading.Lock()
_request_counter: dict[tuple[str, str, int], int] = defaultdict(int)
_duration_buckets: dict[tuple[str, str], list[int]] = defaultdict(
    lambda: [0] * len(_DEFAULT_BUCKETS)
)
_duration_sum: dict[tuple[str, str], float] = defaultdict(float)
_duration_count: dict[tuple[str, str], int] = defaultdict(int)

# Process-wide constants shown in app_info. Imported lazily so importing this
# module never crashes if the version file isn't there yet.
_app_version = "unknown"
_app_build = "unknown"


def configure(version: str | None = None, build: str | None = None) -> None:
    """Set the strings shown in the ``app_info`` gauge. Idempotent."""
    global _app_version, _app_build
    if version:
        _app_version = str(version)
    if build:
        _app_build = str(build)


def observe_request(method: str, path: str, status: int, duration_seconds: float) -> None:
    """Record one HTTP request. Called from ``after_request``.

    ``path`` should be the *route rule* (``url_rule.rule``) not the literal
    URL — see the module docstring on cardinality.
    """
    method = (method or "?").upper()
    path = path or "?"
    try:
        status_i = int(status)
    except (TypeError, ValueError):
        status_i = 0
    try:
        d = float(duration_seconds)
    except (TypeError, ValueError):
        d = 0.0
    with _lock:
        _request_counter[(method, path, status_i)] += 1
        bkey = (method, path)
        buckets = _duration_buckets[bkey]
        # Store non-cumulative — only the smallest fitting bucket. ``render``
        # turns this into the cumulative form Prometheus expects. (Storing
        # cumulative directly + cumulating again in render = double counting.)
        for i, edge in enumerate(_DEFAULT_BUCKETS):
            if d <= edge:
                buckets[i] += 1
                break
        _duration_sum[bkey] += d
        _duration_count[bkey] += 1


def _escape_label_value(v: str) -> str:
    # Per spec: backslash, double quote, and newline must be escaped.
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _label_string(pairs: Iterable[tuple[str, str]]) -> str:
    inner = ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in pairs)
    return "{" + inner + "}" if inner else ""


def render() -> str:
    """Render current metric state as Prometheus text exposition (UTF-8 string).

    Content-Type for the response: ``text/plain; version=0.0.4; charset=utf-8``.
    """
    lines: list[str] = []
    with _lock:
        # ── http_requests_total ────────────────────────────────────────────
        lines.append("# HELP http_requests_total Total HTTP requests processed.")
        lines.append("# TYPE http_requests_total counter")
        for (method, path, status), n in sorted(_request_counter.items()):
            labels = _label_string([
                ("method", method), ("path", path), ("status", str(status)),
            ])
            lines.append(f"http_requests_total{labels} {n}")

        # ── http_request_duration_seconds histogram ────────────────────────
        lines.append("# HELP http_request_duration_seconds HTTP request latency.")
        lines.append("# TYPE http_request_duration_seconds histogram")
        for (method, path), buckets in sorted(_duration_buckets.items()):
            cumulative = 0
            for i, edge in enumerate(_DEFAULT_BUCKETS):
                cumulative += buckets[i]
                # NB: histogram bucket counts are CUMULATIVE — bucket[le=0.05]
                # contains everything ≤ 0.05, including ≤ 0.025. We accumulate
                # incrementally rather than re-summing for each bucket.
                labels = _label_string([
                    ("method", method), ("path", path), ("le", f"{edge}"),
                ])
                lines.append(f"http_request_duration_seconds_bucket{labels} {cumulative}")
            # +Inf bucket = total count (every observation is ≤ +Inf)
            total = _duration_count[(method, path)]
            labels_inf = _label_string([
                ("method", method), ("path", path), ("le", "+Inf"),
            ])
            lines.append(f"http_request_duration_seconds_bucket{labels_inf} {total}")
            sum_labels = _label_string([("method", method), ("path", path)])
            lines.append(
                f"http_request_duration_seconds_sum{sum_labels} "
                f"{_duration_sum[(method, path)]:.6f}"
            )
            lines.append(f"http_request_duration_seconds_count{sum_labels} {total}")

        # ── app_info ───────────────────────────────────────────────────────
        lines.append("# HELP app_info Static information about the service.")
        lines.append("# TYPE app_info gauge")
        info_labels = _label_string([
            ("version", _app_version), ("build", _app_build),
        ])
        lines.append(f"app_info{info_labels} 1")

        # ── process timestamp (cheap freshness check on the scrape) ────────
        lines.append("# HELP app_scrape_timestamp_seconds Unix time of this scrape.")
        lines.append("# TYPE app_scrape_timestamp_seconds gauge")
        lines.append(f"app_scrape_timestamp_seconds {time.time():.3f}")

    return "\n".join(lines) + "\n"


def reset_for_tests() -> None:
    """Clear all counters. Test-only; do not call in production."""
    with _lock:
        _request_counter.clear()
        _duration_buckets.clear()
        _duration_sum.clear()
        _duration_count.clear()
