"""Tests for src.metrics + the /metrics endpoint."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics.reset_for_tests()
    yield
    metrics.reset_for_tests()


def test_observe_request_increments_counter():
    metrics.observe_request("GET", "/foo", 200, 0.012)
    metrics.observe_request("GET", "/foo", 200, 0.05)
    metrics.observe_request("POST", "/foo", 500, 1.2)
    out = metrics.render()
    assert 'http_requests_total{method="GET",path="/foo",status="200"} 2' in out
    assert 'http_requests_total{method="POST",path="/foo",status="500"} 1' in out


def test_render_uses_cumulative_buckets():
    # Three observations: 0.001s, 0.05s, 0.5s. Buckets should be cumulative.
    for d in (0.001, 0.05, 0.5):
        metrics.observe_request("GET", "/x", 200, d)
    out = metrics.render()
    # le=0.005 contains only the 0.001s sample → 1
    assert 'http_request_duration_seconds_bucket{method="GET",path="/x",le="0.005"} 1' in out
    # le=0.05 contains 0.001 + 0.05 → 2
    assert 'http_request_duration_seconds_bucket{method="GET",path="/x",le="0.05"} 2' in out
    # le=0.5 contains all three → 3
    assert 'http_request_duration_seconds_bucket{method="GET",path="/x",le="0.5"} 3' in out
    # +Inf is total count
    assert 'http_request_duration_seconds_bucket{method="GET",path="/x",le="+Inf"} 3' in out
    assert 'http_request_duration_seconds_count{method="GET",path="/x"} 3' in out


def test_render_includes_app_info_and_scrape_timestamp():
    metrics.configure(version="9.9.9", build="abc1234")
    out = metrics.render()
    assert 'app_info{version="9.9.9",build="abc1234"} 1' in out
    assert "app_scrape_timestamp_seconds" in out
    # HELP/TYPE comments must appear
    assert "# TYPE http_requests_total counter" in out
    assert "# TYPE http_request_duration_seconds histogram" in out


def test_label_value_escapes_special_chars():
    # The spec requires escaping " and \ and \n in label values.
    metrics.observe_request("GET", '/with"quote', 200, 0.01)
    out = metrics.render()
    assert '/with\\"quote' in out


def test_observe_handles_garbage_inputs_without_raising():
    metrics.observe_request("GET", "/foo", "not-a-number", "still-bad")  # type: ignore[arg-type]
    out = metrics.render()
    assert 'http_requests_total{method="GET",path="/foo",status="0"} 1' in out


def test_metrics_endpoint_returns_text(monkeypatch, tmp_path):
    monkeypatch.setenv("BANKROLL_DB", str(tmp_path / "bankroll.db"))
    from src import app as _app
    client = _app.app.test_client()
    # Hit a couple of endpoints first so there's something to render
    client.get("/cache/stats")
    client.get("/cache/stats")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.content_type.startswith("text/plain")
    body = r.get_data(as_text=True)
    # Route-rule label, not literal path — confirms we're using url_rule.rule
    assert "/cache/stats" in body
    assert "http_requests_total" in body


def test_metrics_endpoint_excludes_self_from_histogram(monkeypatch, tmp_path):
    monkeypatch.setenv("BANKROLL_DB", str(tmp_path / "bankroll.db"))
    from src import app as _app
    client = _app.app.test_client()
    # Repeated /metrics scrapes should not pile counters onto themselves.
    client.get("/metrics")
    client.get("/metrics")
    r = client.get("/metrics")
    body = r.get_data(as_text=True)
    assert 'path="/metrics"' not in body
