"""Tests for JSON log formatting + request-context filter."""
from __future__ import annotations

import json
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logging_config import JsonFormatter, RequestContextFilter  # noqa: E402


def _make_record(msg="hi", **extra):
    rec = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_json_formatter_emits_valid_json():
    rec = _make_record(msg="server up")
    line = JsonFormatter().format(rec)
    payload = json.loads(line)
    assert payload["msg"] == "server up"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "t"
    assert "ts" in payload


def test_json_formatter_passes_through_extras():
    rec = _make_record(msg="bet placed", request_id="abc123", stake=12.5)
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["request_id"] == "abc123"
    assert payload["stake"] == 12.5


def test_json_formatter_handles_nonserializable_extras():
    class Foo:
        def __repr__(self):
            return "<Foo>"

    rec = _make_record(msg="x", obj=Foo())
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["obj"] == "<Foo>"


def test_json_formatter_includes_exception_info():
    try:
        raise ValueError("boom")
    except ValueError:
        rec = logging.LogRecord(
            name="t", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="oops", args=(), exc_info=sys.exc_info(),
        )
    payload = json.loads(JsonFormatter().format(rec))
    assert "ValueError: boom" in payload["exc"]


def test_request_context_filter_no_op_outside_request():
    f = RequestContextFilter()
    rec = _make_record()
    assert f.filter(rec) is True
    assert not hasattr(rec, "request_id")


def test_request_context_filter_injects_request_id_inside_flask_ctx(monkeypatch, tmp_path):
    monkeypatch.setenv("BANKROLL_DB", str(tmp_path / "bankroll.db"))
    from src import app as _app
    f = RequestContextFilter()
    with _app.app.test_request_context("/"):
        from flask import g
        g.request_id = "xyz999"
        rec = _make_record()
        f.filter(rec)
        assert rec.request_id == "xyz999"


def test_request_context_filter_does_not_clobber_explicit_id(monkeypatch, tmp_path):
    monkeypatch.setenv("BANKROLL_DB", str(tmp_path / "bankroll.db"))
    from src import app as _app
    f = RequestContextFilter()
    with _app.app.test_request_context("/"):
        from flask import g
        g.request_id = "from-flask"
        rec = _make_record(request_id="explicit")
        f.filter(rec)
        assert rec.request_id == "explicit"
