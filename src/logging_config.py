"""Centralised logging setup.

Call ``configure_logging()`` once at process start. Every module should then
use ``logging.getLogger(__name__)`` — no direct ``print`` calls for anything
worth seeing later.

Env vars
--------
``LOG_LEVEL``     : DEBUG / INFO / WARNING / ERROR (default INFO)
``LOG_DIR``       : directory for rotating file handler (default ./logs)
``LOG_JSON``      : set to ``1`` to emit structured JSON logs (for shipping)
``LOG_MAX_BYTES`` : rotation threshold (default 10 MiB)
``LOG_BACKUPS``   : rotation history (default 7)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """One JSON object per line — easy to grep, easy to ship to ELK/loki."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Pass through any ad-hoc keys the caller attached via ``extra=``
        for k, v in record.__dict__.items():
            if k in ("args", "asctime", "created", "exc_info", "exc_text",
                     "filename", "funcName", "levelname", "levelno", "lineno",
                     "message", "module", "msecs", "msg", "name", "pathname",
                     "process", "processName", "relativeCreated", "stack_info",
                     "thread", "threadName", "taskName"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except TypeError:
                payload[k] = repr(v)
        return json.dumps(payload, separators=(",", ":"))


class RequestContextFilter(logging.Filter):
    """Logging filter that auto-injects ``request_id`` from ``flask.g``.

    Without this, every log line that wants the request id has to remember
    to pass ``extra={"request_id": g.request_id}`` — they don't, so log
    lines generated inside library code (data_collector, kelly, …) lose
    the trace breadcrumb. This filter reads ``flask.g.request_id`` if a
    request context is active and stitches it onto the record. No-op
    outside a request (e.g. background threads, scripts).
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            from flask import g, has_request_context
        except ImportError:
            return True
        if has_request_context():
            rid = getattr(g, "request_id", None)
            if rid and not hasattr(record, "request_id"):
                record.request_id = rid
        return True


_CONFIGURED = False


def configure_logging(app_logger: logging.Logger | None = None) -> logging.Logger:
    """Idempotent root-logger setup. Returns the root logger for convenience."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return root

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)

    # Clear any default handlers (Flask's WSGI adds one at import time)
    for h in list(root.handlers):
        root.removeHandler(h)

    use_json = os.environ.get("LOG_JSON", "").strip() in ("1", "true", "yes")
    fmt_text = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
    formatter: logging.Formatter = JsonFormatter() if use_json else logging.Formatter(fmt_text)

    # Auto-inject ``request_id`` from flask.g when in request context
    request_filter = RequestContextFilter()

    # Console — stderr, respects LOG_LEVEL
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    stream.setLevel(level)
    stream.addFilter(request_filter)
    root.addHandler(stream)

    # Rotating file handler — always text even if console is JSON, for grepability
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=int(os.environ.get("LOG_MAX_BYTES", 10 * 1024 * 1024)),
            backupCount=int(os.environ.get("LOG_BACKUPS", 7)),
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        file_handler.addFilter(request_filter)
        root.addHandler(file_handler)
    except Exception as e:  # noqa: BLE001
        root.warning(f"file handler setup failed: {e}")

    # Tame very-chatty third-party loggers
    for noisy in ("urllib3", "requests", "werkzeug", "nba_api"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if app_logger is not None:
        app_logger.handlers = []  # Flask adds its own; reuse root
        app_logger.propagate = True
        app_logger.setLevel(level)

    _CONFIGURED = True
    root.info(
        "logging configured",
        extra={"level": level_name, "json": use_json, "log_dir": str(log_dir)},
    )
    return root
