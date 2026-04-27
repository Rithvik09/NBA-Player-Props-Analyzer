"""Smoke tests for walk-forward backtest harness."""
from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_walkforward import _hit, run  # noqa: E402


def test_hit_over_wins():
    assert _hit(actual=30, predicted=28, line=25.5) == 1


def test_hit_over_loses():
    assert _hit(actual=20, predicted=28, line=25.5) == 0


def test_hit_under_wins():
    assert _hit(actual=20, predicted=22, line=25.5) == 1


def test_hit_exact_line_pushes_to_none():
    assert _hit(actual=25.5, predicted=28, line=25.5) is None


def test_hit_explicit_side_overrides_inference():
    # Predicted < line, but caller forces 'over' → over wins because actual > line
    assert _hit(actual=30, predicted=20, line=25.5, side_hint="over") == 1


def test_run_buckets_predictions_into_windows(tmp_path):
    db = str(tmp_path / "test.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            prop_type TEXT,
            line REAL,
            predicted_value REAL,
            confidence REAL,
            actual_result REAL
        )
    """)
    # Window 1: 3 picks, 2 win, 1 lose
    rows = [
        ("2025-01-01T00:00:00", "points", 25.5, 28, 0.65, 30),  # over win
        ("2025-01-02T00:00:00", "points", 25.5, 28, 0.65, 20),  # over loss
        ("2025-01-03T00:00:00", "points", 25.5, 28, 0.65, 30),  # over win
        # Window 2: 1 pick, win
        ("2025-01-20T00:00:00", "points", 25.5, 28, 0.70, 31),
    ]
    conn.executemany(
        "INSERT INTO prediction_logs (timestamp, prop_type, line, predicted_value, "
        "confidence, actual_result) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()

    result = run(db, window_days=14, conf_threshold=0.60, out_csv=None)
    assert len(result) == 2
    assert result[0]["n"] == 3
    assert abs(result[0]["hit_rate"] - 2 / 3) < 1e-9
    assert result[1]["n"] == 1
    assert result[1]["hit_rate"] == 1.0


def test_run_filters_below_threshold(tmp_path):
    db = str(tmp_path / "test.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, prop_type TEXT, line REAL,
            predicted_value REAL, confidence REAL, actual_result REAL
        )
    """)
    conn.executemany(
        "INSERT INTO prediction_logs (timestamp, prop_type, line, predicted_value, "
        "confidence, actual_result) VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("2025-01-01T00:00:00", "points", 25.5, 28, 0.55, 30),  # below 0.6 → skip
            ("2025-01-02T00:00:00", "points", 25.5, 28, 0.70, 30),  # included
        ],
    )
    conn.commit()
    conn.close()

    result = run(db, window_days=30, conf_threshold=0.60, out_csv=None)
    assert len(result) == 1
    assert result[0]["n"] == 1
