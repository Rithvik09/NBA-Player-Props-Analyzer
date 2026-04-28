"""Smoke tests for scripts/backup_db.sh.

Skipped when sqlite3/gzip aren't on the host (e.g. minimal CI sandboxes).
The behaviour we care about: a backup of a live DB is (a) created, (b) passes
integrity_check, and (c) round-trips back to a usable SQLite file.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "backup_db.sh"


def _have(*tools: str) -> bool:
    return all(shutil.which(t) for t in tools)


pytestmark = pytest.mark.skipif(
    not _have("sqlite3", "gzip", "bash"),
    reason="needs sqlite3/gzip/bash on PATH",
)


def _make_db(path: Path):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE bets(id INTEGER PRIMARY KEY, stake REAL)")
    conn.executemany("INSERT INTO bets(stake) VALUES (?)", [(1.0,), (2.0,), (3.0,)])
    conn.commit()
    conn.close()


def test_backup_creates_compressed_file(tmp_path):
    db = tmp_path / "live.db"
    _make_db(db)
    env = os.environ.copy()
    env["BACKUP_DIR"] = str(tmp_path / "out")
    r = subprocess.run(
        ["bash", str(SCRIPT), str(db)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    out_dir = tmp_path / "out"
    files = list(out_dir.glob("live_*.db.gz"))
    assert len(files) == 1
    assert files[0].stat().st_size > 0


def test_backup_round_trips_data(tmp_path):
    db = tmp_path / "live.db"
    _make_db(db)
    env = os.environ.copy()
    env["BACKUP_DIR"] = str(tmp_path / "out")
    subprocess.run(
        ["bash", str(SCRIPT), str(db)],
        env=env, check=True, timeout=30, capture_output=True,
    )
    backup_gz = next((tmp_path / "out").glob("live_*.db.gz"))
    # Decompress and read back — values should match exactly
    subprocess.run(["gunzip", "-k", str(backup_gz)], check=True)
    restored = backup_gz.with_suffix("")  # strip .gz
    rows = sqlite3.connect(str(restored)).execute(
        "SELECT stake FROM bets ORDER BY id"
    ).fetchall()
    assert rows == [(1.0,), (2.0,), (3.0,)]


def test_backup_missing_file_errors(tmp_path):
    env = os.environ.copy()
    env["BACKUP_DIR"] = str(tmp_path)
    r = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path / "does-not-exist.db")],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert r.returncode != 0
    assert "not found" in r.stderr


def test_backup_keep_flag_validates_integer(tmp_path):
    db = tmp_path / "live.db"
    _make_db(db)
    env = os.environ.copy()
    env["BACKUP_DIR"] = str(tmp_path / "out")
    r = subprocess.run(
        ["bash", str(SCRIPT), str(db), "--keep", "not-a-number"],
        env=env, capture_output=True, text=True, timeout=10,
    )
    assert r.returncode != 0
    assert "non-negative integer" in r.stderr
