"""No script may put the DSN somewhere another user can read it.

``ps`` shows every process's full argv to every account on the box, and a
training run holds that command line for hours. ``scripts/run_daily_jobs.sh``
passed ``--db "$DATABASE_URL"`` to five commands, which was pure redundancy
on top of the ``export DATABASE_URL`` the script already does — every
``--db`` defaults to ``None``, meaning "read DATABASE_URL".

Grep-shaped tests, because the thing being pinned is the text of a command
line: nothing else observes it before `ps` does.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

SHELL_SCRIPTS = sorted(SCRIPTS.glob("*.sh"))

# ``--db $DATABASE_URL`` / ``--dbname="$DATABASE_URL"`` and friends.
_DSN_IN_ARGV_RE = re.compile(
    r"--(?:db|dbname|database[-_]url)[= ]+[\"']?\$\{?(?:DATABASE_URL|DB_DSN)"
)


def test_there_are_shell_scripts_to_check():
    """Guard against the glob silently matching nothing."""
    assert SHELL_SCRIPTS


@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_no_shell_script_passes_the_dsn_in_argv(script):
    hits = [
        f"{script.name}:{n}: {line.strip()}"
        for n, line in enumerate(script.read_text().splitlines(), 1)
        if _DSN_IN_ARGV_RE.search(line)
    ]
    assert not hits, "DSN on a command line (visible in ps):\n" + "\n".join(hits)


def test_daily_jobs_still_exports_the_dsn():
    """Dropping ``--db`` is only safe because the export is still there."""
    text = (SCRIPTS / "run_daily_jobs.sh").read_text()
    assert "export DATABASE_URL" in text


def test_fit_parlay_correlations_does_not_print_the_dsn():
    """That print goes straight into the cron log."""
    text = (SCRIPTS / "fit_parlay_correlations.py").read_text()
    assert "{db_path}" not in text


def test_launchd_example_tells_the_operator_to_lock_it_down():
    """The plist placeholder is replaced with a real password; launchd
    does not care about its mode, so nothing else would say it."""
    text = (SCRIPTS / "macos.launchd.com.example.nba-props-analyzer.plist").read_text()
    assert "chmod 600" in text
