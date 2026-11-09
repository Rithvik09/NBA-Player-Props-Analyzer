"""`scripts/serve.sh` must load `.env`, because nothing else does.

The repository has carried a `.env` holding ODDS_API_KEY and DATABASE_URL
since October and nothing ever read it: python-dotenv is not installed and
no module calls a loader — every apparent hit is `os.environ`. So the odds
poller and the betting-splits poller could not have run even with their
flags set, because the key was never in any process's environment.

These tests pin three contracts:
  - values in `.env` reach the served process
  - a value already in the environment WINS over `.env`, so launchd or a
    one-off `FOO=bar ./scripts/serve.sh` is not silently overridden by a
    stale file
  - comments, blanks and quotes are handled, and a DSN containing `=` in
    its query string survives intact

`serve.sh` ends in `exec gunicorn`, so these run it with a stub on PATH
that records its environment and exits, rather than binding a port.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "serve.sh"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists() or shutil.which("bash") is None,
    reason="needs scripts/serve.sh and bash",
)


@pytest.fixture
def run_serve(tmp_path):
    """Run serve.sh with a stub gunicorn that dumps the environment it got."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    dump = tmp_path / "env.json"
    stub = bindir / "gunicorn"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"json.dump(dict(os.environ), open({str(dump)!r}, 'w'))\n"
        "sys.exit(0)\n"
    )
    stub.chmod(0o755)

    def _run(env_file_text: str | None, **extra_env):
        # serve.sh cds to the repo root, so .env has to be staged there.
        # Back up any real one and restore it afterwards — a test must never
        # destroy a developer's working file.
        target = REPO / ".env"
        backup = tmp_path / ".env.backup"
        had_real = target.exists()
        if had_real:
            shutil.copy2(target, backup)
        try:
            if env_file_text is None:
                target.unlink(missing_ok=True)
            else:
                target.write_text(env_file_text)
            env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", **extra_env}
            # Keep the caller's real values out of the assertions.
            for key in ("SERVE_TEST_TOKEN",):
                if key not in extra_env:
                    env.pop(key, None)
            subprocess.run(["bash", str(SCRIPT)], env=env, check=True,
                           timeout=60, capture_output=True)
            return json.loads(dump.read_text())
        finally:
            if had_real:
                shutil.copy2(backup, target)
            else:
                target.unlink(missing_ok=True)

    return _run


def test_env_file_values_reach_the_server(run_serve):
    seen = run_serve("SERVE_TEST_TOKEN=from-env-file\n")
    assert seen.get("SERVE_TEST_TOKEN") == "from-env-file"


def test_existing_environment_wins_over_the_file(run_serve):
    """A stale .env must not override an explicitly supplied value."""
    seen = run_serve("SERVE_TEST_TOKEN=from-env-file\n",
                     SERVE_TEST_TOKEN="from-the-caller")
    assert seen.get("SERVE_TEST_TOKEN") == "from-the-caller"


def test_comments_blanks_and_quotes_are_handled(run_serve):
    seen = run_serve(
        "# a comment\n"
        "\n"
        '   SERVE_TEST_TOKEN="quoted value"\n'
        "not a valid line\n"
    )
    assert seen.get("SERVE_TEST_TOKEN") == "quoted value"


def test_a_dsn_with_equals_in_it_survives(run_serve):
    """Split on the first `=` only — a DSN query string contains more."""
    dsn = "postgresql://u@h/db?options=-csearch_path%3Dpublic"
    seen = run_serve(f"SERVE_TEST_TOKEN={dsn}\n")
    assert seen.get("SERVE_TEST_TOKEN") == dsn


def test_missing_env_file_is_not_an_error(run_serve):
    seen = run_serve(None)
    assert "SERVE_TEST_TOKEN" not in seen
