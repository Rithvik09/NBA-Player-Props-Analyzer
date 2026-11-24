# Restore the app and research workspace

Recorded 2026-11-24. Source branch: `research/points-model-campaign`.
This guide describes a functional local setup; it does not certify predictive performance.

## Restore code and environments

```sh
git clone --branch research/points-model-campaign https://github.com/Rithvik09/NBA-Player-Props-Analyzer.git
cd NBA-Player-Props-Analyzer
python3.12 -m venv .venv-serving-legacy
.venv-serving-legacy/bin/python -m pip install -r requirements.serving-legacy.lock
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

The verified serving runtime was Python 3.12.5 on macOS arm64. Keep serving and
research environments separate: serialized estimators use different sklearn
versions. The lock files pin Python packages, not Python itself, PostgreSQL,
OS libraries, provider availability, or future package-index availability.

## Database and private configuration

Install/start PostgreSQL (project CI uses PostgreSQL 14; local client observed
18.0). Create an app database for your local role with `createdb nba_props`.
Copy `.env.example` to `.env`, replace `YOUR_LOCAL_ROLE`, and restore private
credentials from your password manager or private archive. Never commit `.env`.
Original configuration used DATABASE_URL, ODDS_API_KEY, DUNKAPI_API_KEY and
BETTING_SPLITS_POLL; secret values are deliberately absent from GitHub.

For a custom-format database backup, restore into a NEW empty database:

```sh
createdb nba_props_restored
pg_restore --no-owner --no-privileges --dbname=nba_props_restored /path/to/nba_props_pg14.dump
```

The private backup includes `nba_props_pg14.dump`, created with PostgreSQL 14 tools.
Use matching PostgreSQL 14 tools when restoring to version 14. A PostgreSQL 18
client dump (`nba_props.dump`, also retained) contains a transaction_timeout setting
that PostgreSQL 14 rejects; use the pg14 dump instead. On the original Mac the
matching tools are in `/opt/homebrew/opt/postgresql@14/bin/`.

Point DATABASE_URL to that restored database. Do not run test schema resets on it.
The old `basketball_data.db` is retained evidence, not the PostgreSQL app database.

## Start safely

Export DATABASE_URL in the shell when using `run.py`; it does not read `.env`.
Use a fresh, unique output path for the manifest:

```sh
export DATABASE_URL=postgresql://YOUR_LOCAL_ROLE@localhost/nba_props
.venv-serving-legacy/bin/python -m scripts.serving_manifest --output /tmp/new-serving-release/manifest.json
.venv-serving-legacy/bin/python -m scripts.serving_manifest --verify /tmp/new-serving-release/manifest.json
APP_BACKGROUND_TASKS=0 NIGHTLY_RETRAIN=0 ODDS_API_KEY='' SERVING_MANIFEST_PATH=/tmp/new-serving-release/manifest.json .venv-serving-legacy/bin/python run.py
```

Open http://127.0.0.1:5000 and check `/healthz`. `scripts/serve.sh` loads `.env`
and runs gunicorn from PATH; activate the serving environment first if using it.
Keep APP_BACKGROUND_TASKS=0 until deliberately enabling individual workflows.
Interactive requests may still fetch public provider data. Do not enable broad
Odds API polling, automatic retraining, or promote research models accidentally.

## Re-run verification

Create a fresh local disposable database (the suite resets its public schema):

```sh
createdb codex_serving_test_restore
APP_BACKGROUND_TASKS=0 NIGHTLY_RETRAIN=0 ODDS_API_KEY='' TEST_DATABASE_URL=postgresql:///codex_serving_test_restore .venv-serving-legacy/bin/python -m pytest -q
.venv-serving-legacy/bin/python -m scripts.audit_legacy_runtime --output /tmp/new-model-audit.json
```

2026-11-24 verification: 999 passed, 3 failed in 257.46 seconds. Two poller tests
expected background tasks enabled; all six tests in test_betting_splits_poller.py
passed separately with APP_BACKGROUND_TASKS=1 (their fixture prevents real threads).
The remaining backup test hard-codes `nba_props_test` in its expected filename,
so it fails with the correctly named isolated test database. No tests were edited.
There were 219 warnings, including research-model version mismatches in the serving
environment and numerical warnings. All 44 serving artifacts loaded and all 84
inference checks passed. Real Flask/model/PostgreSQL prediction, exact replay and
PUSH grading passed using canned provider data. Browser homepage/log/accuracy/bias
views worked. The UI season label and automatic-retraining explanation are stale.
Live NBA source availability was not established; official collection times out.

## Research evidence is not in Git

The ignored `logs/` directory is approximately 3.3 GB and includes raw collection
evidence, timestamps, datasets, model candidates, protocols, budget reservations,
notification state, and experiment locks. It must be restored from the private
archive to reproduce existing experiments. GitHub code and this document alone
cannot recreate historical quotes or point-in-time evidence. Do not reconstruct
old forecasts from newly fetched data or reset attempt/budget records.

Restore archived files to their original relative paths. Keep `.env` private.
Virtual environments, bytecode and pytest caches are intentionally excluded from
the archive; recreate environments from the lock files. A Git bundle preserves
local refs separately. Linked worktrees are not independent backups of this repo.

Current frozen studies must fail closed on source drift: v1/v5 have known source
mismatches and the points validator routes to v2/v6. Preserve the protocols and
models; do not adjust hashes to force acceptance. See PROSPECTIVE_POINTS_VALIDATION.md
and SOURCE_BLOCKERS.md. Research history-volume replay also checks an old protected
state snapshot; later heartbeat updates can cause that check to fail legitimately.
Neither deleting locks nor overwriting old evidence is a replay repair.

The scheduled monitor uses the original absolute project path. Deleting/moving it
requires separately updating or pausing that automation; GitHub does not run it.

## Before deleting the original folder

Verify the pushed branch, private archive checksum and Git bundle. Keep the archive
and checksum off this Mac (external drive or private storage) and test restoration.
A private archive on this same disk is protection against folder deletion, not disk
failure, and does not itself reclaim much space. No deletion is authorized by this
restoration guide.
