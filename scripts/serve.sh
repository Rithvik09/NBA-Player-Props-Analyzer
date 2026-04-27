#!/usr/bin/env bash
# Production entrypoint — runs the Flask app under gunicorn with sane defaults.
#
# Env vars:
#   PORT          — listen port (default 8000)
#   HOST          — bind address (default 0.0.0.0)
#   GUNICORN_WORKERS — worker process count (default: 2*CPU + 1)
#   GUNICORN_TIMEOUT — request timeout in seconds (default 120, NBA API is slow)
#   GUNICORN_LOG_LEVEL — debug|info|warning|error (default info)
#
# Example:
#   PORT=8080 GUNICORN_WORKERS=4 ./scripts/serve.sh
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${GUNICORN_WORKERS:-$(python3 -c 'import os; print(2*os.cpu_count()+1)')}"
TIMEOUT="${GUNICORN_TIMEOUT:-120}"
LOG_LEVEL="${GUNICORN_LOG_LEVEL:-info}"

cd "$(dirname "$0")/.."

# Confirm models exist before binding the port — fail fast in CI/CD.
if [[ ! -f models/model_metadata.json ]]; then
  echo "[serve] WARNING: models/model_metadata.json not found — /healthz will return 503" >&2
fi

exec gunicorn \
  --bind "${HOST}:${PORT}" \
  --workers "${WORKERS}" \
  --timeout "${TIMEOUT}" \
  --log-level "${LOG_LEVEL}" \
  --access-logfile - \
  --error-logfile - \
  --worker-class sync \
  --preload \
  src.app:app
