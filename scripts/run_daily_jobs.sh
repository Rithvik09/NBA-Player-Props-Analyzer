#!/usr/bin/env bash
set -euo pipefail

# Standalone daily job runner (works even if Flask is not running).
# Runs:
#  1) daily precompute (DVP + defenders)
#  2) daily incremental training (fast)
#  3) optional weekly batch training (full retrain + evaluation)
#
# Config via env vars:
#  DB_PATH (default: basketball_data.db)
#  MODELS_DIR (default: models)
#  INCREMENTAL_SEASON (default: auto-detected current season)
#  INCREMENTAL_MAX_PLAYERS (default: 200)
#  BATCH_WEEKLY (default: 1)
#  BATCH_WEEKDAY (default: 1; 1=Mon .. 7=Sun)
#  BATCH_SEASONS (default: 2021-22,2022-23,2023-24,2024-25,2025-26)
#  BATCH_MAX_PLAYERS (default: 400)
#
# Safety:
#  - lockfile to prevent overlapping runs
#
# Usage:
#  ./scripts/run_daily_jobs.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DB_PATH="${DB_PATH:-basketball_data.db}"
MODELS_DIR="${MODELS_DIR:-models}"
INCREMENTAL_SEASON="${INCREMENTAL_SEASON:-}"
INCREMENTAL_MAX_PLAYERS="${INCREMENTAL_MAX_PLAYERS:-200}"

BATCH_WEEKLY="${BATCH_WEEKLY:-1}"
BATCH_WEEKDAY="${BATCH_WEEKDAY:-1}" # 1=Mon .. 7=Sun (date +%u)
BATCH_SEASONS="${BATCH_SEASONS:-2021-22,2022-23,2023-24,2024-25,2025-26}"
BATCH_MAX_PLAYERS="${BATCH_MAX_PLAYERS:-400}"

LOCK_DIR="${LOCK_DIR:-/tmp/nba_props_analyzer_daily_jobs.lock}"

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$$" > "$LOCK_DIR/pid"
    trap 'rm -rf "$LOCK_DIR"' EXIT
  else
    echo "[daily-jobs] another run is already in progress (lock: $LOCK_DIR). exiting."
    exit 0
  fi
}

acquire_lock

echo "[daily-jobs] starting at $(date)"
echo "[daily-jobs] DB_PATH=$DB_PATH MODELS_DIR=$MODELS_DIR"
echo "[daily-jobs] INCREMENTAL_SEASON=$INCREMENTAL_SEASON INCREMENTAL_MAX_PLAYERS=$INCREMENTAL_MAX_PLAYERS"
echo "[daily-jobs] BATCH_WEEKLY=$BATCH_WEEKLY BATCH_WEEKDAY=$BATCH_WEEKDAY BATCH_SEASONS=$BATCH_SEASONS BATCH_MAX_PLAYERS=$BATCH_MAX_PLAYERS"

PYTHONUNBUFFERED=1 python3 -u scripts/update_precomputed.py --db "$DB_PATH"

if [[ -z "$INCREMENTAL_SEASON" ]]; then
  INCREMENTAL_SEASON="$(python3 - <<'PY'
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models import EnhancedMLPredictor
print(EnhancedMLPredictor().current_season)
PY
)"
fi

today_dow="$(date +%u)"  # 1..7 (Mon..Sun)
if [[ "$BATCH_WEEKLY" == "1" || "$BATCH_WEEKLY" == "true" || "$BATCH_WEEKLY" == "yes" ]]; then
  if [[ "$today_dow" == "$BATCH_WEEKDAY" ]]; then
    echo "[daily-jobs] weekly batch day matched (dow=$today_dow). running batch training..."
    PYTHONUNBUFFERED=1 python3 -u scripts/train_models.py \
      --seasons "$BATCH_SEASONS" \
      --max-players "$BATCH_MAX_PLAYERS" \
      --models-dir "$MODELS_DIR" \
      --db "$DB_PATH"
    echo "[daily-jobs] skipping incremental update on batch day (batch retrain supersedes it)."
  else
    echo "[daily-jobs] weekly batch skipped (today dow=$today_dow, configured=$BATCH_WEEKDAY)"
    PYTHONUNBUFFERED=1 python3 -u scripts/update_incremental_models.py \
      --season "$INCREMENTAL_SEASON" \
      --max-players "$INCREMENTAL_MAX_PLAYERS" \
      --models-dir "$MODELS_DIR" \
      --db "$DB_PATH"
  fi
else
  echo "[daily-jobs] weekly batch disabled"
  PYTHONUNBUFFERED=1 python3 -u scripts/update_incremental_models.py \
    --season "$INCREMENTAL_SEASON" \
    --max-players "$INCREMENTAL_MAX_PLAYERS" \
    --models-dir "$MODELS_DIR" \
    --db "$DB_PATH"
fi

echo "[daily-jobs] done at $(date)"


