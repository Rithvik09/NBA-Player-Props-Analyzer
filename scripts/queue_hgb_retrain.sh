#!/usr/bin/env bash
# Queue the EnhancedMLPredictor.train (HGB-stacked path) to run after the
# XGBoost retrain finishes. Why two passes:
#   - scripts/train_models.py uses XGBoost and benefits only from the
#     data_collector improvements on this branch.
#   - EnhancedMLPredictor.train uses HGB+stacking and benefits from EVERY
#     improvement — per-prop monotonic, NaN imputation, reliability +
#     volatility weighting, outlier capping, isotonic threshold bump.
#
# Whichever pass runs LAST wins the on-disk joblibs that production
# inference loads. Running HGB last makes the in-models.py changes the
# canonical production behaviour.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WAIT_PID="${WAIT_PID:-}"
NUM_PLAYERS="${NUM_PLAYERS:-400}"
NUM_SEASONS="${NUM_SEASONS:-5}"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/retrain_hgb_$TS.log"

{
  echo "[queue] started at $(date)"
  echo "[queue] WAIT_PID=$WAIT_PID NUM_PLAYERS=$NUM_PLAYERS NUM_SEASONS=$NUM_SEASONS"

  if [ -n "$WAIT_PID" ]; then
    echo "[queue] polling PID $WAIT_PID until it exits ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
      sleep 60
    done
    echo "[queue] PID $WAIT_PID exited at $(date)"
  fi

  # Backup XGBoost-written joblibs before HGB pass overwrites them.
  BACKUP_DIR="$ROOT_DIR/models.backup.xgb_$TS"
  echo "[queue] backing up XGBoost models -> $BACKUP_DIR"
  cp -R "$ROOT_DIR/models" "$BACKUP_DIR"

  echo "[queue] starting EnhancedMLPredictor.train ..."
  PYTHONUNBUFFERED=1 python3 -u -c "
from src.basketball_betting_helper import BasketballBettingHelper
h = BasketballBettingHelper()
result = h.retrain(min_new_samples=0, num_players=$NUM_PLAYERS, num_seasons=$NUM_SEASONS)
print('=== retrain result ===')
print(result)
"

  echo "[queue] HGB retrain finished at $(date)"
} > "$LOG" 2>&1
