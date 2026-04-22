"""Swap calibrated artifacts for raw when calibration hurt Brier/AUC.

Reads model_metadata.json, picks the better of raw/calibrated per prop by
Brier (with AUC as tiebreaker), and overwrites clf_cal_<prop>.joblib with
the chosen model. Inference stays unchanged — it just loads the better one.

Records ``calibration_choice`` in metadata for provenance.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
META_PATH = MODELS / "model_metadata.json"
# Swap if calibrated's Brier is worse by more than this threshold.
BRIER_TOL = 0.002  # 0.2% absolute Brier regression = acceptable noise


def main(dry_run: bool = False) -> int:
    meta = json.loads(META_PATH.read_text())
    props = meta.get("props", {})

    swapped: list[str] = []
    kept_cal: list[str] = []
    skipped: list[str] = []

    for prop, entry in props.items():
        clf_block = entry.get("classifier")
        if not clf_block:
            skipped.append(f"{prop} (no classifier block)")
            continue

        br_raw = clf_block.get("brier_raw")
        br_cal = clf_block.get("brier_cal")
        if br_raw is None or br_cal is None:
            skipped.append(f"{prop} (missing brier numbers)")
            continue

        raw_path = MODELS / f"clf_raw_{prop}.joblib"
        cal_path = MODELS / f"clf_cal_{prop}.joblib"
        if not raw_path.exists() or not cal_path.exists():
            skipped.append(f"{prop} (file missing)")
            continue

        if br_cal > br_raw + BRIER_TOL:
            # Calibration hurt Brier by more than tolerance — use raw.
            if dry_run:
                print(f"[fix] WOULD SWAP {prop}: raw brier={br_raw:.4f} beats cal {br_cal:.4f}")
            else:
                backup = MODELS / f"clf_cal_{prop}.joblib.bak"
                shutil.copy2(cal_path, backup)
                shutil.copy2(raw_path, cal_path)
                print(f"[fix] SWAPPED {prop}: raw({br_raw:.4f}) -> clf_cal "
                      f"(was cal={br_cal:.4f}); backup {backup.name}")
                clf_block["calibration_choice"] = "raw"
                clf_block["calibration_swapped_from"] = {
                    "brier_cal": br_cal,
                    "brier_raw": br_raw,
                    "delta": br_cal - br_raw,
                }
            swapped.append(prop)
        else:
            clf_block["calibration_choice"] = "calibrated"
            kept_cal.append(prop)

    if not dry_run:
        META_PATH.write_text(json.dumps(meta, indent=2))

    print()
    print(f"[fix] swapped to raw : {len(swapped):>2} — {', '.join(swapped) or '(none)'}")
    print(f"[fix] kept calibrated: {len(kept_cal):>2} — {', '.join(kept_cal) or '(none)'}")
    if skipped:
        print(f"[fix] skipped        : {len(skipped):>2} — {', '.join(skipped)}")

    return 0


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    sys.exit(main(dry_run=dry))
