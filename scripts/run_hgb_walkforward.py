"""Run EnhancedMLPredictor.walk_forward_evaluate on real data and write
its per-prop Brier numbers under ``models/model_metadata.json``'s
``walk_forward_hgb`` key — leaving the XGBoost ``walk_forward`` block
intact for side-by-side comparison.

Footprint: 200 players × 3 seasons. Smaller than the full 400 × 5
retrain because walk-forward retrains the model on each fold, so
compute scales with samples × n_folds. With nba_api cache warm from
yesterday's runs, data collection is fast.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collector import TrainingDataCollector
from src.models import EnhancedMLPredictor
from src.precomputed_store import PrecomputedStore


NUM_PLAYERS = int(os.environ.get("WF_NUM_PLAYERS", 200))
NUM_SEASONS = int(os.environ.get("WF_NUM_SEASONS", 3))
N_FOLDS = int(os.environ.get("WF_N_FOLDS", 5))
FOLD_DAYS = int(os.environ.get("WF_FOLD_DAYS", 30))


def main() -> int:
    print(f"[wf] config: players={NUM_PLAYERS} seasons={NUM_SEASONS} folds={N_FOLDS}")
    t0 = time.time()

    collector = TrainingDataCollector()
    player_ids = collector.get_active_player_ids(n=NUM_PLAYERS)
    seasons = collector._get_seasons(num_seasons=NUM_SEASONS)
    print(f"[wf] {len(player_ids)} players, seasons={seasons}")

    pre = PrecomputedStore("basketball_data.db")
    pdata = pre.refresh(force=False)
    dvp_map = pdata.get("dvp") or {}
    dvp_avgs = pdata.get("dvp_pos_avgs") or {}
    defs_map = pdata.get("defenders") or {}

    print("[wf] collecting samples ...")
    samples = collector.collect_bulk(
        player_ids, seasons=seasons,
        dvp_map=dvp_map, dvp_pos_avgs=dvp_avgs, defenders_map=defs_map,
    )
    print(f"[wf] collected {len(samples)} samples in {time.time() - t0:.0f}s")
    if not samples:
        print("[wf] no samples — aborting")
        return 1

    p = EnhancedMLPredictor()
    print(f"[wf] running walk_forward_evaluate ...")
    t1 = time.time()
    result = p.walk_forward_evaluate(
        samples,
        n_folds=N_FOLDS,
        fold_days=FOLD_DAYS,
        min_train_samples=200,
        write_metadata=False,
    )
    print(f"[wf] eval took {time.time() - t1:.0f}s")

    # Write under walk_forward_hgb so XGBoost's walk_forward stays intact
    meta_path = "models/model_metadata.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"props": {}}
    meta.setdefault("props", {})

    print("\n=== HGB walk-forward summary ===")
    print(f"{'prop':18s} {'n_folds':8s} {'brier_cal_mean':18s} {'auc_cal_mean':14s} {'rmse_mean':10s}")
    for prop, info in (result.get("per_prop") or {}).items():
        folds = info.get("folds") or []
        if not folds:
            continue
        n = len(folds)
        bc = sum(f.get("brier_cal", 0.0) for f in folds) / n
        ac = sum(f.get("auc_cal", 0.0) for f in folds) / n
        rmse_vals = [f.get("rmse") for f in folds if "rmse" in f and f["rmse"] is not None]
        rmse = sum(rmse_vals) / len(rmse_vals) if rmse_vals else float("nan")
        print(f"{prop:18s} {n:<8d} {bc:<18.4f} {ac:<14.4f} {rmse:<10.4f}")

        meta["props"].setdefault(prop, {})
        meta["props"][prop]["walk_forward_hgb"] = {
            "n_folds": n,
            "brier_cal_mean": bc,
            "auc_cal_mean": ac,
            "rmse_mean": rmse,
            "folds": folds,
        }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n[wf] wrote walk_forward_hgb blocks to {meta_path}")
    print(f"[wf] total wall-clock: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
