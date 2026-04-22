"""End-to-end audit of trained model artifacts + metadata + calibration health.

Checks:
  - every expected joblib exists and unpickles
  - each classifier (raw, calibrated) has predict_proba
  - each regressor has predict
  - calibrator wraps an isotonic model
  - metadata.json consistency (prop list, required fields, sane numbers)
  - calibration improved over raw (brier_cal <= brier_raw within tolerance)
  - per-prop sanity: AUC in [0.5, 1.0], Brier in [0, 0.35]
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
META_PATH = MODELS / "model_metadata.json"


EXPECTED_PROPS = [
    "points", "assists", "rebounds", "steals", "blocks",
    "turnovers", "three_pointers",
    "pts_reb", "pts_ast", "ast_reb", "pts_ast_reb", "stl_blk",
    "double_double", "triple_double",
]

# Binaries don't have a reg_ model (by design)
BINARY_PROPS = {"double_double", "triple_double"}


class Issue:
    def __init__(self, severity: str, where: str, msg: str):
        self.severity = severity  # "FAIL" | "WARN" | "INFO"
        self.where = where
        self.msg = msg

    def __repr__(self):
        return f"[{self.severity}] {self.where}: {self.msg}"


def audit() -> list[Issue]:
    issues: list[Issue] = []

    # 1. metadata integrity
    if not META_PATH.exists():
        issues.append(Issue("FAIL", "meta", "model_metadata.json missing"))
        return issues

    try:
        meta = json.loads(META_PATH.read_text())
    except Exception as e:
        issues.append(Issue("FAIL", "meta", f"not valid JSON: {e}"))
        return issues

    props_meta = meta.get("props", {})
    if not props_meta:
        issues.append(Issue("FAIL", "meta", "metadata has no 'props' dict"))
        return issues

    for prop in EXPECTED_PROPS:
        if prop not in props_meta:
            issues.append(Issue("FAIL", f"meta/{prop}",
                                "missing from model_metadata.json"))
            continue

        entry = props_meta[prop]
        required = {"n_examples"}
        missing = required - entry.keys()
        if missing:
            issues.append(Issue("WARN", f"meta/{prop}",
                                f"missing fields: {sorted(missing)}"))

        # Classifier sub-block sanity
        clf_block = entry.get("classifier") or {}
        for k in ("auc_raw", "auc_cal", "brier_raw", "brier_cal"):
            v = clf_block.get(k)
            if v is None:
                issues.append(Issue("WARN", f"meta/{prop}",
                                    f"classifier.{k} missing"))
                continue
            if k.startswith("auc"):
                if not (0.0 < v < 1.0):
                    issues.append(Issue("WARN", f"meta/{prop}",
                                        f"classifier.{k}={v} outside (0,1)"))
                if v < 0.5:
                    issues.append(Issue("WARN", f"meta/{prop}",
                                        f"classifier.{k}={v:.3f} < 0.5 (worse than coin flip)"))
            else:  # brier
                if not (0.0 <= v <= 0.35):
                    issues.append(Issue("WARN", f"meta/{prop}",
                                        f"classifier.{k}={v:.3f} outside plausible [0, 0.35]"))

        # Calibration should not substantially HURT — allow 0.005 drift.
        # If fix_calibration.py has already swapped to raw for this prop,
        # the regression is informational-only.
        choice = clf_block.get("calibration_choice", "calibrated")
        br_raw = clf_block.get("brier_raw")
        br_cal = clf_block.get("brier_cal")
        if br_raw is not None and br_cal is not None:
            if br_cal > br_raw + 0.005 and choice == "calibrated":
                issues.append(Issue("WARN", f"meta/{prop}",
                                    f"calibration HURT Brier: raw={br_raw:.4f} "
                                    f"-> cal={br_cal:.4f} (+{br_cal-br_raw:.4f}) "
                                    f"(run scripts/fix_calibration.py)"))

    # 2. artifact existence + loadability
    for prop in EXPECTED_PROPS:
        # Classifier raw
        p = MODELS / f"clf_raw_{prop}.joblib"
        if not p.exists():
            issues.append(Issue("FAIL", f"clf_raw/{prop}", "file missing"))
        else:
            try:
                m = joblib.load(p)
                if not hasattr(m, "predict_proba"):
                    issues.append(Issue("FAIL", f"clf_raw/{prop}",
                                        "loaded but no .predict_proba"))
            except Exception as e:
                issues.append(Issue("FAIL", f"clf_raw/{prop}",
                                    f"unpickle failed: {e}"))

        # Classifier calibrated
        p = MODELS / f"clf_cal_{prop}.joblib"
        if not p.exists():
            issues.append(Issue("FAIL", f"clf_cal/{prop}", "file missing"))
        else:
            try:
                m = joblib.load(p)
                if not hasattr(m, "predict_proba"):
                    issues.append(Issue("FAIL", f"clf_cal/{prop}",
                                        "loaded but no .predict_proba"))
                # If fix_calibration.py swapped to raw for this prop, it's
                # deliberately a bare XGBClassifier (no .calibrator). Only
                # warn when choice is still "calibrated".
                prop_meta = props_meta.get(prop, {}).get("classifier", {})
                swap_choice = prop_meta.get("calibration_choice", "calibrated")
                if swap_choice == "calibrated":
                    inner = getattr(m, "calibrator", None)
                    if inner is None:
                        issues.append(Issue("WARN", f"clf_cal/{prop}",
                                            "no .calibrator attribute (not isotonic-wrapped?)"))
                    base = getattr(m, "base_estimator", None)
                    if base is None:
                        issues.append(Issue("WARN", f"clf_cal/{prop}",
                                            "no .base_estimator attribute"))
            except Exception as e:
                issues.append(Issue("FAIL", f"clf_cal/{prop}",
                                    f"unpickle failed: {e}"))

        # Regressor (skip binaries by design)
        if prop in BINARY_PROPS:
            continue
        p = MODELS / f"reg_{prop}.joblib"
        if not p.exists():
            issues.append(Issue("FAIL", f"reg/{prop}", "file missing"))
        else:
            try:
                m = joblib.load(p)
                if not hasattr(m, "predict"):
                    issues.append(Issue("FAIL", f"reg/{prop}",
                                        "loaded but no .predict"))
            except Exception as e:
                issues.append(Issue("FAIL", f"reg/{prop}",
                                    f"unpickle failed: {e}"))

    # 3. smoke-test predict_proba on EVERY prop — catches silent breakage
    import pandas as pd
    for prop in EXPECTED_PROPS:
        p = MODELS / f"clf_cal_{prop}.joblib"
        if not p.exists():
            continue
        try:
            m = joblib.load(p)
            schema = props_meta[prop].get("classifier", {}).get("schema", {})
            cols = schema.get("numeric_features") or []
            # Resolve the actual feature-name list the XGB booster expects
            actual = getattr(m, "estimator",
                             getattr(m, "base_estimator", m))
            feat_names = None
            if hasattr(actual, "feature_names_in_"):
                feat_names = list(actual.feature_names_in_)
            elif hasattr(actual, "get_booster"):
                try:
                    feat_names = actual.get_booster().feature_names
                except Exception:
                    feat_names = None
            if feat_names:
                cols = feat_names
            if not cols:
                issues.append(Issue("WARN", f"predict/{prop}",
                                    "no feature-name list discoverable"))
                continue
            X = pd.DataFrame([{c: 0.0 for c in cols}])
            proba = m.predict_proba(X)[:, 1]
            if not (0.0 <= float(proba[0]) <= 1.0):
                issues.append(Issue("FAIL", f"predict/{prop}",
                                    f"proba out of [0,1]: {proba[0]}"))
        except Exception as e:
            issues.append(Issue("FAIL", f"predict/{prop}",
                                f"smoke-predict failed: {e}"))

    # 4. regression smoke-test on every non-binary prop
    for prop in EXPECTED_PROPS:
        if prop in BINARY_PROPS:
            continue
        p = MODELS / f"reg_{prop}.joblib"
        if not p.exists():
            continue
        try:
            m = joblib.load(p)
            feat_names = None
            if hasattr(m, "feature_names_in_"):
                feat_names = list(m.feature_names_in_)
            elif hasattr(m, "get_booster"):
                try:
                    feat_names = m.get_booster().feature_names
                except Exception:
                    feat_names = None
            if not feat_names:
                issues.append(Issue("WARN", f"predict_reg/{prop}",
                                    "no feature-name list discoverable"))
                continue
            X = pd.DataFrame([{c: 0.0 for c in feat_names}])
            yhat = m.predict(X)
            if not np.isfinite(float(yhat[0])):
                issues.append(Issue("FAIL", f"predict_reg/{prop}",
                                    f"non-finite prediction: {yhat[0]}"))
        except Exception as e:
            issues.append(Issue("FAIL", f"predict_reg/{prop}",
                                f"reg-smoke-predict failed: {e}"))

    return issues


def main():
    print(f"[audit] models dir: {MODELS}")
    print(f"[audit] metadata:   {META_PATH}")
    print()

    try:
        issues = audit()
    except Exception:
        traceback.print_exc()
        sys.exit(2)

    fails = [i for i in issues if i.severity == "FAIL"]
    warns = [i for i in issues if i.severity == "WARN"]

    if not fails and not warns:
        print("[audit] ALL CLEAN — no issues detected.")
        return 0

    if warns:
        print(f"[audit] {len(warns)} warnings:")
        for i in warns:
            print("  ", i)
    if fails:
        print()
        print(f"[audit] {len(fails)} FAILURES:")
        for i in fails:
            print("  ", i)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
