#!/usr/bin/env python3
"""
Comprehensive model testing script.
Evaluates all trained models on test data and reports accuracy metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, brier_score_loss, log_loss
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import EnhancedMLPredictor
from src.precomputed_store import PrecomputedStore
from src.ml_features import build_feature_vector, build_classifier_vector, NUMERIC_FEATURE_KEYS, CLASSIFIER_EXTRA_KEYS

# Import the training example builder
# Add scripts directory to path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from train_models import build_training_examples, Example, _sportsbook_center, _sample_lines


def load_models(models_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load all trained models"""
    models = {}
    all_props = ["points", "assists", "rebounds", "steals", "blocks", "turnovers", "three_pointers",
                 "pts_reb", "pts_ast", "ast_reb", "pts_ast_reb", "stl_blk", "double_double", "triple_double"]
    
    for prop in all_props:
        models[prop] = {}
        
        # Load regression model (if exists)
        reg_path = os.path.join(models_dir, f"reg_{prop}.joblib")
        if os.path.exists(reg_path):
            models[prop]["regressor"] = joblib.load(reg_path)
        
        # Load raw classifier
        clf_raw_path = os.path.join(models_dir, f"clf_raw_{prop}.joblib")
        if os.path.exists(clf_raw_path):
            models[prop]["classifier_raw"] = joblib.load(clf_raw_path)
        
        # Load calibrated classifier
        clf_cal_path = os.path.join(models_dir, f"clf_cal_{prop}.joblib")
        if os.path.exists(clf_cal_path):
            models[prop]["classifier_cal"] = joblib.load(clf_cal_path)
    
    return models


def _align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Align test features to the exact columns (and order) the model was trained on.
    Drops any new columns and fills missing ones with 0.
    """
    # Try direct feature_names_in_ first
    feature_names = getattr(model, "feature_names_in_", None)

    # Some wrapped models (like calibrated models) may keep feature_names on base_estimator
    if feature_names is None and hasattr(model, "base_estimator"):
        feature_names = getattr(model.base_estimator, "feature_names_in_", None)

    if feature_names is None:
        # Fallback: assume X already matches training schema
        return X

    return X.reindex(columns=list(feature_names), fill_value=0.0)


def evaluate_regression_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate regression model and return metrics"""
    X_aligned = _align_features(model, X_test)
    y_pred = model.predict(X_aligned)
    
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    
    # R² score
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Mean percentage error
    mpe = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100)
    
    # Accuracy within different thresholds
    within_1 = float(np.mean(np.abs(y_test - y_pred) <= 1.0) * 100)
    within_2 = float(np.mean(np.abs(y_test - y_pred) <= 2.0) * 100)
    within_3 = float(np.mean(np.abs(y_test - y_pred) <= 3.0) * 100)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "mean_percentage_error": mpe,
        "within_1": within_1,
        "within_2": within_2,
        "within_3": within_3,
        "mean_actual": float(np.mean(y_test)),
        "mean_predicted": float(np.mean(y_pred)),
    }


def evaluate_classifier_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate classifier model and return metrics"""
    if len(np.unique(y_test)) < 2:
        return {"error": "Not enough class diversity in test set"}
    
    X_aligned = _align_features(model, X_test)
    proba = model.predict_proba(X_aligned)[:, 1]
    
    auc = float(roc_auc_score(y_test, proba))
    brier = float(brier_score_loss(y_test, proba))
    
    # Log loss
    log_loss_val = float(log_loss(y_test, proba))
    
    # Expected Value (assuming -110 odds)
    ev = float(np.mean((proba * (100/110)) - ((1 - proba) * (110/100))))
    
    # Binary accuracy at different thresholds
    thresholds = [0.4, 0.5, 0.6]
    accuracies = {}
    for thresh in thresholds:
        y_pred_binary = (proba >= thresh).astype(int)
        acc = float(np.mean(y_pred_binary == y_test) * 100)
        accuracies[f"accuracy_at_{thresh}"] = acc
    
    # Calibration metrics
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calibration_error = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (proba > bin_lower) & (proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_test[in_bin].mean()
            avg_confidence_in_bin = proba[in_bin].mean()
            calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return {
        "auc": auc,
        "brier_score": brier,
        "log_loss": log_loss_val,
        "expected_value": ev,
        "calibration_error": float(calibration_error),
        "mean_predicted_prob": float(np.mean(proba)),
        "mean_actual_rate": float(np.mean(y_test)),
        **accuracies
    }


def test_all_models(models_dir: str = "models", db_path: str = "basketball_data.db", 
                    seasons: List[str] = None, max_players: int = 200) -> Dict[str, Any]:
    """Test all models on test data"""
    
    print("=" * 80)
    print("🧪 MODEL ACCURACY TESTING")
    print("=" * 80)
    
    # Load models
    print("\n📦 Loading models...")
    models = load_models(models_dir)
    print(f"   Loaded models for {len(models)} props")
    
    # Load metadata
    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    # Build test examples
    print("\n📊 Building test examples...")
    predictor = EnhancedMLPredictor(model_dir=models_dir)
    precomputed = PrecomputedStore(db_path)
    
    if seasons is None:
        seasons = [predictor.current_season]
    
    all_examples = []
    for season in seasons:
        print(f"   Processing season {season}...")
        examples = build_training_examples(season=season, max_players=max_players, 
                                         predictor=predictor, precomputed=precomputed)
        all_examples.extend(examples)
    
    print(f"   Total examples: {len(all_examples)}")
    
    # Group by prop and evaluate
    by_prop: Dict[str, List[Example]] = {}
    for ex in all_examples:
        by_prop.setdefault(ex.prop_type, []).append(ex)
    
    results = {}
    
    print("\n" + "=" * 80)
    print("📈 EVALUATION RESULTS")
    print("=" * 80)
    
    for prop in sorted(by_prop.keys()):
        print(f"\n{'='*80}")
        print(f"🏀 {prop.upper()}")
        print(f"{'='*80}")
        
        exs = sorted(by_prop[prop], key=lambda e: e.game_date)
        
        # Use same 80/20 split as training
        cut = int(len(exs) * 0.8)
        train_exs = exs[:cut]
        test_exs = exs[cut:]
        
        print(f"   Test examples: {len(test_exs)} (from {len(exs)} total)")
        
        if not test_exs:
            print("   ⚠️  No test examples available")
            continue
        
        X_test = pd.concat([e.X for e in test_exs], ignore_index=True)
        y_test = np.array([e.y for e in test_exs], dtype=float)
        
        prop_results = {
            "n_test_examples": len(test_exs),
            "n_train_examples": len(train_exs),
            "n_total_examples": len(exs),
        }
        
        # Test regression model
        is_binary = prop in ("double_double", "triple_double")
        if not is_binary and "regressor" in models[prop]:
            print(f"\n   📊 Regression Model:")
            reg_metrics = evaluate_regression_model(models[prop]["regressor"], X_test, y_test)
            prop_results["regression"] = reg_metrics
            
            # Compare with metadata
            meta_reg = metadata["props"][prop]
            print(f"      RMSE: {reg_metrics['rmse']:.3f} (metadata: {meta_reg.get('rmse', 'N/A')})")
            print(f"      MAE:  {reg_metrics['mae']:.3f} (metadata: {meta_reg.get('mae', 'N/A')})")
            print(f"      R²:   {reg_metrics['r2_score']:.3f} (metadata: {meta_reg.get('r2_score', 'N/A')})")
            print(f"      Within 1: {reg_metrics['within_1']:.1f}%")
            print(f"      Within 2: {reg_metrics['within_2']:.1f}%")
            print(f"      Within 3: {reg_metrics['within_3']:.1f}%")
            print(f"      Mean Actual: {reg_metrics['mean_actual']:.2f}")
            print(f"      Mean Predicted: {reg_metrics['mean_predicted']:.2f}")
        
        # Test classifier models
        print(f"\n   🎯 Classification Models:")
        
        # Build classifier test set (with line sampling)
        Xc_test_parts = []
        yc_test_parts = []
        for e, yv in zip(test_exs, y_test):
            raw = e.X.iloc[0].to_dict()
            center = _sportsbook_center(raw)
            for ln in _sample_lines(float(center), prop):
                Xc_test_parts.append(build_classifier_vector(raw, line=ln).X)
                yc_test_parts.append(1 if float(yv) > float(ln) else 0)
        
        if Xc_test_parts:
            Xc_test = pd.concat(Xc_test_parts, ignore_index=True)
            yc_test = np.array(yc_test_parts, dtype=int)
            
            print(f"      Test samples (with line sampling): {len(Xc_test)}")
            
            # Test raw classifier
            if "classifier_raw" in models[prop]:
                print(f"\n      🔹 Raw Classifier:")
                clf_raw_metrics = evaluate_classifier_model(
                    models[prop]["classifier_raw"], Xc_test, yc_test
                )
                prop_results["classifier_raw"] = clf_raw_metrics
                
                meta_clf = metadata["props"][prop].get("classifier", {})
                print(f"         AUC:  {clf_raw_metrics['auc']:.3f} (metadata: {meta_clf.get('auc_raw', 'N/A')})")
                print(f"         Brier: {clf_raw_metrics['brier_score']:.4f} (metadata: {meta_clf.get('brier_raw', 'N/A')})")
                print(f"         Log Loss: {clf_raw_metrics['log_loss']:.4f}")
                print(f"         Expected Value: {clf_raw_metrics['expected_value']:.4f}")
                print(f"         Calibration Error: {clf_raw_metrics['calibration_error']:.4f}")
                print(f"         Accuracy @ 0.5: {clf_raw_metrics.get('accuracy_at_0.5', 'N/A'):.1f}%")
            
            # Test calibrated classifier
            if "classifier_cal" in models[prop]:
                print(f"\n      🔹 Calibrated Classifier:")
                clf_cal_metrics = evaluate_classifier_model(
                    models[prop]["classifier_cal"], Xc_test, yc_test
                )
                prop_results["classifier_cal"] = clf_cal_metrics
                
                meta_clf = metadata["props"][prop].get("classifier", {})
                print(f"         AUC:  {clf_cal_metrics['auc']:.3f} (metadata: {meta_clf.get('auc_cal', 'N/A')})")
                print(f"         Brier: {clf_cal_metrics['brier_score']:.4f} (metadata: {meta_clf.get('brier_cal', 'N/A')})")
                print(f"         Log Loss: {clf_cal_metrics['log_loss']:.4f}")
                print(f"         Expected Value: {clf_cal_metrics['expected_value']:.4f}")
                print(f"         Calibration Error: {clf_cal_metrics['calibration_error']:.4f}")
                print(f"         Accuracy @ 0.5: {clf_cal_metrics.get('accuracy_at_0.5', 'N/A'):.1f}%")
        
        results[prop] = prop_results
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    print("\n🎯 Regression Models (RMSE):")
    for prop in sorted(results.keys()):
        if "regression" in results[prop]:
            rmse = results[prop]["regression"]["rmse"]
            within_2 = results[prop]["regression"]["within_2"]
            print(f"   {prop:20s} | RMSE: {rmse:6.3f} | Within 2: {within_2:5.1f}%")
    
    print("\n🎲 Classification Models (AUC - Calibrated):")
    for prop in sorted(results.keys()):
        if "classifier_cal" in results[prop]:
            auc = results[prop]["classifier_cal"]["auc"]
            ev = results[prop]["classifier_cal"]["expected_value"]
            print(f"   {prop:20s} | AUC: {auc:.3f} | EV: {ev:+.4f}")
    
    # Save results
    results_path = os.path.join(models_dir, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--db", default="basketball_data.db")
    ap.add_argument("--seasons", default=None, help="Comma-separated seasons like 2023-24,2024-25")
    ap.add_argument("--max-players", type=int, default=200)
    args = ap.parse_args()
    
    seasons = None
    if args.seasons:
        seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    
    test_all_models(
        models_dir=args.models_dir,
        db_path=args.db,
        seasons=seasons,
        max_players=args.max_players
    )
