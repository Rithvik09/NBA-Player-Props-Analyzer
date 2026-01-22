from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

from .ml_features import NUMERIC_FEATURE_KEYS, CLASSIFIER_EXTRA_KEYS


@dataclass
class IncrementalModelBundle:
    scaler_reg: StandardScaler
    reg: SGDRegressor
    scaler_clf: StandardScaler
    clf: SGDClassifier
    is_initialized: bool  # kept for backwards compatibility; not used for classifier first-call logic


class IncrementalModelManager:
    """
    Online (incremental) models per prop using partial_fit.

    - Regression: SGDRegressor (Huber loss) for predicted value
    - Classification: SGDClassifier (log_loss) for P(y > line)

    Models are persisted in `models_dir/incremental/`.
    """

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.root = os.path.join(models_dir, "incremental")
        os.makedirs(self.root, exist_ok=True)
        self.meta_path = os.path.join(self.root, "meta.json")
        self.meta: dict[str, Any] = self._load_meta()
        self.bundles: dict[str, IncrementalModelBundle] = {}

        # Avoid excessive IO: only persist every N updates per prop (configurable).
        try:
            self.save_every = int(os.environ.get("INCREMENTAL_SAVE_EVERY", "500"))
        except Exception:
            self.save_every = 500
        self._updates_since_save: dict[str, int] = {}  # prop -> count

    def _load_meta(self) -> dict[str, Any]:
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_meta(self):
        try:
            with open(self.meta_path, "w") as f:
                json.dump(self.meta, f, indent=2)
        except Exception:
            return

    def _paths(self, prop: str) -> dict[str, str]:
        return {
            "scaler_reg": os.path.join(self.root, f"scaler_reg_{prop}.joblib"),
            "reg": os.path.join(self.root, f"reg_{prop}.joblib"),
            "scaler_clf": os.path.join(self.root, f"scaler_clf_{prop}.joblib"),
            "clf": os.path.join(self.root, f"clf_{prop}.joblib"),
        }

    def load_or_init(self, prop: str) -> IncrementalModelBundle:
        if prop in self.bundles:
            return self.bundles[prop]

        p = self._paths(prop)
        if all(os.path.exists(p[k]) for k in p):
            try:
                bundle = IncrementalModelBundle(
                    scaler_reg=joblib.load(p["scaler_reg"]),
                    reg=joblib.load(p["reg"]),
                    scaler_clf=joblib.load(p["scaler_clf"]),
                    clf=joblib.load(p["clf"]),
                    is_initialized=True,
                )
            except Exception:
                # Corrupted partial writes (e.g. after ENOSPC) should not brick training.
                bundle = IncrementalModelBundle(
                    scaler_reg=StandardScaler(with_mean=True, with_std=True),
                    reg=SGDRegressor(
                        loss="huber",
                        alpha=1e-4,
                        learning_rate="invscaling",
                        eta0=0.01,
                        power_t=0.25,
                        max_iter=1,
                        tol=None,
                        random_state=42,
                    ),
                    scaler_clf=StandardScaler(with_mean=True, with_std=True),
                    clf=SGDClassifier(
                        loss="log_loss",
                        alpha=1e-4,
                        learning_rate="optimal",
                        max_iter=1,
                        tol=None,
                        random_state=42,
                    ),
                    is_initialized=False,
                )
        else:
            bundle = IncrementalModelBundle(
                scaler_reg=StandardScaler(with_mean=True, with_std=True),
                reg=SGDRegressor(
                    loss="huber",
                    alpha=1e-4,
                    learning_rate="invscaling",
                    eta0=0.01,
                    power_t=0.25,
                    max_iter=1,
                    tol=None,
                    random_state=42,
                ),
                scaler_clf=StandardScaler(with_mean=True, with_std=True),
                clf=SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    learning_rate="optimal",
                    max_iter=1,
                    tol=None,
                    random_state=42,
                ),
                is_initialized=False,
            )

        self.bundles[prop] = bundle
        return bundle

    def save(self, prop: str):
        bundle = self.bundles.get(prop)
        if not bundle:
            return
        p = self._paths(prop)
        joblib.dump(bundle.scaler_reg, p["scaler_reg"])
        joblib.dump(bundle.reg, p["reg"])
        joblib.dump(bundle.scaler_clf, p["scaler_clf"])
        joblib.dump(bundle.clf, p["clf"])

    def save_all(self):
        for prop in list(self.bundles.keys()):
            self.save(prop)

    def _maybe_save(self, prop: str):
        if self.save_every <= 1:
            self.save(prop)
            return
        n = int(self._updates_since_save.get(prop, 0)) + 1
        self._updates_since_save[prop] = n
        if n >= self.save_every:
            self.save(prop)
            self._updates_since_save[prop] = 0

    def partial_fit_reg(self, prop: str, X: np.ndarray, y: np.ndarray):
        b = self.load_or_init(prop)
        # scale (partial_fit scaler)
        b.scaler_reg.partial_fit(X)
        Xs = b.scaler_reg.transform(X)
        b.reg.partial_fit(Xs, y)
        b.is_initialized = True
        self._maybe_save(prop)

    def partial_fit_clf(self, prop: str, X: np.ndarray, y: np.ndarray):
        b = self.load_or_init(prop)
        b.scaler_clf.partial_fit(X)
        Xs = b.scaler_clf.transform(X)
        # Always pass classes on the first call (sklearn requirement).
        if not hasattr(b.clf, "classes_"):
            b.clf.partial_fit(Xs, y, classes=np.array([0, 1], dtype=int))
        else:
            b.clf.partial_fit(Xs, y)
        b.is_initialized = True
        self._maybe_save(prop)

    def predict(self, prop: str, X_reg: np.ndarray, X_clf: np.ndarray) -> dict[str, float] | None:
        b = self.load_or_init(prop)
        if not hasattr(b.clf, "classes_"):
            return None
        Xc = b.scaler_clf.transform(X_clf)
        proba = float(b.clf.predict_proba(Xc)[0, 1])
        pred = None
        try:
            Xr = b.scaler_reg.transform(X_reg)
            pred = float(b.reg.predict(Xr)[0])
        except Exception:
            pred = None
        return {"predicted_value": pred, "over_probability": proba}


def reg_feature_order() -> list[str]:
    return list(NUMERIC_FEATURE_KEYS)


def clf_feature_order() -> list[str]:
    return list(NUMERIC_FEATURE_KEYS) + list(CLASSIFIER_EXTRA_KEYS)


