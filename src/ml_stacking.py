"""Stacking blender — combine multiple base classifiers via a logistic head.

Why stack instead of just averaging? Different boosters make different
errors on different feature regimes (HGB tends to underfit thin tails,
XGBoost can overfit small classes). A logistic blender learns which
booster to trust *as a function of* the features that matter — typically
``line_distance`` (how far the line is from the player's mean) and the
prediction itself.

The blender is fit on a held-out set of base-model probabilities so it
doesn't see training data twice. At serve time it takes the same per-base
probabilities + meta features and outputs a single calibrated probability.

Tested only with sklearn; no XGBoost dependency in the stacker itself
(it just consumes whatever probabilities are passed in).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    LogisticRegression = None


@dataclass
class StackingBlender:
    """Logistic-regression second stage on base classifier outputs.

    Inputs at fit time:
      - ``base_probs``: shape (n, k) — k base classifier probabilities
      - ``meta``: shape (n, m) — m meta features (e.g. line_distance, edge)
      - ``y``: shape (n,) — 0/1 labels

    The blender concatenates ``[base_probs, meta]`` and fits an L2-regularised
    logistic. At serve time the same concatenation feeds ``predict_proba``.
    """
    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 42
    base_names: list[str] = field(default_factory=list)
    meta_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        if LogisticRegression is None:
            raise ImportError("sklearn is required for StackingBlender")
        self._model: LogisticRegression | None = None

    @staticmethod
    def _stack(base_probs: np.ndarray, meta: np.ndarray | None) -> np.ndarray:
        bp = np.asarray(base_probs, dtype=float)
        if bp.ndim == 1:
            bp = bp.reshape(-1, 1)
        if meta is None:
            return bp
        m = np.asarray(meta, dtype=float)
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        if m.shape[0] != bp.shape[0]:
            raise ValueError(
                f"base_probs and meta row count mismatch: {bp.shape[0]} vs {m.shape[0]}"
            )
        return np.hstack([bp, m])

    def fit(
        self,
        base_probs: np.ndarray,
        y: np.ndarray,
        meta: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "StackingBlender":
        X = self._stack(base_probs, meta)
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise ValueError(
                "stacking blender needs both classes in fit set; got only "
                f"{np.unique(y_arr).tolist()}"
            )
        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self._model.fit(X, y_arr, sample_weight=sample_weight)
        return self

    def predict_proba(
        self,
        base_probs: np.ndarray,
        meta: np.ndarray | None = None,
    ) -> np.ndarray:
        """Returns shape (n,) probability of class 1 (the "over wins" event)."""
        if self._model is None:
            raise RuntimeError("call .fit() first")
        X = self._stack(base_probs, meta)
        return self._model.predict_proba(X)[:, 1]

    @property
    def coefficients(self) -> dict:
        """Inspect what the blender learned. Useful for sanity-checking that
        the blender isn't ignoring a base classifier."""
        if self._model is None:
            raise RuntimeError("call .fit() first")
        names = list(self.base_names) + list(self.meta_names)
        coefs = self._model.coef_[0]
        # Only return names if dimensions match — otherwise emit anonymous keys
        if len(names) == coefs.shape[0]:
            return {n: float(c) for n, c in zip(names, coefs)}
        return {f"x{i}": float(c) for i, c in enumerate(coefs)}


class StackedCalibratedClassifier:
    """Sklearn-shaped adapter exposing ``predict_proba(X) -> (n, 2)`` over a
    StackingBlender on top of two calibrated base classifiers.

    Why this wrapper exists: downstream inference code (e.g.
    ``EnhancedMLPredictor.predict_prop`` in src/models.py) does
    ``classifier.predict_proba(X)[:, 1]`` and reads ``feature_names_in_``
    for column alignment. By mirroring that interface, the blender slots
    into existing inference paths with zero call-site changes.

    The two base classifiers are expected to be FITTED + CALIBRATED already
    (we don't re-fit them). The blender consumes their probabilities for
    class 1 and produces a single combined probability that we expand into
    the (n, 2) form sklearn callers expect.
    """

    def __init__(self, base_a, base_b, blender: "StackingBlender"):
        self.base_a = base_a
        self.base_b = base_b
        self.blender = blender
        # Inherit feature names from base_a — both bases see the same X
        names = getattr(base_a, "feature_names_in_", None)
        if names is None:
            inner = getattr(base_a, "estimator", getattr(base_a, "base_estimator", None))
            names = getattr(inner, "feature_names_in_", None)
        if names is not None:
            self.feature_names_in_ = np.asarray(names)
        # Sklearn duck-typing: many internal helpers check ``classes_``
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        p_a = self.base_a.predict_proba(X)[:, 1]
        p_b = self.base_b.predict_proba(X)[:, 1]
        bp = np.column_stack([p_a, p_b])
        p_combined = self.blender.predict_proba(bp)
        return np.column_stack([1.0 - p_combined, p_combined])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
