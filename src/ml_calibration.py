from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class IsotonicCalibratedModel:
    """
    Wrap a probabilistic classifier with an isotonic calibrator.

    Persistable via joblib.
    """

    base_estimator: object
    calibrator: IsotonicRegression

    def predict_proba(self, X):
        # base probability of class 1
        p = self.base_estimator.predict_proba(X)[:, 1]
        p_cal = self.calibrator.transform(p)
        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.vstack([1 - p_cal, p_cal]).T


