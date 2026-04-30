"""Quantile-regression head for one-sided prediction intervals.

Why this matters
----------------
The mean-regression head (HistGradientBoosting / XGBoost) gives ``ŷ`` — a
point estimate. To bet OVER, what we actually care about is ``P(y > line)``.
We currently estimate that by passing ``ŷ`` plus the line into a separate
classifier head, but that's a noisy two-stage process.

Quantile regression sidesteps it: fit ``q_τ(x)`` directly, the value such
that ``P(y < q_τ(x)) = τ``. With τ ∈ {0.10, 0.50, 0.90}:

  - ``q_0.50`` = median, robust point estimate (better than mean for
    heavy-tailed prop distributions like 3PM where 0 is common)
  - ``q_0.90`` = "90% of the time the player scores below this" → if the
    line is below q_0.90, P(over) is at least 10% by construction
  - ``q_0.10`` = lower-bound, the symmetric thing for unders

The helper here wraps three HGB regressors into one object that returns a
dict per row. Tested on synthetic data; production training script can
swap it in by replacing the existing regressor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:  # pragma: no cover - sklearn is a hard dep
    HistGradientBoostingRegressor = None


@dataclass
class QuantileEnsemble:
    """Three independent HGB regressors at quantiles τ ∈ {low, mid, high}.

    Default τ = (0.10, 0.50, 0.90) gives a non-parametric 80% PI plus a
    median point estimate. ``predict_intervals(X)`` returns lower/median/upper
    arrays of shape (n,). Joint monotonicity is NOT enforced — sklearn's
    quantile loss can occasionally produce crossing predictions on tiny
    test sets; ``_repair_crossings`` sorts each row before returning so
    downstream code can rely on ``lower <= median <= upper``.
    """
    quantile_low: float = 0.10
    quantile_mid: float = 0.50
    quantile_high: float = 0.90
    learning_rate: float = 0.05
    max_iter: int = 200
    max_depth: int | None = 6
    random_state: int = 42
    # Optional integer array (n_features,) of {-1, 0, +1}. Same constraint
    # is applied to all three quantile heads — if more recent_avg is
    # supposed to push the median up, it should push the 10th and 90th
    # percentiles up too.
    monotonic_cst: "np.ndarray | None" = None

    def __post_init__(self):
        if HistGradientBoostingRegressor is None:
            raise ImportError("sklearn is required for QuantileEnsemble")
        if not 0 < self.quantile_low < self.quantile_mid < self.quantile_high < 1:
            raise ValueError(
                f"quantiles must be 0 < lo < mid < hi < 1, got "
                f"{self.quantile_low}/{self.quantile_mid}/{self.quantile_high}"
            )
        self._models = {}
        # Conformal-calibration state. ``_conformity_offset`` is the q̂ from
        # split conformal prediction (CQR, Romano et al. 2019); positive
        # widens the PI, negative tightens it. ``None`` means "not
        # calibrated yet", in which case predict_intervals returns the raw
        # quantile-regression bounds.
        self._conformity_offset: float | None = None
        self._conformal_alpha: float | None = None

    def _make(self, q: float) -> "HistGradientBoostingRegressor":
        kwargs = dict(
            loss="quantile",
            quantile=q,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        if self.monotonic_cst is not None:
            kwargs["monotonic_cst"] = self.monotonic_cst
        return HistGradientBoostingRegressor(**kwargs)

    def fit(self, X, y, sample_weight=None) -> "QuantileEnsemble":
        for name, q in (
            ("low", self.quantile_low),
            ("mid", self.quantile_mid),
            ("high", self.quantile_high),
        ):
            m = self._make(q)
            m.fit(X, y, sample_weight=sample_weight)
            self._models[name] = m
        # Mimic sklearn's regressor interface so this can be slotted into
        # downstream code that does estimator.feature_names_in_ inspection
        # for column alignment (see EnhancedMLPredictor._align_to_model).
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(list(X.columns))
        return self

    def predict(self, X) -> np.ndarray:
        """Drop-in regressor interface — returns the median forecast (q_mid).

        Why median rather than mean: heavy-tailed prop distributions (3PM, blocks,
        steals) have right-skewed real distributions where mean overshoots the
        bulk. Median is robust to those tails and produces a saner point estimate
        for downstream blending with the line-distance probability.
        """
        if not self._models:
            raise RuntimeError("call .fit() first")
        return self._models["mid"].predict(X)

    @staticmethod
    def _repair_crossings(lo, mid, hi):
        # Stack and sort each row so lo <= mid <= hi. Cheap; preserves
        # marginal calibration on the average row even when individual
        # quantile fits cross.
        stacked = np.stack([lo, mid, hi], axis=1)
        stacked.sort(axis=1)
        return stacked[:, 0], stacked[:, 1], stacked[:, 2]

    def _raw_intervals(self, X):
        """Quantile-regressor outputs without the conformity offset applied.

        Used by :meth:`calibrate` to derive the offset, and by
        :meth:`predict_intervals` which adds the offset on top.
        """
        if not self._models:
            raise RuntimeError("call .fit() first")
        lo = self._models["low"].predict(X)
        mid = self._models["mid"].predict(X)
        hi = self._models["high"].predict(X)
        return lo, mid, hi

    def calibrate(self, X, y, alpha: float | None = None) -> "QuantileEnsemble":
        """Split-conformal calibration of the (lo, hi) interval (CQR).

        Why this exists: quantile regression gives ``q_τ(x)`` *as fitted on
        training data*. On unseen data the actual coverage of [q_lo, q_hi]
        often deviates from ``1 - 2τ`` because of distribution shift, finite
        sample noise, and quantile crossing. CQR fixes this with a single
        held-out set:

          1. Compute conformity scores ``e_i = max(q_lo(x_i) - y_i,
             y_i - q_hi(x_i))`` on the calibration set. Negative ``e_i``
             means the point fell inside the band; positive means outside.
          2. Take the (1-α)(1+1/n)-quantile, q̂.
          3. At test time return ``[q_lo(x) - q̂, q_hi(x) + q̂]``. Marginal
             coverage is provably ≥ 1-α on iid test data from the same
             distribution.

        ``alpha`` defaults to ``1 - (quantile_high - quantile_low)`` so an
        ensemble at (0.10, 0.50, 0.90) calibrates an 80% PI by default.
        Persists ``_conformity_offset`` so :meth:`predict_intervals` and
        :meth:`prob_over` use it on subsequent calls.
        """
        if not self._models:
            raise RuntimeError("call .fit() first")
        if alpha is None:
            alpha = 1.0 - (self.quantile_high - self.quantile_low)
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        y_arr = np.asarray(y, dtype=float)
        lo_raw, _mid, hi_raw = self._raw_intervals(X)
        # Repair crossings before scoring — we don't want a flipped pair to
        # produce a spurious "outside" score.
        lo_raw, _mid, hi_raw = self._repair_crossings(lo_raw, _mid, hi_raw)
        e = np.maximum(lo_raw - y_arr, y_arr - hi_raw)
        n = len(e)
        if n < 2:
            # Not enough points for a meaningful quantile — leave uncalibrated.
            return self
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = min(max(k, 1), n)
        # ``np.partition`` is O(n); we just need the kth-smallest.
        self._conformity_offset = float(np.partition(e, k - 1)[k - 1])
        self._conformal_alpha = float(alpha)
        return self

    def predict_intervals(self, X) -> dict:
        """Return ``{"lower": np.ndarray, "median": np.ndarray, "upper": np.ndarray}``.

        If :meth:`calibrate` has been called, the lower/upper bands are
        widened (or tightened) by ``_conformity_offset`` to give marginal
        ``1 - alpha`` coverage on test data from the same distribution.
        """
        lo, mid, hi = self._raw_intervals(X)
        if self._conformity_offset is not None:
            lo = lo - self._conformity_offset
            hi = hi + self._conformity_offset
        lo, mid, hi = self._repair_crossings(lo, mid, hi)
        return {"lower": lo, "median": mid, "upper": hi}

    def prob_over(self, X, line: float | Sequence[float]) -> np.ndarray:
        """Crude calibrated P(y > line) using piecewise-linear interpolation
        across the three quantile points.

        We know ``CDF(lower) = q_low`` and ``CDF(upper) = q_high``. For a
        value ``L`` between them we interpolate; outside, we clamp. This is
        cruder than full conformal prediction but works well enough as a
        side probability without a separate classifier.
        """
        intervals = self.predict_intervals(X)
        lo, mid, hi = intervals["lower"], intervals["median"], intervals["upper"]
        line_arr = np.asarray(line, dtype=float)
        if line_arr.shape == ():
            line_arr = np.full_like(lo, float(line))
        # CDF at our three knots: q_low, q_mid, q_high
        q_low, q_mid, q_high = self.quantile_low, self.quantile_mid, self.quantile_high
        cdf = np.empty_like(lo)
        # Below lower
        below_lo = line_arr <= lo
        cdf[below_lo] = 0.0
        # Above upper
        above_hi = line_arr >= hi
        cdf[above_hi] = 1.0
        # Between lower and mid
        in_lo_mid = (~below_lo) & (line_arr <= mid)
        if in_lo_mid.any():
            denom = (mid - lo)[in_lo_mid]
            denom = np.where(denom > 1e-9, denom, 1.0)
            frac = (line_arr - lo)[in_lo_mid] / denom
            cdf[in_lo_mid] = q_low + frac * (q_mid - q_low)
        # Between mid and upper
        in_mid_hi = (~above_hi) & (line_arr > mid)
        if in_mid_hi.any():
            denom = (hi - mid)[in_mid_hi]
            denom = np.where(denom > 1e-9, denom, 1.0)
            frac = (line_arr - mid)[in_mid_hi] / denom
            cdf[in_mid_hi] = q_mid + frac * (q_high - q_mid)
        # P(y > line) = 1 - CDF(line)
        return 1.0 - cdf
