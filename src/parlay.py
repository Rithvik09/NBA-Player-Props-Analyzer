"""Parlay probability + expected-value estimator via Gaussian copula.

The naive parlay probability — ``∏ p_i`` — assumes legs are independent.
They never are: two props on the same player correlate (a big-scoring night
tends to drive rebounds + assists too), and props on teammates correlate
weakly via pace. The Gaussian copula lets us bake that in.

Inputs
------
Each leg is a ``ParlayLeg``:
    prob          — model's estimated probability THIS leg hits
    american_odds — sportsbook price on that leg
    player_id, team_id, prop  — identifiers used to look up a correlation prior

Output
------
    {
        "naive_prob":       ∏ p_i  — what a bookie assumes (or wants you to),
        "correlated_prob":  copula estimate,
        "parlay_decimal":   product of leg decimal odds,
        "ev_per_dollar":    correlated_prob · (parlay_decimal − 1) − (1 − correlated_prob),
    }

The correlation prior defaults come from published NBA prop-market research
(e.g. same-player PTS↔AST ≈ 0.35). Replace ``CORRELATION_PRIORS`` with
empirical residuals from your graded history once you have enough rows.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correlation priors
# ---------------------------------------------------------------------------
# Key = (scope, prop_a, prop_b)  with prop ordering alphabetically. ``scope``
# is 'same_player', 'same_team', or 'opp_team'. Missing pairs default to 0.
CORRELATION_PRIORS: dict[tuple[str, str, str], float] = {
    # Same-player bundle — strongly positive: a good night drives everything.
    ("same_player", "points",    "rebounds"):       0.25,
    ("same_player", "assists",   "points"):         0.35,
    ("same_player", "points",    "three_pointers"): 0.40,
    ("same_player", "assists",   "rebounds"):       0.15,
    ("same_player", "blocks",    "rebounds"):       0.35,
    ("same_player", "steals",    "points"):         0.15,
    ("same_player", "points",    "turnovers"):      0.10,
    ("same_player", "assists",   "turnovers"):      0.30,
    ("same_player", "pts_reb",   "rebounds"):       0.70,
    ("same_player", "points",    "pts_reb"):        0.80,
    ("same_player", "assists",   "pts_ast"):        0.75,
    ("same_player", "points",    "pts_ast"):        0.80,
    ("same_player", "pts_ast_reb", "rebounds"):     0.60,
    ("same_player", "ast_reb",   "rebounds"):       0.65,
    ("same_player", "double_double", "rebounds"):   0.50,
    ("same_player", "double_double", "points"):     0.45,
    ("same_player", "triple_double", "assists"):    0.55,

    # Same-team: pace boost lifts teammates, but a hogged-ball night hurts
    # others' assists. Small positive net.
    ("same_team", "points",   "points"):    0.08,
    ("same_team", "rebounds", "rebounds"):  0.05,
    ("same_team", "assists",  "points"):    0.10,
    ("same_team", "assists",  "assists"):  -0.15,  # ball-distribution fight

    # Opposing team: a blowout suppresses one side, lifts the other on garbage
    # time — net near zero. Small negative for volume props.
    ("opp_team", "points",   "points"):   -0.05,
    ("opp_team", "rebounds", "rebounds"): -0.03,
}


def _canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Empirical overlay: scripts/fit_parlay_correlations.py writes JSON of
# residual-correlations from graded predictions. If present, these override
# hand-tuned priors per pair.
# ---------------------------------------------------------------------------

EMPIRICAL_CORRELATIONS: dict[tuple[str, str, str], float] = {}
EMPIRICAL_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "empirical_correlations.json",
)


def load_empirical_correlations(path: str | None = None) -> int:
    """Load fitted correlations from disk. Returns count loaded.

    Idempotent: re-loading replaces the in-memory overlay.
    """
    global EMPIRICAL_CORRELATIONS
    p = path or EMPIRICAL_PATH_DEFAULT
    if not os.path.exists(p):
        EMPIRICAL_CORRELATIONS = {}
        return 0
    try:
        with open(p) as f:
            blob = json.load(f)
    except (OSError, ValueError) as e:
        log.warning("[parlay] failed to load empirical correlations: %s", e)
        EMPIRICAL_CORRELATIONS = {}
        return 0
    pairs = blob.get("pairs", {}) if isinstance(blob, dict) else {}
    overlay: dict[tuple[str, str, str], float] = {}
    for key, val in pairs.items():
        try:
            scope, a, b = key.split("|")
            r = float(val.get("r", 0.0)) if isinstance(val, dict) else float(val)
            overlay[(scope, a.lower(), b.lower())] = r
        except (ValueError, TypeError, AttributeError):
            continue
    EMPIRICAL_CORRELATIONS = overlay
    log.info("[parlay] loaded %d empirical correlations from %s", len(overlay), p)
    return len(overlay)


# Try once at import — fails silently if file absent
try:
    load_empirical_correlations()
except Exception:  # noqa: BLE001
    pass


def default_correlation(scope: str, prop_a: str, prop_b: str) -> float:
    """Look up prior, with empirical overlay taking precedence.

    Resolution order:
      1. EMPIRICAL_CORRELATIONS (fitted from graded history)
      2. CORRELATION_PRIORS (hand-tuned)
      3. Same-prop-same-player → 1.0
      4. 0.0
    """
    a, b = _canonical(prop_a.lower(), prop_b.lower())
    v = EMPIRICAL_CORRELATIONS.get((scope, a, b))
    if v is not None:
        return v
    v = CORRELATION_PRIORS.get((scope, a, b))
    if v is not None:
        return v
    # Same-prop on same-player = perfect correlation (same event)
    if scope == "same_player" and a == b:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Leg model
# ---------------------------------------------------------------------------

@dataclass
class ParlayLeg:
    prob: float
    american_odds: float
    prop: str
    player_id: int | None = None
    team_id: int | None = None
    side: str = "over"


def _scope(leg_a: ParlayLeg, leg_b: ParlayLeg) -> str:
    if leg_a.player_id is not None and leg_a.player_id == leg_b.player_id:
        return "same_player"
    if leg_a.team_id is not None and leg_a.team_id == leg_b.team_id:
        return "same_team"
    # Heuristic: if team ids differ, treat as opp_team IFF both sides set
    if (leg_a.team_id is not None and leg_b.team_id is not None
            and leg_a.team_id != leg_b.team_id):
        return "opp_team"
    return "unrelated"


def build_correlation_matrix(legs: Sequence[ParlayLeg]) -> np.ndarray:
    n = len(legs)
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            scope = _scope(legs[i], legs[j])
            if scope == "unrelated":
                m[i, j] = m[j, i] = 0.0
            else:
                side_sign = 1.0 if legs[i].side == legs[j].side else -1.0
                m[i, j] = m[j, i] = side_sign * default_correlation(
                    scope, legs[i].prop, legs[j].prop
                )
    return _project_psd(m)


def _project_psd(m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Nudge a near-PSD correlation matrix to PSD by floor-clipping eigenvalues."""
    w, v = np.linalg.eigh(m)
    if np.any(w < eps):
        w = np.maximum(w, eps)
        m = v @ np.diag(w) @ v.T
        # re-normalise diagonal to 1
        d = np.sqrt(np.diag(m))
        m = m / np.outer(d, d)
    return m


# ---------------------------------------------------------------------------
# Copula simulation
# ---------------------------------------------------------------------------

def _std_normal_ppf(p: float) -> float:
    """Φ⁻¹ — no scipy dependency, Beasley–Springer–Moro."""
    # Clamp to keep finite
    p = max(1e-9, min(1 - 1e-9, float(p)))
    # Wichura AS241 approximation (good to ~1e-10)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        num = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        den = (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1
        return num / den
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        num = ((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]
        den = (((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1
    return num / den


def parlay_probability(
    legs: Sequence[ParlayLeg],
    n_samples: int = 20_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Correlation-aware parlay hit probability via Gaussian copula MC."""
    if not legs:
        return {"naive_prob": 0.0, "correlated_prob": 0.0,
                "parlay_decimal": 0.0, "ev_per_dollar": 0.0}

    rng = rng or np.random.default_rng()
    probs = np.array([max(1e-9, min(1 - 1e-9, l.prob)) for l in legs])
    # Z-thresholds: sample leg i "hits" iff Z_i <= Φ⁻¹(p_i)
    thresholds = np.array([_std_normal_ppf(p) for p in probs])

    corr = build_correlation_matrix(legs)
    # Cholesky for sampling correlated standard normals
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Fallback: eigendecomp (handles PSD with zero eigenvalue)
        w, v = np.linalg.eigh(corr)
        L = v @ np.diag(np.sqrt(np.maximum(w, 0)))

    z = rng.standard_normal(size=(n_samples, len(legs))) @ L.T
    hits = (z <= thresholds).all(axis=1)
    correlated_prob = float(hits.mean())

    naive_prob = float(np.prod(probs))
    # Parlay payout = product of decimal odds
    from .bankroll import american_to_decimal
    parlay_decimal = float(np.prod([american_to_decimal(l.american_odds) for l in legs]))
    ev_per_dollar = correlated_prob * (parlay_decimal - 1) - (1 - correlated_prob)

    return {
        "naive_prob": naive_prob,
        "correlated_prob": correlated_prob,
        "parlay_decimal": parlay_decimal,
        "ev_per_dollar": ev_per_dollar,
        "correlation_matrix": corr.tolist(),
        "n_samples": n_samples,
    }
