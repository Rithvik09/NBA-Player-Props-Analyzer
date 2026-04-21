"""Parlay copula tests."""
from __future__ import annotations

import math

import numpy as np

from src.parlay import (
    ParlayLeg,
    build_correlation_matrix,
    default_correlation,
    parlay_probability,
    _std_normal_ppf,
)


def test_ppf_tails():
    # Φ⁻¹ should be anti-symmetric about 0.5
    assert abs(_std_normal_ppf(0.5)) < 1e-6
    assert abs(_std_normal_ppf(0.95) + _std_normal_ppf(0.05)) < 1e-3


def test_independent_matches_product():
    legs = [
        ParlayLeg(prob=0.6, american_odds=-110, prop="points",
                  player_id=1, team_id=10),
        ParlayLeg(prob=0.5, american_odds=-110, prop="rebounds",
                  player_id=2, team_id=20),
    ]
    r = parlay_probability(legs, n_samples=30_000,
                           rng=np.random.default_rng(42))
    # Independent legs → correlated ≈ naive
    assert math.isclose(r["correlated_prob"], r["naive_prob"], abs_tol=0.01)


def test_same_player_positive_corr_boosts():
    legs = [
        ParlayLeg(prob=0.6, american_odds=-110, prop="points",
                  player_id=1, team_id=10),
        ParlayLeg(prob=0.55, american_odds=-110, prop="assists",
                  player_id=1, team_id=10),
    ]
    r = parlay_probability(legs, n_samples=30_000,
                           rng=np.random.default_rng(7))
    # Same-player PTS+AST correlation ~0.35 → correlated > naive
    assert r["correlated_prob"] > r["naive_prob"]


def test_opposite_sides_anti_correlate():
    legs = [
        ParlayLeg(prob=0.6, american_odds=-110, prop="points",
                  player_id=1, team_id=10, side="over"),
        ParlayLeg(prob=0.6, american_odds=-110, prop="assists",
                  player_id=1, team_id=10, side="under"),
    ]
    r = parlay_probability(legs, n_samples=30_000,
                           rng=np.random.default_rng(11))
    # Over points + under assists — opposite sides flip the sign → correlated < naive
    assert r["correlated_prob"] < r["naive_prob"]


def test_correlation_matrix_psd():
    legs = [ParlayLeg(prob=0.5, american_odds=-110, prop=p, player_id=1, team_id=1)
            for p in ("points", "rebounds", "assists", "steals")]
    m = build_correlation_matrix(legs)
    w = np.linalg.eigvalsh(m)
    assert (w >= -1e-6).all()
    # Diagonals exactly 1
    for i in range(len(legs)):
        assert math.isclose(m[i, i], 1.0, rel_tol=1e-9)


def test_default_correlation_symmetric():
    a = default_correlation("same_player", "points", "rebounds")
    b = default_correlation("same_player", "rebounds", "points")
    assert a == b


def test_empty_legs():
    r = parlay_probability([])
    assert r["correlated_prob"] == 0.0
