"""Tests for v4 schema (CLV columns), Shin's devig, and bootstrap PIs."""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bankroll import (  # noqa: E402
    BankrollTracker,
    american_to_implied_prob,
    devig_shin,
    devig_two_way,
)
from src.uncertainty import (  # noqa: E402
    bootstrap_pi,
    prob_over_from_samples,
    residual_quantile_interval,
)


# -------------------------------------------------------------------- CLV
def test_record_closing_line_computes_clv(tmp_path):
    db = str(tmp_path / "bankroll.db")
    bt = BankrollTracker(db)
    bt.set_balance(1000.0)
    bet_id = bt.record_bet(
        player_name="Tatum", prop_type="points", side="over", line=27.5,
        american_odds=-110, our_prob=0.60, stake=50.0,
    )
    out = bt.record_closing_line(bet_id, closing_line=27.5, closing_odds=-130)
    # closing -130 → implied 0.5652; our_prob 0.60 → CLV ≈ 0.0348
    assert 0.025 < out["clv"] < 0.045

    # Persisted on the row
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT closing_line, closing_odds, clv FROM bankroll_bets WHERE id = ?",
        (bet_id,),
    ).fetchone()
    conn.close()
    assert row[0] == 27.5
    assert row[1] == -130.0
    assert row[2] is not None


def test_clv_summary_aggregates(tmp_path):
    db = str(tmp_path / "bankroll.db")
    bt = BankrollTracker(db)
    bt.set_balance(1000.0)
    for odds_close in (-130, -150):
        bid = bt.record_bet(
            player_name="X", prop_type="points", side="over", line=20.5,
            american_odds=-110, our_prob=0.60, stake=10,
        )
        bt.record_closing_line(bid, closing_odds=odds_close, closing_line=20.5)
    s = bt.clv_summary()
    assert s["n"] == 2
    assert s["mean_clv"] > 0


def test_v4_schema_idempotent(tmp_path):
    db = str(tmp_path / "bankroll.db")
    BankrollTracker(db)
    BankrollTracker(db)  # second init must not error
    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(bankroll_bets)").fetchall()}
    conn.close()
    for c in ("closing_line", "closing_odds", "clv"):
        assert c in cols


# -------------------------------------------------------------------- Shin
def test_shin_devig_sums_to_one():
    p_o, p_u, z = devig_shin(-110, -110)
    # Symmetric input — falls back to multiplicative; both 0.5
    assert abs(p_o + p_u - 1.0) < 1e-9
    assert abs(p_o - 0.5) < 1e-9


def test_shin_devig_asymmetric_market():
    # -200 / +170 is a typical asymmetric prop
    p_o, p_u, z = devig_shin(-200, +170)
    assert abs(p_o + p_u - 1.0) < 1e-6
    # The favourite (over @ -200) should have a fair prob less than its raw
    # implied (since Shin attributes more vig to the favourite)
    raw_o = american_to_implied_prob(-200)
    assert p_o < raw_o
    assert 0.0 <= z < 1.0


def test_shin_vs_multiplicative_differ_on_asymmetric():
    sm_o, _ = devig_two_way(-200, +170)
    sh_o, _, _ = devig_shin(-200, +170)
    # Shin and multiplicative must disagree (otherwise Shin adds nothing)
    assert abs(sm_o - sh_o) > 1e-4


# -------------------------------------------------------------------- PIs
def test_residual_quantile_interval_brackets_point():
    residuals = np.random.default_rng(0).normal(0, 3, size=200).tolist()
    lo, hi = residual_quantile_interval(25.0, residuals, alpha=0.10)
    assert lo < 25.0 < hi
    # 90% PI for N(0, 3) ≈ ±1.645·3 ≈ ±4.94
    assert 3.5 < (hi - lo) / 2 < 6.5


def test_residual_quantile_interval_short_residuals_fallback():
    lo, hi = residual_quantile_interval(25.0, [], alpha=0.10)
    assert lo < 25.0 < hi


def test_residual_quantile_interval_alpha_validation():
    with pytest.raises(ValueError):
        residual_quantile_interval(0, [0, 1, 2], alpha=1.5)


def test_bootstrap_pi_brackets_point_and_emits_samples():
    residuals = np.random.default_rng(1).normal(0, 4, size=300).tolist()
    out = bootstrap_pi(30.0, residuals, alpha=0.10, n_boot=500)
    assert out["lower"] < 30.0 < out["upper"]
    assert out["samples"].shape == (500,)


def test_prob_over_from_samples_monotone():
    samples = np.array([10, 20, 25, 30, 40])
    p_lo = prob_over_from_samples(samples, line=15)
    p_hi = prob_over_from_samples(samples, line=35)
    assert p_lo > p_hi
    assert p_hi == 0.2  # only 40 > 35
