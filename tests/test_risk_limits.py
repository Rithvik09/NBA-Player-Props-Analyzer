"""Tests for batch-3 risk circuit breakers (drawdown, exposure, concentration)
and the CRRA-utility Kelly variant."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bankroll import (  # noqa: E402
    BankrollTracker,
    RiskLimitError,
    RiskLimits,
    kelly_stake,
    kelly_stake_crra,
)


# ---------------------------------------------------------------- drawdown
def test_peak_balance_seeded_on_first_set(tmp_path):
    bt = BankrollTracker(str(tmp_path / "b.db"))
    bt.set_balance(1000.0)
    assert bt.get_peak_balance() == 1000.0


def test_peak_only_moves_up(tmp_path):
    bt = BankrollTracker(str(tmp_path / "b.db"))
    bt.set_balance(1000.0)
    bt.set_balance(800.0)  # downward
    assert bt.get_peak_balance() == 1000.0  # unchanged
    bt.set_balance(1500.0)  # new high
    assert bt.get_peak_balance() == 1500.0


def test_drawdown_zero_at_peak(tmp_path):
    bt = BankrollTracker(str(tmp_path / "b.db"))
    bt.set_balance(1000.0)
    assert bt.get_drawdown()["drawdown"] == 0.0


def test_drawdown_after_losses(tmp_path):
    bt = BankrollTracker(str(tmp_path / "b.db"))
    bt.set_balance(1000.0)
    bt.set_balance(750.0)
    dd = bt.get_drawdown()
    assert dd["peak"] == 1000.0
    assert dd["current"] == 750.0
    assert abs(dd["drawdown"] - 0.25) < 1e-9


def test_drawdown_circuit_breaker_blocks_new_bet(tmp_path):
    rl = RiskLimits(max_drawdown=0.20)
    bt = BankrollTracker(str(tmp_path / "b.db"), risk_limits=rl)
    bt.set_balance(1000.0)
    bt.set_balance(750.0)  # 25% drawdown — over the 20% cap
    with pytest.raises(RiskLimitError, match="drawdown"):
        bt.record_bet(player_name="X", prop_type="points", side="over",
                      line=20.5, american_odds=-110, our_prob=0.6, stake=10)


def test_drawdown_breaker_lifts_when_balance_recovers(tmp_path):
    rl = RiskLimits(max_drawdown=0.20)
    bt = BankrollTracker(str(tmp_path / "b.db"), risk_limits=rl)
    bt.set_balance(1000.0)
    bt.set_balance(750.0)  # blocked
    with pytest.raises(RiskLimitError):
        bt.record_bet(player_name="X", prop_type="points", side="over",
                      line=20.5, american_odds=-110, our_prob=0.6, stake=10)
    bt.set_balance(900.0)  # 10% drawdown — under cap
    bid = bt.record_bet(player_name="X", prop_type="points", side="over",
                        line=20.5, american_odds=-110, our_prob=0.6, stake=10)
    assert bid > 0


# -------------------------------------------------------- daily exposure
def test_daily_exposure_cap_blocks_overshoot(tmp_path):
    rl = RiskLimits(max_daily_exposure_pct=0.10)  # 10% of bankroll/day
    bt = BankrollTracker(str(tmp_path / "b.db"), risk_limits=rl)
    bt.set_balance(1000.0)
    # First bet at $80 — fine (8%)
    bt.record_bet(player_name="A", prop_type="points", side="over",
                  line=20.5, american_odds=-110, our_prob=0.55, stake=80)
    # Second bet at $30 — pushes to $110 = 11%, blocked
    with pytest.raises(RiskLimitError, match="daily exposure"):
        bt.record_bet(player_name="B", prop_type="points", side="over",
                      line=22.5, american_odds=-110, our_prob=0.55, stake=30)


def test_daily_exposure_cap_allows_at_limit(tmp_path):
    rl = RiskLimits(max_daily_exposure_pct=0.10)
    bt = BankrollTracker(str(tmp_path / "b.db"), risk_limits=rl)
    bt.set_balance(1000.0)
    # Exactly at the limit ($100) — allowed
    bid = bt.record_bet(player_name="A", prop_type="points", side="over",
                        line=20.5, american_odds=-110, our_prob=0.55, stake=100)
    assert bid > 0


# ----------------------------------------------------- player concentration
def test_player_concentration_cap_blocks(tmp_path):
    rl = RiskLimits(max_player_concentration_pct=0.05)  # 5% per player
    bt = BankrollTracker(str(tmp_path / "b.db"), risk_limits=rl)
    bt.set_balance(1000.0)
    bt.record_bet(player_name="Tatum", prop_type="points", side="over",
                  line=27.5, american_odds=-110, our_prob=0.6, stake=40)
    # Second Tatum bet pushes to $70 = 7%, blocked
    with pytest.raises(RiskLimitError, match="concentration"):
        bt.record_bet(player_name="Tatum", prop_type="rebounds", side="over",
                      line=8.5, american_odds=-110, our_prob=0.55, stake=30)


def test_player_concentration_only_counts_open(tmp_path):
    rl = RiskLimits(max_player_concentration_pct=0.05)
    bt = BankrollTracker(str(tmp_path / "b.db"), risk_limits=rl)
    bt.set_balance(1000.0)
    bid = bt.record_bet(player_name="Tatum", prop_type="points", side="over",
                        line=27.5, american_odds=-110, our_prob=0.6, stake=40)
    bt.settle(bid, "win")  # closes the slot — Tatum exposure now $0
    # Next Tatum bet should pass since prior is settled
    bid2 = bt.record_bet(player_name="Tatum", prop_type="rebounds", side="over",
                         line=8.5, american_odds=-110, our_prob=0.55, stake=40)
    assert bid2 > 0


def test_concentration_disabled_when_no_limit(tmp_path):
    bt = BankrollTracker(str(tmp_path / "b.db"))  # no limits
    bt.set_balance(1000.0)
    # Stack massive Tatum exposure — must allow when no cap
    for _ in range(5):
        bt.record_bet(player_name="Tatum", prop_type="points", side="over",
                      line=27.5, american_odds=-110, our_prob=0.6, stake=100)


# ------------------------------------------------------------ env-driven
def test_risk_limits_from_env(monkeypatch):
    monkeypatch.setenv("MAX_DRAWDOWN", "0.30")
    monkeypatch.setenv("MAX_DAILY_EXPOSURE_PCT", "0.15")
    monkeypatch.delenv("MAX_PLAYER_CONCENTRATION_PCT", raising=False)
    rl = RiskLimits.from_env()
    assert rl.max_drawdown == 0.30
    assert rl.max_daily_exposure_pct == 0.15
    assert rl.max_player_concentration_pct is None


def test_risk_limits_from_env_ignores_garbage(monkeypatch):
    monkeypatch.setenv("MAX_DRAWDOWN", "not-a-number")
    rl = RiskLimits.from_env()
    assert rl.max_drawdown is None


# --------------------------------------------------------------- CRRA Kelly
def test_crra_gamma_one_matches_log_kelly():
    # γ=1 → log-utility → standard Kelly
    crra = kelly_stake_crra(0.58, -110, 1000.0, risk_aversion=1.0, max_fraction=1.0)
    log_kelly = kelly_stake(0.58, -110, 1000.0, kelly_fraction=1.0,
                            max_fraction=1.0)
    assert abs(crra.full_kelly - log_kelly.full_kelly) < 1e-6


def test_crra_higher_gamma_smaller_stake():
    base = kelly_stake_crra(0.58, -110, 1000.0, risk_aversion=1.0, max_fraction=1.0)
    averse = kelly_stake_crra(0.58, -110, 1000.0, risk_aversion=4.0, max_fraction=1.0)
    very_averse = kelly_stake_crra(0.58, -110, 1000.0, risk_aversion=10.0,
                                    max_fraction=1.0)
    # Strictly monotone: more risk aversion → smaller stake fraction
    assert base.stake_fraction > averse.stake_fraction > very_averse.stake_fraction


def test_crra_zero_stake_at_no_edge():
    # 0.45 < break-even at -110 (~0.5238) — no edge, stake must be zero
    out = kelly_stake_crra(0.45, -110, 1000.0, risk_aversion=2.0)
    assert out.stake_dollars == 0.0


def test_crra_respects_max_fraction_cap():
    out = kelly_stake_crra(0.80, -110, 1000.0, risk_aversion=1.0, max_fraction=0.05)
    assert out.stake_fraction <= 0.05 + 1e-9
