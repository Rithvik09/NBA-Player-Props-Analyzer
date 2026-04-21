"""Kelly math + bankroll ledger tests."""
from __future__ import annotations

import math

import pytest

from src.bankroll import (
    BankrollTracker,
    american_to_decimal,
    american_to_implied_prob,
    devig_two_way,
    kelly_stake,
)


# ---------------- odds conversions ----------------------------------------

def test_american_to_decimal_neg():
    assert math.isclose(american_to_decimal(-110), 1.9090909, rel_tol=1e-5)


def test_american_to_decimal_pos():
    assert math.isclose(american_to_decimal(+150), 2.5, rel_tol=1e-9)


def test_implied_prob_neg():
    # -110 → ~0.5238
    assert math.isclose(american_to_implied_prob(-110), 0.5238095, rel_tol=1e-5)


def test_devig_symmetric():
    # symmetric -110 / -110 → 50/50 after devig
    o, u = devig_two_way(-110, -110)
    assert math.isclose(o, 0.5, rel_tol=1e-6)
    assert math.isclose(u, 0.5, rel_tol=1e-6)


# ---------------- kelly stake ---------------------------------------------

def test_no_edge_zero_stake():
    ks = kelly_stake(our_prob=0.5, american_odds=-110, bankroll=1000)
    assert ks.stake_dollars == 0
    assert ks.full_kelly <= 0


def test_positive_edge_stake():
    # p=0.60 on -110 → decisive edge
    ks = kelly_stake(our_prob=0.60, american_odds=-110, bankroll=1000,
                     kelly_fraction=0.25, max_fraction=0.05)
    assert ks.edge > 0
    assert ks.full_kelly > 0
    assert 0 < ks.stake_dollars <= 50.0  # capped at 5%


def test_cap_limits_stake():
    # Huge probability → cap should bind
    ks = kelly_stake(our_prob=0.95, american_odds=+100, bankroll=1000,
                     kelly_fraction=1.0, max_fraction=0.05)
    assert ks.stake_dollars == 50.0


def test_ev_per_dollar_consistent():
    ks = kelly_stake(our_prob=0.55, american_odds=-110, bankroll=0)
    # EV/$1 = p*b - q
    b = american_to_decimal(-110) - 1
    expected = 0.55 * b - 0.45
    assert math.isclose(ks.ev_per_dollar, expected, rel_tol=1e-6)


# ---------------- bankroll ledger -----------------------------------------

def test_bankroll_win_flow(tmp_path):
    db = str(tmp_path / "bankroll.db")
    br = BankrollTracker(db)
    assert br.get_balance(default=1000.0) == 1000.0

    bet_id = br.record_bet(
        player_name="Jayson Tatum", prop_type="points", side="over",
        line=27.5, american_odds=-110, our_prob=0.6, stake=50,
    )
    assert br.get_balance() == 950.0  # debited

    result = br.settle(bet_id, "win")
    # profit on 50 at -110 ≈ 50 * 0.9091 = 45.45; balance back to 1000 + 45.45
    assert math.isclose(result["pnl"], 45.4545, rel_tol=1e-4)
    assert math.isclose(result["new_balance"], 995.4545 + 50.0, rel_tol=1e-4)


def test_bankroll_loss_flow(tmp_path):
    db = str(tmp_path / "bankroll.db")
    br = BankrollTracker(db)
    br.set_balance(500.0)
    bet_id = br.record_bet(
        player_name="x", prop_type="points", side="under",
        line=30, american_odds=+120, our_prob=0.45, stake=100,
    )
    assert br.get_balance() == 400.0
    res = br.settle(bet_id, "loss")
    assert res["pnl"] == -100.0
    assert res["new_balance"] == 400.0  # loss leaves it at 400


def test_bankroll_push_flow(tmp_path):
    db = str(tmp_path / "bankroll.db")
    br = BankrollTracker(db)
    br.set_balance(200.0)
    bet_id = br.record_bet(
        player_name="x", prop_type="rebounds", side="over",
        line=10.5, american_odds=-115, our_prob=0.5, stake=25,
    )
    res = br.settle(bet_id, "push")
    assert res["pnl"] == 0.0
    assert res["new_balance"] == 200.0


def test_double_settle_rejected(tmp_path):
    br = BankrollTracker(str(tmp_path / "bankroll.db"))
    br.set_balance(100.0)
    bid = br.record_bet(player_name=None, prop_type="p", side="o", line=1,
                        american_odds=-110, our_prob=0.6, stake=10)
    br.settle(bid, "win")
    with pytest.raises(ValueError):
        br.settle(bid, "loss")


def test_summary_rolls_up(tmp_path):
    br = BankrollTracker(str(tmp_path / "bankroll.db"))
    br.set_balance(1000.0)
    ids = []
    for i in range(4):
        ids.append(br.record_bet(player_name=f"p{i}", prop_type="x", side="o",
                                 line=10, american_odds=-110, our_prob=0.55,
                                 stake=50))
    br.settle(ids[0], "win")
    br.settle(ids[1], "loss")
    br.settle(ids[2], "push")
    # leave ids[3] open
    s = br.summary()
    assert s["settled_bets"] == 3
    assert s["wins"] == 1
    assert s["losses"] == 1
    assert s["pushes"] == 1
    assert s["open_bets"] == 1
