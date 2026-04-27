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
    kelly_stake_correlated,
    kelly_stake_three_outcome,
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


# ---------------- three-outcome Kelly ------------------------------------

def test_three_outcome_reduces_to_classic_when_no_push():
    a = kelly_stake_three_outcome(p_win=0.60, p_push=0.0, american_odds=-110,
                                   bankroll=1000, kelly_fraction=0.25)
    b = kelly_stake(our_prob=0.60, american_odds=-110, bankroll=1000,
                    kelly_fraction=0.25)
    assert math.isclose(a.full_kelly, b.full_kelly, rel_tol=1e-9)
    assert math.isclose(a.stake_dollars, b.stake_dollars, rel_tol=1e-6)


def test_three_outcome_push_dampens_stake():
    # Holding p_loss fixed, converting win mass into push mass shrinks Kelly
    # (push neither grows nor decays bankroll, but you give up positive EV).
    no_push = kelly_stake_three_outcome(p_win=0.60, p_push=0.00,
                                         american_odds=-110, bankroll=1000,
                                         kelly_fraction=1.0, max_fraction=1.0)
    with_push = kelly_stake_three_outcome(p_win=0.50, p_push=0.10,
                                           american_odds=-110, bankroll=1000,
                                           kelly_fraction=1.0, max_fraction=1.0)
    # Same p_loss=0.40, but with_push has lower p_win → smaller stake
    assert with_push.full_kelly < no_push.full_kelly
    assert with_push.full_kelly > 0


def test_three_outcome_no_edge_zero_stake():
    ks = kelly_stake_three_outcome(p_win=0.40, p_push=0.10,
                                    american_odds=-110, bankroll=1000)
    assert ks.stake_dollars == 0.0


def test_three_outcome_renormalises_inconsistent_probs():
    # p_win + p_push > 1 → should renormalise rather than crash
    ks = kelly_stake_three_outcome(p_win=0.7, p_push=0.6,
                                    american_odds=-110, bankroll=1000)
    # Should not raise; ev_per_dollar finite
    assert math.isfinite(ks.ev_per_dollar)


# ---------------- correlation-aware Kelly --------------------------------

def test_correlated_kelly_independence_matches_solo():
    import numpy as np
    bets = [
        {"prob": 0.60, "american_odds": -110},
        {"prob": 0.58, "american_odds": -110},
    ]
    ind = kelly_stake_correlated(bets, bankroll=1000,
                                  correlation_matrix=np.eye(2),
                                  kelly_fraction=0.25, max_fraction_per_bet=0.05,
                                  max_total_fraction=1.0, n_samples=10_000,
                                  seed=1)
    # Solo Kelly for each
    a = kelly_stake(our_prob=0.60, american_odds=-110, bankroll=1000,
                    kelly_fraction=0.25, max_fraction=0.05)
    b = kelly_stake(our_prob=0.58, american_odds=-110, bankroll=1000,
                    kelly_fraction=0.25, max_fraction=0.05)
    # MC noise ± a few percent of stake
    assert abs(ind[0]["stake_dollars"] - a.stake_dollars) <= 5.0
    assert abs(ind[1]["stake_dollars"] - b.stake_dollars) <= 5.0


def test_correlated_kelly_high_corr_shrinks():
    import numpy as np
    bets = [
        {"prob": 0.60, "american_odds": -110},
        {"prob": 0.60, "american_odds": -110},
    ]
    independent = kelly_stake_correlated(bets, bankroll=1000,
                                          correlation_matrix=np.eye(2),
                                          kelly_fraction=0.25,
                                          max_fraction_per_bet=0.05,
                                          max_total_fraction=1.0,
                                          n_samples=10_000, seed=2)
    corr_high = np.array([[1.0, 0.95], [0.95, 1.0]])
    correlated = kelly_stake_correlated(bets, bankroll=1000,
                                         correlation_matrix=corr_high,
                                         kelly_fraction=0.25,
                                         max_fraction_per_bet=0.05,
                                         max_total_fraction=1.0,
                                         n_samples=10_000, seed=2)
    # Highly correlated → joint stake should not exceed independent total
    sum_ind = sum(b["stake_fraction"] for b in independent)
    sum_corr = sum(b["stake_fraction"] for b in correlated)
    assert sum_corr <= sum_ind + 1e-6


def test_correlated_kelly_respects_total_cap():
    import numpy as np
    bets = [{"prob": 0.65, "american_odds": -110} for _ in range(5)]
    res = kelly_stake_correlated(bets, bankroll=1000,
                                  correlation_matrix=np.eye(5),
                                  kelly_fraction=1.0,
                                  max_fraction_per_bet=0.10,
                                  max_total_fraction=0.20,
                                  n_samples=5_000, seed=3)
    total_frac = sum(b["stake_fraction"] for b in res)
    assert total_frac <= 0.20 + 1e-6


def test_correlated_kelly_empty_input():
    assert kelly_stake_correlated([], bankroll=1000) == []


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


def test_schema_migrations_idempotent(tmp_path):
    """Re-instantiating BankrollTracker on the same DB must be a no-op."""
    import sqlite3
    from src.bankroll import _TARGET_VERSION
    db = str(tmp_path / "bankroll.db")
    BankrollTracker(db)
    BankrollTracker(db)  # second time — must not raise
    BankrollTracker(db)  # third time — must not raise

    conn = sqlite3.connect(db)
    v = conn.execute("PRAGMA user_version").fetchone()[0]
    assert v == _TARGET_VERSION

    cols = [r[1] for r in conn.execute("PRAGMA table_info(bankroll_bets)").fetchall()]
    # New columns from v2/v3 should be present
    for must in ("kelly_fraction", "edge", "ev_per_dollar",
                  "prediction_log_id", "model_version"):
        assert must in cols, f"missing column {must}"
    conn.close()


def test_schema_migrates_legacy_v1_database(tmp_path):
    """Bootstrap a v1-only DB by hand, then open via tracker → should fast-forward."""
    import sqlite3
    from src.bankroll import _TARGET_VERSION
    db = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE bankroll_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            balance REAL NOT NULL,
            currency TEXT NOT NULL DEFAULT 'USD',
            updated_utc TEXT NOT NULL
        );
        CREATE TABLE bankroll_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placed_utc TEXT NOT NULL,
            player_name TEXT, prop_type TEXT, side TEXT,
            line REAL, american_odds REAL, our_prob REAL,
            stake REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            result TEXT, pnl REAL, settled_utc TEXT
        );
        PRAGMA user_version = 1;
    """)
    conn.commit()
    conn.close()

    BankrollTracker(db)  # should run v2, v3 migrations

    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == _TARGET_VERSION
    cols = [r[1] for r in conn.execute("PRAGMA table_info(bankroll_bets)").fetchall()]
    assert "kelly_fraction" in cols
    assert "model_version" in cols
    conn.close()


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
