"""Kelly-criterion stake sizing + bankroll tracker.

The model's ``over_probability`` is the edge input. Combine with the sportsbook
price to get the *fractional* Kelly stake that maximises expected log-growth:

    b        = decimal odds − 1         (net payoff on a 1-unit stake)
    p        = our probability
    q        = 1 − p
    f*       = (b·p − q) / b            (full Kelly)
    stake_$  = bankroll · kelly_fraction · max(0, f*)

We default to ``kelly_fraction = 0.25`` (quarter-Kelly) — full-Kelly is very
volatile in a world where p is estimated, not known.

Also exports a tiny SQLite bankroll-tracker used by ``/bankroll`` endpoints.
"""
from __future__ import annotations

import math
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable


# ---------------------------------------------------------------------------
# Odds conversions
# ---------------------------------------------------------------------------

def american_to_decimal(american: float) -> float:
    """Convert American odds (-110, +150, ...) to decimal (1.909, 2.5, ...)."""
    american = float(american)
    if american == 0:
        raise ValueError("american odds cannot be 0")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def american_to_implied_prob(american: float) -> float:
    """Implied probability baked into American odds (before vig removal)."""
    dec = american_to_decimal(american)
    return 1.0 / dec


def devig_two_way(over_american: float, under_american: float) -> tuple[float, float]:
    """Remove the vig by normalising two implied probs to sum to 1."""
    o = american_to_implied_prob(over_american)
    u = american_to_implied_prob(under_american)
    total = o + u
    if total <= 0:
        return 0.5, 0.5
    return o / total, u / total


# ---------------------------------------------------------------------------
# Kelly stake sizing
# ---------------------------------------------------------------------------

@dataclass
class KellyStake:
    edge: float             # p - implied_prob (positive = value bet)
    full_kelly: float       # f* — full-Kelly fraction of bankroll
    stake_fraction: float   # full_kelly · kelly_fraction, clamped to [0, cap]
    stake_dollars: float    # bankroll · stake_fraction
    ev_per_dollar: float    # expected value per $1 staked


def kelly_stake(
    our_prob: float,
    american_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_fraction: float = 0.05,
) -> KellyStake:
    """Compute recommended stake for a single two-way prop bet.

    Parameters
    ----------
    our_prob
        Model's estimated probability the bet wins (e.g. prob Over).
    american_odds
        Sportsbook price on that side (e.g. -110).
    bankroll
        Current bankroll in dollars.
    kelly_fraction
        Scale applied to full Kelly. 0.25 = quarter-Kelly (default; robust to
        miscalibration). Set higher at your peril.
    max_fraction
        Hard cap on stake as a fraction of bankroll, independent of Kelly.

    Returns a ``KellyStake`` dataclass. Stake is zero if no edge.
    """
    p = max(0.0, min(1.0, float(our_prob)))
    q = 1.0 - p
    b = american_to_decimal(american_odds) - 1.0  # net payoff
    if b <= 0:
        return KellyStake(edge=0.0, full_kelly=0.0, stake_fraction=0.0,
                          stake_dollars=0.0, ev_per_dollar=0.0)
    implied = 1.0 / (b + 1.0)
    edge = p - implied
    ev_per_dollar = p * b - q  # EV per $1 staked
    full_kelly = (b * p - q) / b  # classic Kelly
    if full_kelly <= 0 or bankroll <= 0:
        return KellyStake(edge=edge, full_kelly=full_kelly, stake_fraction=0.0,
                          stake_dollars=0.0, ev_per_dollar=ev_per_dollar)
    fraction = min(full_kelly * kelly_fraction, max_fraction)
    fraction = max(0.0, fraction)
    return KellyStake(
        edge=edge,
        full_kelly=full_kelly,
        stake_fraction=fraction,
        stake_dollars=round(bankroll * fraction, 2),
        ev_per_dollar=ev_per_dollar,
    )


# ---------------------------------------------------------------------------
# Bankroll tracker (thin SQLite wrapper)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bankroll_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    balance REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    updated_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bankroll_bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_utc TEXT NOT NULL,
    player_name TEXT,
    prop_type TEXT,
    side TEXT,
    line REAL,
    american_odds REAL,
    our_prob REAL,
    stake REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    result TEXT,                -- 'win' | 'loss' | 'push' | null while open
    pnl REAL,                   -- realised P&L once settled
    settled_utc TEXT
);
CREATE INDEX IF NOT EXISTS idx_bankroll_bets_status ON bankroll_bets(status);
"""

_LOCK = threading.Lock()


class BankrollTracker:
    """SQLite-backed bankroll + bet ledger. Single-row ``bankroll_state``."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    # ------------------------------------------------------------------ infra

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with _LOCK, self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ state

    def get_balance(self, default: float = 1000.0) -> float:
        with _LOCK, self._conn() as conn:
            row = conn.execute(
                "SELECT balance FROM bankroll_state WHERE id = 1"
            ).fetchone()
            if row is None:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO bankroll_state (id, balance, updated_utc) VALUES (1, ?, ?)",
                    (float(default), now),
                )
                return float(default)
            return float(row[0])

    def set_balance(self, amount: float) -> float:
        now = datetime.now(timezone.utc).isoformat()
        with _LOCK, self._conn() as conn:
            conn.execute(
                "INSERT INTO bankroll_state (id, balance, updated_utc) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET balance = excluded.balance, updated_utc = excluded.updated_utc",
                (float(amount), now),
            )
        return float(amount)

    # -------------------------------------------------------------------- bets

    def record_bet(
        self,
        *,
        player_name: str | None,
        prop_type: str,
        side: str,
        line: float,
        american_odds: float,
        our_prob: float,
        stake: float,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with _LOCK, self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO bankroll_bets
                   (placed_utc, player_name, prop_type, side, line, american_odds, our_prob, stake)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, player_name, prop_type, side, float(line),
                 float(american_odds), float(our_prob), float(stake)),
            )
            # debit bankroll
            conn.execute(
                "UPDATE bankroll_state SET balance = balance - ?, updated_utc = ? WHERE id = 1",
                (float(stake), now),
            )
            return int(cur.lastrowid)

    def settle(self, bet_id: int, result: str) -> dict:
        """Settle a bet. ``result`` ∈ {'win', 'loss', 'push'}."""
        if result not in ("win", "loss", "push"):
            raise ValueError(f"bad result: {result}")
        now = datetime.now(timezone.utc).isoformat()
        with _LOCK, self._conn() as conn:
            row = conn.execute(
                "SELECT stake, american_odds, status FROM bankroll_bets WHERE id = ?",
                (int(bet_id),),
            ).fetchone()
            if row is None:
                raise KeyError(f"no bet {bet_id}")
            stake, odds, status = row
            if status != "open":
                raise ValueError(f"bet {bet_id} already {status}")

            if result == "win":
                profit = float(stake) * (american_to_decimal(float(odds)) - 1.0)
                credit = float(stake) + profit
                pnl = profit
            elif result == "push":
                credit = float(stake)
                pnl = 0.0
            else:  # loss
                credit = 0.0
                pnl = -float(stake)

            conn.execute(
                """UPDATE bankroll_bets
                   SET status = 'settled', result = ?, pnl = ?, settled_utc = ?
                   WHERE id = ?""",
                (result, pnl, now, int(bet_id)),
            )
            conn.execute(
                "UPDATE bankroll_state SET balance = balance + ?, updated_utc = ? WHERE id = 1",
                (credit, now),
            )

            balance = conn.execute(
                "SELECT balance FROM bankroll_state WHERE id = 1"
            ).fetchone()[0]
            return {"bet_id": int(bet_id), "result": result, "pnl": pnl,
                    "new_balance": float(balance)}

    def list_bets(self, status: str | None = None, limit: int = 100) -> list[dict]:
        sql = "SELECT * FROM bankroll_bets"
        params: tuple = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY placed_utc DESC LIMIT ?"
        with _LOCK, self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (*params, int(limit))).fetchall()
            return [dict(r) for r in rows]

    def summary(self) -> dict:
        with _LOCK, self._conn() as conn:
            balance = conn.execute(
                "SELECT balance FROM bankroll_state WHERE id = 1"
            ).fetchone()
            stats = conn.execute("""
                SELECT
                    COUNT(*)                                  AS n,
                    SUM(CASE WHEN result='win'  THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN result='push' THEN 1 ELSE 0 END) AS pushes,
                    COALESCE(SUM(pnl), 0)                     AS total_pnl,
                    COALESCE(SUM(stake), 0)                   AS total_staked
                FROM bankroll_bets
                WHERE status = 'settled'
            """).fetchone()
            open_stats = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(stake), 0) FROM bankroll_bets WHERE status='open'"
            ).fetchone()
        n, wins, losses, pushes, pnl, staked = stats
        open_n, open_stake = open_stats
        roi = (pnl / staked) if staked else 0.0
        return {
            "balance": float(balance[0]) if balance else 0.0,
            "settled_bets": int(n or 0),
            "wins": int(wins or 0),
            "losses": int(losses or 0),
            "pushes": int(pushes or 0),
            "total_pnl": float(pnl or 0.0),
            "total_staked": float(staked or 0.0),
            "roi": float(roi),
            "open_bets": int(open_n or 0),
            "open_exposure": float(open_stake or 0.0),
        }
