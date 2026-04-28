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


def devig_shin(
    over_american: float,
    under_american: float,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> tuple[float, float, float]:
    """Shin's (1992) two-way devig — a sharper alternative to multiplicative.

    The vig isn't uniformly distributed across both sides: bookmakers price
    in adverse-selection risk, which falls disproportionately on the
    perceived favourite. Shin parameterises the book's "z" (insider-trader
    fraction) and inverts to recover the fair probability:

        π_i = (sqrt(z² + 4·(1−z)·q_i² / Σq) − z) / (2·(1−z))

    where ``q_i`` is the raw implied probability of side i. We solve for z
    by binary search on the constraint ``π_o + π_u = 1``.

    Returns ``(p_over_fair, p_under_fair, z)``. Falls back to multiplicative
    devig if Shin doesn't converge (e.g. equal odds — z is undefined).
    """
    q_o = american_to_implied_prob(over_american)
    q_u = american_to_implied_prob(under_american)
    overround = q_o + q_u
    if overround <= 1.0 + 1e-12 or abs(q_o - q_u) < 1e-12:
        # No vig (or perfectly symmetric) → Shin is degenerate; fall back
        po, pu = devig_two_way(over_american, under_american)
        return po, pu, 0.0

    def _pi(q: float, z: float, denom: float) -> float:
        # denom = q_o + q_u (the overround)
        radicand = z * z + 4.0 * (1.0 - z) * q * q / denom
        return (math.sqrt(max(0.0, radicand)) - z) / (2.0 * (1.0 - z))

    lo, hi = 0.0, min(0.999, overround - 1.0 + 0.5)  # z ∈ [0, vig + buffer)
    for _ in range(max_iter):
        z = 0.5 * (lo + hi)
        po = _pi(q_o, z, overround)
        pu = _pi(q_u, z, overround)
        s = po + pu
        if abs(s - 1.0) < tol:
            return po, pu, z
        # The function π_o(z) + π_u(z) is monotone decreasing in z over [0,1).
        # If sum is too high, push z higher; too low, lower.
        if s > 1.0:
            lo = z
        else:
            hi = z
    # Did not converge — fall back to safe devig
    po, pu = devig_two_way(over_american, under_american)
    return po, pu, 0.0


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


def kelly_stake_three_outcome(
    p_win: float,
    p_push: float,
    american_odds: float,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_fraction: float = 0.05,
) -> KellyStake:
    """Push-aware Kelly for three-outcome bets (win / push / lose).

    Maximises ``E[log(1 + f·X)]`` where the random return X is:
        +b  with prob p_win
         0  with prob p_push   (stake refunded — no growth, no decay)
        -1  with prob p_loss = 1 − p_win − p_push

    Closed-form optimum:
        f* = (p_win · b − p_loss) / (b · (p_win + p_loss))

    Reduces to classic Kelly when p_push == 0.
    """
    p_w = max(0.0, min(1.0, float(p_win)))
    p_p = max(0.0, min(1.0, float(p_push)))
    if p_w + p_p > 1.0:
        # Renormalise — caller passed inconsistent probs
        s = p_w + p_p
        p_w, p_p = p_w / s, p_p / s
    p_l = max(0.0, 1.0 - p_w - p_p)
    b = american_to_decimal(american_odds) - 1.0
    if b <= 0:
        return KellyStake(edge=0.0, full_kelly=0.0, stake_fraction=0.0,
                          stake_dollars=0.0, ev_per_dollar=0.0)
    implied = 1.0 / (b + 1.0)
    edge = p_w - implied
    ev_per_dollar = p_w * b - p_l  # push contributes 0
    denom = b * (p_w + p_l)
    full_kelly = ((p_w * b) - p_l) / denom if denom > 0 else 0.0
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


def kelly_stake_correlated(
    bets: list[dict],
    bankroll: float,
    correlation_matrix=None,
    kelly_fraction: float = 0.25,
    max_fraction_per_bet: float = 0.05,
    max_total_fraction: float = 0.20,
    n_samples: int = 20_000,
    seed: int = 42,
) -> list[dict]:
    """Joint-Kelly stake sizing for N simultaneous correlated bets.

    Single-bet Kelly oversizes when bets are positively correlated:
    losing on one bet makes losing on the others more likely, so the
    growth-optimal joint stake is smaller than the sum of solo stakes.

    Approach: Monte-Carlo simulate joint win/loss outcomes via the
    Gaussian copula on the supplied correlation matrix, then climb
    ``E[log(1 + Σ f_i X_i)]`` by coordinate-wise solo-Kelly shrinkage.
    Doesn't need scipy — closed-form per-step.

    Parameters
    ----------
    bets
        List of ``{"prob": p, "american_odds": odds}`` dicts.
    correlation_matrix
        ``N×N`` numpy-like; ``None`` → identity (independence). Diagonals 1.
    max_total_fraction
        Hard cap on Σ f_i. Stops over-exposure when many small edges pile up.

    Returns a list of dicts mirroring ``bets`` augmented with
    ``stake_fraction`` and ``stake_dollars``. Single-bet input reduces to
    ``kelly_stake`` (within MC noise).
    """
    import numpy as np
    if not bets:
        return []
    n = len(bets)
    p = np.array([max(1e-9, min(1 - 1e-9, float(b["prob"]))) for b in bets])
    b_payout = np.array([american_to_decimal(float(bb["american_odds"])) - 1.0
                          for bb in bets])

    # Solo full-Kelly per bet — starting point + shrinkage anchor
    solo_full = np.maximum(0.0, (b_payout * p - (1 - p)) / np.maximum(b_payout, 1e-12))

    if correlation_matrix is None:
        corr = np.eye(n)
    else:
        corr = np.asarray(correlation_matrix, dtype=float)
        if corr.shape != (n, n):
            corr = np.eye(n)

    # Sample correlated win/loss outcomes via Gaussian copula
    rng = np.random.default_rng(seed)
    try:
        L = np.linalg.cholesky(corr + 1e-9 * np.eye(n))
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(corr)
        L = v @ np.diag(np.sqrt(np.maximum(w, 0)))
    z = rng.standard_normal(size=(n_samples, n)) @ L.T
    # Sample wins iff Z_i <= Φ⁻¹(p_i). Use scipy if available, else AS241.
    try:
        from scipy.stats import norm
        thresholds = norm.ppf(p)
    except Exception:  # noqa: BLE001
        from .parlay import _std_normal_ppf
        thresholds = np.array([_std_normal_ppf(float(pi)) for pi in p])
    win_mat = (z <= thresholds).astype(float)  # shape (S, n) ∈ {0, 1}
    # Per-bet returns: +b_i on win, -1 on loss
    X = win_mat * b_payout + (1.0 - win_mat) * (-1.0)  # (S, n)

    def neg_log_growth(f: np.ndarray) -> float:
        port = 1.0 + X @ f
        port = np.maximum(port, 1e-9)  # guard against bankrupt sample
        return -float(np.mean(np.log(port)))

    # First find UNSCALED joint full-Kelly via coordinate descent. Then
    # apply ``kelly_fraction`` as the safety multiplier and cap.
    # Bound each f_i at solo_full[i] (full-Kelly never exceeds solo Kelly under
    # positive correlation), but allow up to 1.0 in the rare zero-correlation
    # multi-bet case.
    upper_per_bet = np.minimum(np.maximum(solo_full, 1e-12), 1.0)
    f = solo_full.copy()  # starting point
    for _ in range(10):
        improved = False
        for i in range(n):
            base_obj = neg_log_growth(f)
            best_obj, best_fi = base_obj, f[i]
            # Search 8 grid points between 0 and the per-bet upper bound
            for frac_of_upper in (0.0, 0.25, 0.5, 0.75, 1.0):
                trial = float(frac_of_upper * upper_per_bet[i])
                f_try = f.copy()
                f_try[i] = trial
                obj = neg_log_growth(f_try)
                if obj < best_obj - 1e-9:
                    best_obj, best_fi = obj, trial
            if abs(best_fi - f[i]) > 1e-12:
                f[i] = best_fi
                improved = True
        if not improved:
            break

    # f is now the unscaled joint-Kelly optimum. Apply safety multiplier + caps.
    f_scaled = f * kelly_fraction
    f_scaled = np.minimum(f_scaled, max_fraction_per_bet)
    f_scaled = np.maximum(f_scaled, 0.0)
    total = f_scaled.sum()
    if total > max_total_fraction and total > 0:
        f_scaled *= max_total_fraction / total

    out = []
    for i, bb in enumerate(bets):
        frac = float(f_scaled[i])
        out.append({
            **bb,
            "solo_full_kelly": float(solo_full[i]),
            "joint_full_kelly": float(f[i]),
            "stake_fraction": frac,
            "stake_dollars": round(float(bankroll) * frac, 2) if bankroll > 0 else 0.0,
        })
    return out


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


def kelly_stake_crra(
    our_prob: float,
    american_odds: float,
    bankroll: float,
    risk_aversion: float = 2.0,
    max_fraction: float = 0.05,
) -> KellyStake:
    """CRRA-utility (constant relative risk aversion) Kelly variant.

    Pure log-utility (γ=1) is "full Kelly" — but log-utility is risk-neutral
    in a log sense, which is more aggressive than most humans actually are
    when their savings are on the line. CRRA with γ>1 (typical empirical
    estimate γ≈2-4) penalises bankroll variance, producing a smaller stake
    than full Kelly.

    For a binary bet with payoff b on win (prob p) and -1 on loss (prob q):
      U(f) = p · ((1+fb)^(1-γ)) / (1-γ)  +  q · ((1-f)^(1-γ)) / (1-γ)
    Maximising over f gives the closed form (γ != 1):
      f* = [ (p·b)^(1/γ) − q^(1/γ) ] / [ b · q^(1/γ) + (p·b)^(1/γ) ]

    γ=1 reduces to log-Kelly (handled via fallback). Output is otherwise
    interface-compatible with kelly_stake_two_outcome.
    """
    p = max(0.0, min(1.0, float(our_prob)))
    q = 1.0 - p
    b = american_to_decimal(american_odds) - 1.0
    gamma = float(risk_aversion)
    if b <= 0 or bankroll <= 0:
        return KellyStake(0.0, 0.0, 0.0, 0.0, 0.0)
    implied = 1.0 / (b + 1.0)
    edge = p - implied
    ev_per_dollar = p * b - q
    if abs(gamma - 1.0) < 1e-9:
        # Log-utility — falls back to standard Kelly
        full = (b * p - q) / b
    else:
        # Closed-form CRRA: f* = (A − B) / (b·B + A) where A = (pb)^(1/γ),
        # B = q^(1/γ). Numerically stable for any γ > 0.
        if p <= 0 or q <= 0:
            full = 0.0
        else:
            A = (p * b) ** (1.0 / gamma)
            B = q ** (1.0 / gamma)
            denom = b * B + A
            full = (A - B) / denom if denom > 0 else 0.0
    if full <= 0:
        return KellyStake(edge, full, 0.0, 0.0, ev_per_dollar)
    fraction = min(full, max_fraction)
    return KellyStake(
        edge=edge,
        full_kelly=full,
        stake_fraction=fraction,
        stake_dollars=round(bankroll * fraction, 2),
        ev_per_dollar=ev_per_dollar,
    )


# ---------------------------------------------------------------------------
# Risk limits — concentration / drawdown / exposure circuit breakers
# ---------------------------------------------------------------------------

@dataclass
class RiskLimits:
    """Guardrails enforced at ``record_bet`` time.

    All three default to off (None). Set via env vars from the app entry
    point or pass explicitly to ``BankrollTracker(...)``. When a limit
    fires, ``record_bet`` raises ``RiskLimitError`` and does NOT debit
    the bankroll — the caller decides whether to reduce stake and retry.
    """
    # Pause new bets when (peak − current) / peak exceeds this. Default 0.25.
    max_drawdown: float | None = None
    # Cap total open stake placed today as a fraction of bankroll.
    max_daily_exposure_pct: float | None = None
    # Cap total open stake on any single player as a fraction of bankroll.
    max_player_concentration_pct: float | None = None

    @classmethod
    def from_env(cls) -> "RiskLimits":
        """Build from env vars (all optional). Empty/missing = no limit."""
        def _f(key: str) -> float | None:
            raw = os.environ.get(key, "").strip()
            if not raw:
                return None
            try:
                return float(raw)
            except ValueError:
                return None
        return cls(
            max_drawdown=_f("MAX_DRAWDOWN"),
            max_daily_exposure_pct=_f("MAX_DAILY_EXPOSURE_PCT"),
            max_player_concentration_pct=_f("MAX_PLAYER_CONCENTRATION_PCT"),
        )


class RiskLimitError(ValueError):
    """Raised when a circuit breaker refuses a bet. Message names the rule."""


# ---------------------------------------------------------------------------
# Bankroll tracker (thin SQLite wrapper)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Schema migrations
#
# Versioned, append-only list of (version, ddl) pairs. ``_ensure_schema``
# fast-forwards from whatever version the DB is at to ``_TARGET_VERSION``.
# Adding a new field? Append a new tuple — never edit a past one.
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, str]] = [
    (1, """
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
            result TEXT,
            pnl REAL,
            settled_utc TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_bankroll_bets_status ON bankroll_bets(status);
    """),
    # v2: track Kelly metadata + originating prediction for analytics
    (2, """
        ALTER TABLE bankroll_bets ADD COLUMN kelly_fraction REAL;
        ALTER TABLE bankroll_bets ADD COLUMN edge REAL;
        ALTER TABLE bankroll_bets ADD COLUMN ev_per_dollar REAL;
        ALTER TABLE bankroll_bets ADD COLUMN prediction_log_id INTEGER;
        CREATE INDEX IF NOT EXISTS idx_bankroll_bets_placed_utc
            ON bankroll_bets(placed_utc);
    """),
    # v3: tag bets with the model version that produced them (for ROI-by-model)
    (3, """
        ALTER TABLE bankroll_bets ADD COLUMN model_version TEXT;
    """),
    # v4: closing line value tracking. ``closing_*`` columns are filled in
    # after the bet is placed (e.g. via a tip-off cron); CLV = our_prob -
    # implied(closing_odds) and is the single best leading indicator that the
    # model is sharper than the market.
    (4, """
        ALTER TABLE bankroll_bets ADD COLUMN closing_line REAL;
        ALTER TABLE bankroll_bets ADD COLUMN closing_odds REAL;
        ALTER TABLE bankroll_bets ADD COLUMN clv REAL;
    """),
    # v5: peak-balance tracker so drawdown can be computed without a full scan
    # of the bet log. Updated whenever balance moves up; never reset by code.
    (5, """
        ALTER TABLE bankroll_state ADD COLUMN peak_balance REAL;
    """),
]

_TARGET_VERSION = max(v for v, _ in _MIGRATIONS)

_LOCK = threading.Lock()


def _current_user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _set_user_version(conn: sqlite3.Connection, v: int) -> None:
    # PRAGMA doesn't accept parameters — but we control v inside this module,
    # so f-string is safe (and only ints).
    conn.execute(f"PRAGMA user_version = {int(v)}")


class BankrollTracker:
    """SQLite-backed bankroll + bet ledger. Single-row ``bankroll_state``."""

    def __init__(self, db_path: str, risk_limits: RiskLimits | None = None):
        self.db_path = db_path
        # Default to env-driven limits so a deploy can flip on circuit
        # breakers without code changes. Pass an explicit ``risk_limits=``
        # to override (useful in tests).
        self.risk_limits = risk_limits if risk_limits is not None else RiskLimits.from_env()
        self._ensure_schema()

    # ------------------------------------------------------------------ infra

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        """Apply pending migrations idempotently.

        Uses ``PRAGMA user_version`` as the version pin. Each migration
        runs in its own transaction so a partial failure leaves the DB
        at the last successful version.
        """
        with _LOCK, self._conn() as conn:
            current = _current_user_version(conn)
            for version, ddl in _MIGRATIONS:
                if version <= current:
                    continue
                # ALTER TABLE ADD COLUMN is idempotent-by-error-handling: on
                # a partially-migrated DB (where someone created columns
                # outside the migration system), swallow "duplicate column".
                for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError as e:
                        if "duplicate column" in str(e).lower():
                            continue
                        raise
                _set_user_version(conn, version)
                conn.commit()

    # ------------------------------------------------------------------ state

    def get_balance(self, default: float = 1000.0) -> float:
        with _LOCK, self._conn() as conn:
            row = conn.execute(
                "SELECT balance FROM bankroll_state WHERE id = 1"
            ).fetchone()
            if row is None:
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT INTO bankroll_state (id, balance, updated_utc, peak_balance) "
                    "VALUES (1, ?, ?, ?)",
                    (float(default), now, float(default)),
                )
                return float(default)
            return float(row[0])

    def set_balance(self, amount: float) -> float:
        now = datetime.now(timezone.utc).isoformat()
        with _LOCK, self._conn() as conn:
            # Peak is the running high-water mark. We seed it on first write
            # (so a brand-new bankroll has a sensible peak from minute 1) and
            # bump it whenever the new balance exceeds the prior peak.
            conn.execute(
                """INSERT INTO bankroll_state (id, balance, updated_utc, peak_balance)
                   VALUES (1, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE
                     SET balance = excluded.balance,
                         updated_utc = excluded.updated_utc,
                         peak_balance = MAX(COALESCE(peak_balance, 0), excluded.balance)""",
                (float(amount), now, float(amount)),
            )
        return float(amount)

    def get_peak_balance(self) -> float:
        """Return the all-time-high balance recorded by ``set_balance`` /
        bumped by the ``settle()`` credit path. Used for drawdown."""
        with _LOCK, self._conn() as conn:
            row = conn.execute(
                "SELECT balance, peak_balance FROM bankroll_state WHERE id = 1"
            ).fetchone()
        if row is None:
            return 0.0
        bal, peak = row
        # Defensive: if peak was never set on a legacy row, treat current as peak.
        return float(peak) if peak is not None else float(bal or 0.0)

    def get_drawdown(self) -> dict:
        """Compute current peak-to-trough drawdown.

        Returns ``{peak, current, drawdown}`` where ``drawdown`` is in
        ``[0, 1)`` — 0 means we're at the peak, 0.25 means down a quarter.
        """
        with _LOCK, self._conn() as conn:
            row = conn.execute(
                "SELECT balance, peak_balance FROM bankroll_state WHERE id = 1"
            ).fetchone()
        if row is None:
            return {"peak": 0.0, "current": 0.0, "drawdown": 0.0}
        cur = float(row[0] or 0.0)
        peak = float(row[1]) if row[1] is not None else cur
        if peak <= 0:
            return {"peak": peak, "current": cur, "drawdown": 0.0}
        return {
            "peak": peak,
            "current": cur,
            "drawdown": max(0.0, (peak - cur) / peak),
        }

    def daily_exposure(self, *, date_iso: str | None = None) -> float:
        """Sum of stake on bets placed today (UTC) regardless of status.

        Used to enforce ``max_daily_exposure_pct``. We deliberately count
        already-settled bets too — that's "intent to risk", which is what
        the cap is meant to limit; otherwise a fast-settling slate would
        let a user blow through the cap by churning bets.
        """
        day = (date_iso or datetime.now(timezone.utc).date().isoformat())[:10]
        with _LOCK, self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(stake), 0) FROM bankroll_bets "
                "WHERE substr(placed_utc, 1, 10) = ?",
                (day,),
            ).fetchone()
        return float(row[0] or 0.0)

    def player_open_exposure(self, player_name: str) -> float:
        """Sum of stake on currently-open bets for one player. Used for
        per-player concentration limits."""
        with _LOCK, self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(stake), 0) FROM bankroll_bets "
                "WHERE status = 'open' AND player_name = ?",
                (str(player_name),),
            ).fetchone()
        return float(row[0] or 0.0)

    def _check_risk_limits(self, *, player_name: str | None, stake: float) -> None:
        """Apply ``self.risk_limits`` to a proposed bet. Raises on violation.

        Order: drawdown (cheapest, killswitch) → daily exposure → per-player
        concentration. Raised errors carry which rule fired so the caller
        can build a useful HTTP 4xx response.
        """
        rl = self.risk_limits
        if rl is None:
            return
        bal = self.get_balance()
        if rl.max_drawdown is not None:
            dd = self.get_drawdown()["drawdown"]
            if dd >= float(rl.max_drawdown):
                raise RiskLimitError(
                    f"drawdown circuit breaker: current drawdown {dd:.3f} "
                    f">= cap {float(rl.max_drawdown):.3f}"
                )
        if rl.max_daily_exposure_pct is not None and bal > 0:
            today = self.daily_exposure()
            cap = float(rl.max_daily_exposure_pct) * bal
            if today + float(stake) > cap:
                raise RiskLimitError(
                    f"daily exposure cap: today's stake {today + stake:.2f} "
                    f"would exceed cap {cap:.2f} ({rl.max_daily_exposure_pct:.2%} of bankroll)"
                )
        if rl.max_player_concentration_pct is not None and bal > 0 and player_name:
            on_player = self.player_open_exposure(player_name)
            cap = float(rl.max_player_concentration_pct) * bal
            if on_player + float(stake) > cap:
                raise RiskLimitError(
                    f"player concentration cap: open stake on {player_name} "
                    f"{on_player + stake:.2f} would exceed cap {cap:.2f} "
                    f"({rl.max_player_concentration_pct:.2%} of bankroll)"
                )

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
        kelly_fraction: float | None = None,
        edge: float | None = None,
        ev_per_dollar: float | None = None,
        prediction_log_id: int | None = None,
        model_version: str | None = None,
    ) -> int:
        """Record a placed bet. The optional kwargs are populated by v2/v3
        schema columns; passing them at write-time enables ROI-by-model
        and joins back to ``prediction_logs`` for explainability.

        Risk limits (``self.risk_limits``) are evaluated BEFORE the row is
        inserted; a violation raises ``RiskLimitError`` and leaves the
        bankroll untouched.
        """
        # Circuit-breaker checks happen outside the lock — they only read.
        # On violation, raise before we touch the DB.
        self._check_risk_limits(player_name=player_name, stake=float(stake))
        now = datetime.now(timezone.utc).isoformat()
        with _LOCK, self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO bankroll_bets
                   (placed_utc, player_name, prop_type, side, line,
                    american_odds, our_prob, stake,
                    kelly_fraction, edge, ev_per_dollar,
                    prediction_log_id, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, player_name, prop_type, side, float(line),
                 float(american_odds), float(our_prob), float(stake),
                 None if kelly_fraction is None else float(kelly_fraction),
                 None if edge is None else float(edge),
                 None if ev_per_dollar is None else float(ev_per_dollar),
                 None if prediction_log_id is None else int(prediction_log_id),
                 model_version),
            )
            # debit bankroll
            conn.execute(
                "UPDATE bankroll_state SET balance = balance - ?, updated_utc = ? WHERE id = 1",
                (float(stake), now),
            )
            return int(cur.lastrowid)

    def record_closing_line(
        self,
        bet_id: int,
        *,
        closing_line: float | None = None,
        closing_odds: float | None = None,
    ) -> dict:
        """Stamp a bet with the line/odds that closed the market.

        CLV is computed as ``our_prob − implied(closing_odds)``. A consistently
        positive CLV is the strongest evidence the model is profitably ahead
        of the market — far stronger than realised ROI on a small sample.
        """
        with _LOCK, self._conn() as conn:
            row = conn.execute(
                "SELECT our_prob FROM bankroll_bets WHERE id = ?",
                (int(bet_id),),
            ).fetchone()
            if row is None:
                raise KeyError(f"no bet {bet_id}")
            our_prob = float(row[0]) if row[0] is not None else None
            clv = None
            if our_prob is not None and closing_odds is not None:
                clv = our_prob - american_to_implied_prob(float(closing_odds))
            conn.execute(
                """UPDATE bankroll_bets
                   SET closing_line = ?, closing_odds = ?, clv = ?
                   WHERE id = ?""",
                (
                    None if closing_line is None else float(closing_line),
                    None if closing_odds is None else float(closing_odds),
                    None if clv is None else float(clv),
                    int(bet_id),
                ),
            )
            return {"bet_id": int(bet_id), "clv": clv}

    def clv_summary(self) -> dict:
        """Mean CLV across bets where closing odds have been recorded."""
        with _LOCK, self._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*), AVG(clv), MIN(clv), MAX(clv)
                   FROM bankroll_bets
                   WHERE clv IS NOT NULL"""
            ).fetchone()
        n, mean_clv, min_clv, max_clv = row
        return {
            "n": int(n or 0),
            "mean_clv": float(mean_clv) if mean_clv is not None else 0.0,
            "min_clv": float(min_clv) if min_clv is not None else 0.0,
            "max_clv": float(max_clv) if max_clv is not None else 0.0,
        }

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
            # Credit the bankroll, then bump peak_balance if this settlement
            # set a new high-water mark (drives the drawdown calculation).
            conn.execute(
                """UPDATE bankroll_state
                   SET balance = balance + ?,
                       updated_utc = ?,
                       peak_balance = MAX(COALESCE(peak_balance, 0), balance + ?)
                   WHERE id = 1""",
                (credit, now, credit),
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
