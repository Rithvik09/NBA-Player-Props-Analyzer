from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
try:
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"Warning: XGBoost not available ({type(e).__name__}), falling back to HistGradientBoosting")
    XGBRegressor = None
    XGBClassifier = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available, using default hyperparameters")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog  # noqa: E402
from nba_api.stats.endpoints import CommonPlayerInfo  # noqa: E402
from nba_api.stats.static import teams  # noqa: E402

from src.models import EnhancedMLPredictor  # noqa: E402
from src.ml_features import NUMERIC_FEATURE_KEYS, CLASSIFIER_EXTRA_KEYS, build_feature_vector, build_classifier_vector  # noqa: E402
from src.precomputed_store import PrecomputedStore  # noqa: E402
from src.ml_calibration import IsotonicCalibratedModel  # noqa: E402
from src.arena_data import calculate_travel_metrics, ARENA_DATA  # noqa: E402


def _compute_injury_trajectory_from_df(gl_df: pd.DataFrame, up_to_idx: int) -> tuple[float, float]:
    """Compute games_since_return and missed_games_before_return from a game log DataFrame.

    Scans date gaps in the historical slice (rows 0..up_to_idx-1) to find the most
    recent gap > 5 days (proxy for injury absence).
    """
    hist = gl_df.iloc[:up_to_idx]
    if len(hist) < 2 or "GAME_DATE" not in hist.columns:
        return 0.0, 0.0
    try:
        dates = pd.to_datetime(hist["GAME_DATE"], errors="coerce").dropna().sort_values(ascending=False)
        if len(dates) < 2:
            return 0.0, 0.0
        dates_list = dates.tolist()
        games_since_return = 0
        missed_games = 0.0
        found_gap = False
        for i in range(len(dates_list) - 1):
            gap = (dates_list[i] - dates_list[i + 1]).days
            if gap > 5 and not found_gap:
                missed_games = max(0.0, (gap - 2) / 2.0)
                found_gap = True
                break
            elif not found_gap:
                games_since_return += 1
        return float(games_since_return), float(missed_games)
    except Exception:
        return 0.0, 0.0


def _compute_implied_game_total(team_ctx: dict, opp_ctx: dict) -> float:
    """Estimate the implied game total from team and opponent pace."""
    t_pace = float((team_ctx or {}).get("pace", 100.0))
    o_pace = float((opp_ctx or {}).get("pace", 100.0))
    avg_pace = (t_pace + o_pace) / 2.0
    return float(avg_pace * 2.0 * 1.1)


def _compute_primary_defender_active(
    primary_def: dict | None,
    gl_df: pd.DataFrame,
    up_to_idx: int,
) -> float:
    """Estimate whether the primary defender was active around the game at up_to_idx.

    Uses the player's own game log as a proxy: if the opponent team had a game
    within 2 days of this game, we assume the primary defender was active (1.0).
    Falls back to 1.0 (active) when unknown — conservative assumption.
    """
    if not primary_def:
        return 1.0
    # We don't have the defender's game log in training, but we can use the
    # fact that the opponent played this game (opp_id is known) as a proxy.
    # If the primary defender has a score01 > 0 they are an active elite defender.
    score = float((primary_def or {}).get("score01", 0.0) or 0.0)
    return 1.0 if score > 0 else 0.0


def _estimate_opp_lineup_changes(dvp_rolling: dict, opp_id: int) -> float:
    """Estimate opponent lineup instability from rolling DVP variance.

    If the 5-game rolling DVP deviates significantly from the 10-game rolling DVP,
    it suggests the opponent's defensive personnel has been changing.
    Uses the absolute difference in pts-allowed between the two windows as a proxy.
    """
    r5  = dvp_rolling.get((opp_id, 5),  {})
    r10 = dvp_rolling.get((opp_id, 10), {})
    if not r5 or not r10:
        return 0.0
    pts5  = float(r5.get("pts",  0.0))
    pts10 = float(r10.get("pts", 0.0))
    # Normalize: a 2+ pt swing per game suggests ~1 lineup change
    delta = abs(pts5 - pts10)
    return min(3.0, delta / 2.0)


def _load_precomputed_player_data(db_path: str) -> dict[str, Any]:
    """Load all player-level and team-level precomputed data from the DB into memory.

    Returns a dict with:
      - player tables: {table_alias: {player_id: row_dict}}
      - team tables:   {table_alias: {team_id: row_dict}}
      - referee:       {ref_name: row_dict}
      - dvp_rolling:   {(team_id, window): row_dict}
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    def _load_player_table(table: str) -> dict[int, dict]:
        try:
            c.execute(f"SELECT * FROM {table}")
            return {int(row["player_id"]): dict(row) for row in c.fetchall()}
        except Exception:
            return {}

    def _load_team_table(table: str) -> dict[int, dict]:
        try:
            c.execute(f"SELECT * FROM {table}")
            return {int(row["team_id"]): dict(row) for row in c.fetchall()}
        except Exception:
            return {}

    def _load_vs_opponent() -> dict[tuple, dict]:
        try:
            c.execute("SELECT * FROM player_vs_opponent")
            result: dict[tuple, dict] = {}
            for row in c.fetchall():
                key = (int(row["player_id"]), int(row["opponent_team_id"]))
                result[key] = dict(row)
            return result
        except Exception:
            return {}

    def _load_dvp_rolling() -> dict[tuple, dict]:
        try:
            c.execute("SELECT * FROM dvp_rolling")
            result: dict[tuple, dict] = {}
            for row in c.fetchall():
                key = (int(row["team_id"]), int(row["window"]))
                result[key] = dict(row)
            return result
        except Exception:
            return {}

    def _load_referee_stats() -> dict[str, dict]:
        try:
            c.execute("SELECT * FROM referee_stats")
            return {str(row["ref_name"]): dict(row) for row in c.fetchall()}
        except Exception:
            return {}

    data: dict[str, Any] = {
        # Player tables
        "advanced":      _load_player_table("player_advanced_stats"),
        "clutch":        _load_player_table("player_clutch_stats"),
        "hustle":        _load_player_table("player_hustle_stats"),
        "shot_prof":     _load_player_table("player_shot_profile"),
        "play_types":    _load_player_table("player_play_types"),
        "on_off":        _load_player_table("player_on_off"),
        "shot_zones":    _load_player_table("player_shot_zones"),
        "q_splits":      _load_player_table("player_quarter_splits"),
        "tracking":      _load_player_table("player_tracking_stats"),
        "scoring":       _load_player_table("player_scoring_breakdown"),
        "yoy":           _load_player_table("player_yoy_stats"),
        # Team tables
        "team_stats":      _load_team_table("team_stats"),
        "team_foul":       _load_team_table("team_foul_rates"),
        "rest_splits":     _load_team_table("team_rest_splits"),
        "standings":       _load_team_table("team_standings"),
        "opp_shot_zones":  _load_team_table("team_opp_shot_zones"),
        "opp_synergy":     _load_team_table("team_synergy_defense"),
        "home_away":       _load_team_table("team_home_away_splits"),
        "lineup_stats":    _load_team_table("team_lineup_stats"),
        "injury_status":   _load_team_table("team_injury_status"),
        # Special structures
        "vs_opponent":   _load_vs_opponent(),
        "dvp_rolling":   _load_dvp_rolling(),
        "referee":       _load_referee_stats(),
    }
    conn.close()

    totals = {k: len(v) for k, v in data.items()}
    print(f"[train] precomputed data loaded: {totals}")
    return data


STAT_TARGETS = {
    "points": "PTS",
    "assists": "AST",
    "rebounds": "REB",
    "steals": "STL",
    "blocks": "BLK",
    "turnovers": "TOV",
    "three_pointers": "FG3M",
}

COMBO_TARGETS = {
    "pts_reb": ["PTS", "REB"],
    "pts_ast": ["PTS", "AST"],
    "ast_reb": ["AST", "REB"],
    "pts_ast_reb": ["PTS", "AST", "REB"],
    "stl_blk": ["STL", "BLK"],
}


def _parse_matchup(matchup: str):
    m = str(matchup or "")
    parts = m.split(" ")
    if len(parts) < 3:
        return None, None, None
    team = parts[0]
    if "vs" in parts:
        opp = parts[-1]
        return team, opp, True
    if "@" in parts:
        opp = parts[-1]
        return team, opp, False
    return team, parts[-1], None


def _team_id(abbrev: str | None) -> int | None:
    if not abbrev:
        return None
    # normalize a few variants
    alias = {"NOR": "NOP", "PHO": "PHX", "UTH": "UTA"}
    abbrev = alias.get(abbrev, abbrev)
    t = teams.find_team_by_abbreviation(abbrev)
    return int(t["id"]) if t else None


# Defaults — replaced by main() when CLI flags are supplied.
_TRAIN_CONFIG: dict = {
    "recency_half_life": None,
    "minutes_weight": False,
    "min_sample_weight": 0.1,
    "prune_features": 0.0,
}


@dataclass
class Example:
    prop_type: str
    X: pd.DataFrame
    y: float
    game_date: pd.Timestamp
    # optional: minutes played in the training-source game — used for
    # minutes-weighted sample weights (filters out garbage-time distortion)
    minutes: float = 0.0


def compute_sample_weights(
    examples: list,
    *,
    recency_half_life_days: float | None = 365.0,
    minutes_weight: bool = True,
    min_weight: float = 0.1,
    today: pd.Timestamp | None = None,
) -> np.ndarray:
    """Per-sample weight = recency_weight × minutes_weight, floored at ``min_weight``.

    Parameters
    ----------
    recency_half_life_days
        Half-life for exponential decay. ``None`` disables recency weighting
        (all samples equal). 365 ≈ last season counts half as much as this one.
    minutes_weight
        If True, scale by ``clip(minutes / typical_minutes, 0.4, 1.5)`` so
        blowout/injury games with <10 mins don't get full weight.
    min_weight
        Lower bound on the final weight — prevents ancient/garbage-time rows
        from collapsing to ~0 and starving the optimiser.
    """
    # Use tz-naive "today" — game dates from nba_api are tz-naive strings
    # (e.g. '2024-11-21'), so subtracting a tz-aware Timestamp raises TypeError.
    if today is None:
        today = pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp(today)
        if today.tzinfo is not None:
            today = today.tz_localize(None)
    minutes_list = [float(getattr(e, "minutes", 0.0) or 0.0) for e in examples]
    nonzero_mins = [m for m in minutes_list if m > 0]
    # np.median raises RuntimeWarning on empty input — guard explicitly.
    typical_min = float(np.median(nonzero_mins)) if nonzero_mins else 28.0

    weights = np.empty(len(examples), dtype=float)
    for i, e in enumerate(examples):
        w = 1.0
        # recency
        if recency_half_life_days and recency_half_life_days > 0 and e.game_date is not None:
            gd = pd.to_datetime(e.game_date)
            # Defensive: strip tz if the source ever produces aware timestamps
            if getattr(gd, "tzinfo", None) is not None:
                gd = gd.tz_localize(None)
            days_old = max(0.0, (today - gd).days)
            # half-life decay: 2 ** (-days/half_life)
            w *= 2.0 ** (-days_old / float(recency_half_life_days))
        # minutes
        if minutes_weight:
            mins = float(getattr(e, "minutes", 0.0) or 0.0)
            scale = 1.0 if mins <= 0 else max(0.4, min(1.5, mins / typical_min))
            w *= scale
        weights[i] = max(min_weight, w)
    return weights


def prune_low_importance_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    drop_fraction: float = 0.30,
    sample_weight: np.ndarray | None = None,
) -> tuple[list[str], dict]:
    """Fit a cheap XGB regressor, return columns in the TOP (1-drop_fraction).

    Returns ``(keep_cols, report)`` where report contains per-feature gain.
    Fail-safe: on any error returns all columns + empty report.
    """
    if not XGBOOST_AVAILABLE or X_train.empty:
        return list(X_train.columns), {}
    try:
        probe = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            tree_method="hist", verbosity=0,
        )
        probe.fit(X_train, y_train, sample_weight=sample_weight)
        importances = np.asarray(probe.feature_importances_, dtype=float)
        total = importances.sum() or 1.0
        share = importances / total
        order = np.argsort(share)[::-1]  # high → low
        cumulative = np.cumsum(share[order])
        # Keep features up to (1 - drop_fraction) cumulative gain
        cutoff_idx = int(np.searchsorted(cumulative, 1.0 - float(drop_fraction)))
        cutoff_idx = max(cutoff_idx, 10)  # always keep at least 10 features
        keep_idx = sorted(order[: cutoff_idx + 1].tolist())
        keep_cols = [X_train.columns[i] for i in keep_idx]
        report = {
            X_train.columns[i]: float(share[i])
            for i in range(len(share))
        }
        return keep_cols, report
    except Exception:
        return list(X_train.columns), {}


def build_training_examples(
    season: str,
    max_players: int,
    predictor: EnhancedMLPredictor,
    precomputed: PrecomputedStore,
    db_path: str = "basketball_data.db",
) -> list[Example]:
    # Load all player precomputed data once (shot zones, tracking, clutch, etc.)
    _precomp = _load_precomputed_player_data(db_path)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            players_df = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, measure_type_detailed_defense="Base", timeout=60
            ).get_data_frames()[0]
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[train] API timeout on attempt {attempt+1}/{max_retries}, retrying in 10s...")
                time.sleep(10)
            else:
                print(f"[train] Failed to fetch player stats for {season} after {max_retries} attempts")
                raise
    players_df = players_df.sort_values("MIN", ascending=False).head(max_players)

    pre = precomputed.refresh(force=True)
    dvp_map = pre.get("dvp", {})
    dvp_pos_avgs = pre.get("dvp_pos_avgs", {})
    defenders_map = pre.get("defenders", {})

    examples: list[Example] = []

    warnings.filterwarnings("ignore", category=FutureWarning)

    players_list = list(players_df.to_dict("records"))
    for i, prow in enumerate(players_list, start=1):
        player_id = int(prow["PLAYER_ID"])
        if i % 10 == 0:
            print(f"[train] players processed: {i}/{len(players_list)}")

        # --- Load per-player precomputed data (no extra API calls) ---
        _adv  = _precomp["advanced"].get(player_id, {})
        _clt  = _precomp["clutch"].get(player_id, {})
        _hsl  = _precomp["hustle"].get(player_id, {})
        _sp   = _precomp["shot_prof"].get(player_id, {})
        _pt   = _precomp["play_types"].get(player_id, {})
        _oo   = _precomp["on_off"].get(player_id, {})
        _sz   = _precomp["shot_zones"].get(player_id, {})
        _qs   = _precomp["q_splits"].get(player_id, {})
        _trk  = _precomp["tracking"].get(player_id, {})
        _scr  = _precomp["scoring"].get(player_id, {})
        _yoy  = _precomp["yoy"].get(player_id, {})
        # Player vs opponent lookup (keyed by (player_id, opp_team_id) — resolved per game below)
        _vs_opp_map = _precomp["vs_opponent"]
        
        # Retry logic for player game logs
        gl = None
        for attempt in range(3):
            try:
                gl = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=60).get_data_frames()[0]
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"[train] Skipping player {player_id} after 3 failed attempts")
                    
        if gl is None:
            continue
        if gl is None or gl.empty:
            continue

        # Also fetch playoff + play-in logs so intensity features have non-zero training exposure
        for _stype in ("Playoffs", "PlayIn"):
            try:
                _extra = playergamelog.PlayerGameLog(
                    player_id=player_id, season=season,
                    season_type_all_star=_stype, timeout=60,
                ).get_data_frames()[0]
                time.sleep(0.6)
                if _extra is not None and len(_extra) > 0:
                    gl = pd.concat([gl, _extra], ignore_index=True)
            except Exception:
                pass

        gl = gl.copy()
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"], format="mixed", errors="coerce")
        gl = gl.sort_values("GAME_DATE")


        try:
            info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            raw_pos = str(info["POSITION"].iloc[0] if "POSITION" in info.columns else "").upper()
            # Extract bio data for training features
            try:
                _birthdate = str(info["BIRTHDATE"].iloc[0] if "BIRTHDATE" in info.columns else "")
                if _birthdate and _birthdate != "nan":
                    _bd = pd.to_datetime(_birthdate, errors='coerce')
                    # Store birthdate; age will be computed per-game relative to game date below
                    _player_birthdate = _bd if _bd is not pd.NaT else None
                    _player_age = 26.0  # fallback; overridden per-game below
                else:
                    _player_birthdate = None
                    _player_age = 26.0
            except Exception:
                _player_birthdate = None
                _player_age = 26.0
            try:
                _exp_str = str(info["SEASON_EXP"].iloc[0] if "SEASON_EXP" in info.columns else "5")
                _player_exp = float(_exp_str) if _exp_str and _exp_str != "nan" else 5.0
            except Exception:
                _player_exp = 5.0
            try:
                _ht_str = str(info["HEIGHT"].iloc[0] if "HEIGHT" in info.columns else "6-6")
                _ht_parts = _ht_str.split("-")
                _player_height = float(_ht_parts[0]) * 12 + float(_ht_parts[1]) if len(_ht_parts) == 2 else 78.0
            except Exception:
                _player_height = 78.0
            try:
                _player_weight = float(info["WEIGHT"].iloc[0] if "WEIGHT" in info.columns else 220.0)
            except Exception:
                _player_weight = 220.0
        except Exception:
            raw_pos = ""
            _player_birthdate = None
            _player_age = 26.0; _player_exp = 5.0; _player_height = 78.0; _player_weight = 220.0
        if "C" in raw_pos and "G" not in raw_pos:
            dvp_pos = "C"
            pos_group = "C"
        elif "G" in raw_pos and "F" not in raw_pos:
            dvp_pos = "SG"
            pos_group = "G"
        elif "F" in raw_pos and "C" not in raw_pos and "G" not in raw_pos:
            dvp_pos = "SF"
            pos_group = "F"
        elif "C" in raw_pos:
            dvp_pos = "C"
            pos_group = "C"
        elif "G" in raw_pos:
            dvp_pos = "SG"
            pos_group = "G"
        else:
            dvp_pos = "SF"
            pos_group = "F"


        # Default target_col for per-idx features that compute before the STAT_TARGETS loop.
        # The STAT_TARGETS loop rebinds this each pass; this default only matters for code
        # paths that reference target_col outside that loop (e.g. home/away split at ~line 691).
        target_col = "PTS"

        for idx in range(10, len(gl)):
          try:
            hist = gl.iloc[:idx]
            row = gl.iloc[idx]

            team_abbrev, opp_abbrev, is_home = _parse_matchup(row.get("MATCHUP"))
            team_id = _team_id(team_abbrev)
            opp_id = _team_id(opp_abbrev)
            if not team_id or not opp_id:
                continue

            # --- Intensity / playoff context features ---
            _gid_col = "Game_ID" if "Game_ID" in gl.columns else ("GAME_ID" if "GAME_ID" in gl.columns else None)
            _gid_str = str(row.get(_gid_col, "")) if _gid_col else ""
            _prefix = _gid_str[:3] if len(_gid_str) >= 3 else ""
            is_playoff = 1.0 if _prefix == "004" else 0.0
            is_play_in = 1.0 if _prefix == "005" else 0.0
            series_game_num = 0.0
            team_series_wins_in = 0.0
            opp_series_wins_in = 0.0
            is_elimination_game = 0.0
            if is_playoff == 1.0 and opp_abbrev and _gid_col and "MATCHUP" in hist.columns:
                _prior_po = hist[
                    (hist[_gid_col].astype(str).str[:3] == "004")
                    & hist["MATCHUP"].astype(str).str.contains(opp_abbrev, na=False)
                ]
                series_game_num = float(len(_prior_po) + 1)
                if "WL" in _prior_po.columns and len(_prior_po) > 0:
                    team_series_wins_in = float((_prior_po["WL"] == "W").sum())
                    opp_series_wins_in = float((_prior_po["WL"] == "L").sum())
                if team_series_wins_in >= 3.0 or opp_series_wins_in >= 3.0:
                    is_elimination_game = 1.0
            _home_bit = 1.0 if bool(is_home) else 0.0
            playoff_home = is_playoff * _home_bit
            playoff_away = is_playoff * (1.0 - _home_bit)

            last5 = hist.tail(5)
            minutes = hist["MIN"].tolist()
            mins_last5 = float(last5["MIN"].mean()) if not last5.empty else float(hist["MIN"].mean())
            mins_season = float(hist["MIN"].mean()) if len(hist) else 0.0

            fg_pct_recent = float(last5["FG_PCT"].mean()) if not last5.empty and "FG_PCT" in last5.columns else 0.0
            fg3_pct_recent = float(last5["FG3_PCT"].mean()) if not last5.empty and "FG3_PCT" in last5.columns else 0.0
            ft_pct_recent = float(last5["FT_PCT"].mean()) if not last5.empty and "FT_PCT" in last5.columns else 0.0
            fga_per_game = float(hist["FGA"].mean()) if "FGA" in hist.columns else 0.0
            fg3a_per_game = float(hist["FG3A"].mean()) if "FG3A" in hist.columns else 0.0
            fta_per_game = float(hist["FTA"].mean()) if "FTA" in hist.columns else 0.0

            oreb_per_game = float(hist["OREB"].mean()) if "OREB" in hist.columns else 0.0
            dreb_per_game = float(hist["DREB"].mean()) if "DREB" in hist.columns else 0.0

            plus_minus_avg = float(hist["PLUS_MINUS"].mean()) if "PLUS_MINUS" in hist.columns else 0.0
            fouls_per_game = float(hist["PF"].mean()) if "PF" in hist.columns else 0.0
            win_rate_last10 = float((hist.tail(10)["WL"] == "W").sum() / min(10, len(hist))) if "WL" in hist.columns else 0.5

            total_fga = hist["FGA"].sum() if "FGA" in hist.columns else 1
            total_pts = hist["PTS"].sum() if "PTS" in hist.columns else 0
            points_per_shot = total_pts / max(total_fga, 1)
            
            total_ast = hist["AST"].sum() if "AST" in hist.columns else 0
            total_tov = hist["TOV"].sum() if "TOV" in hist.columns else 1
            ast_to_tov_ratio = total_ast / max(total_tov, 1)
            
            total_reb = hist["REB"].sum() if "REB" in hist.columns else 0
            total_min = hist["MIN"].sum() if "MIN" in hist.columns else 1
            reb_rate_per_36 = (total_reb / max(total_min, 1)) * 36
            
            recent_pts_per_shot = last5["PTS"].sum() / max(last5["FGA"].sum(), 1) if not last5.empty and "FGA" in last5.columns else points_per_shot
            scoring_efficiency_trend = recent_pts_per_shot - points_per_shot
            
            recent_usage = last5["FGA"].mean() if not last5.empty and "FGA" in last5.columns else 0
            season_usage = hist["FGA"].mean() if "FGA" in hist.columns else 0
            usage_trend = recent_usage - season_usage
            
            minutes_volatility = float(hist["MIN"].std()) if "MIN" in hist.columns else 0.0
            
            if "PLUS_MINUS" in hist.columns:
                blowout_games = (hist["PLUS_MINUS"].abs() > 15).sum()
                close_games = (hist["PLUS_MINUS"].abs() <= 5).sum()
                blowout_game_pct = blowout_games / max(len(hist), 1)
                close_game_pct = close_games / max(len(hist), 1)
            else:
                blowout_game_pct = 0.0
                close_game_pct = 0.0
            
            pts_mean = hist["PTS"].mean() if "PTS" in hist.columns else 0
            pts_std = hist["PTS"].std() if "PTS" in hist.columns else 0
            consistency_score = max(0.0, 1.0 - (pts_std / max(pts_mean, 1)))
            
            ceiling_threshold = pts_mean * 1.5
            ceiling_games = (hist["PTS"] >= ceiling_threshold).sum() if "PTS" in hist.columns else 0
            ceiling_game_frequency = ceiling_games / max(len(hist), 1)

            # --- Pre-computed features: replace zero defaults with real values ---
            # Date / schedule features
            _cur_date = pd.to_datetime(row.get("GAME_DATE"))
            if len(hist) >= 1 and "GAME_DATE" in hist.columns:
                _last_date = hist.iloc[-1]["GAME_DATE"]
                _days_gap = int((_cur_date - _last_date).days)
                _rest_days = float(max(0, _days_gap - 1))
                _is_b2b = 1.0 if _rest_days == 0 else 0.0
                _days_since_last = float(_days_gap)
            else:
                _rest_days = 2.0; _is_b2b = 0.0; _days_since_last = 2.0
            _hist_dates = hist["GAME_DATE"] if "GAME_DATE" in hist.columns else pd.Series(dtype="datetime64[ns]")
            _7d_ago = _cur_date - pd.Timedelta(days=7)
            _games_in_7d = int((_hist_dates >= _7d_ago).sum()) if len(_hist_dates) > 0 else 3

            # Compute player age relative to this game's date (not today's date)
            if _player_birthdate is not None:
                _player_age = float((_cur_date - _player_birthdate).days / 365.25)

            # Travel / arena features
            _arena_info = ARENA_DATA.get(int(team_id) if team_id else 0, {})
            try:
                _prev_matchup = hist.iloc[-1].get("MATCHUP") if len(hist) >= 1 else None
                _prev_team_abbr = _parse_matchup(_prev_matchup)[0] if _prev_matchup else None
                _prev_team_id = _team_id(_prev_team_abbr) if _prev_team_abbr else None
                _tz_change, _coast_to_coast, _travel_dist = calculate_travel_metrics(_prev_team_id, team_id, is_home)
            except Exception:
                _tz_change, _coast_to_coast, _travel_dist = 0.0, 0.0, 0.0

            # Season-phase calendar features
            # NBA regular season runs roughly Oct 22 – Apr 14
            _season_year = _cur_date.year if _cur_date.month >= 10 else _cur_date.year - 1
            _season_start = pd.Timestamp(f"{_season_year}-10-22")
            _season_end   = pd.Timestamp(f"{_season_year + 1}-04-14")
            _days_into_season = max(0, (_cur_date - _season_start).days)
            _season_total_days = max(1, (_season_end - _season_start).days)
            _games_remaining_approx = max(0, int(82 * (1.0 - _days_into_season / _season_total_days)))
            # 0=early (first 20%), 1=mid (20-80%), 2=late (last 20%)
            _phase_frac = _days_into_season / _season_total_days
            _season_phase_numeric = 0.0 if _phase_frac < 0.20 else (2.0 if _phase_frac > 0.80 else 1.0)

            # Minutes fatigue
            _mins_arr = hist["MIN"].values.astype(float) if "MIN" in hist.columns else np.full(len(hist), mins_season)
            _mins_l3 = float(np.mean(_mins_arr[-3:])) if len(_mins_arr) >= 3 else mins_season
            _mins_l7 = float(np.mean(_mins_arr[-7:])) if len(_mins_arr) >= 7 else mins_season
            _total_mins_l3 = float(np.sum(_mins_arr[-3:])) if len(_mins_arr) >= 3 else _mins_l3 * 3
            _total_mins_l5 = float(np.sum(_mins_arr[-5:])) if len(_mins_arr) >= 5 else mins_last5 * 5
            _total_mins_l7 = float(np.sum(_mins_arr[-7:])) if len(_mins_arr) >= 7 else _mins_l7 * 7
            _fatigue = min(1.0, _total_mins_l5 / max(mins_season * 5, 1.0)) if mins_season > 0 else 0.5

            # Rebound / FT rates (player-level, not prop-specific)
            _total_reb_pg = oreb_per_game + dreb_per_game
            _oreb_rate = oreb_per_game / max(_total_reb_pg, 0.1)
            _dreb_rate = dreb_per_game / max(_total_reb_pg, 0.1)
            _total_reb_rate = _total_reb_pg / max(mins_season / 36.0, 0.1) if mins_season > 0 else 0.1
            _ft_rate = fta_per_game / max(fga_per_game, 1.0)
            _foul_draw = fta_per_game / max(mins_season / 36.0, 0.1) if mins_season > 0 else 0.0
            _fouls_drawn_pg = fta_per_game * 0.44  # proxy: FTA drives foul count
            _and_one_freq = min(0.3, fta_per_game / max(fga_per_game * 3, 1.0))

            # Performance splits (W vs L)
            if "WL" in hist.columns and "PTS" in hist.columns:
                _win_m = hist["WL"] == "W"
                _perf_lead = float(hist.loc[_win_m, "PTS"].mean()) if _win_m.any() else pts_mean
                _perf_trail = float(hist.loc[~_win_m, "PTS"].mean()) if (~_win_m).any() else pts_mean
            else:
                _perf_lead = pts_mean; _perf_trail = pts_mean

            team_ctx = predictor.get_team_context(team_id)
            opp_ctx = predictor.get_team_context(opp_id)

            team_style = (team_ctx or {}).get("style", {}) or {}
            opp_style = (opp_ctx or {}).get("style", {}) or {}
            opp_base = (opp_ctx or {}).get("base", {}) or {}
            league_avgs = (opp_ctx or {}).get("league_avgs", {}) or {}

            dvp = dvp_map.get((int(opp_id), dvp_pos))
            dvp_avg = dvp_pos_avgs.get(dvp_pos, {})
            dvp_deltas = {}
            dvp_gp = 0
            if dvp:
                dvp_gp = int(dvp.get("gp", 0) or 0)
                for k in ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]:
                    dvp_deltas[f"dvp_{k}_delta"] = float(dvp.get(k, 0.0) - float(dvp_avg.get(k, 0.0) or 0.0))

            special_defenders = defenders_map.get((int(opp_id), pos_group), [])
            primary_def = special_defenders[0] if special_defenders else None

            # --- Team-level precomputed context (DB tables) ---
            _ts   = _precomp["team_stats"].get(int(opp_id), {})
            _tf   = _precomp["team_foul"].get(int(opp_id), {})
            _rst  = _precomp["rest_splits"].get(int(opp_id), {})
            _std_team = _precomp["standings"].get(int(team_id), {})
            _std_opp  = _precomp["standings"].get(int(opp_id), {})
            _osz  = _precomp["opp_shot_zones"].get(int(opp_id), {})
            _osy  = _precomp["opp_synergy"].get(int(opp_id), {})

            # Rolling DVP (5-game and 10-game windows for the opponent)
            _dvp_r5  = _precomp["dvp_rolling"].get((int(opp_id), 5), {})
            _dvp_r10 = _precomp["dvp_rolling"].get((int(opp_id), 10), {})
            _dvp_avg_row = dvp_avg  # position-level season average for delta computation
            _dvp_pts_delta_last5  = float(_dvp_r5.get("pts",  0.0)) - float(_dvp_avg_row.get("pts",  0.0) or 0.0)
            _dvp_pts_delta_last10 = float(_dvp_r10.get("pts", 0.0)) - float(_dvp_avg_row.get("pts",  0.0) or 0.0)
            _dvp_reb_delta_last5  = float(_dvp_r5.get("reb",  0.0)) - float(_dvp_avg_row.get("reb",  0.0) or 0.0)
            _dvp_ast_delta_last5  = float(_dvp_r5.get("ast",  0.0)) - float(_dvp_avg_row.get("ast",  0.0) or 0.0)
            _dvp_fg3m_delta_last5 = float(_dvp_r5.get("fg3m", 0.0)) - float(_dvp_avg_row.get("fg3m", 0.0) or 0.0)

            # Opponent foul rates (from team_stats and team_foul_rates)
            _opp_foul_rate_season = float(_ts.get("foul_rate_season", _tf.get("foul_rate_season", 0.0)))
            _opp_foul_rate_last5  = float(_ts.get("foul_rate_last5",  _tf.get("foul_rate_last5",  0.0)))

            # Team standings context
            _team_win_pct     = float(_std_team.get("win_pct", 0.5))
            _team_conf_rank   = float(_std_team.get("conf_rank", 8))
            _team_streak      = float(_std_team.get("current_streak", 0))
            _team_l10_wins    = float(_std_team.get("l10_wins", 5))
            _opp_win_pct      = float(_std_opp.get("win_pct", 0.5))
            _opp_conf_rank    = float(_std_opp.get("conf_rank", 8))
            _opp_streak       = float(_std_opp.get("current_streak", 0))
            _opp_l10_wins     = float(_std_opp.get("l10_wins", 5))
            _opp_win_rate_l10 = _opp_l10_wins / 10.0
            _games_back       = float(_std_team.get("games_back", 5.0))

            # Opponent rest splits (b2b vs rested)
            _opp_b2b_def_rating   = float(_rst.get("b2b_def_rating",   (opp_ctx or {}).get("defensive_rating", 110.0)))
            _opp_b2b_pace         = float(_rst.get("b2b_pace",         (opp_ctx or {}).get("pace", 100.0)))
            _opp_b2b_pts_allowed  = float(_rst.get("b2b_pts_allowed",  0.0))
            _opp_rested_def_rating= float(_rst.get("rested_def_rating",(opp_ctx or {}).get("defensive_rating", 110.0)))
            _opp_rested_pace      = float(_rst.get("rested_pace",      (opp_ctx or {}).get("pace", 100.0)))

            # Infer opponent back-to-back from rest splits availability (proxy: b2b_def_rating populated)
            _opp_is_b2b = 1.0 if (_rst and _rst.get("b2b_def_rating") and _is_b2b == 0.0) else 0.0
            # Rest advantage: player rest days minus assumed opponent rest days
            _opp_rest_days_est = 0.0 if _opp_is_b2b else 2.0
            _rest_advantage = _rest_days - _opp_rest_days_est
            _rest_advantage_abs = abs(_rest_advantage)
            _both_rested = 1.0 if (_rest_days >= 2 and _opp_rest_days_est >= 2) else 0.0

            # Player vs specific opponent
            _pvo = _vs_opp_map.get((player_id, int(opp_id)), {})
            _vs_opp_gp      = float(_pvo.get("gp", 0))
            _vs_opp_avg_pts = float(_pvo.get("avg_stat_pts", 0.0))
            _vs_opp_fg_pct  = float(_pvo.get("fg_pct", 0.0))
            _vs_opp_ts_pct  = float(_pvo.get("ts_pct", 0.0))
            _vs_opp_avg_min = float(_pvo.get("avg_min", 0.0))

            # --- Previously "unavailable" features — now computed from existing data ---
            _games_since_return, _missed_before = _compute_injury_trajectory_from_df(gl, idx)
            _implied_total = _compute_implied_game_total(team_ctx, opp_ctx)
            _primary_def_active = _compute_primary_defender_active(primary_def, gl, idx)
            _opp_lineup_changes = _estimate_opp_lineup_changes(_precomp["dvp_rolling"], int(opp_id))

            # Per-100 possessions (needs pace from team_ctx)
            _pace = float((team_ctx or {}).get("pace", 100.0))
            _poss_pg = _pace * (mins_season / 48.0) if mins_season > 0 else 50.0
            _p100 = 100.0 / max(_poss_pg, 1.0)
            _pts_p100 = pts_mean * _p100
            _ast_p100 = float(hist["AST"].mean() if "AST" in hist.columns else 0) * _p100
            _reb_p100 = _total_reb_pg * _p100
            _stl_p100 = float(hist["STL"].mean() if "STL" in hist.columns else 0) * _p100
            _blk_p100 = float(hist["BLK"].mean() if "BLK" in hist.columns else 0) * _p100
            _tov_p100 = (total_tov / max(len(hist), 1)) * _p100

            # --- Stub-replacement pre-computations ---
            # 1. Recent away streak: consecutive away games from end of hist
            _away_streak = 0
            if "MATCHUP" in hist.columns and len(hist) > 0:
                for _i in range(len(hist) - 1, -1, -1):
                    if "@" in str(hist.iloc[_i]["MATCHUP"]):
                        _away_streak += 1
                    else:
                        break

            # 2. Season phase (month-based): 0=early, 1=mid, 2=late, 3=playoff push
            _month = _cur_date.month
            _season_phase = 0 if _month in (10, 11) else (1 if _month in (12, 1) else (2 if _month in (2, 3) else 3))

            # 3. Win pct vs this opponent from hist
            _vs_opp_games = hist[hist["MATCHUP"].str.contains(str(opp_abbrev), na=False)] if ("MATCHUP" in hist.columns and opp_abbrev) else pd.DataFrame()
            _vs_team_win_pct = float((_vs_opp_games["WL"] == "W").mean()) if (len(_vs_opp_games) > 0 and "WL" in _vs_opp_games.columns) else 0.5

            # 4 & 5. Home/away performance split (target_col-dependent part moved
            # inside make_features so each prop gets its OWN home/away split instead
            # of everything defaulting to PTS).
            _home_games = hist[hist["MATCHUP"].str.contains("vs.", na=False)] if "MATCHUP" in hist.columns else pd.DataFrame()
            _away_games = hist[hist["MATCHUP"].str.contains("@", na=False)] if "MATCHUP" in hist.columns else pd.DataFrame()
            _is_away_now = "@" in str(row.get("MATCHUP", ""))

            def make_features(stat_values, last5_avg, season_avg, stddev):
                # Per-prop home/away target split (target_col is bound in the enclosing
                # STAT_TARGETS/COMBO_TARGETS loop when make_features is called).
                _target_mean = float(season_avg)
                if target_col in _home_games.columns:
                    _home_avg_target = float(_home_games[target_col].mean()) if len(_home_games) > 0 else _target_mean
                else:
                    _home_avg_target = _target_mean
                if target_col in _away_games.columns:
                    _away_avg_target = float(_away_games[target_col].mean()) if len(_away_games) > 0 else _target_mean
                else:
                    _away_avg_target = _target_mean
                _vs_team_home_away_split = _home_avg_target - _away_avg_target
                _player_vs_arena = (_away_avg_target - _target_mean) if _is_away_now else (_home_avg_target - _target_mean)
                # --- Prop-specific time-series features (computed from stat_values) ---
                _sv = stat_values  # shorthand
                # EWM
                _sv_s = pd.Series(_sv)
                _ewm03 = float(_sv_s.ewm(alpha=0.3).mean().iloc[-1]) if len(_sv) > 0 else season_avg
                _ewm05 = float(_sv_s.ewm(alpha=0.5).mean().iloc[-1]) if len(_sv) > 0 else season_avg
                # Rolling game-count averages
                _r7 = float(np.mean(_sv[-7:])) if len(_sv) >= 7 else season_avg
                _r14 = float(np.mean(_sv[-14:])) if len(_sv) >= 14 else season_avg
                _r30 = float(np.mean(_sv[-30:])) if len(_sv) >= 30 else season_avg
                # Trend slopes
                def _slope(arr):
                    if len(arr) < 2: return 0.0
                    try: return float(np.polyfit(range(len(arr)), arr, 1)[0])
                    except: return 0.0
                _t5 = _slope(_sv[-5:])
                _t10 = _slope(_sv[-10:]) if len(_sv) >= 10 else 0.0
                # Volatility ratio
                _vol = float(np.std(_sv[-5:]) / max(stddev, 0.01)) if len(_sv) >= 5 else 1.0
                # Momentum (last5 vs prev5)
                _prev5 = _sv[-10:-5] if len(_sv) >= 10 else _sv[:max(1, len(_sv)//2)]
                _prev5_avg = float(np.mean(_prev5)) if len(_prev5) > 0 else season_avg
                _momentum = (last5_avg - _prev5_avg) / max(abs(_prev5_avg), 0.1)
                # Games above season avg in 7/14 game windows
                _g_above7 = int(sum(1 for v in _sv[-7:] if v > season_avg)) if len(_sv) >= 7 else 0
                _g_above14 = int(sum(1 for v in _sv[-14:] if v > season_avg)) if len(_sv) >= 14 else 0
                # Streaks (over/under rolling line)
                _line_proxy = float(np.mean(_sv[-10:])) if len(_sv) >= 10 else season_avg
                _c_over = 0; _c_under = 0
                for _v in reversed(_sv[-10:] if len(_sv) >= 10 else _sv):
                    if _v > _line_proxy:
                        if _c_under > 0: break
                        _c_over += 1
                    else:
                        if _c_over > 0: break
                        _c_under += 1
                _hot = 1.0 if len(_sv) >= 3 and all(v > season_avg for v in _sv[-3:]) else 0.0
                return {
                    "recent_avg": float(last5_avg),
                    "season_avg": float(season_avg),
                    "stddev": float(stddev),
                    "games_played": len(stat_values),
                    "max_recent": float(max(stat_values)),
                    "min_recent": float(min(stat_values)),
                    "mins_last5": mins_last5,
                    "mins_season": mins_season,
                    # Tier 1 additions
                    "fg_pct_recent": fg_pct_recent,
                    "fg3_pct_recent": fg3_pct_recent,
                    "ft_pct_recent": ft_pct_recent,
                    "fga_per_game": fga_per_game,
                    "fg3a_per_game": fg3a_per_game,
                    "fta_per_game": fta_per_game,
                    "oreb_per_game": oreb_per_game,
                    "dreb_per_game": dreb_per_game,
                    "plus_minus_avg": plus_minus_avg,
                    "fouls_per_game": fouls_per_game,
                    "win_rate_last10": win_rate_last10,
                    "rest_days": _rest_days,
                    "is_home_game": is_home,
                    "recent_away_streak": float(_away_streak),
                    "team_pace": float(team_ctx.get("pace", 100.0)),
                    "opp_pace": float(opp_ctx.get("pace", 100.0)),
                    "team_off_rating": float(team_ctx.get("offensive_rating", 110.0)),
                    "team_def_rating": float(team_ctx.get("defensive_rating", 110.0)),
                    "opp_off_rating": float(opp_ctx.get("offensive_rating", 110.0)),
                    "opp_def_rating": float(opp_ctx.get("defensive_rating", 110.0)),
                    "team_key_players_out": int(_precomp["injury_status"].get(int(team_id) if team_id else 0, {}).get("key_players_out", 0)),
                    "opp_key_players_out":  int(_precomp["injury_status"].get(int(opp_id)   if opp_id   else 0, {}).get("key_players_out", 0)),
                    "opp_injury_impact":    float(_precomp["injury_status"].get(int(opp_id) if opp_id   else 0, {}).get("total_impact", 0.0)),
                    "team_pts_fb": float(team_style.get("pts_fb", 0.0)),
                    "opp_pts_fb_allowed": float(opp_style.get("opp_pts_fb", 0.0)),
                    "team_pts_off_tov": float(team_style.get("pts_off_tov", 0.0)),
                    "opp_pts_off_tov_allowed": float(opp_style.get("opp_pts_off_tov", 0.0)),
                    "opp_pts_paint": float(opp_style.get("pts_paint", 0.0)),
                    "opp_fga": float(opp_base.get("fga", 0.0)),
                    "opp_fg_pct": float(opp_base.get("fg_pct", 0.47)),
                    "opp_fg3a": float(opp_base.get("fg3a", 0.0)),
                    "opp_fg3_pct": float(opp_base.get("fg3_pct", 0.36)),
                    "opp_tov": float(opp_base.get("tov", 0.0)),
                    "opp_stl": float(opp_base.get("stl", 0.0)),
                    "opp_blk": float(opp_base.get("blk", 0.0)),
                    "lg_pts_fb": float(league_avgs.get("pts_fb", 12.0)),
                    "lg_opp_pts_fb": float(league_avgs.get("opp_pts_fb", 12.0)),
                    "lg_pts_off_tov": float(league_avgs.get("pts_off_tov", 16.0)),
                    "lg_opp_pts_off_tov": float(league_avgs.get("opp_pts_off_tov", 16.0)),
                    "lg_fga": float(league_avgs.get("fga", 86.0)),
                    "lg_fg_pct": float(league_avgs.get("fg_pct", 0.47)),
                    "lg_fg3a": float(league_avgs.get("fg3a", 35.0)),
                    "lg_tov": float(league_avgs.get("tov", 14.0)),
                    "lg_stl": float(league_avgs.get("stl", 7.0)),
                    "vs_team_avg": _vs_opp_avg_pts,
                    "matchup_games": int(_vs_opp_gp),
                    "dvp_gp": dvp_gp,
                    "primary_defender_score01": float((primary_def or {}).get("score01", 0.0) or 0.0),
                    "points_per_shot": float(points_per_shot),
                    "ast_to_tov_ratio": float(ast_to_tov_ratio),
                    "reb_rate_per_36": float(reb_rate_per_36),
                    "scoring_efficiency_trend": float(scoring_efficiency_trend),
                    "usage_trend": float(usage_trend),
                    "minutes_volatility": float(minutes_volatility),
                    "blowout_game_pct": float(blowout_game_pct),
                    "close_game_pct": float(close_game_pct),
                    "opp_def_rating_home_away_split": float(_precomp["home_away"].get(int(opp_id) if opp_id else 0, {}).get("home_away_def_split", 0.0)),
                    "opp_blocks_per_game_last5": float(_ts.get("opp_blk_last5", opp_base.get("blk", 0.0))),
                    "opp_steals_per_game_last5": float(_ts.get("opp_stl_last5", opp_base.get("stl", 0.0))),
                    "days_rest_opponent": _opp_rest_days_est,
                    "opponent_back_to_back": _opp_is_b2b,
                    "playoff_implications": 0,
                    "rivalry_game": 0,
                    "national_tv_game": 0,  # Not available in game log data
                    "season_phase": float(_season_phase),
                    "primary_teammate_out": 0,
                    "secondary_teammate_out": 0,
                    "new_teammate_games": 0,
                    "lineup_stability_score": float(_precomp["lineup_stats"].get(int(team_id) if team_id else 0, {}).get("lineup_continuity", 1.0)),
                    "bench_strength": float(_precomp["lineup_stats"].get(int(team_id) if team_id else 0, {}).get("bench_strength", 0.0)),
                    "pts_vs_top10_defenses": float(pts_mean),
                    "pts_vs_bottom10_defenses": float(pts_mean),
                    "consistency_score": float(consistency_score),
                    "ceiling_game_frequency": float(ceiling_game_frequency),
                        "def_fg_pct_allowed": 0.0,
                        "def_rating_individual": 0.0,
                        "pnr_ball_handler_pct": float(_pt.get("pnr_bh_poss_pct",    0.0)),
                        "pnr_roll_man_pct":    float(_pt.get("pnr_roll_poss_pct",  0.0)),
                        "isolation_pct":       float(_pt.get("iso_poss_pct",        0.0)),
                        "spot_up_pct":         float(_pt.get("spotup_poss_pct",     0.0)),
                        "post_up_pct":         float(_pt.get("postup_poss_pct",     0.0)),
                        "transition_pct":      float(_pt.get("transition_poss_pct", 0.0)),
                        "consecutive_over_games": float(_c_over),
                        "consecutive_under_games": float(_c_under),
                        "hot_hand_indicator": _hot,
                        "recent_variance_spike": float(np.std(_sv[-3:]) / max(stddev, 0.1) - 1.0) if len(_sv) >= 3 else 0.0,
                        "playoff_seeding_impact": 0.5,
                        "tanking_indicator": 1.0 if _team_conf_rank >= 13 and _team_win_pct < 0.35 else 0.0,
                        "must_win_situation": 1.0 if _team_conf_rank <= 10 and _games_back <= 3.0 else 0.0,
                        "games_back_from_playoff": _games_back,
                        "fourth_quarter_usage_rate": 0.25 if mins_season > 30 else 0.18,
                        "garbage_time_minutes_pct": float(blowout_game_pct * 0.15),
                        "typical_substitution_minute": float(min(48.0, mins_season + 3.0)),
                        "crunch_time_usage": 0.28 if mins_season > 28 else 0.15,
                        "career_vs_defender": 0.0,
                        "recent_vs_defender": 0.0,
                        "player_vs_arena": float(_player_vs_arena),
                        "avg_shot_distance": float(_sz.get("rim_fga_pct", 0.25)) * 3.0 + float(_sz.get("paint_fga_pct", 0.20)) * 8.0 + float(_sz.get("midrange_fga_pct", 0.20)) * 16.0 + float(_sz.get("corner3_fga_pct", 0.10)) * 22.0 + float(_sz.get("above_break3_fga_pct", 0.25)) * 24.0,
                        "contested_shot_pct": float(_sp.get("tight_shot_freq", 0.5)),
                        "open_shot_pct": float(_sp.get("open_shot_freq", 0.3)),
                        "wide_open_shot_pct": max(0.0, float(_sp.get("open_shot_freq", 0.5)) - float(_sp.get("tight_shot_freq", 0.3))),
                        "catch_and_shoot_pct": float(_sp.get("catch_shoot_freq", 0.3)),
                        "pull_up_shot_pct": float(_sp.get("pullup_freq", 0.3)),
                        "paint_touch_frequency": float(hist[hist['PTS'] > 0]['PTS'].count() * 0.1) if 'PTS' in hist.columns else 0.0,
                        "corner_three_pct": float(fg3_pct_recent),
                        "above_break_three_pct": float(fg3_pct_recent),
                        "restricted_area_fg_pct": float(fg_pct_recent),
                        "mid_range_frequency": float(_sz.get("midrange_fga_pct", 0.0)),
                        "shot_quality_vs_expected": 0.0,
                        "avg_shot_clock_time": 12.0,
                        "late_clock_shot_frequency": 0.15,
                        "early_clock_shot_frequency": 0.25,
                        "touches_per_game": float(hist["FGA"].mean() + hist["AST"].mean() if "FGA" in hist.columns and "AST" in hist.columns else 0.0),
                        "avg_dribbles_per_touch": float(_trk.get("avg_drib_per_touch", 2.0)),
                        "avg_seconds_per_touch":  float(_trk.get("time_of_poss_pg", 2.5)) * 60.0 / max(float(_trk.get("touches_pg", 50.0)), 1.0),
                        "elbow_touches_per_game": float(_trk.get("elbow_touches_pg", 0.0)),
                        "post_touches_per_game":  float(_trk.get("paint_touches_pg", 0.0)),
                        "paint_touches_per_game": float(hist["REB"].mean() * 0.5 if "REB" in hist.columns else 0.0),
                        "front_court_touches_per_game": float(hist["AST"].mean() + hist["FGA"].mean() if "AST" in hist.columns and "FGA" in hist.columns else 0.0),
                        "time_of_possession_per_game": float(_trk.get("time_of_poss_pg", mins_season * 0.25)),
                        "touches_per_possession": float(_trk.get("touches_pg", 50.0)) / max(float((team_ctx or {}).get("pace", 100.0)), 1.0),
                        "avg_points_per_touch":   float(pts_mean) / max(float(_trk.get("touches_pg", 50.0)), 1.0),
                        "net_rating_with_starters": 0.0,
                        "usage_rate_with_star_out": float(hist["FGA"].mean() * 1.1 if "FGA" in hist.columns else 0.0),
                        "minutes_with_starting_lineup_pct": 0.65 if mins_season > 25 else 0.35,
                        "five_man_unit_net_rating": float(_precomp["lineup_stats"].get(int(team_id) if team_id else 0, {}).get("top_lineup_net_rating", 0.0)),
                        "lineups_played_count": float(_precomp["lineup_stats"].get(int(team_id) if team_id else 0, {}).get("lineups_played_count", 1.0)),
                        "is_home_game_num": float(is_home if is_home is not None else 1),
                        "dvp_pts_delta": float(dvp_deltas.get("dvp_pts_delta", 0.0)),
                        "dvp_reb_delta": float(dvp_deltas.get("dvp_reb_delta", 0.0)),
                        "dvp_ast_delta": float(dvp_deltas.get("dvp_ast_delta", 0.0)),
                        "dvp_stl_delta": float(dvp_deltas.get("dvp_stl_delta", 0.0)),
                        "dvp_blk_delta": float(dvp_deltas.get("dvp_blk_delta", 0.0)),
                        "dvp_tov_delta": float(dvp_deltas.get("dvp_tov_delta", 0.0)),
                        "dvp_fg3m_delta": float(dvp_deltas.get("dvp_fg3m_delta", 0.0)),
                        "usage_rate": float(hist["FGA"].mean() + hist["FTA"].mean() * 0.44 + hist["TOV"].mean() if all(c in hist.columns for c in ["FGA", "FTA", "TOV"]) else 0.0) / max(mins_season, 1) * 48,
                        "true_shooting_pct": float(pts_mean / (2 * (fga_per_game + 0.44 * fta_per_game)) if (fga_per_game + 0.44 * fta_per_game) > 0 else 0.0),
                        "effective_fg_pct": float((_adv.get("efg_pct") or
                            (((hist["FGM"].mean() if "FGM" in hist.columns else fga_per_game * 0.47)
                              + 0.5 * (hist["FG3M"].mean() if "FG3M" in hist.columns else fg3a_per_game * 0.36))
                             / max(fga_per_game, 1.0)))),
                        "net_rating": float(hist["PLUS_MINUS"].mean() if "PLUS_MINUS" in hist.columns else 0.0),
                        "assist_percentage": float(hist["AST"].mean() / max(fga_per_game, 1) if "AST" in hist.columns else 0.0),
                        "rebound_percentage": float((oreb_per_game + dreb_per_game) / 100.0),
                        "last_3_games_trend": float(hist[target_col].tail(3).mean() - season_avg if len(hist) >= 3 else 0.0),
                        "last_5_games_trend": float(last5_avg - season_avg),
                        "last_10_games_trend": float(hist[target_col].tail(10).mean() - season_avg if len(hist) >= 10 else 0.0),
                        "games_above_season_avg_last5": float(sum(1 for v in hist[target_col].tail(5) if v > season_avg)),
                        "is_back_to_back": _is_b2b,
                        "days_since_last_game": _days_since_last,
                        "games_in_last_7_days": float(_games_in_7d),
                        "travel_distance": float(_travel_dist),
                        "time_zone_change": float(_tz_change),
                        "arena_altitude":   float(_arena_info.get("altitude", 0.0)),
                        "arena_capacity":   float(_arena_info.get("capacity", 18000.0)),
                        "home_court_advantage_rating": 3.5 if (is_home is not None and is_home) else (-3.5 if is_home is not None else 0.0),
                        "vs_team_win_pct": float(_vs_team_win_pct),
                        "vs_team_last_season_avg": float(_vs_opp_avg_pts) if _vs_opp_gp > 0 else float(season_avg),
                        "vs_team_home_away_split": float(_vs_team_home_away_split),
                        "clutch_minutes_per_game": float(_clt.get("clutch_min_pg", mins_season * 0.15)),
                        "paint_fga_per_game": float(_sz.get("paint_fga_pct", 0.40)) * fga_per_game,
                        "mid_range_fga_per_game": float(_sz.get("midrange_fga_pct", 0.25)) * fga_per_game,
                        "rim_fga_per_game": float(_sz.get("rim_fga_pct", 0.35)) * fga_per_game,
                        "corner_3_pct": float(fg3_pct_recent * 1.05),
                        "above_break_3_pct": fg3_pct_recent,
                        "coast_to_coast": float(_coast_to_coast),
                        "top_lineup_minutes_pct": 0.65 if mins_season > 25 else 0.35,
                        "model_accuracy_player": 0.70,
                        "avg_prediction_error_player": float(stddev * 0.5),
                        "calibration_score_player": 0.75,
                        # Tier 7: Time-Series Features (computed from stat_values)
                        "rolling_7day_avg": _r7,
                        "rolling_14day_avg": _r14,
                        "rolling_30day_avg": _r30,
                        "ewm_alpha_0.3": _ewm03,
                        "ewm_alpha_0.5": _ewm05,
                        "trend_slope_10games": _t10,
                        "trend_slope_5games": _t5,
                        "volatility_ratio": _vol,
                        "momentum_score": _momentum,
                        "games_above_season_avg_7day": float(_g_above7),
                        "games_above_season_avg_14day": float(_g_above14),
                        # Tier 7: Enhanced Matchup Features
                        "head_to_head_avg": _vs_opp_avg_pts if _vs_opp_gp > 0 else float(season_avg),
                        "head_to_head_games": int(_vs_opp_gp),
                        "position_vs_position_dvp": float(dvp_deltas.get("dvp_pts_delta", 0.0)),
                        "matchup_pace": float(((team_ctx or {}).get("pace", 100.0) + (opp_ctx or {}).get("pace", 100.0)) / 2.0),
                        "defender_switching_frequency": 0.0,
                        "historical_game_script_avg": float(blowout_game_pct - close_game_pct),
                        # Tier 8: Relative Rest Advantage
                        "rest_advantage": _rest_advantage,
                        "rest_advantage_abs": _rest_advantage_abs,
                        "both_teams_rested": _both_rested,
                        # Tier 8: Opponent Recent Form
                        "opp_def_rating_last5": float(_ts.get("opp_def_rating_last5", (opp_ctx or {}).get("defensive_rating", 110.0))),
                        # Use season def_rating as baseline for last10 (dvp_rolling has pts-allowed, not def_rating)
                        "opp_def_rating_last10": float((opp_ctx or {}).get("defensive_rating", 110.0)),
                        "opp_def_rating_trend": float(_dvp_pts_delta_last5),
                        # opp_pace_last5: use opp season pace (dvp_rolling "pts" is pts-allowed, not pace)
                        "opp_pace_last5": float((opp_ctx or {}).get("pace", 100.0)),
                        "opp_win_rate_last10": _opp_win_rate_l10,
                        # Tier 8: Player Age & Experience (DB precomputed > CommonPlayerInfo)
                        "player_age": float(_adv.get("age", _player_age)),
                        "years_experience": float(_adv.get("years_experience", _player_exp)),
                        "is_rookie": 1.0 if float(_adv.get("years_experience", _player_exp)) <= 1 else 0.0,
                        "is_veteran": 1.0 if float(_adv.get("years_experience", _player_exp)) >= 10 else 0.0,
                        # Tier 8: Game Script Prediction
                        "expected_game_script": float(blowout_game_pct - close_game_pct),
                        "blowout_probability": float(blowout_game_pct),
                        "close_game_probability": float(close_game_pct),
                        # Tier 8: Quarter Performance (default — need play-by-play)
                        "first_quarter_avg": float(_qs.get("q1_avg", last5_avg * 0.25)),
                        "fourth_quarter_avg": float(_qs.get("q4_avg", last5_avg * 0.25)),
                        "clutch_performance_score": float(consistency_score),
                        # Tier 8: Shot Selection Quality (defaults)
                        "shot_selection_rating": float(fg_pct_recent),
                        "bad_shot_frequency": float(1.0 - fg_pct_recent),
                        "shot_clock_management": 12.0,
                        # Tier 8: Team Chemistry (defaults)
                        "teammate_chemistry_score": 0.5,
                        "lineup_continuity": float(_precomp["lineup_stats"].get(int(team_id) if team_id else 0, {}).get("lineup_continuity", 0.7)),
                        "team_win_streak": max(0.0, float(_team_streak)),
                        "team_loss_streak": max(0.0, float(-_team_streak)),
                        # Tier 8: Enhanced Defensive Matchup (from team_special_defenders + player_advanced_stats)
                        "primary_defender_rating": float((primary_def or {}).get("def_rating", 110.0)),
                        "primary_defender_age":    float(_precomp["advanced"].get(int((primary_def or {}).get("player_id", 0) or 0), {}).get("age", 26.0)),
                        "defender_size_mismatch":  float(_precomp["advanced"].get(int((primary_def or {}).get("player_id", 0) or 0), {}).get("height_inches", 78.0)) - float(_adv.get("height_inches", 78.0)),
                        "defender_recent_form":    float((primary_def or {}).get("score01", 0.5)),
                        # Tier 9: Pace-Adjusted Per-100 Stats
                        "pts_per_100": _pts_p100,
                        "ast_per_100": _ast_p100,
                        "reb_per_100": _reb_p100,
                        "stl_per_100": _stl_p100,
                        "blk_per_100": _blk_p100,
                        "tov_per_100": _tov_p100,
                        # Tier 9: Free Throw Rate & Foul Drawing
                        "ft_rate": _ft_rate,
                        "fouls_drawn_per_game": _fouls_drawn_pg,
                        "ft_attempts_per_game": fta_per_game,
                        "and_one_frequency": _and_one_freq,
                        "foul_drawing_ability": _foul_draw,
                        # Tier 9: Rebounding Rates
                        "oreb_rate": _oreb_rate,
                        "dreb_rate": _dreb_rate,
                        "total_reb_rate": _total_reb_rate,
                        "rebound_contested_pct": min(1.0, float(_hsl.get("contested_shots_pg", 3.0)) / 10.0),
                        "rebound_positioning_score": float(reb_rate_per_36 / max(_total_reb_pg * 36 / max(mins_season, 1), 0.1)),
                        # Tier 9: Points in Paint (estimated)
                        "paint_pts_per_game": float(_scr.get("pct_pts_paint", 0.35)) * pts_mean,
                        "paint_attempts_per_game": float(_sz.get("paint_fga_pct", 0.40)) * fga_per_game,
                        "paint_touch_to_points": float(_scr.get("pct_pts_paint", 0.35)) * pts_mean / max(float(_sz.get("paint_fga_pct", 0.40)) * fga_per_game, 0.1),
                        "restricted_area_attempts": float(_sz.get("rim_fga_pct", 0.25)) * fga_per_game,
                        # Tier 9: Game Situation Performance (from W/L splits)
                        "performance_when_leading": _perf_lead,
                        "performance_when_trailing": _perf_trail,
                        "performance_when_tied": float(season_avg),
                        "performance_in_overtime": float(season_avg),
                        "performance_by_score_differential": float(_perf_lead - _perf_trail),
                        # Tier 10: Minutes Fatigue
                        "minutes_last_3_games": _total_mins_l3,
                        "minutes_last_5_games": _total_mins_l5,
                        "minutes_last_7_games": _total_mins_l7,
                        "avg_minutes_last_3": _mins_l3,
                        "minutes_fatigue_score": _fatigue,
                        # Tier 10: Player-Level Advanced Metrics (from precomputed DB)
                        "player_off_rating": float(_adv.get("off_rating", (team_ctx or {}).get("offensive_rating", 110.0))),
                        "player_def_rating": float(_adv.get("def_rating", (team_ctx or {}).get("defensive_rating", 110.0))),
                        "player_pace":       float(_adv.get("pace",       (team_ctx or {}).get("pace", 100.0))),
                        "player_height_inches": float(_adv.get("height_inches", _player_height)),
                        "player_weight":     float(_adv.get("weight", _player_weight)),
                        "net_rating_player": float(_adv.get("net_rating", 0.0)),
                        "fta_rate_player": _ft_rate,
                        "pct_fga_2pt": float(1.0 - (fg3a_per_game / max(fga_per_game, 1.0))),
                        "pct_fga_3pt": float(fg3a_per_game / max(fga_per_game, 1.0)),
                        # Official advanced metrics (from player_advanced_stats)
                        "usg_pct_official":  float(_adv.get("usg_pct",  0.18)),
                        "ts_pct_official":   float(_adv.get("ts_pct",   0.55)),
                        "efg_pct_official":  float(_adv.get("efg_pct",  0.50)),
                        "ast_pct_official":  float(_adv.get("ast_pct",  0.15)),
                        "oreb_pct_official": float(_adv.get("oreb_pct", 0.05)),
                        "dreb_pct_official": float(_adv.get("dreb_pct", 0.15)),
                        "reb_pct_official":  float(_adv.get("reb_pct",  0.10)),
                        "pie":               float(_adv.get("pie",       0.10)),
                        # Clutch stats (from player_clutch_stats)
                        "clutch_pts_per_game":  float(_clt.get("clutch_pts_pg",     0.0)),
                        "clutch_fg_pct":        float(_clt.get("clutch_fg_pct",     0.45)),
                        "clutch_fg3_pct":       float(_clt.get("clutch_fg3_pct",    0.33)),
                        "clutch_fta_per_game":  float(_clt.get("clutch_fta_pg",     0.0)),
                        "clutch_plus_minus":    float(_clt.get("clutch_plus_minus", 0.0)),
                        "clutch_min_per_game":  float(_clt.get("clutch_min_pg",     0.0)),
                        "clutch_games":         float(_clt.get("clutch_games",      0)),
                        # Hustle stats (from player_hustle_stats)
                        "contested_shots_per_game": float(_hsl.get("contested_shots_pg", 3.0)),
                        "deflections_per_game":     float(_hsl.get("deflections_pg",     1.0)),
                        "charges_drawn_per_game":   float(_hsl.get("charges_drawn_pg",   0.1)),
                        "screen_assists_per_game":  float(_hsl.get("screen_assists_pg",  0.5)),
                        # Shot profile (from player_shot_profile)
                        "open_shot_fg_pct":     float(_sp.get("open_shot_fg_pct",  0.50)),
                        "open_shot_frequency":  float(_sp.get("open_shot_freq",    0.30)),
                        "tight_shot_fg_pct":    float(_sp.get("tight_shot_fg_pct", 0.38)),
                        "tight_shot_frequency": float(_sp.get("tight_shot_freq",   0.15)),
                        "catch_shoot_fg_pct":   float(_sp.get("catch_shoot_fg_pct",0.40)),
                        "catch_shoot_frequency":float(_sp.get("catch_shoot_freq",  0.25)),
                        "pullup_fg_pct":        float(_sp.get("pullup_fg_pct",     0.40)),
                        "pullup_frequency":     float(_sp.get("pullup_freq",       0.20)),
                        # Play types (from player_play_types)
                        "iso_poss_pct":        float(_pt.get("iso_poss_pct",        0.0)),
                        "iso_ppp":             float(_pt.get("iso_ppp",             0.9)),
                        "pnr_bh_poss_pct":     float(_pt.get("pnr_bh_poss_pct",    0.0)),
                        "pnr_bh_ppp":          float(_pt.get("pnr_bh_ppp",         0.9)),
                        "pnr_roll_poss_pct":   float(_pt.get("pnr_roll_poss_pct",  0.0)),
                        "pnr_roll_ppp":        float(_pt.get("pnr_roll_ppp",        0.9)),
                        "spotup_poss_pct":     float(_pt.get("spotup_poss_pct",    0.0)),
                        "spotup_ppp":          float(_pt.get("spotup_ppp",          1.0)),
                        "transition_poss_pct": float(_pt.get("transition_poss_pct", 0.0)),
                        "transition_ppp":      float(_pt.get("transition_ppp",      1.1)),
                        "postup_poss_pct":     float(_pt.get("postup_poss_pct",    0.0)),
                        "cut_poss_pct":        float(_pt.get("cut_poss_pct",        0.0)),
                        # On/off (from player_on_off)
                        "on_court_net_rating":  float(_oo.get("on_court_net_rating",  0.0)),
                        "off_court_net_rating": float(_oo.get("off_court_net_rating", 0.0)),
                        "on_off_differential":  float(_oo.get("on_off_differential",  0.0)),
                        "on_court_plus_minus":  float(hist["PLUS_MINUS"].mean() if "PLUS_MINUS" in hist.columns else 0.0),
                        "off_court_plus_minus": float(_oo.get("off_court_net_rating", 0.0)),
                        # Shot zones (from player_shot_zones)
                        "rim_fga_pct":          float(_sz.get("rim_fga_pct",          0.25)),
                        "rim_fg_pct":           float(_sz.get("rim_fg_pct",           0.62)),
                        "paint_fga_pct":        float(_sz.get("paint_fga_pct",        0.30)),
                        "paint_fg_pct":         float(_sz.get("paint_fg_pct",         0.55)),
                        "midrange_fga_pct":     float(_sz.get("midrange_fga_pct",     0.20)),
                        "midrange_fg_pct":      float(_sz.get("midrange_fg_pct",      0.42)),
                        "corner3_fga_pct":      float(_sz.get("corner3_fga_pct",      0.10)),
                        "corner3_fg_pct":       float(_sz.get("corner3_fg_pct",       0.38)),
                        "above_break3_fga_pct": float(_sz.get("above_break3_fga_pct", 0.25)),
                        "above_break3_fg_pct":  float(_sz.get("above_break3_fg_pct",  0.35)),
                        # Quarter splits (from player_quarter_splits)
                        "q1_avg":          float(_qs.get("q1_avg",    0.0)),
                        "q2_avg":          float(_qs.get("q2_avg",    0.0)),
                        "q3_avg":          float(_qs.get("q3_avg",    0.0)),
                        "q4_avg":          float(_qs.get("q4_avg",    0.0)),
                        "q4_min_per_game": float(_qs.get("q4_min_pg", 0.0)),
                        # Tracking stats (from player_tracking_stats)
                        "tracking_avg_speed":          float(_trk.get("avg_speed",          4.5)),
                        "tracking_avg_speed_off":      float(_trk.get("avg_speed_off",      4.8)),
                        "tracking_avg_speed_def":      float(_trk.get("avg_speed_def",      4.2)),
                        "tracking_dist_miles":         float(_trk.get("dist_miles",         2.5)),
                        "tracking_dist_miles_off":     float(_trk.get("dist_miles_off",     1.3)),
                        "tracking_dist_miles_def":     float(_trk.get("dist_miles_def",     1.2)),
                        "tracking_touches_pg":         float(_trk.get("touches_pg",        50.0)),
                        "tracking_time_of_poss_pg":    float(_trk.get("time_of_poss_pg",   2.5)),
                        "tracking_avg_drib_per_touch": float(_trk.get("avg_drib_per_touch", 1.5)),
                        "tracking_passes_made_pg":     float(_trk.get("passes_made_pg",    30.0)),
                        "tracking_potential_ast_pg":   float(_trk.get("potential_ast_pg",   5.0)),
                        "tracking_secondary_ast_pg":   float(_trk.get("secondary_ast_pg",   1.0)),
                        # Scoring breakdown (from player_scoring_breakdown)
                        "pct_pts_3pt":      float(_scr.get("pct_pts_3pt",      0.25)),
                        "pct_pts_paint":    float(_scr.get("pct_pts_paint",    0.30)),
                        "pct_pts_ft":       float(_scr.get("pct_pts_ft",       0.15)),
                        "pct_pts_midrange": float(_scr.get("pct_pts_midrange", 0.20)),
                        "pct_uast_fgm":     float(_scr.get("pct_uast_fgm",    0.40)),
                        "pct_ast_fgm":      float(_scr.get("pct_ast_fgm",     0.60)),
                        "pct_pts_in_paint": float(_scr.get("pct_pts_paint",    0.30)),
                        "pct_pts_off_tov":  float(_scr.get("pct_pts_off_tov", 0.10)),
                        "pct_pts_fb":       float(_scr.get("pct_pts_fb",      0.12)),
                        # YoY stats (from player_yoy_stats)
                        "yoy_pts_change":    float(_yoy.get("yoy_pts_change",   0.0)),
                        "yoy_ts_change":     float(_yoy.get("yoy_ts_change",    0.0)),
                        "yoy_usage_change":  float(_yoy.get("yoy_usage_change", 0.0)),
                        "seasons_in_league": float(_yoy.get("seasons_in_league", 5)),
                        # Season-phase calendar features
                        "days_into_season":       float(_days_into_season),
                        "games_remaining_approx": float(_games_remaining_approx),
                        "season_phase_numeric":   _season_phase_numeric,
                        # Referee tendencies (aggregate over all refs for this game — use league avg)
                        "ref_foul_rate": float(np.mean([v["foul_rate"] for v in _precomp["referee"].values()]) if _precomp["referee"] else 0.0),
                        "ref_home_bias": float(np.mean([v["home_win_pct"] for v in _precomp["referee"].values()]) if _precomp["referee"] else 0.5),
                        "ref_pace_tendency": float(np.mean([v["pace"] for v in _precomp["referee"].values()]) if _precomp["referee"] else 100.0),
                        # Rolling DVP (opponent's recent defensive form)
                        "dvp_pts_delta_last5":  _dvp_pts_delta_last5,
                        "dvp_pts_delta_last10": _dvp_pts_delta_last10,
                        "dvp_reb_delta_last5":  _dvp_reb_delta_last5,
                        "dvp_ast_delta_last5":  _dvp_ast_delta_last5,
                        "dvp_fg3m_delta_last5": _dvp_fg3m_delta_last5,
                        # Opponent foul tendency
                        "opp_foul_rate_per48": _opp_foul_rate_season,
                        "opp_foul_rate_last5": _opp_foul_rate_last5,
                        # Opponent rest splits
                        "opp_b2b_def_rating":    _opp_b2b_def_rating,
                        "opp_b2b_pace":          _opp_b2b_pace,
                        "opp_b2b_pts_allowed":   _opp_b2b_pts_allowed,
                        "opp_rested_def_rating": _opp_rested_def_rating,
                        "opp_rested_pace":       _opp_rested_pace,
                        # Team standings context
                        "team_win_pct":       _team_win_pct,
                        "team_conf_rank":     _team_conf_rank,
                        "team_current_streak": _team_streak,
                        "team_l10_wins":      _team_l10_wins,
                        "opp_win_pct":        _opp_win_pct,
                        "opp_conf_rank":      _opp_conf_rank,
                        "opp_current_streak": _opp_streak,
                        "opp_l10_wins":       _opp_l10_wins,
                        # Player vs specific opponent
                        "vs_opp_gp":      _vs_opp_gp,
                        "vs_opp_avg_pts": _vs_opp_avg_pts,
                        "vs_opp_fg_pct":  _vs_opp_fg_pct,
                        "vs_opp_ts_pct":  _vs_opp_ts_pct,
                        "vs_opp_avg_min": _vs_opp_avg_min,
                        # Opponent shot zone defense (from team_opp_shot_zones)
                        "opp_rim_fg_pct_allowed":         float(_osz.get("rim_fg_pct_allowed",         0.62)),
                        "opp_paint_fg_pct_allowed":       float(_osz.get("paint_fg_pct_allowed",       0.55)),
                        "opp_midrange_fg_pct_allowed":    float(_osz.get("midrange_fg_pct_allowed",    0.42)),
                        "opp_corner3_fg_pct_allowed":     float(_osz.get("corner3_fg_pct_allowed",     0.38)),
                        "opp_above_break3_fg_pct_allowed":float(_osz.get("above_break3_fg_pct_allowed",0.35)),
                        # Opponent synergy defense (from team_synergy_defense)
                        "opp_pnr_ppp_allowed":        float(_osy.get("pnr_ppp_allowed",        0.9)),
                        "opp_iso_ppp_allowed":         float(_osy.get("iso_ppp_allowed",         0.9)),
                        "opp_spotup_ppp_allowed":      float(_osy.get("spotup_ppp_allowed",      1.0)),
                        "opp_transition_ppp_allowed":  float(_osy.get("transition_ppp_allowed",  1.1)),
                        "opp_postup_ppp_allowed":      float(_osy.get("postup_ppp_allowed",      0.9)),
                        # Shot zone matchup advantages (player efficiency vs opponent zone defense)
                        "rim_shot_quality_matchup":          float(_sz.get("rim_fg_pct",          0.62)) - float(_osz.get("rim_fg_pct_allowed",         0.62)),
                        "midrange_shot_quality_matchup":     float(_sz.get("midrange_fg_pct",     0.42)) - float(_osz.get("midrange_fg_pct_allowed",    0.42)),
                        "corner3_shot_quality_matchup":      float(_sz.get("corner3_fg_pct",      0.38)) - float(_osz.get("corner3_fg_pct_allowed",     0.38)),
                        "above_break3_shot_quality_matchup": float(_sz.get("above_break3_fg_pct", 0.35)) - float(_osz.get("above_break3_fg_pct_allowed",0.35)),
                        # Synergy play type matchup advantages (player PPP vs opponent PPP allowed)
                        "pnr_matchup_advantage":        float(_pt.get("pnr_bh_ppp",     0.9)) - float(_osy.get("pnr_ppp_allowed",       0.9)),
                        "iso_matchup_advantage":        float(_pt.get("iso_ppp",         0.9)) - float(_osy.get("iso_ppp_allowed",        0.9)),
                        "spotup_matchup_advantage":     float(_pt.get("spotup_ppp",      1.0)) - float(_osy.get("spotup_ppp_allowed",     1.0)),
                        "transition_matchup_advantage": float(_pt.get("transition_ppp",  1.1)) - float(_osy.get("transition_ppp_allowed", 1.1)),
                        # Injury trajectory (derived from date gaps in game log)
                        "games_since_return":         _games_since_return,
                        "missed_games_before_return": _missed_before,
                        # Implied game total (from team + opponent pace)
                        "implied_game_total": _implied_total,
                        # Defender health proxy (score01 > 0 means active elite defender)
                        "primary_defender_active": _primary_def_active,
                        # Opponent lineup instability (rolling DVP variance proxy)
                        "opp_lineup_changes_last5": _opp_lineup_changes,
                        # Market / odds features (zeros during training — populated at inference)
                        "opening_line": 0.0,
                        "current_line": 0.0,
                        "line_movement": 0.0,
                        "line_movement_pct": 0.0,
                        "implied_over_prob": 0.5,
                        "implied_under_prob": 0.5,
                        "market_consensus_std": 0.0,
                        "sharp_action_score": 0.0,
                        "line_velocity": 0.0,
                        "stale_line_flag": 0.0,
                        "bookmaker_count": 0.0,
                        "is_playoff": is_playoff,
                        "is_play_in": is_play_in,
                        "series_game_num": series_game_num,
                        "team_series_wins_in": team_series_wins_in,
                        "opp_series_wins_in": opp_series_wins_in,
                        "is_elimination_game": is_elimination_game,
                        "playoff_home": playoff_home,
                        "playoff_away": playoff_away,
                    }

            for prop_type, target_col in STAT_TARGETS.items():
                values = hist[target_col].tolist()
                if len(values) < 8:
                    continue
                last5_avg = float(last5[target_col].mean()) if not last5.empty else float(hist[target_col].mean())
                season_avg = float(hist[target_col].mean())
                stddev = float(np.std(values))
                X = build_feature_vector(make_features(values, last5_avg, season_avg, stddev)).X
                y = float(row[target_col])
                _mins = float(row.get("MIN", 0.0) or 0.0)
                examples.append(Example(prop_type=prop_type, X=X, y=y,
                                        game_date=pd.to_datetime(row["GAME_DATE"]),
                                        minutes=_mins))

            for combo_prop, cols in COMBO_TARGETS.items():
                combo_series = hist[cols].sum(axis=1)
                values = combo_series.tolist()
                if len(values) < 8:
                    continue
                last5_avg = float(combo_series.tail(5).mean()) if len(combo_series) >= 5 else float(combo_series.mean())
                season_avg = float(combo_series.mean())
                stddev = float(np.std(values))
                X = build_feature_vector(make_features(values, last5_avg, season_avg, stddev)).X
                y = float(sum(float(row.get(c, 0.0)) for c in cols))
                _mins_c = float(row.get("MIN", 0.0) or 0.0)
                examples.append(Example(prop_type=combo_prop, X=X, y=y,
                                        game_date=pd.to_datetime(row["GAME_DATE"]),
                                        minutes=_mins_c))

            stats_now = [float(row.get(c, 0.0)) for c in ["PTS", "REB", "AST", "STL", "BLK"]]
            dd_y = 1.0 if sum(1 for s in stats_now if s >= 10.0) >= 2 else 0.0
            td_y = 1.0 if sum(1 for s in stats_now if s >= 10.0) >= 3 else 0.0
            pts_values = hist["PTS"].tolist()
            if len(pts_values) >= 8:
                last5_avg = float(last5["PTS"].mean()) if not last5.empty else float(hist["PTS"].mean())
                season_avg = float(hist["PTS"].mean())
                stddev = float(np.std(pts_values))
                X = build_feature_vector(make_features(pts_values, last5_avg, season_avg, stddev)).X
                _mins_b = float(row.get("MIN", 0.0) or 0.0)
                examples.append(Example(prop_type="double_double", X=X, y=dd_y,
                                        game_date=pd.to_datetime(row["GAME_DATE"]),
                                        minutes=_mins_b))
                examples.append(Example(prop_type="triple_double", X=X, y=td_y,
                                        game_date=pd.to_datetime(row["GAME_DATE"]),
                                        minutes=_mins_b))

          except Exception as _row_err:
            print(f"[train] skipping row idx={idx} player={player_id}: {_row_err}")
            continue

    return examples


def _sample_lines(y: float, prop_type: str) -> list[float]:
    if prop_type in ("double_double", "triple_double"):
        return [0.5]

    step = 0.5
    if prop_type in ("three_pointers", "blocks", "steals"):
        step = 0.5
    elif prop_type in ("turnovers",):
        step = 0.5
    elif prop_type in ("assists", "rebounds"):
        step = 0.5
    else:
        step = 0.5

    lines = [
        max(0.0, y - 2 * step),
        max(0.0, y - step),
        max(0.0, y),
        y + step,
        y + 2 * step,
    ]
    out = []
    for ln in lines:
        out.append(round(float(ln) * 2) / 2.0)
    seen = set()
    final = []
    for ln in out:
        if ln not in seen:
            seen.add(ln)
            final.append(ln)
    return final


def _sportsbook_center(raw_features: dict[str, Any]) -> float:
    recent = float(raw_features.get("recent_avg", 0.0) or 0.0)
    season = float(raw_features.get("season_avg", recent) or recent)
    return 0.7 * recent + 0.3 * season


def _walk_forward_splits(n: int, min_train: int = 200, n_folds: int = 5) -> list[tuple[int, int]]:
    if n <= (min_train + 50):
        return []
    test_size = max(50, int(n * 0.10))
    splits: list[tuple[int, int]] = []
    train_end = min_train
    while train_end + test_size <= n and len(splits) < n_folds:
        splits.append((train_end, train_end + test_size))
        train_end += test_size
    return splits


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        return 0.0


def optimize_hyperparameters_regression(X_train, y_train, X_val, y_val, n_trials=50):
    if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
        return None
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 0,
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_hyperparameters_classification(X_train, y_train, X_val, y_val, n_trials=50):
    if not OPTUNA_AVAILABLE or not XGBOOST_AVAILABLE:
        return None
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 0,
            'eval_metric': 'auc',
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        auc = _safe_auc(y_val, proba)
        return auc
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def create_regressor(use_xgboost=True, optimized_params=None):
    if use_xgboost and XGBOOST_AVAILABLE:
        if optimized_params:
            return XGBRegressor(**optimized_params)
        return XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method='hist',
            verbosity=0,
        )
    else:
        return HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.06,
            max_iter=400,
            random_state=42,
        )


def create_classifier(use_xgboost=True, optimized_params=None):
    if use_xgboost and XGBOOST_AVAILABLE:
        if optimized_params:
            return XGBClassifier(**optimized_params)
        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method='hist',
            verbosity=0,
            eval_metric='auc',
        )
    else:
        return HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.06,
            max_iter=350,
            random_state=42,
        )


def train_and_save(models_dir: str, examples: list[Example]) -> dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)

    # Group by prop_type
    by_prop: dict[str, list[Example]] = {}
    for ex in examples:
        by_prop.setdefault(ex.prop_type, []).append(ex)

    metadata: dict[str, Any] = {"schema": {"numeric_features": NUMERIC_FEATURE_KEYS}, "props": {}}

    for prop, exs in by_prop.items():
        exs = sorted(exs, key=lambda e: e.game_date)
        cut = int(len(exs) * 0.8)
        train_exs = exs[:cut]
        test_exs = exs[cut:]

        X_train = pd.concat([e.X for e in train_exs], ignore_index=True)
        y_train = np.array([e.y for e in train_exs], dtype=float)
        X_test = pd.concat([e.X for e in test_exs], ignore_index=True)
        y_test = np.array([e.y for e in test_exs], dtype=float)
        # Deduplicate columns — XGBoost fails with duplicate col names
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test  = X_test.loc[:,  ~X_test.columns.duplicated()]

        # Per-sample weights: recency-decay × minutes-weight (both opt-in).
        # Applied to regression + classification fits below.
        _reg_sample_weight = compute_sample_weights(
            train_exs,
            recency_half_life_days=_TRAIN_CONFIG.get("recency_half_life"),
            minutes_weight=bool(_TRAIN_CONFIG.get("minutes_weight")),
            min_weight=float(_TRAIN_CONFIG.get("min_sample_weight", 0.1)),
        ) if len(train_exs) else None

        # Optional: prune low-importance features to speed up fits + reduce
        # overfitting. Returns the subset of ``X_train.columns`` to keep.
        _prune_frac = float(_TRAIN_CONFIG.get("prune_features", 0.0) or 0.0)
        if _prune_frac > 0 and len(X_train) > 100:
            _pre_prune_n = X_train.shape[1]  # capture BEFORE subsetting
            keep_cols, importance_report = prune_low_importance_features(
                X_train, y_train,
                drop_fraction=_prune_frac,
                sample_weight=_reg_sample_weight,
            )
            X_train = X_train[keep_cols]
            X_test = X_test[keep_cols]
            metadata["props"].setdefault(prop, {})["pruned_feature_count"] = len(keep_cols)
            metadata["props"][prop]["pre_prune_feature_count"] = _pre_prune_n
            print(f"[prune] {prop}: kept {len(keep_cols)}/{_pre_prune_n} features "
                  f"({100.0 * (1 - len(keep_cols)/_pre_prune_n):.1f}% dropped)")

        metadata["props"].setdefault(prop, {})
        metadata["props"][prop].update({
            "n_examples": int(len(exs)),
        })

        wf_splits = _walk_forward_splits(len(exs), min_train=200, n_folds=5)
        if wf_splits:
            wf_rows = []
            for tr_end, te_end in wf_splits:
                tr = exs[:tr_end]
                te = exs[tr_end:te_end]
                if not tr or not te:
                    continue

                Xc_tr_parts, yc_tr_parts = [], []
                for e in tr:
                    raw = e.X.iloc[0].to_dict()
                    center = _sportsbook_center(raw)
                    for ln in _sample_lines(center, prop):
                        Xc_tr_parts.append(build_classifier_vector(raw, line=ln).X)
                        yc_tr_parts.append(1 if float(e.y) > float(ln) else 0)

                Xc_te_parts, yc_te_parts = [], []
                for e in te:
                    raw = e.X.iloc[0].to_dict()
                    center = _sportsbook_center(raw)
                    for ln in _sample_lines(center, prop):
                        Xc_te_parts.append(build_classifier_vector(raw, line=ln).X)
                        yc_te_parts.append(1 if float(e.y) > float(ln) else 0)

                if not Xc_tr_parts or not Xc_te_parts:
                    continue

                Xc_tr = pd.concat(Xc_tr_parts, ignore_index=True)
                yc_tr = np.array(yc_tr_parts, dtype=int)
                Xc_te = pd.concat(Xc_te_parts, ignore_index=True)
                yc_te = np.array(yc_te_parts, dtype=int)

                # Deduplicate columns — XGBoost fails with duplicate col names
                # (pandas returns a DataFrame instead of Series on df[col])
                Xc_tr = Xc_tr.loc[:, ~Xc_tr.columns.duplicated()]
                Xc_te = Xc_te.loc[:, ~Xc_te.columns.duplicated()]

                clf_wf = create_classifier(use_xgboost=True)
                clf_wf.fit(Xc_tr, yc_tr)
                proba_raw = clf_wf.predict_proba(Xc_te)[:, 1]
                auc_raw = _safe_auc(yc_te, proba_raw)
                brier_raw = float(brier_score_loss(yc_te, proba_raw))

                cal_cut = int(len(Xc_tr) * 0.8)
                Xc_fit, Xc_cal = Xc_tr.iloc[:cal_cut], Xc_tr.iloc[cal_cut:]
                yc_fit, yc_cal = yc_tr[:cal_cut], yc_tr[cal_cut:]

                clf_fit = create_classifier(use_xgboost=True)
                clf_fit.fit(Xc_fit, yc_fit)
                raw_cal = clf_fit.predict_proba(Xc_cal)[:, 1] if len(Xc_cal) else None
                auc_cal = 0.0
                brier_cal = 0.0
                if raw_cal is not None and len(Xc_cal) and len(np.unique(yc_cal)) > 1:
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(raw_cal, yc_cal)
                    cal_model = IsotonicCalibratedModel(base_estimator=clf_fit, calibrator=iso)
                    proba_cal = cal_model.predict_proba(Xc_te)[:, 1]
                    auc_cal = _safe_auc(yc_te, proba_cal)
                    brier_cal = float(brier_score_loss(yc_te, proba_cal))

                wf_rows.append({
                    "train_end": int(tr_end),
                    "test_end": int(te_end),
                    "n_train_samples": int(len(Xc_tr)),
                    "n_test_samples": int(len(Xc_te)),
                    "auc_raw": float(auc_raw),
                    "brier_raw": float(brier_raw),
                    "auc_cal": float(auc_cal),
                    "brier_cal": float(brier_cal),
                })

            if wf_rows:
                auc_raw_mean = float(np.mean([r["auc_raw"] for r in wf_rows]))
                brier_raw_mean = float(np.mean([r["brier_raw"] for r in wf_rows]))
                auc_cal_mean = float(np.mean([r["auc_cal"] for r in wf_rows]))
                brier_cal_mean = float(np.mean([r["brier_cal"] for r in wf_rows]))

                metadata["props"][prop]["walk_forward"] = {
                    "n_folds": int(len(wf_rows)),
                    "auc_raw_mean": auc_raw_mean,
                    "brier_raw_mean": brier_raw_mean,
                    "auc_cal_mean": auc_cal_mean,
                    "brier_cal_mean": brier_cal_mean,
                    "folds": wf_rows,
                }
                print(
                    f"[wf] {prop}: auc_raw={auc_raw_mean:.3f} brier_raw={brier_raw_mean:.3f} "
                    f"auc_cal={auc_cal_mean:.3f} brier_cal={brier_cal_mean:.3f} folds={len(wf_rows)}"
                )

        is_binary = prop in ("double_double", "triple_double")
        if not is_binary:
            use_optuna = OPTUNA_AVAILABLE and len(X_train) > 500
            optimized_params = None
            if use_optuna:
                print(f"[optuna] Optimizing hyperparameters for {prop} regression...")
                val_cut = int(len(X_train) * 0.9)
                X_train_opt, X_val_opt = X_train.iloc[:val_cut], X_train.iloc[val_cut:]
                y_train_opt, y_val_opt = y_train[:val_cut], y_train[val_cut:]
                optimized_params = optimize_hyperparameters_regression(
                    X_train_opt, y_train_opt, X_val_opt, y_val_opt, n_trials=30
                )
                if optimized_params:
                    print(f"[optuna] Best params: {optimized_params}")
            
            reg = create_regressor(use_xgboost=True, optimized_params=optimized_params)
            # Recency/minutes weighting supplied via CLI flags; None disables.
            try:
                reg.fit(X_train, y_train, sample_weight=_reg_sample_weight)
            except TypeError:  # older sklearn/XGB without sample_weight in .fit
                reg.fit(X_train, y_train)
            pred = reg.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, pred))) if len(y_test) else 0.0

            resid = y_test - pred
            resid_std = float(np.std(resid)) if len(resid) else 1.0
            
            mae = float(np.mean(np.abs(resid))) if len(resid) else 0.0
            r2_score = 1.0 - (np.sum(resid**2) / np.sum((y_test - np.mean(y_test))**2)) if len(y_test) > 1 else 0.0

            out_path = os.path.join(models_dir, f"reg_{prop}.joblib")
            joblib.dump(reg, out_path)

            metadata["props"][prop].update({
                "type": "regression",
                "rmse": rmse,
                "resid_std": resid_std,
                "mae": mae,
                "r2_score": r2_score,
                "model_type": "xgboost" if XGBOOST_AVAILABLE else "hist_gradient_boosting",
            })
        else:
            metadata["props"][prop].update({
                "type": "classifier_only",
            })

        Xc_train_parts = []
        yc_train_parts = []
        _clf_train_weights: list[float] = []
        _clf_train_bands: list[int] = []  # line-distance band per row, for stratified cal split
        for i, (e, yv) in enumerate(zip(train_exs, y_train)):
            raw = e.X.iloc[0].to_dict()
            center = _sportsbook_center(raw)
            _w = float(_reg_sample_weight[i]) if _reg_sample_weight is not None else 1.0
            _c_safe = max(float(center), 1.0)
            for ln in _sample_lines(float(center), prop):
                Xc_train_parts.append(build_classifier_vector(raw, line=ln).X)
                yc_train_parts.append(1 if float(yv) > float(ln) else 0)
                _clf_train_weights.append(_w)
                # Bucket by relative distance from sportsbook center.
                # Bands: 0=center (<5%), 1=near (5-15%), 2=mid (15-30%), 3=far (>=30%).
                _rel = abs(float(ln) - float(center)) / _c_safe
                if _rel < 0.05:
                    _clf_train_bands.append(0)
                elif _rel < 0.15:
                    _clf_train_bands.append(1)
                elif _rel < 0.30:
                    _clf_train_bands.append(2)
                else:
                    _clf_train_bands.append(3)

        Xc_test_parts = []
        yc_test_parts = []
        for e, yv in zip(test_exs, y_test):
            raw = e.X.iloc[0].to_dict()
            center = _sportsbook_center(raw)
            for ln in _sample_lines(float(center), prop):
                Xc_test_parts.append(build_classifier_vector(raw, line=ln).X)
                yc_test_parts.append(1 if float(yv) > float(ln) else 0)

        if Xc_train_parts and Xc_test_parts:
            Xc_train = pd.concat(Xc_train_parts, ignore_index=True)
            yc_train = np.array(yc_train_parts, dtype=int)
            Xc_test = pd.concat(Xc_test_parts, ignore_index=True)
            yc_test = np.array(yc_test_parts, dtype=int)
            # Deduplicate columns — XGBoost fails with duplicate col names
            Xc_train = Xc_train.loc[:, ~Xc_train.columns.duplicated()]
            Xc_test  = Xc_test.loc[:,  ~Xc_test.columns.duplicated()]

            use_optuna = OPTUNA_AVAILABLE and len(Xc_train) > 500
            optimized_params = None
            if use_optuna:
                print(f"[optuna] Optimizing hyperparameters for {prop} classification...")
                val_cut = int(len(Xc_train) * 0.9)
                Xc_train_opt, Xc_val_opt = Xc_train.iloc[:val_cut], Xc_train.iloc[val_cut:]
                yc_train_opt, yc_val_opt = yc_train[:val_cut], yc_train[val_cut:]
                optimized_params = optimize_hyperparameters_classification(
                    Xc_train_opt, yc_train_opt, Xc_val_opt, yc_val_opt, n_trials=30
                )
                if optimized_params:
                    print(f"[optuna] Best params: {optimized_params}")
            
            clf = create_classifier(use_xgboost=True, optimized_params=optimized_params)
            _clf_w_arr = np.array(_clf_train_weights, dtype=float) if _clf_train_weights else None
            if _clf_w_arr is not None and len(_clf_w_arr) != len(Xc_train):
                # Defensive: mismatch means something changed upstream — fall back to unweighted
                _clf_w_arr = None
            try:
                if _clf_w_arr is not None:
                    clf.fit(Xc_train, yc_train, sample_weight=_clf_w_arr)
                else:
                    clf.fit(Xc_train, yc_train)
            except TypeError:
                clf.fit(Xc_train, yc_train)

            proba = clf.predict_proba(Xc_test)[:, 1]
            auc = _safe_auc(yc_test, proba)
            brier = float(brier_score_loss(yc_test, proba))
            
            log_loss = float(-np.mean(yc_test * np.log(proba + 1e-15) + (1 - yc_test) * np.log(1 - proba + 1e-15))) if len(yc_test) > 0 else 0.0
            vig = 0.045
            ev = float(np.mean((proba * (100/110)) - ((1 - proba) * (110/100)))) if len(proba) > 0 else 0.0

            # ------------------------------------------------------------------
            # Stratified calibration split by line-distance band.
            # Naive prefix-cut (.iloc[:80%]) lets the cal slice be dominated by
            # whatever band happens to land at the tail, poisoning isotonic fit
            # on extreme lines (which classify trivially as 0 or 1). Strat-sample
            # by band so cal has representative slice across the whole curve.
            # ------------------------------------------------------------------
            _bands_arr = np.array(_clf_train_bands, dtype=int) if _clf_train_bands else None
            if _bands_arr is not None and len(_bands_arr) == len(Xc_train) and len(np.unique(_bands_arr)) >= 2:
                rng = np.random.default_rng(42)
                fit_mask = np.zeros(len(Xc_train), dtype=bool)
                for b in np.unique(_bands_arr):
                    idx = np.where(_bands_arr == b)[0]
                    rng.shuffle(idx)
                    n_fit = max(1, int(len(idx) * 0.8))
                    fit_mask[idx[:n_fit]] = True
                cal_mask = ~fit_mask
                # Edge case: degenerate split (one side empty) -> fall back to prefix
                if cal_mask.sum() == 0 or fit_mask.sum() == 0:
                    cal_cut = int(len(Xc_train) * 0.8)
                    fit_mask = np.zeros(len(Xc_train), dtype=bool)
                    fit_mask[:cal_cut] = True
                    cal_mask = ~fit_mask
                Xc_fit = Xc_train.iloc[fit_mask]
                Xc_cal = Xc_train.iloc[cal_mask]
                yc_fit = yc_train[fit_mask]
                yc_cal = yc_train[cal_mask]
            else:
                # Fallback: original prefix-cut behaviour
                cal_cut = int(len(Xc_train) * 0.8)
                fit_mask = np.zeros(len(Xc_train), dtype=bool)
                fit_mask[:cal_cut] = True
                cal_mask = ~fit_mask
                Xc_fit, Xc_cal = Xc_train.iloc[:cal_cut], Xc_train.iloc[cal_cut:]
                yc_fit, yc_cal = yc_train[:cal_cut], yc_train[cal_cut:]

            clf2 = create_classifier(use_xgboost=True, optimized_params=optimized_params)
            _clf_w_fit = None
            if _clf_w_arr is not None:
                _clf_w_fit = _clf_w_arr[fit_mask]
            try:
                if _clf_w_fit is not None:
                    clf2.fit(Xc_fit, yc_fit, sample_weight=_clf_w_fit)
                else:
                    clf2.fit(Xc_fit, yc_fit)
            except TypeError:
                clf2.fit(Xc_fit, yc_fit)
            raw_cal = clf2.predict_proba(Xc_cal)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_cal, yc_cal)
            calibrator = IsotonicCalibratedModel(base_estimator=clf2, calibrator=iso)

            proba_cal = calibrator.predict_proba(Xc_test)[:, 1]
            auc_cal = _safe_auc(yc_test, proba_cal)
            brier_cal = float(brier_score_loss(yc_test, proba_cal))

            joblib.dump(clf, os.path.join(models_dir, f"clf_raw_{prop}.joblib"))
            joblib.dump(calibrator, os.path.join(models_dir, f"clf_cal_{prop}.joblib"))

            metadata["props"][prop].update({
                "classifier": {
                    "schema": {"numeric_features": NUMERIC_FEATURE_KEYS + CLASSIFIER_EXTRA_KEYS},
                    "auc_raw": auc,
                    "brier_raw": brier,
                    "auc_cal": auc_cal,
                    "brier_cal": brier_cal,
                    "log_loss": log_loss,
                    "expected_value": ev,
                    "n_class_samples_train": int(len(Xc_train)),
                    "n_class_samples_test": int(len(Xc_test)),
                    "model_type": "xgboost" if XGBOOST_AVAILABLE else "hist_gradient_boosting",
                }
            })

    def build_dd_label(row_dict):
        stats = [row_dict["PTS"], row_dict["REB"], row_dict["AST"], row_dict["STL"], row_dict["BLK"]]
        return 1 if sum(1 for s in stats if s >= 10) >= 2 else 0

    def build_td_label(row_dict):
        stats = [row_dict["PTS"], row_dict["REB"], row_dict["AST"], row_dict["STL"], row_dict["BLK"]]
        return 1 if sum(1 for s in stats if s >= 10) >= 3 else 0

    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None, help="Season like 2025-26 (defaults to current)")
    ap.add_argument("--seasons", default=None, help="Comma-separated seasons like 2023-24,2024-25,2025-26")
    ap.add_argument("--max-players", type=int, default=200)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--db", default="basketball_data.db")
    # Sample-weight knobs — set to None/0 to disable a dimension.
    ap.add_argument("--recency-half-life", type=float, default=365.0,
                    help="Half-life (days) for exponential recency decay. 0 to disable.")
    ap.add_argument("--minutes-weight", action="store_true",
                    help="Scale sample weight by clip(minutes/median, 0.4, 1.5).")
    ap.add_argument("--min-sample-weight", type=float, default=0.1,
                    help="Floor on per-sample weight.")
    ap.add_argument("--prune-features", type=float, default=0.0,
                    help="Drop this fraction of least-important features (0.30 = drop bottom 30%%).")
    args = ap.parse_args()

    # Stash on module-level so train_and_save can read without threading a param
    global _TRAIN_CONFIG
    _TRAIN_CONFIG = {
        "recency_half_life": args.recency_half_life if args.recency_half_life > 0 else None,
        "minutes_weight": bool(args.minutes_weight),
        "min_sample_weight": args.min_sample_weight,
        "prune_features": float(args.prune_features),
    }

    if args.seasons:
        seasons = [s.strip() for s in str(args.seasons).split(",") if s.strip()]
    else:
        seasons = [args.season or EnhancedMLPredictor().current_season]
    predictor = EnhancedMLPredictor(model_dir=args.models_dir)
    precomputed = PrecomputedStore(args.db)

    try:
        print(f"[train] building examples seasons={seasons} max_players={args.max_players} ...")
        examples = []
        for s in seasons:
            print(f"[train] season {s}: building examples...")
            examples.extend(build_training_examples(season=s, max_players=args.max_players, predictor=predictor, precomputed=precomputed, db_path=args.db))
        print(f"[train] total examples: {len(examples)}")

        print("[train] training + saving models ...")
        meta = train_and_save(args.models_dir, examples)
        print("[train] done. props trained:", list(meta.get("props", {}).keys()))
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


