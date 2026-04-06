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


@dataclass
class Example:
    prop_type: str
    X: pd.DataFrame
    y: float
    game_date: pd.Timestamp


def build_training_examples(
    season: str,
    max_players: int,
    predictor: EnhancedMLPredictor,
    precomputed: PrecomputedStore,
) -> list[Example]:
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

        gl = gl.copy()
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"], format="mixed", errors="coerce")
        gl = gl.sort_values("GAME_DATE")


        try:
            info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            raw_pos = str(info["POSITION"].iloc[0] if "POSITION" in info.columns else "").upper()
        except Exception:
            raw_pos = ""
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


        for idx in range(10, len(gl)):
            hist = gl.iloc[:idx]
            row = gl.iloc[idx]

            team_abbrev, opp_abbrev, is_home = _parse_matchup(row.get("MATCHUP"))
            team_id = _team_id(team_abbrev)
            opp_id = _team_id(opp_abbrev)
            if not team_id or not opp_id:
                continue

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

            team_ctx = predictor.get_team_context(team_id, include_injuries=False, season=season)
            opp_ctx = predictor.get_team_context(opp_id, include_injuries=False, season=season)

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

            def make_features(stat_values, last5_avg, season_avg, stddev):
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
                    "rest_days": 2,
                    "is_home_game": is_home,
                    "recent_away_streak": 0,
                    "team_pace": float(team_ctx.get("pace", 100.0)),
                    "opp_pace": float(opp_ctx.get("pace", 100.0)),
                    "team_off_rating": float(team_ctx.get("offensive_rating", 110.0)),
                    "team_def_rating": float(team_ctx.get("defensive_rating", 110.0)),
                    "opp_off_rating": float(opp_ctx.get("offensive_rating", 110.0)),
                    "opp_def_rating": float(opp_ctx.get("defensive_rating", 110.0)),
                    "team_key_players_out": 0,
                    "opp_key_players_out": 0,
                    "opp_injury_impact": 0.0,
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
                    "vs_team_avg": 0.0,
                    "matchup_games": 0,
                    "dvp_gp": dvp_gp,
                    **dvp_deltas,
                    "primary_defender_score01": float((primary_def or {}).get("score01", 0.0) or 0.0),
                    "points_per_shot": float(points_per_shot),
                    "ast_to_tov_ratio": float(ast_to_tov_ratio),
                    "reb_rate_per_36": float(reb_rate_per_36),
                    "scoring_efficiency_trend": float(scoring_efficiency_trend),
                    "usage_trend": float(usage_trend),
                    "minutes_volatility": float(minutes_volatility),
                    "blowout_game_pct": float(blowout_game_pct),
                    "close_game_pct": float(close_game_pct),
                    "opp_def_rating_last5": float(opp_ctx.get("defensive_rating", 110.0)),
                    "opp_def_rating_trend": 0.0,
                    "opp_def_rating_home_away_split": 0.0,
                    "opp_blocks_per_game_last5": float(opp_base.get("blk", 0.0)),
                    "opp_steals_per_game_last5": float(opp_base.get("stl", 0.0)),
                    "days_rest_opponent": 2,
                    "opponent_back_to_back": 0,
                    "playoff_implications": 0,
                    "rivalry_game": 0,
                    "national_tv_game": 0,
                    "season_phase": 0,
                    "primary_teammate_out": 0,
                    "secondary_teammate_out": 0,
                    "new_teammate_games": 0,
                    "lineup_stability_score": 1.0,
                    "bench_strength": 0.0,
                    "pts_vs_top10_defenses": float(pts_mean),
                    "pts_vs_bottom10_defenses": float(pts_mean),
                    "consistency_score": float(consistency_score),
                    "ceiling_game_frequency": float(ceiling_game_frequency),
                    "def_fg_pct_allowed": 0.0,
                    "def_rating_individual": 0.0,
                    "deflections_per_game": 0.0,
                    "contested_shots_per_game": 0.0,
                    "pnr_ball_handler_pct": 0.0,
                        "pnr_roll_man_pct": 0.0,
                        "isolation_pct": 0.0,
                        "spot_up_pct": 0.0,
                        "post_up_pct": 0.0,
                        "transition_pct": 0.0,
                        "consecutive_over_games": 0,
                        "consecutive_under_games": 0,
                        "hot_hand_indicator": 0.0,
                        "recent_variance_spike": float(hist.tail(3)["PTS"].std() / max(hist["PTS"].std(), 0.1) - 1.0) if "PTS" in hist.columns and len(hist) >= 3 else 0.0,
                        "playoff_seeding_impact": 0.5,
                        "tanking_indicator": 0.0,
                        "must_win_situation": 0.0,
                        "games_back_from_playoff": 0.0,
                        "fourth_quarter_usage_rate": 0.25 if mins_season > 30 else 0.18,
                        "garbage_time_minutes_pct": float(blowout_game_pct * 0.15),
                        "typical_substitution_minute": float(min(48.0, mins_season + 3.0)),
                        "crunch_time_usage": 0.28 if mins_season > 28 else 0.15,
                        "career_vs_defender": 0.0,
                        "recent_vs_defender": 0.0,
                        "player_vs_arena": 0.0,
                        "avg_shot_distance": 0.0,
                        "contested_shot_pct": 0.5,
                        "open_shot_pct": 0.3,
                        "wide_open_shot_pct": 0.2,
                        "catch_and_shoot_pct": 0.3,
                        "pull_up_shot_pct": 0.3,
                        "paint_touch_frequency": float(hist[hist['PTS'] > 0]['PTS'].count() * 0.1) if 'PTS' in hist.columns else 0.0,
                        "corner_three_pct": float(fg3_pct_recent),
                        "above_break_three_pct": float(fg3_pct_recent),
                        "restricted_area_fg_pct": float(fg_pct_recent),
                        "mid_range_frequency": 0.0,
                        "shot_quality_vs_expected": 0.0,
                        "avg_shot_clock_time": 12.0,
                        "late_clock_shot_frequency": 0.15,
                        "early_clock_shot_frequency": 0.25,
                        "touches_per_game": float(hist["FGA"].mean() + hist["AST"].mean() if "FGA" in hist.columns and "AST" in hist.columns else 0.0),
                        "avg_dribbles_per_touch": 2.0,
                        "avg_seconds_per_touch": 3.0,
                        "elbow_touches_per_game": 0.0,
                        "post_touches_per_game": 0.0,
                        "paint_touches_per_game": float(hist["REB"].mean() * 0.5 if "REB" in hist.columns else 0.0),
                        "front_court_touches_per_game": float(hist["AST"].mean() + hist["FGA"].mean() if "AST" in hist.columns and "FGA" in hist.columns else 0.0),
                        "time_of_possession_per_game": float(mins_season * 0.25),
                        "touches_per_possession": 0.0,
                        "avg_points_per_touch": 0.0,
                        "net_rating_with_starters": 0.0,
                        "usage_rate_with_star_out": float(hist["FGA"].mean() * 1.1 if "FGA" in hist.columns else 0.0),
                        "minutes_with_starting_lineup_pct": 0.65 if mins_season > 25 else 0.35,
                        "five_man_unit_net_rating": 0.0,
                        "on_court_net_rating": float(hist["PLUS_MINUS"].mean() if "PLUS_MINUS" in hist.columns else 0.0),
                        "off_court_net_rating": 0.0,
                        "on_off_differential": float(hist["PLUS_MINUS"].mean() if "PLUS_MINUS" in hist.columns else 0.0),
                        "lineups_played_count": 1.0,
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
                        "effective_fg_pct": float((fga_per_game + 0.5 * fg3a_per_game) / fga_per_game if fga_per_game > 0 else 0.0),
                        "pie": float(hist["PLUS_MINUS"].mean() / 100.0 if "PLUS_MINUS" in hist.columns else 0.0),
                        "net_rating": float(hist["PLUS_MINUS"].mean() if "PLUS_MINUS" in hist.columns else 0.0),
                        "assist_percentage": float(hist["AST"].mean() / max(fga_per_game, 1) if "AST" in hist.columns else 0.0),
                        "rebound_percentage": float((oreb_per_game + dreb_per_game) / 100.0),
                        "last_3_games_trend": float(hist[target_col].tail(3).mean() - season_avg if len(hist) >= 3 else 0.0),
                        "last_5_games_trend": float(last5_avg - season_avg),
                        "last_10_games_trend": float(hist[target_col].tail(10).mean() - season_avg if len(hist) >= 10 else 0.0),
                        "games_above_season_avg_last5": float(sum(1 for v in hist[target_col].tail(5) if v > season_avg)),
                        "is_back_to_back": 0.0,
                        "days_since_last_game": 2.0,
                        "games_in_last_7_days": float(min(len(hist), 3)),
                        "travel_distance": 0.0,
                        "time_zone_change": 0.0,
                        "arena_altitude": 0.0,
                        "arena_capacity": 18000.0,
                        "home_court_advantage_rating": 3.5 if (is_home is not None and is_home) else (-3.5 if is_home is not None else 0.0),
                        "vs_team_win_pct": 0.500,
                        "vs_team_last_season_avg": float(season_avg),
                        "vs_team_home_away_split": 0.0,
                        "clutch_pts_per_game": float(pts_mean * 0.25),
                        "clutch_fg_pct": fg_pct_recent,
                        "clutch_minutes_per_game": float(mins_season * 0.15),
                        "paint_fga_per_game": float(fga_per_game * 0.40),
                        "mid_range_fga_per_game": float(fga_per_game * 0.25),
                        "rim_fga_per_game": float(fga_per_game * 0.35),
                        "corner_3_pct": float(fg3_pct_recent * 1.05),
                        "above_break_3_pct": fg3_pct_recent,
                        "coast_to_coast": 0.0,
                        "on_court_plus_minus": float(hist["PLUS_MINUS"].mean() if "PLUS_MINUS" in hist.columns else 0.0),
                        "off_court_plus_minus": 0.0,
                        "top_lineup_minutes_pct": 0.65 if mins_season > 25 else 0.35,
                        "model_accuracy_player": 0.70,
                        "avg_prediction_error_player": float(stddev * 0.5),
                        "calibration_score_player": 0.75,
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
                examples.append(Example(prop_type=prop_type, X=X, y=y, game_date=pd.to_datetime(row["GAME_DATE"])))

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
                examples.append(Example(prop_type=combo_prop, X=X, y=y, game_date=pd.to_datetime(row["GAME_DATE"])))

            stats_now = [float(row.get(c, 0.0)) for c in ["PTS", "REB", "AST", "STL", "BLK"]]
            dd_y = 1.0 if sum(1 for s in stats_now if s >= 10.0) >= 2 else 0.0
            td_y = 1.0 if sum(1 for s in stats_now if s >= 10.0) >= 3 else 0.0
            pts_values = hist["PTS"].tolist()
            if len(pts_values) >= 8:
                last5_avg = float(last5["PTS"].mean()) if not last5.empty else float(hist["PTS"].mean())
                season_avg = float(hist["PTS"].mean())
                stddev = float(np.std(pts_values))
                X = build_feature_vector(make_features(pts_values, last5_avg, season_avg, stddev)).X
                examples.append(Example(prop_type="double_double", X=X, y=dd_y, game_date=pd.to_datetime(row["GAME_DATE"])))
                examples.append(Example(prop_type="triple_double", X=X, y=td_y, game_date=pd.to_datetime(row["GAME_DATE"])))

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
        for e, yv in zip(train_exs, y_train):
            raw = e.X.iloc[0].to_dict()
            center = _sportsbook_center(raw)
            for ln in _sample_lines(float(center), prop):
                Xc_train_parts.append(build_classifier_vector(raw, line=ln).X)
                yc_train_parts.append(1 if float(yv) > float(ln) else 0)

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
            clf.fit(Xc_train, yc_train)

            proba = clf.predict_proba(Xc_test)[:, 1]
            auc = _safe_auc(yc_test, proba)
            brier = float(brier_score_loss(yc_test, proba))
            
            log_loss = float(-np.mean(yc_test * np.log(proba + 1e-15) + (1 - yc_test) * np.log(1 - proba + 1e-15))) if len(yc_test) > 0 else 0.0
            vig = 0.045
            ev = float(np.mean((proba * (100/110)) - ((1 - proba) * (110/100)))) if len(proba) > 0 else 0.0

            cal_cut = int(len(Xc_train) * 0.8)
            Xc_fit, Xc_cal = Xc_train.iloc[:cal_cut], Xc_train.iloc[cal_cut:]
            yc_fit, yc_cal = yc_train[:cal_cut], yc_train[cal_cut:]

            clf2 = create_classifier(use_xgboost=True, optimized_params=optimized_params)
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
    args = ap.parse_args()

    if args.seasons:
        seasons = [s.strip() for s in str(args.seasons).split(",") if s.strip()]
    else:
        seasons = [args.season or EnhancedMLPredictor().current_season]
    predictor = EnhancedMLPredictor(model_dir=args.models_dir)
    precomputed = PrecomputedStore(args.db)

    print(f"[train] building examples seasons={seasons} max_players={args.max_players} ...")
    examples = []
    for s in seasons:
        print(f"[train] season {s}: building examples...")
        examples.extend(build_training_examples(season=s, max_players=args.max_players, predictor=predictor, precomputed=precomputed))
    print(f"[train] total examples: {len(examples)}")

    print("[train] training + saving models ...")
    meta = train_and_save(args.models_dir, examples)
    print("[train] done. props trained:", list(meta.get("props", {}).keys()))


if __name__ == "__main__":
    main()


