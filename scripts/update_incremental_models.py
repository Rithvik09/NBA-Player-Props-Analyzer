"""
Incremental daily updater:
- fetch new games for top-N players for a season
- build rolling-window features
- partial_fit incremental models (SGD)

This is designed to be fast daily after an initial backfill.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
import sqlite3

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog, CommonPlayerInfo  # noqa: E402
from nba_api.stats.static import teams  # noqa: E402

from src.models import EnhancedMLPredictor  # noqa: E402
from src.precomputed_store import PrecomputedStore  # noqa: E402
from src.incremental_models import IncrementalModelManager  # noqa: E402
from src.ml_features import build_feature_vector, build_classifier_vector  # noqa: E402


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
    alias = {"NOR": "NOP", "PHO": "PHX", "UTH": "UTA"}
    abbrev = alias.get(abbrev, abbrev)
    t = teams.find_team_by_abbreviation(abbrev)
    return int(t["id"]) if t else None


def _sample_lines(center: float, prop_type: str) -> list[float]:
    if prop_type in ("double_double", "triple_double"):
        return [0.5]
    step = 0.5
    lines = [max(0.0, center - 1.0), max(0.0, center - 0.5), max(0.0, center), center + 0.5, center + 1.0]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="basketball_data.db")
    ap.add_argument("--season", required=True)
    ap.add_argument("--max-players", type=int, default=350)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--backfill", action="store_true", help="process all games (ignore state)")
    args = ap.parse_args()

    # This script is a standalone job runner; it should not depend on Flask startup side-effects.
    os.environ["AUTO_JOBS"] = "0"

    predictor = EnhancedMLPredictor(model_dir=args.models_dir)
    precomputed = PrecomputedStore(args.db)
    mm = IncrementalModelManager(args.models_dir)

    # pick top-N players by minutes this season
    players_df = leaguedashplayerstats.LeagueDashPlayerStats(
        season=args.season, measure_type_detailed_defense="Base"
    ).get_data_frames()[0]
    players_df = players_df.sort_values("MIN", ascending=False).head(args.max_players)
    players = list(players_df["PLAYER_ID"].astype(int).tolist())

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS incremental_training_state (
            season TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            last_game_date TEXT,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (season, player_id)
        )
        """
    )
    conn.commit()

    def get_last(player_id: int) -> str | None:
        if args.backfill:
            return None
        cur.execute(
            "SELECT last_game_date FROM incremental_training_state WHERE season = ? AND player_id = ?",
            (args.season, int(player_id)),
        )
        r = cur.fetchone()
        return r[0] if r else None

    def set_last(player_id: int, last_date: str):
        cur.execute(
            "INSERT OR REPLACE INTO incremental_training_state (season, player_id, last_game_date, updated_at) VALUES (?, ?, ?, ?)",
            (args.season, int(player_id), str(last_date), int(time.time())),
        )

    pre = precomputed.refresh(force=True)
    dvp_map = pre.get("dvp", {})
    dvp_pos_avgs = pre.get("dvp_pos_avgs", {})
    defenders_map = pre.get("defenders", {})

    try:
        for idx, player_id in enumerate(players, start=1):
            if idx % 25 == 0:
                print(f"[inc] players processed {idx}/{len(players)}")

            last_seen = get_last(player_id)

            try:
                gl = playergamelog.PlayerGameLog(player_id=player_id, season=args.season).get_data_frames()[0]
            except Exception:
                continue
            if gl is None or gl.empty:
                continue

            gl = gl.copy()
            gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
            gl = gl.sort_values("GAME_DATE")

            # position buckets (for DVP and defender group)
            try:
                info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
                raw_pos = str(info["POSITION"].iloc[0] if "POSITION" in info.columns else "").upper()
            except Exception:
                raw_pos = ""
            if "C" in raw_pos and "G" not in raw_pos:
                dvp_pos, pos_group = "C", "C"
            elif "G" in raw_pos and "F" not in raw_pos:
                dvp_pos, pos_group = "SG", "G"
            elif "F" in raw_pos and "C" not in raw_pos and "G" not in raw_pos:
                dvp_pos, pos_group = "SF", "F"
            elif "C" in raw_pos:
                dvp_pos, pos_group = "C", "C"
            elif "G" in raw_pos:
                dvp_pos, pos_group = "SG", "G"
            else:
                dvp_pos, pos_group = "SF", "F"

            # iterate games and train on new ones
            for g_i in range(10, len(gl)):
                row = gl.iloc[g_i]
                game_date = row["GAME_DATE"].strftime("%Y-%m-%d")
                if last_seen and game_date <= last_seen:
                    continue

                hist = gl.iloc[:g_i]
                last5 = hist.tail(5)
                mins_last5 = float(last5["MIN"].mean()) if not last5.empty else float(hist["MIN"].mean())
                mins_season = float(hist["MIN"].mean()) if len(hist) else 0.0

                # Tier 1: Calculate shooting efficiency & volume from history
                fg_pct_recent = float(last5["FG_PCT"].mean()) if not last5.empty and "FG_PCT" in last5.columns else 0.0
                fg3_pct_recent = float(last5["FG3_PCT"].mean()) if not last5.empty and "FG3_PCT" in last5.columns else 0.0
                ft_pct_recent = float(last5["FT_PCT"].mean()) if not last5.empty and "FT_PCT" in last5.columns else 0.0
                fga_per_game = float(hist["FGA"].mean()) if "FGA" in hist.columns else 0.0
                fg3a_per_game = float(hist["FG3A"].mean()) if "FG3A" in hist.columns else 0.0
                fta_per_game = float(hist["FTA"].mean()) if "FTA" in hist.columns else 0.0

                # Tier 1: Rebounding split
                oreb_per_game = float(hist["OREB"].mean()) if "OREB" in hist.columns else 0.0
                dreb_per_game = float(hist["DREB"].mean()) if "DREB" in hist.columns else 0.0

                # Tier 1: Impact & context
                plus_minus_avg = float(hist["PLUS_MINUS"].mean()) if "PLUS_MINUS" in hist.columns else 0.0
                fouls_per_game = float(hist["PF"].mean()) if "PF" in hist.columns else 0.0
                win_rate_last10 = float((hist.tail(10)["WL"] == "W").sum() / min(10, len(hist))) if "WL" in hist.columns else 0.5

                # Tier 4: Calculated Efficiency Metrics
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
                
                # Tier 4: Opponent-Adjusted Stats
                pts_mean = hist["PTS"].mean() if "PTS" in hist.columns else 0
                pts_std = hist["PTS"].std() if "PTS" in hist.columns else 0
                consistency_score = max(0.0, 1.0 - (pts_std / max(pts_mean, 1)))
                
                ceiling_threshold = pts_mean * 1.5
                ceiling_games = (hist["PTS"] >= ceiling_threshold).sum() if "PTS" in hist.columns else 0
                ceiling_game_frequency = ceiling_games / max(len(hist), 1)

                team_abbrev, opp_abbrev, is_home = _parse_matchup(row.get("MATCHUP"))
                team_id = _team_id(team_abbrev)
                opp_id = _team_id(opp_abbrev)
                if not team_id or not opp_id:
                    continue

                team_ctx = predictor.get_team_context(team_id, include_injuries=False, season=args.season)
                opp_ctx = predictor.get_team_context(opp_id, include_injuries=False, season=args.season)

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
                        # Tier 4: Calculated Efficiency Metrics
                        "points_per_shot": float(points_per_shot),
                        "ast_to_tov_ratio": float(ast_to_tov_ratio),
                        "reb_rate_per_36": float(reb_rate_per_36),
                        "scoring_efficiency_trend": float(scoring_efficiency_trend),
                        "usage_trend": float(usage_trend),
                        "minutes_volatility": float(minutes_volatility),
                        "blowout_game_pct": float(blowout_game_pct),
                        "close_game_pct": float(close_game_pct),
                        # Tier 4: Opponent Defensive Trends (from team context)
                        "opp_def_rating_last5": float(opp_ctx.get("defensive_rating", 110.0)),
                        "opp_def_rating_trend": 0.0,  # Would need historical team data
                        "opp_def_rating_home_away_split": 0.0,
                        "opp_blocks_per_game_last5": float(opp_base.get("blk", 0.0)),
                        "opp_steals_per_game_last5": float(opp_base.get("stl", 0.0)),
                        # Tier 4: Game Context Features
                        "days_rest_opponent": 2,  # Placeholder
                        "opponent_back_to_back": 0,
                        "playoff_implications": 0,
                        "rivalry_game": 0,
                        "national_tv_game": 0,
                        "season_phase": 0,  # 0=early, 1=mid, 2=late
                        # Tier 4: Teammate Impact
                        "primary_teammate_out": 0,
                        "secondary_teammate_out": 0,
                        "new_teammate_games": 0,
                        "lineup_stability_score": 1.0,
                        "bench_strength": 0.0,
                        # Tier 4: Opponent-Adjusted Stats
                        "pts_vs_top10_defenses": float(pts_mean),  # Placeholder
                        "pts_vs_bottom10_defenses": float(pts_mean),
                        "consistency_score": float(consistency_score),
                        "ceiling_game_frequency": float(ceiling_game_frequency),
                        # Tier 4: Advanced Defensive Metrics (would need API call)
                        "def_fg_pct_allowed": 0.0,
                        "def_rating_individual": 0.0,
                        "deflections_per_game": 0.0,
                        "contested_shots_per_game": 0.0,
                        # Tier 4: Play Type Data (would need API call)
                        "pnr_ball_handler_pct": 0.0,
                        "pnr_roll_man_pct": 0.0,
                        "isolation_pct": 0.0,
                        "spot_up_pct": 0.0,
                        "post_up_pct": 0.0,
                        "transition_pct": 0.0,
                        # Tier 5: Player Streaks (calculate from historical)
                        "consecutive_over_games": 0,  # Would need line history
                        "consecutive_under_games": 0,
                        "hot_hand_indicator": 0.0,
                        "recent_variance_spike": float(hist.tail(3)["PTS"].std() / max(hist["PTS"].std(), 0.1) - 1.0) if "PTS" in hist.columns and len(hist) >= 3 else 0.0,
                        # Tier 5: Game Importance (would need standings)
                        "playoff_seeding_impact": 0.5,  # Default mid-season
                        "tanking_indicator": 0.0,
                        "must_win_situation": 0.0,
                        "games_back_from_playoff": 0.0,
                        # Tier 5: Rotation Patterns (estimate from minutes)
                        "fourth_quarter_usage_rate": 0.25 if mins_season > 30 else 0.18,
                        "garbage_time_minutes_pct": float(blowout_game_pct * 0.15),
                        "typical_substitution_minute": float(min(48.0, mins_season + 3.0)),
                        "crunch_time_usage": 0.28 if mins_season > 28 else 0.15,
                        # Tier 5: Specific Matchup History (would need opponent tracking)
                        "career_vs_defender": 0.0,
                        "recent_vs_defender": 0.0,
                        "player_vs_arena": 0.0,
                        # Tier 6: Shot Quality Metrics (estimated from game logs)
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
                        # Tier 6: Touch & Usage (estimated from usage rate)
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
                        # Tier 6: Lineup Context (estimated)
                        "net_rating_with_starters": 0.0,
                        "usage_rate_with_star_out": float(hist["FGA"].mean() * 1.1 if "FGA" in hist.columns else 0.0),
                        "minutes_with_starting_lineup_pct": 0.65 if mins_season > 25 else 0.35,
                        "five_man_unit_net_rating": 0.0,
                        "on_court_net_rating": float(hist["PLUS_MINUS"].mean() if "PLUS_MINUS" in hist.columns else 0.0),
                        "off_court_net_rating": 0.0,
                        "on_off_differential": float(hist["PLUS_MINUS"].mean() if "PLUS_MINUS" in hist.columns else 0.0),
                        "lineups_played_count": 1.0,
                    }

                # train base + combo + dd/td
                train_items = []
                for prop, col in STAT_TARGETS.items():
                    series = hist[col]
                    values = series.tolist()
                    if len(values) < 8:
                        continue
                    last5_avg = float(last5[col].mean()) if not last5.empty else float(series.mean())
                    season_avg = float(series.mean())
                    stddev = float(np.std(values))
                    y = float(row[col])
                    train_items.append((prop, values, last5_avg, season_avg, stddev, y))

                for prop, cols in COMBO_TARGETS.items():
                    series = hist[cols].sum(axis=1)
                    values = series.tolist()
                    if len(values) < 8:
                        continue
                    last5_avg = float(series.tail(5).mean()) if len(series) >= 5 else float(series.mean())
                    season_avg = float(series.mean())
                    stddev = float(np.std(values))
                    y = float(sum(float(row.get(c, 0.0)) for c in cols))
                    train_items.append((prop, values, last5_avg, season_avg, stddev, y))

                stats_now = [float(row.get(c, 0.0)) for c in ["PTS", "REB", "AST", "STL", "BLK"]]
                dd_y = 1.0 if sum(1 for s in stats_now if s >= 10.0) >= 2 else 0.0
                td_y = 1.0 if sum(1 for s in stats_now if s >= 10.0) >= 3 else 0.0
                pts_values = hist["PTS"].tolist()
                if len(pts_values) >= 8:
                    last5_avg = float(last5["PTS"].mean()) if not last5.empty else float(hist["PTS"].mean())
                    season_avg = float(hist["PTS"].mean())
                    stddev = float(np.std(pts_values))
                    train_items.append(("double_double", pts_values, last5_avg, season_avg, stddev, dd_y))
                    train_items.append(("triple_double", pts_values, last5_avg, season_avg, stddev, td_y))

                for prop, values, last5_avg, season_avg, stddev, y in train_items:
                    feats = make_features(values, last5_avg, season_avg, stddev)
                    Xr = build_feature_vector(feats).X.values.astype(float)
                    # reg update (skip for dd/td)
                    if prop not in ("double_double", "triple_double"):
                        mm.partial_fit_reg(prop, Xr, np.array([float(y)], dtype=float))

                    # classifier updates (line aware)
                # sportsbook-like lines are generally set near expected value (blend recent + season)
                center = 0.7 * float(last5_avg) + 0.3 * float(season_avg)
                    for ln in _sample_lines(center, prop):
                        Xc = build_classifier_vector(feats, line=float(ln)).X.values.astype(float)
                        yy = 1 if float(y) > float(ln) else 0
                        mm.partial_fit_clf(prop, Xc, np.array([yy], dtype=int))

                set_last(player_id, game_date)

            conn.commit()
    finally:
        # Best-effort persistence for resuming after interruptions/crashes.
        try:
            conn.commit()
        except Exception:
            pass
        try:
            mm.save_all()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    print("[inc] done.")


if __name__ == "__main__":
    main()


