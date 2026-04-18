from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# Feature schema
NUMERIC_FEATURE_KEYS: list[str] = [
    # player performance distribution
    "recent_avg",
    "season_avg",
    "stddev",
    "games_played",
    "max_recent",
    "min_recent",
    # minutes / schedule
    "mins_last5",
    "mins_season",
    "rest_days",
    "is_home_game_num",
    "recent_away_streak",
    # opponent/team environment
    "team_pace",
    "opp_pace",
    "team_off_rating",
    "team_def_rating",
    "opp_off_rating",
    "opp_def_rating",
    # injuries (coarse)
    "team_key_players_out",
    "opp_key_players_out",
    "opp_injury_impact",
    # team style
    "team_pts_fb",
    "opp_pts_fb_allowed",
    "team_pts_off_tov",
    "opp_pts_off_tov_allowed",
    "opp_pts_paint",
    # base opportunities
    "opp_fga",
    "opp_fg_pct",
    "opp_fg3a",
    "opp_fg3_pct",
    "opp_tov",
    "opp_stl",
    "opp_blk",
    # league baselines
    "lg_pts_fb",
    "lg_opp_pts_fb",
    "lg_pts_off_tov",
    "lg_opp_pts_off_tov",
    "lg_fga",
    "lg_fg_pct",
    "lg_fg3a",
    "lg_tov",
    "lg_stl",
    # matchup history
    "vs_team_avg",
    "matchup_games",
    # DVP deltas
    "dvp_gp",
    "dvp_pts_delta",
    "dvp_reb_delta",
    "dvp_ast_delta",
    "dvp_fg3m_delta",
    "dvp_stl_delta",
    "dvp_blk_delta",
    "dvp_tov_delta",
    # elite defender
    "primary_defender_score01",
    # Tier 1 additions: shooting efficiency & volume (from existing game logs)
    "fg_pct_recent",
    "fg3_pct_recent",
    "ft_pct_recent",
    "fga_per_game",
    "fg3a_per_game",
    "fta_per_game",
    # Tier 1: rebounding split
    "oreb_per_game",
    "dreb_per_game",
    # Tier 1: impact & context
    "plus_minus_avg",
    "fouls_per_game",
    "win_rate_last10",
    # Tier 2 Quick Wins: momentum/trend features (from existing game logs)
    "last_3_games_trend",
    "last_5_games_trend",
    "last_10_games_trend",
    "games_above_season_avg_last5",
    # Tier 2 Quick Wins: schedule/fatigue features (from game dates)
    "is_back_to_back",
    "days_since_last_game",
    "games_in_last_7_days",
    # Tier 2: Advanced Player Metrics (from PlayerDashboardByGeneralSplits)
    "usage_rate",
    "true_shooting_pct",
    "effective_fg_pct",
    "assist_percentage",
    "rebound_percentage",
    "pie",
    # Tier 2: Shot Location Data (from PlayerDashboardByShootingSplits)
    "rim_fga_per_game",
    "paint_fga_per_game",
    "mid_range_fga_per_game",
    "corner_3_pct",
    "above_break_3_pct",
    # Tier 2: Clutch Performance (from PlayerDashboardByClutch)
    "clutch_pts_per_game",
    "clutch_fg_pct",
    "clutch_minutes_per_game",
    # Tier 3: Time Zone & Travel (calculated from locations)
    "time_zone_change",
    "travel_distance",
    "coast_to_coast",
    # Tier 3: Lineup Context (from PlayerDashboardByLineups)
    "on_court_plus_minus",
    "off_court_plus_minus",
    "net_rating",
    "top_lineup_minutes_pct",
    # Tier 3: Opponent History Enhancement (calculated from existing data)
    "vs_team_last_season_avg",
    "vs_team_home_away_split",
    "vs_team_win_pct",
    # Tier 3: Arena/Environmental (static data)
    "arena_altitude",
    "arena_capacity",
    "home_court_advantage_rating",
    # Tier 3: Historical Model Performance (track internally)
    "model_accuracy_player",
    "avg_prediction_error_player",
    "calibration_score_player",
    # Tier 4: Calculated Efficiency Metrics (from existing game logs)
    "points_per_shot",
    "ast_to_tov_ratio",
    "reb_rate_per_36",
    "scoring_efficiency_trend",
    "usage_trend",
    "minutes_volatility",
    "blowout_game_pct",
    "close_game_pct",
    # Tier 4: Opponent Defensive Trends (from team stats)
    "opp_def_rating_home_away_split",
    "opp_blocks_per_game_last5",
    "opp_steals_per_game_last5",
    # Tier 4: Game Context Features (from schedule/standings)
    "days_rest_opponent",
    "opponent_back_to_back",
    "playoff_implications",
    "rivalry_game",
    "national_tv_game",
    "season_phase",
    # Tier 4: Teammate Impact (from roster/injuries)
    "primary_teammate_out",
    "secondary_teammate_out",
    "new_teammate_games",
    "lineup_stability_score",
    "bench_strength",
    # Tier 4: Opponent-Adjusted Stats (calculated from game logs)
    "pts_vs_top10_defenses",
    "pts_vs_bottom10_defenses",
    "consistency_score",
    "ceiling_game_frequency",
    # Tier 4: Advanced Defensive Metrics (from PlayerDashboardByOpponent)
    "def_fg_pct_allowed",
    "def_rating_individual",
    "deflections_per_game",
    "contested_shots_per_game",
    # Tier 4: Play Type Data (from PlayerDashboardByPlayType)
    "pnr_ball_handler_pct",
    "pnr_roll_man_pct",
    "isolation_pct",
    "spot_up_pct",
    "post_up_pct",
    "transition_pct",
    # Tier 5: Player Streaks (from game logs vs betting lines)
    "consecutive_over_games",
    "consecutive_under_games",
    "hot_hand_indicator",
    "recent_variance_spike",
    # Tier 5: Game Importance (calculated from standings/schedule)
    "playoff_seeding_impact",
    "tanking_indicator",
    "must_win_situation",
    "games_back_from_playoff",
    # Tier 5: Rotation Patterns (from play-by-play or game logs)
    "fourth_quarter_usage_rate",
    "garbage_time_minutes_pct",
    "typical_substitution_minute",
    "crunch_time_usage",
    # Tier 5: Specific Matchup History (from PlayerVsPlayer)
    "career_vs_defender",
    "recent_vs_defender",
    "player_vs_arena",
    # Tier 6: Shot Quality Metrics (from ShotChart/Tracking)
    "avg_shot_distance",
    "contested_shot_pct",
    "open_shot_pct",
    "wide_open_shot_pct",
    "catch_and_shoot_pct",
    "pull_up_shot_pct",
    "paint_touch_frequency",
    "corner_three_pct",
    "above_break_three_pct",
    "restricted_area_fg_pct",
    "mid_range_frequency",
    "shot_quality_vs_expected",
    "avg_shot_clock_time",
    "late_clock_shot_frequency",
    "early_clock_shot_frequency",
    # Tier 6: Touch/Usage Details (from Tracking)
    "touches_per_game",
    "avg_dribbles_per_touch",
    "avg_seconds_per_touch",
    "elbow_touches_per_game",
    "post_touches_per_game",
    "paint_touches_per_game",
    "front_court_touches_per_game",
    "time_of_possession_per_game",
    "touches_per_possession",
    "avg_points_per_touch",
    # Tier 6: Lineup Context (from Lineups endpoint)
    "net_rating_with_starters",
    "usage_rate_with_star_out",
    "minutes_with_starting_lineup_pct",
    "five_man_unit_net_rating",
    "lineups_played_count",
    # Tier 7: Time-Series Features (Quick Win #2)
    "rolling_7day_avg",
    "rolling_14day_avg",
    "rolling_30day_avg",
    "ewm_alpha_0.3",  # Exponential weighted moving average
    "ewm_alpha_0.5",
    "trend_slope_10games",  # Linear trend over last 10 games
    "trend_slope_5games",
    "volatility_ratio",  # std/mean ratio
    "momentum_score",  # (last_5_avg - prev_5_avg) / prev_5_avg
    "games_above_season_avg_7day",
    "games_above_season_avg_14day",
    # Tier 7: Enhanced Matchup Features (Quick Win #5)
    "head_to_head_avg",  # Player vs specific opponent team (enhanced)
    "head_to_head_games",  # Number of H2H games
    "position_vs_position_dvp",  # Position-specific DVP
    "matchup_pace",  # Pace when these teams play
    "defender_switching_frequency",  # Does opponent switch on screens?
    "historical_game_script_avg",  # Blowout vs close game impact
    # Tier 8: Relative Rest Advantage
    "rest_advantage",  # Player rest - Opponent rest
    "rest_advantage_abs",  # Absolute difference in rest
    "both_teams_rested",  # Both teams have 2+ days rest
    # Tier 8: Opponent Recent Form
    "opp_def_rating_last5",  # Opponent's defensive rating in last 5 games
    "opp_def_rating_last10",  # Opponent's defensive rating in last 10 games
    "opp_def_rating_trend",  # Is opponent defense improving/declining?
    "opp_pace_last5",  # Opponent's recent pace
    "opp_win_rate_last10",  # Opponent's recent win rate
    # Tier 8: Player Age & Experience
    "player_age",  # Current age
    "years_experience",  # Years in NBA
    "is_rookie",  # 1 if rookie, 0 otherwise
    "is_veteran",  # 1 if 10+ years, 0 otherwise
    # Tier 8: Game Script Prediction
    "expected_game_script",  # Will this be a blowout or close game?
    "blowout_probability",  # Probability of blowout (>15 pt margin)
    "close_game_probability",  # Probability of close game (<5 pt margin)
    # Tier 8: Quarter-Specific Performance
    "first_quarter_avg",  # Average points in 1st quarter
    "fourth_quarter_avg",  # Average points in 4th quarter
    "clutch_performance_score",  # Performance in close games (4th quarter, <5 min, <5 pt diff)
    # Tier 8: Enhanced Shot Selection Quality
    "shot_selection_rating",  # Quality of shots taken (open vs contested)
    "bad_shot_frequency",  # Frequency of forced/bad shots
    "shot_clock_management",  # Performance in different shot clock situations
    # Tier 8: Team Chemistry Metrics
    "teammate_chemistry_score",  # How long have key teammates played together?
    "lineup_continuity",  # How stable is the starting lineup?
    "team_win_streak",  # Current win streak
    "team_loss_streak",  # Current loss streak
    # Tier 8: Enhanced Defensive Matchup Quality
    "primary_defender_rating",  # Defensive rating of primary defender
    "primary_defender_age",  # Age of primary defender
    "defender_size_mismatch",  # Height/weight difference
    "defender_recent_form",  # How has defender been playing recently?
    # Tier 9: Pace-Adjusted Stats (Per 100 Possessions)
    "pts_per_100",  # Points per 100 possessions
    "ast_per_100",  # Assists per 100 possessions
    "reb_per_100",  # Rebounds per 100 possessions
    "stl_per_100",  # Steals per 100 possessions
    "blk_per_100",  # Blocks per 100 possessions
    "tov_per_100",  # Turnovers per 100 possessions
    # Tier 9: Free Throw Rate & Foul Drawing
    "ft_rate",  # Free throw attempts per field goal attempt
    "fouls_drawn_per_game",  # How many fouls player draws (estimated from FTA)
    "ft_attempts_per_game",  # Free throw attempts per game (already have fta_per_game, but keep for consistency)
    "and_one_frequency",  # Frequency of and-1 opportunities (estimated)
    "foul_drawing_ability",  # Fouls drawn per 36 minutes
    # Tier 9: Rebounding Rates & Positioning
    "oreb_rate",  # Offensive rebound percentage
    "dreb_rate",  # Defensive rebound percentage
    "total_reb_rate",  # Total rebound percentage
    "rebound_contested_pct",  # Percentage of rebounds that were contested (estimated)
    "rebound_positioning_score",  # Quality of rebounding position (estimated)
    # Tier 9: Points in the Paint
    "paint_pts_per_game",  # Points scored in the paint
    "paint_attempts_per_game",  # Field goal attempts in the paint
    "paint_touch_to_points",  # Points per paint touch
    "restricted_area_attempts",  # Attempts in restricted area
    # Tier 9: Game Situation Performance
    "performance_when_leading",  # Performance when team is leading
    "performance_when_trailing",  # Performance when team is trailing
    "performance_when_tied",  # Performance when score is tied
    "performance_in_overtime",  # Performance in overtime periods
    "performance_by_score_differential",  # Performance by point differential ranges
    # Tier 10: Minutes Fatigue (Cumulative Workload)
    "minutes_last_3_games",  # Total minutes played in last 3 games
    "minutes_last_5_games",  # Total minutes played in last 5 games
    "minutes_last_7_games",  # Total minutes played in last 7 games
    "avg_minutes_last_3",  # Average minutes per game in last 3 games
    "minutes_fatigue_score",  # Normalized fatigue indicator (0-1 scale)
    # Tier 10: Player-Level Advanced Metrics (from PlayerDashboardByGeneralSplits)
    "player_off_rating",  # Player's individual offensive rating (on-court)
    "player_def_rating",  # Player's individual defensive rating (on-court)
    "player_pace",  # Pace when player is on court
    "fta_rate_player",  # Free throw attempt rate (from PlayerDashboard)
    "pct_fga_2pt",  # Percentage of FGA that are 2-pointers
    "pct_fga_3pt",  # Percentage of FGA that are 3-pointers
    "pct_pts_in_paint",  # Percentage of points from paint
    "pct_pts_off_tov",  # Percentage of points off turnovers
    "pct_pts_fb",  # Percentage of points from fast breaks
    # New: Referee tendencies (precomputed daily from Basketball-Reference)
    "ref_foul_rate",
    "ref_home_bias",
    "ref_pace_tendency",
    # New: Rolling DVP — recent opponent defensive form vs. season average
    "dvp_pts_delta_last5",
    "dvp_pts_delta_last10",
    "dvp_reb_delta_last5",
    "dvp_ast_delta_last5",
    "dvp_fg3m_delta_last5",
    # New: Opponent foul tendency (affects FTA and scoring)
    "opp_foul_rate_per48",
    "opp_foul_rate_last5",
    # New: Return-from-injury trajectory
    "games_since_return",
    "missed_games_before_return",
    # New: Calendar / season-phase position
    "days_into_season",
    "season_phase_numeric",
    "games_remaining_approx",
    # New: Defender health and lineup stability
    "primary_defender_active",
    "opp_lineup_changes_last5",
    # New: Market-implied game environment
    "implied_game_total",
    # Advanced player stats (official NBA API Advanced measure)
    "usg_pct_official",
    "ts_pct_official",
    "efg_pct_official",
    "ast_pct_official",
    "oreb_pct_official",
    "dreb_pct_official",
    "reb_pct_official",
    "net_rating_player",
    # Player bio
    "player_height_inches",
    "player_weight",
    # Clutch performance (last 5 min, within 5 pts)
    "clutch_fg3_pct",
    "clutch_fta_per_game",
    "clutch_plus_minus",
    "clutch_min_per_game",
    "clutch_games",
    # Hustle stats
    "charges_drawn_per_game",
    "screen_assists_per_game",
    # Shot profile by defender distance / shot type
    "open_shot_fg_pct",
    "open_shot_frequency",
    "tight_shot_fg_pct",
    "tight_shot_frequency",
    "catch_shoot_fg_pct",
    "catch_shoot_frequency",
    "pullup_fg_pct",
    "pullup_frequency",
    # Synergy play types
    "iso_poss_pct",
    "iso_ppp",
    "pnr_bh_poss_pct",
    "pnr_bh_ppp",
    "pnr_roll_poss_pct",
    "pnr_roll_ppp",
    "spotup_poss_pct",
    "spotup_ppp",
    "transition_poss_pct",
    "transition_ppp",
    "postup_poss_pct",
    "cut_poss_pct",
    # On/off court differential
    "on_court_net_rating",
    "off_court_net_rating",
    "on_off_differential",
    # Shot zone breakdown
    "rim_fga_pct",
    "rim_fg_pct",
    "paint_fga_pct",
    "paint_fg_pct",
    "midrange_fga_pct",
    "midrange_fg_pct",
    "corner3_fga_pct",
    "corner3_fg_pct",
    "above_break3_fga_pct",
    "above_break3_fg_pct",
    # Quarter splits
    "q1_avg",
    "q2_avg",
    "q3_avg",
    "q4_avg",
    "q4_min_per_game",
    # Opponent shot zone defense
    "opp_rim_fg_pct_allowed",
    "opp_paint_fg_pct_allowed",
    "opp_midrange_fg_pct_allowed",
    "opp_corner3_fg_pct_allowed",
    "opp_above_break3_fg_pct_allowed",
    # Matchup-derived: player shot zone vs opponent zone defense
    "rim_shot_quality_matchup",
    "midrange_shot_quality_matchup",
    "corner3_shot_quality_matchup",
    "above_break3_shot_quality_matchup",
    # Synergy team defense
    "opp_pnr_ppp_allowed",
    "opp_iso_ppp_allowed",
    "opp_spotup_ppp_allowed",
    "opp_transition_ppp_allowed",
    "opp_postup_ppp_allowed",
    # Synergy matchup: player play type PPP vs team defense
    "pnr_matchup_advantage",
    "iso_matchup_advantage",
    "spotup_matchup_advantage",
    "transition_matchup_advantage",
    # Player tracking stats (from player_tracking_stats)
    "tracking_avg_speed",
    "tracking_avg_speed_off",
    "tracking_avg_speed_def",
    "tracking_dist_miles",
    "tracking_dist_miles_off",
    "tracking_dist_miles_def",
    "tracking_touches_pg",
    "tracking_time_of_poss_pg",
    "tracking_avg_drib_per_touch",
    "tracking_passes_made_pg",
    "tracking_potential_ast_pg",
    "tracking_secondary_ast_pg",
    # Scoring breakdown (from player_scoring_breakdown)
    "pct_pts_3pt",
    "pct_pts_paint",
    "pct_pts_ft",
    "pct_pts_midrange",
    "pct_uast_fgm",
    "pct_ast_fgm",
    # Year-over-year development (from player_yoy_stats)
    "yoy_pts_change",
    "yoy_ts_change",
    "yoy_usage_change",
    "seasons_in_league",
    # Opponent rest splits (from team_rest_splits)
    "opp_b2b_def_rating",
    "opp_b2b_pace",
    "opp_b2b_pts_allowed",
    "opp_rested_def_rating",
    "opp_rested_pace",
    # Team standings context (from team_standings)
    "team_win_pct",
    "team_conf_rank",
    "team_current_streak",
    "team_l10_wins",
    "opp_win_pct",
    "opp_conf_rank",
    "opp_current_streak",
    "opp_l10_wins",
    # Player vs specific opponent (from player_vs_opponent)
    "vs_opp_gp",
    "vs_opp_avg_pts",
    "vs_opp_fg_pct",
    "vs_opp_ts_pct",
    "vs_opp_avg_min",
    # Market / odds line movement features (from odds_tracker)
    "opening_line",
    "current_line",
    "line_movement",
    "line_movement_pct",
    "implied_over_prob",
    "implied_under_prob",
    "market_consensus_std",
    "sharp_action_score",
    "line_velocity",
    "stale_line_flag",
    "bookmaker_count",
    # Intensity / playoff context
    "is_playoff",
    "is_play_in",
    "series_game_num",
    "team_series_wins_in",
    "opp_series_wins_in",
    "is_elimination_game",
]

CLASSIFIER_EXTRA_KEYS: list[str] = [
    "line",
]


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, bool):
            return 1.0 if x else 0.0
        return float(x)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class FeatureVector:
    X: pd.DataFrame


def build_feature_vector(raw_features: dict[str, Any]) -> FeatureVector:
    safe = dict(raw_features)
    if "is_home_game_num" not in safe:
        is_home = safe.get("is_home_game", None)
        if is_home is True:
            safe["is_home_game_num"] = 1.0
        elif is_home is False:
            safe["is_home_game_num"] = 0.0
        else:
            safe["is_home_game_num"] = -1.0  # unknown/auto

    row = {k: _to_float(safe.get(k, 0.0)) for k in NUMERIC_FEATURE_KEYS}
    X = pd.DataFrame([row], columns=NUMERIC_FEATURE_KEYS)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return FeatureVector(X=X)


def build_classifier_vector(raw_features: dict[str, Any], line: float) -> FeatureVector:
    safe = dict(raw_features)
    safe["line"] = line
    base = build_feature_vector(safe).X
    base["line"] = _to_float(line, 0.0)
    cols = NUMERIC_FEATURE_KEYS + CLASSIFIER_EXTRA_KEYS
    base = base.reindex(columns=cols, fill_value=0.0)
    return FeatureVector(X=base)


