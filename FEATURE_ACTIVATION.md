# Disposition of the 162 originally disabled features

Implemented in `point_in_time_v3`: 115 recovered, 33 duplicate/covered, 14 source-blocked.

Active means an implemented, gated source path. Coverage and predictive usefulness are separate questions. Defender traits refer to a forecast from earlier matchups, not a confirmed assignment. The October 25 repair qualifies season-specific decay rules for 2023–24, 2024–25 and 2025–26; 2022–23 and current 2026–27 still fail. The seven defender fields are conditionally available in qualifying historical rows, not established current-season evidence. Four market capabilities now have real free live observations and replay, but historical training activation remains blocked. See the dated evidence in SOURCE_BLOCKERS.md.

Coverage: `logs/full_feature_audit/features.csv`. Source-blocker follow-up: `SOURCE_BLOCKERS.md`.

## Recovered

| Feature | Source | Definition |
|---|---|---|
| above_break3_shot_quality_matchup | player_and_opponent_prior_zone_measurements | Player zone FG fraction minus opponent allowed fraction, both completed prior season; explicit interaction useful to models without automatic interactions. Units: fraction difference. |
| and_one_frequency | dated_play_by_play_linked_events | A made field goal followed at the same clock by the same player1-of-1 free throw, linked by player/period/clock. Requires validated shot totals. Units: and-one opportunities/made field goal. |
| avg_shot_distance | dated_play_by_play_shots | Mean measured shot distance over prior20 appearances; requires known distance for every observed FGA, never substitutes zero for an unknown distance. Units: feet. |
| bench_strength | completed_prior_season_bench_boxscores | NBA bench-only team points/game, with the provider Bench filter. Units: bench points/game. |
| blowout_game_pct | dated_team_scores | Share of prior20 team games with final absolute margin>15 or<=5; not player plus-minus, and not predicted probabilities. Units: fraction. |
| blowout_probability | qualified_prior_season_game_script_forecaster | Logistic game-context forecasts fit/calibrated/qualified chronologically before target season. Only published if prior-period Brier beats a calibrated constant and AUC>0.5. Regular season only; failed qualification stays missing. Units: probability. |
| close_game_pct | dated_team_scores | Share of prior20 team games with final absolute margin>15 or<=5; not player plus-minus, and not predicted probabilities. Units: fraction. |
| close_game_probability | qualified_prior_season_game_script_forecaster | Logistic game-context forecasts fit/calibrated/qualified chronologically before target season. Only published if prior-period Brier beats a calibrated constant and AUC>0.5. Regular season only; failed qualification stays missing. Units: probability. |
| corner3_shot_quality_matchup | player_and_opponent_prior_zone_measurements | Player zone FG fraction minus opponent allowed fraction, both completed prior season; explicit interaction useful to models without automatic interactions. Units: fraction difference. |
| crunch_time_usage | completed_prior_season_clutch_advanced | Official clutch USG_PCT, rather than a ramp based on average minutes. Units: fraction. |
| def_fg_pct_allowed | dated_opposing_boxscores | FGM/FGA by teams facing this opponent in prior games, not opponent offensive FG%. Units: fraction. |
| def_rating_individual | qualified_defender_forecast_and_actual_matchup_points | Expected defender actual matchup points conceded per100 tracked partial possessions, prior20 observed games within400 days, min3 games and100 partial possessions. Distinct from official on-court team defensive rating. Units: opponent player points/100 tracked partial possessions. |
| defender_recent_form | qualified_defender_forecast_and_actual_matchup_points | Expected defender prior5 observed-game matchup scoring allowed minus prior20 observed-game rate; min3 games, strictly before prediction date. Units: points/100 partial possessions difference. |
| defender_size_mismatch | qualified_prior_matchup_defender_forecast_and_prior_bio | Regular-season expected defender from prior400-day player/opponent matchup possessions, min20 possessions, share>=40%, lead>=15%; latest earlier team membership must match opponent. Enabled only if preceding regular-season assignment backtest n>=100, accuracy>=50% and beats minutes baseline by>=5 points. Rating and bio use completed previous season; identity is a forecast, not a confirmed assignment. Unknown scheduled game type and postseason remain missing. Units: official on-court defensive rating; lagged years; defender-minus-player inches. |
| dvp_ast_delta | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_ast_delta_last5 | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_blk_delta | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_fg3m_delta | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_fg3m_delta_last5 | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_gp | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_pts_delta | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_pts_delta_last10 | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_pts_delta_last5 | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_reb_delta | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_reb_delta_last5 | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_stl_delta | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| dvp_tov_delta | dated_position_opponent_boxscores | Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing. Units: games; events/team-game difference. |
| early_clock_shot_frequency | completed_prior_season_shot_clock_dashboard | Actual FGA_FREQUENCY for24-22 plus22-18 seconds, and4-0 seconds respectively. No pace-based proxy. Units: fraction. |
| fouls_drawn_per_game | completed_prior_season_player_misc | Actual PFD field from NBA Misc. Units: fouls/game. |
| fourth_quarter_usage_rate | completed_prior_season_fourth_quarter_boxscores | (Q4 FGA+.44*Q4 FTA+Q4 TOV)/Q4 minutes*48; same usage units as main boxscore usage feature. Units: possessions/48min. |
| front_court_touches_per_game | completed_prior_season_possession_tracking | Actual POST_TOUCHES, FRONT_CT_TOUCHES and PTS_PER_PAINT_TOUCH tracking fields. Units: touches/game; paint_touch_to_points=points/touch. |
| games_back_from_playoff | dated_conference_team_records | As-of conference win-fraction rank (not official tie-break seeding); games behind conference10th record from wins/losses differential, floored0. Units: win-fraction rank with tied teams sharing rank; games. |
| games_since_return | official_injury_reports_and_prior_appearances | Prior player appearances since latest confirmed medical Out game, and confirmed Out games in the absence spell. Unknown/unreported injury history remains missing. Units: games. |
| garbage_time_minutes_pct | dated_validated_play_by_play_stints | Actual tracked stints; every player minute total must match recorded source precision (0.12min for exact clocks,0.51min for rounded league logs). Garbage rule: Q4 margin>=20 inside6min or>=15 inside2min. Starting unit exposure requires at least4 actual starters on floor. Units: fraction of tracked minutes. |
| is_elimination_game | dated_postseason_schedule_and_prior_results | Game type from NBA game ID, series state from earlier series games only. Regular-season values are known0. Postseason game logs must be present; no current/future outcomes. Units: boolean; game number; wins. |
| is_play_in | dated_postseason_schedule_and_prior_results | Game type from NBA game ID, series state from earlier series games only. Regular-season values are known0. Postseason game logs must be present; no current/future outcomes. Units: boolean; game number; wins. |
| is_playoff | dated_postseason_schedule_and_prior_results | Game type from NBA game ID, series state from earlier series games only. Regular-season values are known0. Postseason game logs must be present; no current/future outcomes. Units: boolean; game number; wins. |
| iso_matchup_advantage | player_and_opponent_prior_playtype_measurements | Player play-type PPP minus opponent defensive PPP, using completed prior season. Units: points/possession difference. |
| late_clock_shot_frequency | completed_prior_season_shot_clock_dashboard | Actual FGA_FREQUENCY for24-22 plus22-18 seconds, and4-0 seconds respectively. No pace-based proxy. Units: fraction. |
| lg_fg3a | completed_prior_season_league_aggregate | League averages of completed-prior-season team observations. FG percentage is weighted by total makes/attempts. Units: points/game, attempts/game, FG fraction, or events/game. |
| lg_fg_pct | completed_prior_season_league_aggregate | League averages of completed-prior-season team observations. FG percentage is weighted by total makes/attempts. Units: points/game, attempts/game, FG fraction, or events/game. |
| lg_fga | completed_prior_season_league_aggregate | League averages of completed-prior-season team observations. FG percentage is weighted by total makes/attempts. Units: points/game, attempts/game, FG fraction, or events/game. |
| lg_opp_pts_fb | completed_prior_season_league_aggregate | League averages of completed-prior-season team observations. FG percentage is weighted by total makes/attempts. Units: points/game, attempts/game, FG fraction, or events/game. |
| lg_opp_pts_off_tov | completed_prior_season_league_aggregate | League averages of completed-prior-season team observations. FG percentage is weighted by total makes/attempts. Units: points/game, attempts/game, FG fraction, or events/game. |
| lg_pts_fb | completed_prior_season_league_aggregate | League averages of completed-prior-season team observations. FG percentage is weighted by total makes/attempts. Units: points/game, attempts/game, FG fraction, or events/game. |
| lg_pts_off_tov | completed_prior_season_league_aggregate | League averages of completed-prior-season team observations. FG percentage is weighted by total makes/attempts. Units: points/game, attempts/game, FG fraction, or events/game. |
| lg_stl | completed_prior_season_league_aggregate | League averages of completed-prior-season team observations. FG percentage is weighted by total makes/attempts. Units: points/game, attempts/game, FG fraction, or events/game. |
| lg_tov | completed_prior_season_league_aggregate | League averages of completed-prior-season team observations. FG percentage is weighted by total makes/attempts. Units: points/game, attempts/game, FG fraction, or events/game. |
| lineup_continuity | dated_team_rotation_appearances | Rotation continuity=prior5 minutes share belonging to prior20 core players (>=10 prior appearances). New teammate games=appearances with the newest current rotation teammate inside the prior20 player-game window. Units: fraction; games. |
| lineups_played_count | completed_prior_season_five_player_lineups | For lineups containing this player: largest five-player unit minutes / all such minutes; number of observed units. Actual lineups, not a minutes ramp. Units: fraction; lineups. |
| midrange_shot_quality_matchup | player_and_opponent_prior_zone_measurements | Player zone FG fraction minus opponent allowed fraction, both completed prior season; explicit interaction useful to models without automatic interactions. Units: fraction difference. |
| minutes_with_starting_lineup_pct | dated_validated_play_by_play_stints | Actual tracked stints; every player minute total must match recorded source precision (0.12min for exact clocks,0.51min for rounded league logs). Garbage rule: Q4 margin>=20 inside6min or>=15 inside2min. Starting unit exposure requires at least4 actual starters on floor. Units: fraction of tracked minutes. |
| missed_games_before_return | official_injury_reports_and_prior_appearances | Prior player appearances since latest confirmed medical Out game, and confirmed Out games in the absence spell. Unknown/unreported injury history remains missing. Units: games. |
| national_tv_game | historical_game_broadcast_metadata | Recorded national broadcast schedule metadata, not inferred from player fame. Historical schedule publication timestamp is not independently archived. Units: boolean. |
| net_rating_with_starters | prior_season_observed_starting_units_and_official_unit_ratings | Completed-prior-season official five-player unit net ratings, restricted to units observed as actual starters in dated boxscores, weighted by unit minutes. Missing past starters is not filled by team net rating. Units: points/100poss, minute-weighted mean of unit ratings. |
| new_teammate_games | dated_team_rotation_appearances | Rotation continuity=prior5 minutes share belonging to prior20 core players (>=10 prior appearances). New teammate games=appearances with the newest current rotation teammate inside the prior20 player-game window. Units: fraction; games. |
| opp_above_break3_fg_pct_allowed | completed_prior_season_opponent_zones | Actual opponent makes/attempts in each shooting zone; corrected endpoint and preserved source season. Not substituted league averages. Units: fraction. |
| opp_b2b_def_rating | dated_team_history | Prior opponent games classified by actual schedule rest and venue. Rested means at least2 days off. Empty category is missing. Units: ratings/100poss; pace/48min; points/game; rating delta. |
| opp_b2b_pace | dated_team_history | Prior opponent games classified by actual schedule rest and venue. Rested means at least2 days off. Empty category is missing. Units: ratings/100poss; pace/48min; points/game; rating delta. |
| opp_b2b_pts_allowed | dated_team_history | Prior opponent games classified by actual schedule rest and venue. Rested means at least2 days off. Empty category is missing. Units: ratings/100poss; pace/48min; points/game; rating delta. |
| opp_conf_rank | dated_conference_team_records | As-of conference win-fraction rank (not official tie-break seeding); games behind conference10th record from wins/losses differential, floored0. Units: win-fraction rank with tied teams sharing rank; games. |
| opp_corner3_fg_pct_allowed | completed_prior_season_opponent_zones | Actual opponent makes/attempts in each shooting zone; corrected endpoint and preserved source season. Not substituted league averages. Units: fraction. |
| opp_def_rating_home_away_split | dated_team_history | Prior opponent games classified by actual schedule rest and venue. Rested means at least2 days off. Empty category is missing. Units: ratings/100poss; pace/48min; points/game; rating delta. |
| opp_injury_impact | official_timestamped_injury_reports | Confirmed Out only, before prediction cutoff, for the scheduled game. Key role=prior mean minutes>=20; impact=sum prior minutes of absent players/240, capped1. Unknown report is missing. Units: players; impact=fraction of240 team minutes. |
| opp_iso_ppp_allowed | completed_prior_season_defensive_synergy | Actual NBA defensive play-type efficiency, completed prior season. Units: points/possession. |
| opp_key_players_out | official_timestamped_injury_reports | Confirmed Out only, before prediction cutoff, for the scheduled game. Key role=prior mean minutes>=20; impact=sum prior minutes of absent players/240, capped1. Unknown report is missing. Units: players; impact=fraction of240 team minutes. |
| opp_lineup_changes_last5 | dated_starting_player_boxscores | Mean5-minus-overlap of actual starting units over successive opponent games, using only prior games. Units: starting positions changed/game. |
| opp_midrange_fg_pct_allowed | completed_prior_season_opponent_zones | Actual opponent makes/attempts in each shooting zone; corrected endpoint and preserved source season. Not substituted league averages. Units: fraction. |
| opp_paint_fg_pct_allowed | completed_prior_season_opponent_zones | Actual opponent makes/attempts in each shooting zone; corrected endpoint and preserved source season. Not substituted league averages. Units: fraction. |
| opp_pnr_ppp_allowed | completed_prior_season_defensive_synergy | Actual NBA defensive play-type efficiency, completed prior season. Units: points/possession. |
| opp_postup_ppp_allowed | completed_prior_season_defensive_synergy | Actual NBA defensive play-type efficiency, completed prior season. Units: points/possession. |
| opp_pts_fb_allowed | completed_prior_season_team_misc | Actual NBA Misc scoring components from completed prior season, not latest overwritten team snapshots. Units: points/game. |
| opp_pts_off_tov_allowed | completed_prior_season_team_misc | Actual NBA Misc scoring components from completed prior season, not latest overwritten team snapshots. Units: points/game. |
| opp_pts_paint | completed_prior_season_team_misc | Actual NBA Misc scoring components from completed prior season, not latest overwritten team snapshots. Units: points/game. |
| opp_rested_def_rating | dated_team_history | Prior opponent games classified by actual schedule rest and venue. Rested means at least2 days off. Empty category is missing. Units: ratings/100poss; pace/48min; points/game; rating delta. |
| opp_rested_pace | dated_team_history | Prior opponent games classified by actual schedule rest and venue. Rested means at least2 days off. Empty category is missing. Units: ratings/100poss; pace/48min; points/game; rating delta. |
| opp_rim_fg_pct_allowed | completed_prior_season_opponent_zones | Actual opponent makes/attempts in each shooting zone; corrected endpoint and preserved source season. Not substituted league averages. Units: fraction. |
| opp_series_wins_in | dated_postseason_schedule_and_prior_results | Game type from NBA game ID, series state from earlier series games only. Regular-season values are known0. Postseason game logs must be present; no current/future outcomes. Units: boolean; game number; wins. |
| opp_spotup_ppp_allowed | completed_prior_season_defensive_synergy | Actual NBA defensive play-type efficiency, completed prior season. Units: points/possession. |
| opp_transition_ppp_allowed | completed_prior_season_defensive_synergy | Actual NBA defensive play-type efficiency, completed prior season. Units: points/possession. |
| paint_touch_to_points | completed_prior_season_possession_tracking | Actual POST_TOUCHES, FRONT_CT_TOUCHES and PTS_PER_PAINT_TOUCH tracking fields. Units: touches/game; paint_touch_to_points=points/touch. |
| performance_by_score_differential | dated_play_by_play_score_states | Leading-state points/game minus trailing-state points/game, from validated scoring events, not final win/loss averages. Units: points/game difference. |
| performance_in_overtime | dated_play_by_play_score_states | Player points scored while own team leads/trails/is tied BEFORE scoring, or in overtime, averaged over prior observed appearances. PBP points must match NBA boxscore. Units: points/game. |
| performance_when_leading | dated_play_by_play_score_states | Player points scored while own team leads/trails/is tied BEFORE scoring, or in overtime, averaged over prior observed appearances. PBP points must match NBA boxscore. Units: points/game. |
| performance_when_tied | dated_play_by_play_score_states | Player points scored while own team leads/trails/is tied BEFORE scoring, or in overtime, averaged over prior observed appearances. PBP points must match NBA boxscore. Units: points/game. |
| performance_when_trailing | dated_play_by_play_score_states | Player points scored while own team leads/trails/is tied BEFORE scoring, or in overtime, averaged over prior observed appearances. PBP points must match NBA boxscore. Units: points/game. |
| player_age | completed_prior_season_bio | Observed completed-prior-season bio values; age is a lagged season age, not a fabricated exact birthday-age. Units: years; inches; pounds. |
| player_height_inches | completed_prior_season_bio | Observed completed-prior-season bio values; age is a lagged season age, not a fabricated exact birthday-age. Units: years; inches; pounds. |
| player_vs_arena | scheduled_arena_and_dated_player_appearances | Player mean at the actual scheduled arena minus overall prior20 mean. Arena identity comes from game metadata, not home/away guessing; no prior arena appearance is missing. Units: prop-unit difference. |
| player_weight | completed_prior_season_bio | Observed completed-prior-season bio values; age is a lagged season age, not a fabricated exact birthday-age. Units: years; inches; pounds. |
| playoff_away | dated_postseason_schedule_and_prior_results | Game type from NBA game ID, series state from earlier series games only. Regular-season values are known0. Postseason game logs must be present; no current/future outcomes. Units: boolean; game number; wins. |
| playoff_home | dated_postseason_schedule_and_prior_results | Game type from NBA game ID, series state from earlier series games only. Regular-season values are known0. Postseason game logs must be present; no current/future outcomes. Units: boolean; game number; wins. |
| pnr_matchup_advantage | player_and_opponent_prior_playtype_measurements | Player play-type PPP minus opponent defensive PPP, using completed prior season. Units: points/possession difference. |
| post_touches_per_game | completed_prior_season_possession_tracking | Actual POST_TOUCHES, FRONT_CT_TOUCHES and PTS_PER_PAINT_TOUCH tracking fields. Units: touches/game; paint_touch_to_points=points/touch. |
| primary_defender_active | qualified_defender_forecast_and_timestamped_injuries | Availability of the historically expected defender, before adjusting assignment for injury. Out=0; Available or absent from a validated team report=1; uncertain status or missing report remains missing. Units: boolean. |
| primary_defender_age | qualified_prior_matchup_defender_forecast_and_prior_bio | Regular-season expected defender from prior400-day player/opponent matchup possessions, min20 possessions, share>=40%, lead>=15%; latest earlier team membership must match opponent. Enabled only if preceding regular-season assignment backtest n>=100, accuracy>=50% and beats minutes baseline by>=5 points. Rating and bio use completed previous season; identity is a forecast, not a confirmed assignment. Unknown scheduled game type and postseason remain missing. Units: official on-court defensive rating; lagged years; defender-minus-player inches. |
| primary_defender_rating | qualified_prior_matchup_defender_forecast_and_prior_bio | Regular-season expected defender from prior400-day player/opponent matchup possessions, min20 possessions, share>=40%, lead>=15%; latest earlier team membership must match opponent. Enabled only if preceding regular-season assignment backtest n>=100, accuracy>=50% and beats minutes baseline by>=5 points. Rating and bio use completed previous season; identity is a forecast, not a confirmed assignment. Unknown scheduled game type and postseason remain missing. Units: official on-court defensive rating; lagged years; defender-minus-player inches. |
| primary_defender_score01 | qualified_defender_forecast_and_observed_efficiency_percentile | Fraction of prior-season measured defenders with worse scoring allowed/100 partial possessions than expected defender prior20 observed-game rate. Requires100 partial possessions per defender and>=30 comparison defenders. Descriptive efficiency percentile, not calibrated matchup probability. Units: fraction. |
| pts_vs_bottom10_defenses | dated_team_defensive_ranks | Player prior-game PTS split by the opponent defensive-rating rank available BEFORE each past game. Empty split remains missing. Units: points/game. |
| pts_vs_top10_defenses | dated_team_defensive_ranks | Player prior-game PTS split by the opponent defensive-rating rank available BEFORE each past game. Empty split remains missing. Units: points/game. |
| rebound_contested_pct | completed_prior_season_rebounding_tracking | Contested rebounds / rebounds from actual NBA Rebounding tracking, not contested shots. Units: fraction. |
| rim_shot_quality_matchup | player_and_opponent_prior_zone_measurements | Player zone FG fraction minus opponent allowed fraction, both completed prior season; explicit interaction useful to models without automatic interactions. Units: fraction difference. |
| series_game_num | dated_postseason_schedule_and_prior_results | Game type from NBA game ID, series state from earlier series games only. Regular-season values are known0. Postseason game logs must be present; no current/future outcomes. Units: boolean; game number; wins. |
| shot_quality_vs_expected | completed_prior_season_player_zones_and_league_zone_counts | Observed player eFG minus league expected eFG for the same five-zone shot mix. Expectations use completed-prior-season league makes/attempts, not a TS*usage proxy or an unvalidated optical shot-quality claim. Units: eFG fraction above zone expectation. |
| spotup_matchup_advantage | player_and_opponent_prior_playtype_measurements | Player play-type PPP minus opponent defensive PPP, using completed prior season. Units: points/possession difference. |
| team_conf_rank | dated_conference_team_records | As-of conference win-fraction rank (not official tie-break seeding); games behind conference10th record from wins/losses differential, floored0. Units: win-fraction rank with tied teams sharing rank; games. |
| team_key_players_out | official_timestamped_injury_reports | Confirmed Out only, before prediction cutoff, for the scheduled game. Key role=prior mean minutes>=20; impact=sum prior minutes of absent players/240, capped1. Unknown report is missing. Units: players; impact=fraction of240 team minutes. |
| team_pts_fb | completed_prior_season_team_misc | Actual NBA Misc scoring components from completed prior season, not latest overwritten team snapshots. Units: points/game. |
| team_pts_off_tov | completed_prior_season_team_misc | Actual NBA Misc scoring components from completed prior season, not latest overwritten team snapshots. Units: points/game. |
| team_series_wins_in | dated_postseason_schedule_and_prior_results | Game type from NBA game ID, series state from earlier series games only. Regular-season values are known0. Postseason game logs must be present; no current/future outcomes. Units: boolean; game number; wins. |
| top_lineup_minutes_pct | completed_prior_season_five_player_lineups | For lineups containing this player: largest five-player unit minutes / all such minutes; number of observed units. Actual lineups, not a minutes ramp. Units: fraction; lineups. |
| transition_matchup_advantage | player_and_opponent_prior_playtype_measurements | Player play-type PPP minus opponent defensive PPP, using completed prior season. Units: points/possession difference. |
| typical_substitution_minute | dated_play_by_play_substitutions | Mean first substitution-out minute across prior appearances where an explicit outgoing identity was resolved. Units: elapsed game minutes. |
| usage_rate_with_star_out | injury_report_and_prior_team_boxscores | Player prior appearances when the currently absent highest-minute teammate did not play; requires at least3 matching past appearances. No fixed10% multiplier. Units: possessions/48min. |
| vs_team_last_season_avg | completed_previous_season_player_game_logs | Player mean against this opponent in the completed previous season, not current-season head-to-head alias. Units: prop units/game. |

## Duplicate or covered

| Feature | Reason |
|---|---|
| assist_percentage | Official assist rate already represented by ast_pct_official; legacy AST/FGA is mislabeled. |
| bad_shot_frequency | Old formula is1-FG%; no independent bad-shot label. |
| career_vs_defender | Old implementation is team head-to-head, already represented by head_to_head_avg. |
| clutch_performance_score | Old formula duplicates consistency_score; actual clutch stats are retained. |
| current_line | Canonical decision-time classifier input is line, not a second current_line column. |
| dreb_rate | Prior official rebound opportunity rate already covered by dreb_pct_official. |
| expected_game_script | Undefined projection of already measured margin-frequency components, not a separate observation. |
| five_man_unit_net_rating | Same intended actual unit rating as net_rating_with_starters; source recovery belongs to that canonical field. |
| foul_drawing_ability | Old formula duplicates FTA volume/rate. Actual PFD and FTA rates are retained. |
| historical_game_script_avg | Legacy signed blowout/close frequency is covered by actual margin-frequency components. |
| implied_under_prob | Complement of a normalized two-way implied_over_prob; preserve raw under odds for pricing. |
| is_rookie | Threshold alias of completed seasons; no distinct measurement. |
| is_veteran | Threshold alias of completed seasons; no distinct measurement. |
| line_movement_pct | Deterministic normalization of opening/current line and raw line movement; no new source. |
| lineup_stability_score | Same intended rotation-continuity measurement as lineup_continuity. |
| minutes_fatigue_score | Measured minutes_last3/5/7, rest and schedule density already expose its components; old score was constant near1. |
| must_win_situation | Speculative motivation label from rank and games back, already covered by those components. |
| net_rating | Actual player net rating already represented by net_rating_player; legacy value duplicates plus_minus_avg. |
| oreb_rate | Prior official rebound opportunity rate already covered by oreb_pct_official. |
| playoff_implications | Old threshold of calendar phase, rank and games back; retain its measured components. |
| playoff_seeding_impact | Handmade score of rank and calendar phase, already covered by those inputs. |
| position_vs_position_dvp | Exact legacy alias of dvp_pts_delta. |
| primary_teammate_out | Threshold alias of team_key_players_out>=1. |
| rebound_percentage | Official rebound share already represented by reb_pct_official; legacy REB/100 is mislabeled. |
| rebound_positioning_score | Old expression is the same rebound rate divided by itself; real rebounding tracking retained. |
| recent_vs_defender | Old implementation is the same bounded team head-to-head history, not a defender matchup. |
| secondary_teammate_out | Threshold alias of team_key_players_out>=2. |
| shot_clock_management | Undefined aggregate score; retain actual early/late shot frequencies. |
| shot_selection_rating | Old formula duplicates recent FG%; no independent shot-selection measurement. |
| tanking_indicator | Speculative motivation label from rank and win percentage, already covered by measured components. |
| teammate_chemistry_score | Arbitrary combination of continuity and unit rating, not an independent observed statistic. |
| total_reb_rate | Prior official rebound opportunity rate already covered by reb_pct_official. |
| years_experience | Completed seasons already represented by seasons_in_league; draft-year proxy was incorrect. |

## Source-blocked

| Feature | Reason |
|---|---|
| avg_prediction_error_player | Requires actual out-of-sample prediction residuals, not a multiple of volatility. |
| avg_shot_clock_time | Bucket frequencies do not identify actual average shot-clock time; requires shot-level times. |
| bookmaker_count | Requires real cross-book observations at decision time. |
| calibration_score_player | Requires actual out-of-sample probabilities and later outcomes, not a multiple of volatility. |
| defender_switching_frequency | NBA switchesOn is present but zero in all92848 collected matchup rows across493 games; an unpopulated field does not establish real switching counts. Requires a populated tracking source. |
| implied_game_total | Requires actual pregame bookmaker game total history; pace*1.1 is not implied market total. |
| implied_over_prob | Requires actual contemporaneous prices, not synthetic lines. |
| line_movement | Requires timestamped opening and decision-time prop lines. |
| line_velocity | Requires timestamped prop-line changes. |
| market_consensus_std | Requires contemporaneous multi-book prop quotes. |
| model_accuracy_player | Requires logged, out-of-sample pregame predictions joined to later outcomes; no such historical ledger is present. |
| opening_line | Requires decision-time historical prop odds snapshots. |
| sharp_action_score | Requires an objective source/definition of sharp action; not ticket/money imbalance relabeled. |
| stale_line_flag | Requires timestamped quote freshness and cross-book comparison. |

## Verification

691 tests passed. The replay covers 11,548 player-games with stable file inputs and unchanged serving models. No active column is entirely missing or constant across the cohort; no observed fraction violates [0,1]. This does not establish broad coverage or predictive gain. See `logs/source_blocked/report.json` and `logs/full_feature_audit/report.json`. Production player-prop models have not been retrained or promoted.
