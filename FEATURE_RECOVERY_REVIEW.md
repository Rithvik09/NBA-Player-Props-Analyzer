# Initial review of the 162 inactive features

Historical planning document. Superseded by the implemented v3 dispositions in
`FEATURE_ACTIVATION.md` and the measured full-feature audit.

No individual feature is proven essential. These are engineering priorities based on source validity, distinct information and redundancy, not results of an ablation experiment.

The next audited training run does not need to wait for all of these. Keep all currently inactive until sourced, replayed, tested and versioned. Existing schema slots may stay zero for compatibility; retiring a column does not mean deleting raw data.

| Recommendation | Columns |
|---|---:|
| Core basketball-context recovery | 47 |
| Optional development experiment | 59 |
| Retire this column | 40 |
| Monitoring / bet decision layer | 6 |
| Recover and test first | 2 |
| Postseason scope only | 8 |

## Core context and first recovery candidates

| Feature | Why / required replacement |
|---|---|
| `team_key_players_out` | A dated pregame absence can change available minutes and opportunities. Recover timestamped reports and role-weighted impact; an undated current report cannot label historical games. |
| `opp_key_players_out` | A dated pregame absence can change available minutes and opportunities. Recover timestamped reports and role-weighted impact; an undated current report cannot label historical games. |
| `opp_injury_impact` | A dated pregame absence can change available minutes and opportunities. Recover timestamped reports and role-weighted impact; an undated current report cannot label historical games. |
| `dvp_gp` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_pts_delta` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_reb_delta` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_ast_delta` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_fg3m_delta` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_stl_delta` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_blk_delta` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_tov_delta` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `primary_teammate_out` | Retain the role-specific injury concept as a core recovery target. Define whose absence this flag represents using pregame roster/rotation information, and test against weighted absence counts rather than guessing a star ordering. |
| `secondary_teammate_out` | Retain the role-specific injury concept as a core recovery target. Define whose absence this flag represents using pregame roster/rotation information, and test against weighted absence counts rather than guessing a star ordering. |
| `new_teammate_games` | Potential rotation/role context beyond individual histories. Require actual dated player combinations or pregame rotation expectations, with sample-size shrinkage. |
| `shot_quality_vs_expected` | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `avg_shot_clock_time` | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `late_clock_shot_frequency` | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `early_clock_shot_frequency` | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `usage_rate_with_star_out` | Opportunity redistribution is a distinct role signal. Estimate usage conditional on specific missing teammates from earlier minutes/stints, with shrinkage; never FGA*1.1. |
| `minutes_with_starting_lineup_pct` | Potential rotation/role context beyond individual histories. Require actual dated player combinations or pregame rotation expectations, with sample-size shrinkage. |
| `blowout_probability` | Minutes downside is a useful concept to test. Use a validated pregame margin distribution or timestamped spread-based estimate, not player PLUS_MINUS frequency. |
| `lineup_continuity` | Potential rotation/role context beyond individual histories. Require actual dated player combinations or pregame rotation expectations, with sample-size shrinkage. |
| `rebound_contested_pct` | Potentially distinct rebounding opportunity/competition signal for rebound props. Recover actual contested rebound share/chances from tracking; contested shots are unrelated. |
| `dvp_pts_delta_last5` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_pts_delta_last10` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_reb_delta_last5` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_ast_delta_last5` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_fg3m_delta_last5` | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `games_since_return` | Recent confirmed injury return and a reported minutes limit can alter minutes distribution. Calendar gaps alone do not establish an injury or return. Require known-at-the-time status. |
| `missed_games_before_return` | Recent confirmed injury return and a reported minutes limit can alter minutes distribution. Calendar gaps alone do not establish an injury or return. Require known-at-the-time status. |
| `implied_game_total` | Actual pregame bookmaker game total can supply market scoring-environment context. The old pace-derived total was not an implied market total. Requires historical decision-time snapshots. |
| `opp_rim_fg_pct_allowed` | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_paint_fg_pct_allowed` | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_midrange_fg_pct_allowed` | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_corner3_fg_pct_allowed` | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_above_break3_fg_pct_allowed` | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `rim_shot_quality_matchup` | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `midrange_shot_quality_matchup` | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `corner3_shot_quality_matchup` | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `above_break3_shot_quality_matchup` | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `opp_pnr_ppp_allowed` | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_iso_ppp_allowed` | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_spotup_ppp_allowed` | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_transition_ppp_allowed` | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_postup_ppp_allowed` | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `pnr_matchup_advantage` | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `iso_matchup_advantage` | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `spotup_matchup_advantage` | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `transition_matchup_advantage` | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |

The injury/return columns should be designed around player identities, projected minutes, limitations and opportunity redistribution. Weighted counts alone may be weaker than this redesign. Expected minutes and its uncertainty are not a separate existing column among these 162.

## Every inactive feature

| Feature | Decision | Reason |
|---|---|---|
| `team_key_players_out` | Core basketball-context recovery | A dated pregame absence can change available minutes and opportunities. Recover timestamped reports and role-weighted impact; an undated current report cannot label historical games. |
| `opp_key_players_out` | Core basketball-context recovery | A dated pregame absence can change available minutes and opportunities. Recover timestamped reports and role-weighted impact; an undated current report cannot label historical games. |
| `opp_injury_impact` | Core basketball-context recovery | A dated pregame absence can change available minutes and opportunities. Recover timestamped reports and role-weighted impact; an undated current report cannot label historical games. |
| `team_pts_fb` | Optional development experiment | Team style may add context beyond pace and ratings. Require dated prior-game style statistics and verify incremental gain; not a prerequisite for the first audited model. |
| `opp_pts_fb_allowed` | Optional development experiment | Team style may add context beyond pace and ratings. Require dated prior-game style statistics and verify incremental gain; not a prerequisite for the first audited model. |
| `team_pts_off_tov` | Optional development experiment | Team style may add context beyond pace and ratings. Require dated prior-game style statistics and verify incremental gain; not a prerequisite for the first audited model. |
| `opp_pts_off_tov_allowed` | Optional development experiment | Team style may add context beyond pace and ratings. Require dated prior-game style statistics and verify incremental gain; not a prerequisite for the first audited model. |
| `opp_pts_paint` | Optional development experiment | Team style may add context beyond pace and ratings. Require dated prior-game style statistics and verify incremental gain; not a prerequisite for the first audited model. |
| `lg_pts_fb` | Optional development experiment | Prior-only league environment could normalize seasons, but retained team pace, ratings and player histories already describe much of that environment. Use rolling league aggregates, not final same-season averages. |
| `lg_opp_pts_fb` | Optional development experiment | Prior-only league environment could normalize seasons, but retained team pace, ratings and player histories already describe much of that environment. Use rolling league aggregates, not final same-season averages. |
| `lg_pts_off_tov` | Optional development experiment | Prior-only league environment could normalize seasons, but retained team pace, ratings and player histories already describe much of that environment. Use rolling league aggregates, not final same-season averages. |
| `lg_opp_pts_off_tov` | Optional development experiment | Prior-only league environment could normalize seasons, but retained team pace, ratings and player histories already describe much of that environment. Use rolling league aggregates, not final same-season averages. |
| `lg_fga` | Optional development experiment | Prior-only league environment could normalize seasons, but retained team pace, ratings and player histories already describe much of that environment. Use rolling league aggregates, not final same-season averages. |
| `lg_fg_pct` | Optional development experiment | Prior-only league environment could normalize seasons, but retained team pace, ratings and player histories already describe much of that environment. Use rolling league aggregates, not final same-season averages. |
| `lg_fg3a` | Optional development experiment | Prior-only league environment could normalize seasons, but retained team pace, ratings and player histories already describe much of that environment. Use rolling league aggregates, not final same-season averages. |
| `lg_tov` | Optional development experiment | Prior-only league environment could normalize seasons, but retained team pace, ratings and player histories already describe much of that environment. Use rolling league aggregates, not final same-season averages. |
| `lg_stl` | Optional development experiment | Prior-only league environment could normalize seasons, but retained team pace, ratings and player histories already describe much of that environment. Use rolling league aggregates, not final same-season averages. |
| `dvp_gp` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_pts_delta` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_reb_delta` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_ast_delta` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_fg3m_delta` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_stl_delta` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_blk_delta` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_tov_delta` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `primary_defender_score01` | Optional development experiment | Only meaningful with a credible pregame matchup assignment and its uncertainty. An elite player on the roster is not evidence that he will guard this player. |
| `assist_percentage` | Retire this column | Legacy definition is not the named statistic. Retained official prior-season rates or measured boxscore components already cover the intended concept; avoid a misleading duplicate. |
| `rebound_percentage` | Retire this column | Legacy definition is not the named statistic. Retained official prior-season rates or measured boxscore components already cover the intended concept; avoid a misleading duplicate. |
| `net_rating` | Retire this column | Legacy definition is not the named statistic. Retained official prior-season rates or measured boxscore components already cover the intended concept; avoid a misleading duplicate. |
| `top_lineup_minutes_pct` | Retire this column | The old minutes-based ramp is not observed lineup exposure. Retained actual minutes and a tested starting-lineup exposure measure are sufficient first choices. |
| `vs_team_last_season_avg` | Retire this column | Sparse prior-season player-opponent meetings add little distinct coverage beyond retained head-to-head and overall histories; role changes make old meetings less comparable. |
| `model_accuracy_player` | Monitoring / bet decision layer | Useful for monitoring, calibration or abstaining. Must come from logged predictions made before their outcomes. Not required as an input to the points-mean model. |
| `avg_prediction_error_player` | Monitoring / bet decision layer | Useful for monitoring, calibration or abstaining. Must come from logged predictions made before their outcomes. Not required as an input to the points-mean model. |
| `calibration_score_player` | Monitoring / bet decision layer | Useful for monitoring, calibration or abstaining. Must come from logged predictions made before their outcomes. Not required as an input to the points-mean model. |
| `blowout_game_pct` | Optional development experiment | Can recover genuine past team game margins from dated scores; player PLUS_MINUS is not game margin. Historical frequencies are descriptive and should not be labeled predicted probabilities. |
| `close_game_pct` | Optional development experiment | Can recover genuine past team game margins from dated scores; player PLUS_MINUS is not game margin. Historical frequencies are descriptive and should not be labeled predicted probabilities. |
| `opp_def_rating_home_away_split` | Optional development experiment | Already have actual opponent rest and recent pace/ratings. A rest-conditioned or venue-conditioned rate may add signal, but needs enough prior games and shrinkage. |
| `playoff_implications` | Optional development experiment | Actual pregame standings may help late-season context. Test beyond retained calendar phase and recent team form; do not infer effort or minutes restrictions from standings alone. |
| `national_tv_game` | Retire this column | Weak indirect context with no demonstrated incremental value over player, team, calendar and schedule inputs. Do not block training to collect it. |
| `primary_teammate_out` | Core basketball-context recovery | Retain the role-specific injury concept as a core recovery target. Define whose absence this flag represents using pregame roster/rotation information, and test against weighted absence counts rather than guessing a star ordering. |
| `secondary_teammate_out` | Core basketball-context recovery | Retain the role-specific injury concept as a core recovery target. Define whose absence this flag represents using pregame roster/rotation information, and test against weighted absence counts rather than guessing a star ordering. |
| `new_teammate_games` | Core basketball-context recovery | Potential rotation/role context beyond individual histories. Require actual dated player combinations or pregame rotation expectations, with sample-size shrinkage. |
| `lineup_stability_score` | Retire this column | Duplicates the lineup_continuity concept. Recover one well-defined stability measurement, not multiple aliases. |
| `bench_strength` | Optional development experiment | Potential rotation/role context beyond individual histories. Require actual dated player combinations or pregame rotation expectations, with sample-size shrinkage. |
| `pts_vs_top10_defenses` | Retire this column | Hard top10/bottom10 subsets discard games and create small player samples. Prefer continuous opponent ratings plus current player opportunity features. |
| `pts_vs_bottom10_defenses` | Retire this column | Hard top10/bottom10 subsets discard games and create small player samples. Prefer continuous opponent ratings plus current player opportunity features. |
| `def_fg_pct_allowed` | Optional development experiment | Actual opponent shooting allowed may be distinct from opponent offensive FG%. Recover opponent-of-opponent makes/attempts by date, then test against retained defensive rating. |
| `def_rating_individual` | Retire this column | Legacy definition is not the named statistic. Retained official prior-season rates or measured boxscore components already cover the intended concept; avoid a misleading duplicate. |
| `playoff_seeding_impact` | Retire this column | Handcrafted phase/rank score duplicates its components without establishing how much seeding affects the player. Prefer testing raw pregame standings. |
| `tanking_indicator` | Retire this column | Speculative labels from standings are not observed motivation or minutes plans. Use actual pregame availability/rotation news and measured standings when justified. |
| `must_win_situation` | Retire this column | Speculative labels from standings are not observed motivation or minutes plans. Use actual pregame availability/rotation news and measured standings when justified. |
| `games_back_from_playoff` | Optional development experiment | Actual pregame standings may help late-season context. Test beyond retained calendar phase and recent team form; do not infer effort or minutes restrictions from standings alone. |
| `fourth_quarter_usage_rate` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `garbage_time_minutes_pct` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `typical_substitution_minute` | Retire this column | Average minutes plus3 does not identify a substitution pattern. A real projected-minutes distribution is the higher-priority replacement. |
| `crunch_time_usage` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `career_vs_defender` | Retire this column | Old values are team-opponent deltas, not defender matchups. Retained player-opponent history already covers that proxy; do not keep the misleading names. |
| `recent_vs_defender` | Retire this column | Old values are team-opponent deltas, not defender matchups. Retained player-opponent history already covers that proxy; do not keep the misleading names. |
| `player_vs_arena` | Retire this column | Player-by-arena samples are sparse and confounded with opponent and venue. Retained home/away splits and opponent context are the better first choice. |
| `avg_shot_distance` | Retire this column | A zone-bucket weighted estimate mostly repackages retained zone shares. An exact measured distance is an optional future experiment, not required recovery. |
| `shot_quality_vs_expected` | Core basketball-context recovery | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `avg_shot_clock_time` | Core basketball-context recovery | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `late_clock_shot_frequency` | Core basketball-context recovery | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `early_clock_shot_frequency` | Core basketball-context recovery | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `post_touches_per_game` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `front_court_touches_per_game` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `net_rating_with_starters` | Optional development experiment | Potential rotation/role context beyond individual histories. Require actual dated player combinations or pregame rotation expectations, with sample-size shrinkage. |
| `usage_rate_with_star_out` | Core basketball-context recovery | Opportunity redistribution is a distinct role signal. Estimate usage conditional on specific missing teammates from earlier minutes/stints, with shrinkage; never FGA*1.1. |
| `minutes_with_starting_lineup_pct` | Core basketball-context recovery | Potential rotation/role context beyond individual histories. Require actual dated player combinations or pregame rotation expectations, with sample-size shrinkage. |
| `five_man_unit_net_rating` | Retire this column | The old implementation duplicates net_rating_with_starters. Test one actual lineup-efficiency feature; do not keep two names for the same team snapshot. |
| `lineups_played_count` | Retire this column | Raw lineup count strongly reflects sample size. Stable role/continuity measures are a better recovery target. |
| `position_vs_position_dvp` | Retire this column | Legacy definition duplicates dvp_pts_delta. One well-defined position matchup metric is enough for its first experiment. |
| `defender_switching_frequency` | Retire this column | A DVP variance or lineup-change proxy does not measure switching. Do not prioritize unless actual scheme/assignment events become available. |
| `historical_game_script_avg` | Retire this column | Old value is an arbitrary transform, not a validated pregame probability. Do not recover the formula merely to repopulate the column. |
| `player_age` | Optional development experiment | Cheap structural context if sourced as of prediction time. Test incremental benefit beyond retained prior-season experience/profile and recent role; avoid current bio data in historical rows. |
| `years_experience` | Optional development experiment | Cheap structural context if sourced as of prediction time. Test incremental benefit beyond retained prior-season experience/profile and recent role; avoid current bio data in historical rows. |
| `is_rookie` | Retire this column | Threshold aliases of a reliable experience measure. Trees can derive these splits; retain only if a development-period ablation demonstrates extra value. |
| `is_veteran` | Retire this column | Threshold aliases of a reliable experience measure. Trees can derive these splits; retain only if a development-period ablation demonstrates extra value. |
| `expected_game_script` | Retire this column | Old value is an arbitrary transform, not a validated pregame probability. Do not recover the formula merely to repopulate the column. |
| `blowout_probability` | Recover and test first | Minutes downside is a useful concept to test. Use a validated pregame margin distribution or timestamped spread-based estimate, not player PLUS_MINUS frequency. |
| `close_game_probability` | Optional development experiment | Could help minutes-tail prediction if produced by the same validated pregame margin model. Avoid separately inventing this number or treating it as independent of blowout risk. |
| `clutch_performance_score` | Retire this column | Undefined quality score or a renamed simple statistic. Existing measured consistency, shooting, FTA, rebounding and opponent inputs are preferable to the old proxy. |
| `shot_selection_rating` | Retire this column | Undefined quality score or a renamed simple statistic. Existing measured consistency, shooting, FTA, rebounding and opponent inputs are preferable to the old proxy. |
| `bad_shot_frequency` | Retire this column | Undefined quality score or a renamed simple statistic. Existing measured consistency, shooting, FTA, rebounding and opponent inputs are preferable to the old proxy. |
| `shot_clock_management` | Retire this column | Undefined quality score or a renamed simple statistic. Existing measured consistency, shooting, FTA, rebounding and opponent inputs are preferable to the old proxy. |
| `teammate_chemistry_score` | Retire this column | Arbitrary weighted sum of continuity and lineup rating. If those measured inputs help, let the model combine them. |
| `lineup_continuity` | Core basketball-context recovery | Potential rotation/role context beyond individual histories. Require actual dated player combinations or pregame rotation expectations, with sample-size shrinkage. |
| `primary_defender_rating` | Optional development experiment | Only meaningful with a credible pregame matchup assignment and its uncertainty. An elite player on the roster is not evidence that he will guard this player. |
| `primary_defender_age` | Optional development experiment | Only meaningful with a credible pregame matchup assignment and its uncertainty. An elite player on the roster is not evidence that he will guard this player. |
| `defender_size_mismatch` | Optional development experiment | Only meaningful with a credible pregame matchup assignment and its uncertainty. An elite player on the roster is not evidence that he will guard this player. |
| `defender_recent_form` | Optional development experiment | Only meaningful with a credible pregame matchup assignment and its uncertainty. An elite player on the roster is not evidence that he will guard this player. |
| `fouls_drawn_per_game` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `and_one_frequency` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `foul_drawing_ability` | Retire this column | Undefined quality score or a renamed simple statistic. Existing measured consistency, shooting, FTA, rebounding and opponent inputs are preferable to the old proxy. |
| `oreb_rate` | Optional development experiment | Accurate current as-of rebounding opportunity rates may improve on lagged official percentages. Define on-court opportunity denominators carefully; test for rebound props before expanding. |
| `dreb_rate` | Optional development experiment | Accurate current as-of rebounding opportunity rates may improve on lagged official percentages. Define on-court opportunity denominators carefully; test for rebound props before expanding. |
| `total_reb_rate` | Optional development experiment | Accurate current as-of rebounding opportunity rates may improve on lagged official percentages. Define on-court opportunity denominators carefully; test for rebound props before expanding. |
| `rebound_contested_pct` | Core basketball-context recovery | Potentially distinct rebounding opportunity/competition signal for rebound props. Recover actual contested rebound share/chances from tracking; contested shots are unrelated. |
| `rebound_positioning_score` | Retire this column | Undefined quality score or a renamed simple statistic. Existing measured consistency, shooting, FTA, rebounding and opponent inputs are preferable to the old proxy. |
| `paint_touch_to_points` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `performance_when_leading` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `performance_when_trailing` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `performance_when_tied` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `performance_in_overtime` | Optional development experiment | Could describe a distinct role or opportunity if measured from dated events/tracking. Cost and incremental predictive value must be tested against retained minutes, volume and tracking features. |
| `performance_by_score_differential` | Retire this column | Undefined quality score or a renamed simple statistic. Existing measured consistency, shooting, FTA, rebounding and opponent inputs are preferable to the old proxy. |
| `minutes_fatigue_score` | Retire this column | Arbitrary score duplicates actual minutes load, schedule density and rest. Keep the measured components; no separate fatigue score is needed. |
| `dvp_pts_delta_last5` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_pts_delta_last10` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_reb_delta_last5` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_ast_delta_last5` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `dvp_fg3m_delta_last5` | Core basketball-context recovery | Potential prop-specific matchup context, but position labels and small matchup samples can be unstable. Compare against retained opponent ratings and player histories, with shrinkage and prior-only windows. |
| `games_since_return` | Core basketball-context recovery | Recent confirmed injury return and a reported minutes limit can alter minutes distribution. Calendar gaps alone do not establish an injury or return. Require known-at-the-time status. |
| `missed_games_before_return` | Core basketball-context recovery | Recent confirmed injury return and a reported minutes limit can alter minutes distribution. Calendar gaps alone do not establish an injury or return. Require known-at-the-time status. |
| `primary_defender_active` | Optional development experiment | Only meaningful with a credible pregame matchup assignment and its uncertainty. An elite player on the roster is not evidence that he will guard this player. |
| `opp_lineup_changes_last5` | Retire this column | A DVP variance or lineup-change proxy does not measure switching. Do not prioritize unless actual scheme/assignment events become available. |
| `implied_game_total` | Recover and test first | Actual pregame bookmaker game total can supply market scoring-environment context. The old pace-derived total was not an implied market total. Requires historical decision-time snapshots. |
| `player_height_inches` | Optional development experiment | Cheap structural context if sourced as of prediction time. Test incremental benefit beyond retained prior-season experience/profile and recent role; avoid current bio data in historical rows. |
| `player_weight` | Optional development experiment | Cheap structural context if sourced as of prediction time. Test incremental benefit beyond retained prior-season experience/profile and recent role; avoid current bio data in historical rows. |
| `opp_rim_fg_pct_allowed` | Core basketball-context recovery | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_paint_fg_pct_allowed` | Core basketball-context recovery | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_midrange_fg_pct_allowed` | Core basketball-context recovery | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_corner3_fg_pct_allowed` | Core basketball-context recovery | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_above_break3_fg_pct_allowed` | Core basketball-context recovery | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `rim_shot_quality_matchup` | Core basketball-context recovery | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `midrange_shot_quality_matchup` | Core basketball-context recovery | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `corner3_shot_quality_matchup` | Core basketball-context recovery | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `above_break3_shot_quality_matchup` | Core basketball-context recovery | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `opp_pnr_ppp_allowed` | Core basketball-context recovery | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_iso_ppp_allowed` | Core basketball-context recovery | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_spotup_ppp_allowed` | Core basketball-context recovery | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_transition_ppp_allowed` | Core basketball-context recovery | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `opp_postup_ppp_allowed` | Core basketball-context recovery | Potentially distinct style matchup, especially for scoring and threes. Require opponent allowed efficiency by zone/play type from dated prior observations and adequate samples. |
| `pnr_matchup_advantage` | Core basketball-context recovery | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `iso_matchup_advantage` | Core basketball-context recovery | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `spotup_matchup_advantage` | Core basketball-context recovery | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `transition_matchup_advantage` | Core basketball-context recovery | Core player-style versus opponent-defense interaction. Recover dated raw components and evaluate the explicit matchup term, especially with a linear probability model; being derived does not establish redundancy. |
| `opp_b2b_def_rating` | Optional development experiment | Already have actual opponent rest and recent pace/ratings. A rest-conditioned or venue-conditioned rate may add signal, but needs enough prior games and shrinkage. |
| `opp_b2b_pace` | Optional development experiment | Already have actual opponent rest and recent pace/ratings. A rest-conditioned or venue-conditioned rate may add signal, but needs enough prior games and shrinkage. |
| `opp_b2b_pts_allowed` | Optional development experiment | Already have actual opponent rest and recent pace/ratings. A rest-conditioned or venue-conditioned rate may add signal, but needs enough prior games and shrinkage. |
| `opp_rested_def_rating` | Optional development experiment | Already have actual opponent rest and recent pace/ratings. A rest-conditioned or venue-conditioned rate may add signal, but needs enough prior games and shrinkage. |
| `opp_rested_pace` | Optional development experiment | Already have actual opponent rest and recent pace/ratings. A rest-conditioned or venue-conditioned rate may add signal, but needs enough prior games and shrinkage. |
| `team_conf_rank` | Optional development experiment | Actual pregame standings may help late-season context. Test beyond retained calendar phase and recent team form; do not infer effort or minutes restrictions from standings alone. |
| `opp_conf_rank` | Optional development experiment | Actual pregame standings may help late-season context. Test beyond retained calendar phase and recent team form; do not infer effort or minutes restrictions from standings alone. |
| `opening_line` | Optional development experiment | Relevant to line-aware over/under probability or pricing, not automatically to the stat-mean model. Requires timestamped historical lines/prices matching the decision time; synthetic lines cannot validate market value. |
| `current_line` | Retire this column | Duplicates the classifier extra feature line. Keep one canonical decision-time line; this source is still necessary for actual betting evaluation, not as a second numeric column. |
| `line_movement` | Optional development experiment | Relevant to line-aware over/under probability or pricing, not automatically to the stat-mean model. Requires timestamped historical lines/prices matching the decision time; synthetic lines cannot validate market value. |
| `line_movement_pct` | Retire this column | Derived from opening/current line and already represented by raw movement and line level. Start with the measured components before adding normalization experiments. |
| `implied_over_prob` | Optional development experiment | Relevant to line-aware over/under probability or pricing, not automatically to the stat-mean model. Requires timestamped historical lines/prices matching the decision time; synthetic lines cannot validate market value. |
| `implied_under_prob` | Retire this column | Complementary to a properly normalized two-way over probability. Retain both prices in raw odds data, but avoid a redundant model column. |
| `market_consensus_std` | Optional development experiment | Relevant to line-aware over/under probability or pricing, not automatically to the stat-mean model. Requires timestamped historical lines/prices matching the decision time; synthetic lines cannot validate market value. |
| `sharp_action_score` | Retire this column | No objectively defined label or calibrated computation currently exists. A fabricated smart-money score is not a recoverable measurement. |
| `line_velocity` | Monitoring / bet decision layer | Primarily a freshness/liquidity/execution check. Keep timestamped raw observations for bet filtering and price selection; not essential for predicting player points. |
| `stale_line_flag` | Monitoring / bet decision layer | Primarily a freshness/liquidity/execution check. Keep timestamped raw observations for bet filtering and price selection; not essential for predicting player points. |
| `bookmaker_count` | Monitoring / bet decision layer | Primarily a freshness/liquidity/execution check. Keep timestamped raw observations for bet filtering and price selection; not essential for predicting player points. |
| `is_playoff` | Postseason scope only | Not needed for the current regular-season contract. A separate postseason dataset/evaluation is required before enabling this feature. |
| `is_play_in` | Postseason scope only | Not needed for the current regular-season contract. A separate postseason dataset/evaluation is required before enabling this feature. |
| `series_game_num` | Postseason scope only | Not needed for the current regular-season contract. A separate postseason dataset/evaluation is required before enabling this feature. |
| `team_series_wins_in` | Postseason scope only | Not needed for the current regular-season contract. A separate postseason dataset/evaluation is required before enabling this feature. |
| `opp_series_wins_in` | Postseason scope only | Not needed for the current regular-season contract. A separate postseason dataset/evaluation is required before enabling this feature. |
| `is_elimination_game` | Postseason scope only | Not needed for the current regular-season contract. A separate postseason dataset/evaluation is required before enabling this feature. |
| `playoff_home` | Postseason scope only | Not needed for the current regular-season contract. A separate postseason dataset/evaluation is required before enabling this feature. |
| `playoff_away` | Postseason scope only | Not needed for the current regular-season contract. A separate postseason dataset/evaluation is required before enabling this feature. |

## How to establish usefulness

Freeze feature groups before fitting. Use training/development chronology for inclusion decisions, keep calibration later and disjoint, and reserve a genuinely unused evaluation population. The previously inspected 2025–26 cohort is not a fresh blind test. Compare each recovered group against the 254-feature contract using paired game-level evaluation; evaluate prop-specific benefits rather than assuming points results transfer to rebounds or assists. Report calibration, log loss/Brier and regression error, with AUC as an additional metric. Establish betting value only on timestamped real lines/prices after costs.

An individual-column ablation inside a group can then identify redundant members. No feature should be called essential just because it sounds relevant, has variance, or scored highly in a model trained with leakage. There is no reliable feature-importance record in the shipped metadata that establishes necessity for these columns.

## Deep-context priorities

Injuries, role changes, shot quality and player-style versus opponent-defense
matchups are intended core capabilities. An unavailable historical input is a
recovery task, not evidence that the capability should be removed. Individual
columns still need testing once properly measured.

The active contract already includes attempts-weighted recent FG/3P/FT rates,
TS/eFG, completed-prior-season zone efficiency/frequency, catch-and-shoot/pull-up
profiles, tracking and play types. Two gaps deserve particular attention:

1. These detailed player profiles are lagged priors. Current as-of-game form,
   attempts and role changes should supplement them, with shrinkage for small
   samples. Do not mistake previous-season coverage for current-form coverage.
2. The opponent-zone collector searches for zone columns in
   `LeagueDashOppPtShot`, whose installed schema contains aggregate shooting
   fields. Its fallback values can masquerade as real opponent zone measurements.
   `LeagueDashTeamShotLocations` exposes the actual zone groups; validate its
   opponent mode and date cutoffs before replacing the collector and storing
   season/as-of observations. This is concrete source-repair work, not a reason
   to abandon matchup shooting features.

Historical injury recovery needs a publication timestamp and the game's tipoff.
Status-to-role features should represent the injured player, expected minutes,
usage/assist/rebound opportunity transfer, and return limitations. A current
ESPN report or an absence inferred from the completed game's boxscore is not
valid historical pregame status. The official NBA injury page currently serves
updated report links, but historical-season URLs tested during this review
redirected to the current season; archive completeness is not yet established.

Primary source references:

- [NBA official injury reports](https://official.nba.com/nba-injury-report-2026-27-season/)
- [NBA opponent shooting by zone](https://www.nba.com/stats/teams/opponent-shooting?DistanceRange=By%20Zone)

No features were re-enabled and no model was retrained during this necessity
review. The review changes recovery priorities; it does not certify that the
missing data have already been reconstructed.
