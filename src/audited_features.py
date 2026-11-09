"""Versioned point-in-time feature contract for new models.

The legacy 416-column order is preserved. Unsupported measurements are explicitly
inactive, rather than fabricated from unrelated statistics. Both historical and
live callers use this builder with the same dated input contract. Legacy model
artifacts must continue to use their original builder until retrained.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.ml_features import NUMERIC_FEATURE_KEYS

POLICY_VERSION = 'point_in_time_v4'

@dataclass(frozen=True)
class Spec:
    source: str
    units: str
    definition: str
    status: str = 'active'

SPECS: dict[str, Spec] = {}

def register(names, source, units, definition, status='active'):
    for name in names.split():
        if name in SPECS:
            raise ValueError('Duplicate feature specification: '+name)
        SPECS[name] = Spec(source, units, definition, status)

# Each feature is explicitly assigned below. No unreviewed catch-all activation.
register('recent_avg season_avg max_recent min_recent rolling_7day_avg rolling_14day_avg rolling_30day_avg ewm_alpha_0.3 ewm_alpha_0.5', 'player_boxscores', 'prop units', 'Prior results only: recent=5 games, season=20-game bounded history; rolling legacy day names use 7/14/30 games (30 capped at20).')
register('stddev', 'player_boxscores', 'prop units', 'Population standard deviation of prior20 appearances.')
register('games_played games_above_season_avg_last5 consecutive_over_games consecutive_under_games games_above_season_avg_7day games_above_season_avg_14day', 'player_boxscores', 'games', 'Counts in prior bounded history; over/under refers to prior10 mean, never bookmaker lines.')
register('last_3_games_trend last_5_games_trend last_10_games_trend', 'player_boxscores', 'prop units', 'PriorN mean minus prior20 mean.')
register('trend_slope_10games trend_slope_5games', 'player_boxscores', 'prop units/game', 'OLS slope over lastN prior appearances.')
register('hot_hand_indicator', 'player_boxscores', 'boolean', 'All last3 prior results exceed prior20 mean.')
register('volatility_ratio momentum_score recent_variance_spike consistency_score ceiling_game_frequency', 'player_boxscores', 'ratio', 'Volatility/momentum from prior prop results; consistency=max(0,1-sample_std/mean), ceiling share>=1.5*mean.')
register('mins_last5 mins_season avg_minutes_last_3', 'player_boxscores', 'minutes/game', 'Prior5/prior20/prior3 appearance means.')
register('minutes_last_3_games minutes_last_5_games minutes_last_7_games', 'player_boxscores', 'minutes', 'Sum over lastN prior appearances.')
register('minutes_volatility', 'player_boxscores', 'minutes', 'Sample standard deviation of prior20 minutes.')
register('rest_days days_since_last_game games_in_last_7_days recent_away_streak', 'player_boxscores', 'days or games (see definition)', 'rest_days=calendar gap-1; days_since_last_game=gap; games_in_last_7_days counts [date-7,date); away streak counts consecutive @ appearances.')
register('is_home_game_num is_back_to_back', 'schedule', 'boolean; home unknown=-1', 'Known next-game venue; player rest_days==0 indicates consecutive calendar days.')
register('fg_pct_recent fg3_pct_recent ft_pct_recent', 'player_boxscores', 'fraction', 'Last5 aggregate makes / attempts, not mean of percentages; zero attempts ->0.')
register('fga_per_game fg3a_per_game fta_per_game oreb_per_game dreb_per_game fouls_per_game ft_attempts_per_game', 'player_boxscores', 'events/game', 'Arithmetic mean of prior20 boxscore counts.')
register('plus_minus_avg on_court_plus_minus', 'player_boxscores', 'points/game', 'Mean boxscore PLUS_MINUS; not possession-adjusted net rating.')
register('win_rate_last10', 'player_boxscores', 'fraction', 'Wins in player last10 appearances / appearances; distinct from team schedule wins.')
register('usage_rate', 'player_boxscores', 'possessions/48 minutes', '(FGA+.44*FTA+TOV)/MIN*48; not official usage percentage.')
register('true_shooting_pct effective_fg_pct', 'player_boxscores', 'fraction (EFG may exceed1)', 'PTS/[2*(FGA+.44*FTA)] and (FGM+.5*FG3M)/FGA from prior20 totals.')
register('points_per_shot ast_to_tov_ratio reb_rate_per_36 ft_rate fta_rate_player pct_fga_2pt pct_fga_3pt pct_pts_3pt pct_pts_ft', 'player_boxscores', 'ratio except reb_rate_per_36=rebounds/36 minutes', 'Ratios of prior20 aggregate counts; ft_rate=FTA/FGA; shot/point shares use actual totals.')
register('scoring_efficiency_trend usage_trend', 'player_boxscores', 'points/FGA delta or FGA/game delta', 'Last5 points/FGA minus prior20; last5 FGA/game minus prior20.')
register('team_pace opp_pace matchup_pace opp_pace_last5', 'dated_team_boxscores', 'possessions/48 minutes', 'Prior10 team and opponent pace means; matchup mean of both; opponent last5.')
register('team_off_rating team_def_rating opp_off_rating opp_def_rating opp_def_rating_last5 opp_def_rating_last10 opp_def_rating_trend', 'dated_team_boxscores', 'points/100 possessions', 'Prior10 ratings; opponent trend=last5 defensive rating minus last10.')
register('days_rest_opponent', 'dated_team_boxscores', 'days', 'Prediction day minus latest opponent game day minus1; excludes same-day outcomes.')
register('opponent_back_to_back both_teams_rested', 'dated_team_boxscores', 'boolean', 'Opponent rest==0; both team schedules have at least2 days rest.')
register('rest_advantage rest_advantage_abs', 'dated_team_boxscores', 'days', 'Own team schedule rest minus opponent rest, signed/absolute.')
register('team_win_pct opp_win_pct opp_win_rate_last10', 'dated_team_boxscores', 'fraction', 'Season-to-date schedule wins/decided games; last10 for recent opponent rate.')
register('team_current_streak opp_current_streak team_win_streak team_loss_streak team_l10_wins opp_l10_wins', 'dated_team_boxscores', 'games', 'Actual team schedule, including games player missed; signed streak and last10 wins.')
register('opp_fga opp_fg_pct opp_fg3a opp_fg3_pct opp_tov opp_stl opp_blk opp_blocks_per_game_last5 opp_steals_per_game_last5 opp_foul_rate_per48 opp_foul_rate_last5', 'dated_team_player_boxscores', 'events/game; shooting=fraction; foul_rate_per48=events/48min', 'Aggregate all players per team-game before prediction; last10 means or last5 named windows; fouls normalized for overtime.')
register('vs_team_avg matchup_games head_to_head_avg head_to_head_games vs_opp_gp vs_opp_avg_pts vs_opp_fg_pct vs_opp_ts_pct vs_opp_avg_min vs_team_home_away_split vs_team_win_pct', 'player_boxscores', 'prop units or games; FG/TS/wins=fraction; avg_min=minutes', 'Same-opponent subset of prior20 appearances; points aliases are always points, head_to_head_avg is prop-specific; no meetings ->mean/zero as appropriate.')
register('days_into_season season_phase_numeric games_remaining_approx season_phase', 'calendar', 'days / phase / estimated games', 'Fixed October22-April14 calendar convention; remaining is an approximation, not actual schedule games.')
register('arena_altitude arena_capacity home_court_advantage_rating rivalry_game time_zone_change travel_distance coast_to_coast', 'static_venue_and_schedule', 'altitude=feet; capacity=seats; advantage=points prior; travel=miles estimate; timezone=hours; flags=boolean', 'Venue is actual host arena, previous location from last prior matchup. Rivalry/advantage are explicit fixed priors.')
register('ref_foul_rate ref_home_bias ref_pace_tendency', 'dated_referee_history', 'combined fouls/game; home win fraction; possessions/48min', 'Assigned crew tendencies computed from games strictly before prediction day; warmup uses league priors.')

register('pts_per_100 ast_per_100 reb_per_100 stl_per_100 blk_per_100 tov_per_100', 'prior_boxscores_and_team_pace', 'events/100 estimated possessions', 'Estimate=prior20 events/game *100/(prior10 team pace * prior20 minutes/48); not official on-court possessions.')

# Exact source-to-column mappings: source fields, output names, and provider units.
PLAYER_MAP = {
 'advanced': {'pie':'pie','off_rating':'player_off_rating','def_rating':'player_def_rating','pace':'player_pace','usg_pct':'usg_pct_official','ts_pct':'ts_pct_official','efg_pct':'efg_pct_official','ast_pct':'ast_pct_official','oreb_pct':'oreb_pct_official','dreb_pct':'dreb_pct_official','reb_pct':'reb_pct_official','net_rating':'net_rating_player'},
 'clutch': {'clutch_pts_pg':'clutch_pts_per_game','clutch_fg_pct':'clutch_fg_pct','clutch_min_pg':'clutch_min_per_game','clutch_fg3_pct':'clutch_fg3_pct','clutch_fta_pg':'clutch_fta_per_game','clutch_plus_minus':'clutch_plus_minus','clutch_games':'clutch_games'},
 'hustle': {'deflections_pg':'deflections_per_game','contested_shots_pg':'contested_shots_per_game','charges_drawn_pg':'charges_drawn_per_game','screen_assists_pg':'screen_assists_per_game'},
 'shot_prof': {'open_shot_fg_pct':'open_shot_fg_pct','open_shot_freq':'open_shot_frequency','tight_shot_fg_pct':'tight_shot_fg_pct','tight_shot_freq':'tight_shot_frequency','catch_shoot_fg_pct':'catch_shoot_fg_pct','catch_shoot_freq':'catch_shoot_frequency','pullup_fg_pct':'pullup_fg_pct','pullup_freq':'pullup_frequency'},
 'play_types': {k:k for k in 'iso_poss_pct iso_ppp pnr_bh_poss_pct pnr_bh_ppp pnr_roll_poss_pct pnr_roll_ppp spotup_poss_pct spotup_ppp transition_poss_pct transition_ppp postup_poss_pct cut_poss_pct'.split()},
 'on_off': {k:k for k in 'on_court_net_rating off_court_net_rating on_off_differential'.split()},
 'shot_zones': {k:k for k in 'rim_fga_pct rim_fg_pct paint_fga_pct paint_fg_pct midrange_fga_pct midrange_fg_pct corner3_fga_pct corner3_fg_pct above_break3_fga_pct above_break3_fg_pct'.split()},
 'q_splits': {'q1_avg':'q1_avg','q2_avg':'q2_avg','q3_avg':'q3_avg','q4_avg':'q4_avg','q4_min_pg':'q4_min_per_game'},
 'tracking': {k:'tracking_'+k for k in 'avg_speed avg_speed_off avg_speed_def dist_miles dist_miles_off dist_miles_def touches_pg time_of_poss_pg avg_drib_per_touch passes_made_pg potential_ast_pg secondary_ast_pg'.split()},
 'scoring': {k:k for k in 'pct_pts_paint pct_pts_midrange pct_uast_fgm pct_ast_fgm pct_pts_off_tov pct_pts_fb'.split()},
 'yoy': {k:k for k in 'yoy_pts_change yoy_ts_change yoy_usage_change seasons_in_league'.split()},
}
PLAYER_MAP['tracking'].update({'paint_touches_pg':'paint_touches_per_game','elbow_touches_pg':'elbow_touches_per_game'})
for table, mapping in PLAYER_MAP.items():
    for column, name in mapping.items():
        if table=='yoy':
            unit={'yoy_pts_change':'points/game delta','yoy_ts_change':'fraction delta','yoy_usage_change':'relative FGA/game change','seasons_in_league':'seasons played'}[column]
        elif table=='advanced':
            # PIE subtracts missed shots, fouls and turnovers; it is signed,
            # unlike a bounded make/attempt fraction (NBA Stats glossary).
            unit='signed ratio' if column=='pie' else 'fraction' if column.endswith('pct') else 'possessions/48 minutes' if column=='pace' else 'points/100 possessions'
        elif table=='on_off':unit='points/100 possessions'
        elif table=='tracking':
            unit='mph' if 'speed' in column else 'miles/game' if 'dist_miles' in column else 'minutes/game' if column=='time_of_poss_pg' else 'dribbles/touch' if column=='avg_drib_per_touch' else 'events/game'
        elif table=='play_types':unit='fraction' if column.endswith('pct') else 'points/possession'
        elif table=='clutch':unit='fraction' if 'pct' in column else 'games' if column=='clutch_games' else 'minutes/game' if column=='clutch_min_pg' else 'points/game' if column in ('clutch_pts_pg','clutch_plus_minus') else 'attempts/game'
        elif table=='q_splits':unit='minutes/game' if column=='q4_min_pg' else 'points/game'
        elif table=='hustle':unit='events/game'
        else:unit='fraction'
        register(name, 'previous_season:'+table, unit,
                 f'Completed prior-season {table}.{column}; never current-season snapshot. Missing/null/nonfinite source explicitly recorded, value0.' + (' Defender distance is6+ feet (wide open).' if table=='shot_prof' and column.startswith('open_shot') else ' Defender distance is0-2 feet (very tight).' if table=='shot_prof' and column.startswith('tight_shot') else ''))
ALIASES = {
 'clutch_minutes_per_game':'clutch_min_per_game','off_court_plus_minus':'off_court_net_rating',
 'pnr_ball_handler_pct':'pnr_bh_poss_pct','pnr_roll_man_pct':'pnr_roll_poss_pct','isolation_pct':'iso_poss_pct','spot_up_pct':'spotup_poss_pct','post_up_pct':'postup_poss_pct','transition_pct':'transition_poss_pct',
 'corner_3_pct':'corner3_fg_pct','corner_three_pct':'corner3_fg_pct','above_break_3_pct':'above_break3_fg_pct','above_break_three_pct':'above_break3_fg_pct','restricted_area_fg_pct':'rim_fg_pct','mid_range_frequency':'midrange_fga_pct',
 'wide_open_shot_pct':'open_shot_frequency',
 'contested_shot_pct':'tight_shot_frequency','open_shot_pct':'open_shot_frequency','catch_and_shoot_pct':'catch_shoot_frequency','pull_up_shot_pct':'pullup_frequency',
 'first_quarter_avg':'q1_avg','fourth_quarter_avg':'q4_avg','touches_per_game':'tracking_touches_pg','avg_dribbles_per_touch':'tracking_avg_drib_per_touch','time_of_possession_per_game':'tracking_time_of_poss_pg','pct_pts_in_paint':'pct_pts_paint',
}
for name, source in ALIASES.items():
    register(name,SPECS[source].source,SPECS[source].units,'Exact alias of '+source+'; no boxscore substitute.')
register('paint_touch_frequency', 'previous_season:tracking', 'fraction', 'Actual paint touches / total touches from the same completed-season tracking row.')
register('rim_fga_per_game paint_fga_per_game mid_range_fga_per_game paint_attempts_per_game restricted_area_attempts paint_pts_per_game avg_points_per_touch avg_seconds_per_touch touches_per_possession', 'prior_season_profile_and_prior_boxscores', 'events/game; points/touch; seconds/touch; touches/team possession', 'Explicit estimate: prior-season zone share * current prior20 attempts or points. Touch ratios use completed prior-season tracking with minutes converted to seconds.')

# Insufficient dated source, misleading proxy or unknown historical availability.
DISABLED = {
 'unvalidated_season_league_priors': 'lg_pts_fb lg_opp_pts_fb lg_pts_off_tov lg_opp_pts_off_tov lg_fga lg_fg_pct lg_fg3a lg_tov lg_stl',
 'unmeasured_fatigue': 'minutes_fatigue_score',
 'historical_injury_reports': 'team_key_players_out opp_key_players_out opp_injury_impact primary_teammate_out secondary_teammate_out',
 'historical_team_style_snapshots': 'team_pts_fb opp_pts_fb_allowed team_pts_off_tov opp_pts_off_tov_allowed opp_pts_paint',
 'historical_position_matchups': 'dvp_gp dvp_pts_delta dvp_reb_delta dvp_ast_delta dvp_fg3m_delta dvp_stl_delta dvp_blk_delta dvp_tov_delta position_vs_position_dvp dvp_pts_delta_last5 dvp_pts_delta_last10 dvp_reb_delta_last5 dvp_ast_delta_last5 dvp_fg3m_delta_last5',
 'pregame_defender_assignments': 'primary_defender_score01 primary_defender_rating primary_defender_age defender_size_mismatch defender_recent_form primary_defender_active career_vs_defender recent_vs_defender',
 'dated_lineup_stints': 'top_lineup_minutes_pct new_teammate_games lineup_stability_score bench_strength net_rating_with_starters usage_rate_with_star_out minutes_with_starting_lineup_pct five_man_unit_net_rating lineups_played_count teammate_chemistry_score lineup_continuity defender_switching_frequency opp_lineup_changes_last5',
 'actual_oos_prediction_history': 'model_accuracy_player avg_prediction_error_player calibration_score_player',
 'historical_conference_standings': 'playoff_implications playoff_seeding_impact tanking_indicator must_win_situation games_back_from_playoff team_conf_rank opp_conf_rank',
 'dated_opponent_splits': 'opp_def_rating_home_away_split opp_b2b_def_rating opp_b2b_pace opp_b2b_pts_allowed opp_rested_def_rating opp_rested_pace',
 'play_by_play_or_tracking_measurement': 'fourth_quarter_usage_rate garbage_time_minutes_pct typical_substitution_minute crunch_time_usage shot_quality_vs_expected avg_shot_clock_time late_clock_shot_frequency early_clock_shot_frequency post_touches_per_game front_court_touches_per_game clutch_performance_score shot_selection_rating bad_shot_frequency shot_clock_management fouls_drawn_per_game and_one_frequency foul_drawing_ability rebound_contested_pct rebound_positioning_score paint_touch_to_points performance_when_leading performance_when_trailing performance_when_tied performance_in_overtime performance_by_score_differential',
 'dated_defensive_ranks': 'pts_vs_top10_defenses pts_vs_bottom10_defenses',
 'unmeasured_probabilities': 'historical_game_script_avg expected_game_script blowout_probability close_game_probability implied_game_total',
 'incorrect_statistic_definition': 'assist_percentage rebound_percentage net_rating def_fg_pct_allowed def_rating_individual oreb_rate dreb_rate total_reb_rate blowout_game_pct close_game_pct',
 'historical_prior_season_opponent_splits': 'vs_team_last_season_avg',
 'actual_venue_splits': 'player_vs_arena',
 'unmeasured_shot_distance_or_defender_distance': 'avg_shot_distance',
 'dated_biographical_history': 'player_age years_experience is_rookie is_veteran player_height_inches player_weight',
 'dated_opponent_zone_and_synergy': 'opp_rim_fg_pct_allowed opp_paint_fg_pct_allowed opp_midrange_fg_pct_allowed opp_corner3_fg_pct_allowed opp_above_break3_fg_pct_allowed rim_shot_quality_matchup midrange_shot_quality_matchup corner3_shot_quality_matchup above_break3_shot_quality_matchup opp_pnr_ppp_allowed opp_iso_ppp_allowed opp_spotup_ppp_allowed opp_transition_ppp_allowed opp_postup_ppp_allowed pnr_matchup_advantage iso_matchup_advantage spotup_matchup_advantage transition_matchup_advantage',
 'historical_bookmaker_lines': 'opening_line current_line line_movement line_movement_pct implied_over_prob implied_under_prob market_consensus_std sharp_action_score line_velocity stale_line_flag bookmaker_count',
 'historical_pregame_broadcasts': 'national_tv_game',
 'medical_return_history': 'games_since_return missed_games_before_return',
 'regular_season_only_contract': 'is_playoff is_play_in series_game_num team_series_wins_in opp_series_wins_in is_elimination_game playoff_home playoff_away',
}
INACTIVE_CANDIDATES = dict(DISABLED)
from src.recovered_features import RECOVERED, DUPLICATES, BLOCKED
_original_inactive = {k for names in INACTIVE_CANDIDATES.values() for k in names.split()}
if set(RECOVERED) | set(DUPLICATES) | set(BLOCKED) != _original_inactive:
    raise ValueError(f'Recovery review incomplete: {_original_inactive-(set(RECOVERED)|set(DUPLICATES)|set(BLOCKED))}')
if len(set(RECOVERED) & set(DUPLICATES)) or len(set(RECOVERED) & set(BLOCKED)) or len(set(DUPLICATES) & set(BLOCKED)):
    raise ValueError('Conflicting feature recovery dispositions')
for name in _original_inactive:
    if name in RECOVERED:
        register(name, *RECOVERED[name])
    elif name in DUPLICATES:
        register(name, 'covered_by_canonical_features', 'inactive (0)', DUPLICATES[name], status='duplicate')
    else:
        register(name, 'source_recovery_pending', 'missing source', BLOCKED[name], status='source_blocked')
DISABLED = {source:' '.join(k for k in names.split() if SPECS[k].status!='active') for source,names in INACTIVE_CANDIDATES.items()}
DISABLED = {source:names for source,names in DISABLED.items() if names}
if set(SPECS)!=set(NUMERIC_FEATURE_KEYS):
    raise ValueError(f'Incomplete audit: missing={set(NUMERIC_FEATURE_KEYS)-set(SPECS)} extra={set(SPECS)-set(NUMERIC_FEATURE_KEYS)}')


def ratio(a,b):
    return float(a/b) if b>0 else 0.


def build_audited_features(history, target, player_id, team_id, opponent_id, prediction_date, is_home, inputs):
    """Return all416 values and source gaps. Same API for replay and serving.

    inputs must implement HistoricalInputs' dated team/snapshot interfaces.
    Current/future outcomes are excluded here, even when callers pass them.
    """
    date=pd.Timestamp(prediction_date).normalize()
    hist=history.copy();hist['GAME_DATE']=pd.to_datetime(hist.GAME_DATE).dt.normalize()
    if 'PLAYER_ID' in hist:
        hist=hist[pd.to_numeric(hist.PLAYER_ID)==int(player_id)]
    hist=hist[hist.GAME_DATE<date].sort_values('GAME_DATE').tail(20)
    if hist.empty:raise ValueError('Audited features require prior player boxscores')
    source_target=target
    if isinstance(target,(list,tuple)):
        hist['_AUDITED_TARGET']=hist[list(target)].sum(axis=1)
        target='_AUDITED_TARGET'
    if target not in hist:raise ValueError('Missing target column: '+target)
    if any(not str(g).startswith(('002','004','005')) for g in hist.get('GAME_ID',hist.get('Game_ID',[]))):
        raise ValueError('Unexpected NBA game type in audited replay')
    f={k:0. for k in NUMERIC_FEATURE_KEYS};missing=set()
    def vals(col,frame=hist):return pd.to_numeric(frame.get(col,pd.Series(0.,index=frame.index)),errors='coerce').fillna(0)
    def avg(col,frame=hist):return float(vals(col,frame).mean()) if len(frame) else 0.
    def total(col,frame=hist):return float(vals(col,frame).sum())
    v=vals(target).to_numpy();mean=float(v.mean());std=float(v.std());last5=hist.tail(5)
    f.update(recent_avg=float(v[-5:].mean()),season_avg=mean,stddev=std,games_played=len(v),max_recent=float(v.max()),min_recent=float(v.min()),mins_last5=avg('MIN',last5),mins_season=avg('MIN'),avg_minutes_last_3=avg('MIN',hist.tail(3)),minutes_volatility=float(vals('MIN').std()) if len(hist)>1 else 0.,is_home_game_num=float(is_home) if is_home in (0,1) else -1.)
    for n in (3,5,7):f[f'minutes_last_{n}_games']=total('MIN',hist.tail(n))
    for n in (3,5,10):f[f'last_{n}_games_trend']=float(v[-n:].mean()-mean)
    for n in (7,14,30):f[f'rolling_{n}day_avg']=float(v[-n:].mean())
    for n in (5,7,14):f['games_above_season_avg_'+('last5' if n==5 else f'{n}day')]=float((v[-n:]>mean).sum())
    for alpha in (.3,.5):f[f'ewm_alpha_{alpha}']=float(pd.Series(v).ewm(alpha=alpha).mean().iloc[-1])
    for n in (5,10):f[f'trend_slope_{n}games']=float(np.polyfit(np.arange(len(v[-n:])),v[-n:],1)[0]) if len(v)>1 else 0.
    line=float(v[-10:].mean());over=bool(v[-1]>line);streak=0
    for value in reversed(v[-10:]):
        if bool(value>line)!=over:break
        streak+=1
    f['consecutive_over_games' if over else 'consecutive_under_games']=streak
    f.update(hot_hand_indicator=float(len(v)>=3 and (v[-3:]>mean).all()),recent_variance_spike=float(v[-3:].std()/max(std,.1)-1),volatility_ratio=float(v[-5:].std()/max(std,.01)),momentum_score=float((v[-5:].mean()-v[-10:-5].mean())/max(abs(v[-10:-5].mean()),.1)) if len(v)>=10 else 0.,consistency_score=max(0.,1-float(v.std(ddof=1) if len(v)>1 else 0)/max(mean,1)),ceiling_game_frequency=float((v>=1.5*mean).mean()))
    gap=(date-hist.GAME_DATE.iloc[-1]).days;rest=max(0,gap-1)
    f.update(rest_days=rest,days_since_last_game=gap,is_back_to_back=float(gap==1),games_in_last_7_days=int((hist.GAME_DATE>=date-pd.Timedelta(days=7)).sum()))
    matchups=hist.MATCHUP.astype(str)
    for m in reversed(matchups.tolist()):
        if '@' not in m:break
        f['recent_away_streak']+=1
    for prefix,made,attempt in [('fg','FGM','FGA'),('fg3','FG3M','FG3A'),('ft','FTM','FTA')]:f[prefix+'_pct_recent']=ratio(total(made,last5),total(attempt,last5))
    for key,col in {'fga_per_game':'FGA','fg3a_per_game':'FG3A','fta_per_game':'FTA','oreb_per_game':'OREB','dreb_per_game':'DREB','fouls_per_game':'PF','ft_attempts_per_game':'FTA','plus_minus_avg':'PLUS_MINUS','on_court_plus_minus':'PLUS_MINUS'}.items():f[key]=avg(col)
    f['win_rate_last10']=float((hist.tail(10).WL=='W').mean())
    pts,fga,fta,mins=total('PTS'),total('FGA'),total('FTA'),total('MIN')
    f.update(usage_rate=ratio(fga+.44*fta+total('TOV'),mins)*48,true_shooting_pct=ratio(pts,2*(fga+.44*fta)),effective_fg_pct=ratio(total('FGM')+.5*total('FG3M'),fga),points_per_shot=ratio(pts,fga),ast_to_tov_ratio=ratio(total('AST'),total('TOV')),reb_rate_per_36=ratio(total('REB'),mins)*36,ft_rate=ratio(fta,fga),fta_rate_player=ratio(fta,fga),pct_fga_3pt=ratio(total('FG3A'),fga),pct_fga_2pt=ratio(fga-total('FG3A'),fga),pct_pts_3pt=ratio(3*total('FG3M'),pts),pct_pts_ft=ratio(total('FTM'),pts),scoring_efficiency_trend=ratio(total('PTS',last5),total('FGA',last5))-ratio(pts,fga),usage_trend=avg('FGA',last5)-avg('FGA'))
    for table,mapping in PLAYER_MAP.items():
        row=inputs.precomputed.get(table,{}).get((int(player_id),inputs.season),{})
        for col,key in mapping.items():
            value=row.get(col)
            if value is None or not np.isfinite(float(value)):missing.add(key)
            else:f[key]=float(value)
    for name,source in ALIASES.items():
        f[name]=f[source]
        if source in missing:missing.add(name)
    for zone,name in [('rim','rim_fga_per_game'),('paint','paint_fga_per_game'),('midrange','mid_range_fga_per_game')]:
        f[name]=f[zone+'_fga_pct']*f['fga_per_game']
        if zone+'_fga_pct' in missing:missing.add(name)
    f['paint_attempts_per_game']=f['paint_fga_per_game'];f['restricted_area_attempts']=f['rim_fga_per_game']
    f['paint_touch_frequency']=ratio(f['paint_touches_per_game'],f['tracking_touches_pg'])
    if 'paint_touches_per_game' in missing or 'tracking_touches_pg' in missing:missing.add('paint_touch_frequency')
    f['paint_pts_per_game']=f['pct_pts_paint']*avg('PTS')
    for key,deps,numerator,denominator in [
        ('avg_points_per_touch',['tracking_touches_pg'],avg('PTS'),f['tracking_touches_pg']),
        ('avg_seconds_per_touch',['tracking_time_of_poss_pg','tracking_touches_pg'],f['tracking_time_of_poss_pg']*60,f['tracking_touches_pg'])]:
        f[key]=ratio(numerator,denominator)
        if any(d in missing for d in deps):missing.add(key)
    for name,dep in [('paint_pts_per_game','pct_pts_paint'),('paint_attempts_per_game','paint_fga_per_game'),('restricted_area_attempts','rim_fga_per_game')]:
        if dep in missing:missing.add(name)
    ctx=inputs.team_context(team_id,date);opp=inputs.team_context(opponent_id,date)
    for prefix,c in [('team',ctx),('opp',opp)]:
        f[prefix+'_pace']=c['pace'];f[prefix+'_off_rating']=c['offensive_rating'];f[prefix+'_def_rating']=c['defensive_rating']
    f['matchup_pace']=(f['team_pace']+f['opp_pace'])/2
    for col,key in [('PTS','pts'),('AST','ast'),('REB','reb'),('STL','stl'),('BLK','blk'),('TOV','tov')]:
        f[key+'_per_100']=ratio(avg(col)*100,f['team_pace']*f['mins_season']/48)
    f['touches_per_possession']=ratio(f['tracking_touches_pg'],f['team_pace'])
    if 'tracking_touches_pg' in missing:missing.add('touches_per_possession')
    ownrest=ctx['days_rest'];opprest=opp['days_rest']
    f.update(days_rest_opponent=opprest,opponent_back_to_back=float(opprest==0),rest_advantage=ownrest-opprest,rest_advantage_abs=abs(ownrest-opprest),both_teams_rested=float(ownrest>=2 and opprest>=2))
    for tid,prefix in [(team_id,'team'),(opponent_id,'opp')]:
        games=inputs.prior_team_games(tid,date)
        if not games:
            missing.update((prefix+'_pace',prefix+'_off_rating',prefix+'_def_rating',prefix+'_win_pct',prefix+'_current_streak',prefix+'_l10_wins'))
            if prefix=='opp':missing.update(('days_rest_opponent','opponent_back_to_back','opp_def_rating_last5','opp_def_rating_last10','opp_def_rating_trend','opp_pace_last5','opp_win_rate_last10'))
            else:missing.update(('team_win_streak','team_loss_streak'))
            missing.update(('rest_advantage','rest_advantage_abs','both_teams_rested'))
        season_games=[g for g in games if str(g['game_date'])>=inputs.season[:4]+'-10-01']
        decided=[g for g in season_games if g.get('pts') is not None and g.get('opp_pts') is not None and g['pts']!=g['opp_pts']]
        wins=[g['pts']>g['opp_pts'] for g in decided]
        f[prefix+'_win_pct']=float(np.mean(wins)) if wins else .5
        f[prefix+'_l10_wins']=sum(wins[-10:]);streak=0
        for w in reversed(wins):
            if w!=wins[-1]:break
            streak+=1
        f[prefix+'_current_streak']=streak*(1 if wins and wins[-1] else -1)
        if prefix=='team':f.update(team_win_streak=max(0,f['team_current_streak']),team_loss_streak=max(0,-f['team_current_streak']))
        else:
            f['opp_win_rate_last10']=float(np.mean(wins[-10:])) if wins else .5
            for n,key,field in [(5,'opp_def_rating_last5','def_rating'),(10,'opp_def_rating_last10','def_rating'),(5,'opp_pace_last5','pace')]:
                values=[g[field] for g in games[-n:] if g.get(field) is not None]
                f[key]=float(np.mean(values)) if values else (100 if field=='pace' else 110)
            f['opp_def_rating_trend']=f['opp_def_rating_last5']-f['opp_def_rating_last10']
    box=inputs.prior_team_boxscores(opponent_id,date)
    for key,col,n in [('opp_fga','FGA',10),('opp_fg3a','FG3A',10),('opp_tov','TOV',10),('opp_stl','STL',10),('opp_blk','BLK',10),('opp_blocks_per_game_last5','BLK',5),('opp_steals_per_game_last5','STL',5)]:
        f[key]=float(box.tail(n)[col].mean()) if len(box) else 0.
        if box.empty:missing.add(key)
    for key,made,attempt in [('opp_fg_pct','FGM','FGA'),('opp_fg3_pct','FG3M','FG3A')]:
        f[key]=ratio(box.tail(10)[made].sum(),box.tail(10)[attempt].sum()) if len(box) else 0.
        if box.empty:missing.add(key)
    for n,key in [(10,'opp_foul_rate_per48'),(5,'opp_foul_rate_last5')]:
        f[key]=ratio(box.tail(n).PF.sum(),box.tail(n).MIN.sum()/5)*48 if len(box) else 20.
        if box.empty:missing.add(key)
    from nba_api.stats.static import teams
    opponent=teams.find_team_name_by_id(int(opponent_id));abbrev=opponent['abbreviation']
    vs=hist[matchups.str.split().str[-1]==abbrev]
    f.update(vs_team_avg=avg('PTS',vs),matchup_games=len(vs),head_to_head_games=len(vs),head_to_head_avg=avg(target,vs) if len(vs) else mean,vs_opp_gp=len(vs),vs_opp_avg_pts=avg('PTS',vs),vs_opp_avg_min=avg('MIN',vs),vs_opp_fg_pct=ratio(total('FGM',vs),total('FGA',vs)),vs_opp_ts_pct=ratio(total('PTS',vs),2*(total('FGA',vs)+.44*total('FTA',vs))),vs_team_win_pct=float((vs.WL=='W').mean()) if len(vs) else .5)
    home=hist[matchups.str.contains('vs',regex=False)];away=hist[matchups.str.contains('@',regex=False)]
    f['vs_team_home_away_split']=(avg(target,home) if len(home) else mean)-(avg(target,away) if len(away) else mean)
    year=int(inputs.season[:4]);start=pd.Timestamp(f'{year}-10-22');end=pd.Timestamp(f'{year+1}-04-14');days=max(0,(date-start).days);fraction=days/(end-start).days
    f.update(days_into_season=days,games_remaining_approx=max(0,int(82*(1-fraction))),season_phase_numeric=0 if fraction<.2 else (2 if fraction>.8 else 1),season_phase=0 if date.month in (10,11) else 1 if date.month in (12,1) else 2 if date.month in (2,3) else 3)
    from src.feature_context import RIVALRY_PAIRS
    from src.arena_data import ARENA_DATA,calculate_travel_metrics
    own=teams.find_team_name_by_id(int(team_id));f['rivalry_game']=float((own['abbreviation'],abbrev) in RIVALRY_PAIRS)
    host=team_id if is_home==1 else opponent_id if is_home==0 else None
    arena=ARENA_DATA.get(host,{})
    f.update(arena_altitude=arena.get('altitude',0),arena_capacity=arena.get('capacity',18000),home_court_advantage_rating=3.5 if is_home==1 else -3.5 if is_home==0 else 0.)
    if host:
        last=hist.iloc[-1];parts=str(last.MATCHUP).split();prev_own=int(last.get('TEAM_ID',team_id));prev_opp=teams.find_team_by_abbreviation(parts[-1]);prev=prev_own if 'vs' in str(last.MATCHUP) else prev_opp['id'] if prev_opp else None
        if prev:
            tz,coast,dist=calculate_travel_metrics(prev,host,is_home)
            f.update(time_zone_change=float(tz),coast_to_coast=float(coast),travel_distance=float(dist))
    # Referee assignments supplied independently from the current game's outcomes.
    from src.referee_history import historical_features
    gid=getattr(inputs,'prediction_game_id',None)
    if gid:
        refs=historical_features(inputs.precomputed,str(gid))
        f.update(dict(zip(('ref_foul_rate','ref_home_bias','ref_pace_tendency'),refs)))
        names=inputs.precomputed.get('game_officials',{}).get(str(gid),[])
        if not any(inputs.precomputed.get('referee_history',{}).get((str(gid),name.lower()),{}).get('games',0)>=3 for name in names):
            missing.update(('ref_foul_rate','ref_home_bias','ref_pace_tendency'))
    else:
        f.update(ref_foul_rate=0.,ref_home_bias=.5,ref_pace_tendency=100.)
        missing.update(('ref_foul_rate','ref_home_bias','ref_pace_tendency'))
    from src.recovered_features import apply_recovered
    apply_recovered(f,hist,source_target,player_id,team_id,opponent_id,date,inputs,missing)
    # Unsupported or duplicated columns cannot be reactivated by callers.
    for k in NUMERIC_FEATURE_KEYS:
        if SPECS[k].status!='active':f[k]=0.
    if not all(np.isfinite(float(v)) for v in f.values()):raise ValueError('Nonfinite audited feature')
    return f,sorted(missing)
