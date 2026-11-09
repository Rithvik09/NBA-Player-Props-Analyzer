"""Recovered distinct features and explicit duplicate/source dispositions.

No report or snapshot from the target day's outcomes can enter these values.
Previous-season endpoint measurements are documented as lagged priors. A source
failure remains missing; feature-vector construction preserves it as NaN.
"""
from collections import defaultdict
from functools import lru_cache
import json,re
from pathlib import Path
import numpy as np
import pandas as pd
from src.historical_injuries import normalize_name,load_reports

ROOT=Path(__file__).resolve().parents[1]
SOURCES=ROOT/'logs/feature_recovery'
# name -> (source,units,definition)
RECOVERED={}
def add(names,source,units,definition):
    for name in names.split():RECOVERED[name]=(source,units,definition)

add('team_key_players_out opp_key_players_out opp_injury_impact','official_timestamped_injury_reports','players; impact=fraction of240 team minutes','Confirmed Out only, before prediction cutoff, for the scheduled game. Key role=prior mean minutes>=20; impact=sum prior minutes of absent players/240, capped1. Unknown report is missing.')
add('usage_rate_with_star_out','injury_report_and_prior_team_boxscores','possessions/48min','Player prior appearances for the current team when the currently absent highest-minute teammate did not play; earlier observed roster evidence must place that teammate on this team at each appearance. Requires at least3 matching appearances. Unknown roster history stays missing.')
add('games_since_return missed_games_before_return','official_injury_reports_and_prior_appearances','games','Prior player appearances since latest confirmed medical Out game, and confirmed Out games in the absence spell. Unknown/unreported injury history remains missing.')
add('team_pts_fb opp_pts_fb_allowed team_pts_off_tov opp_pts_off_tov_allowed opp_pts_paint','completed_prior_season_team_misc','points/game','Actual NBA Misc scoring components from completed prior season, not latest overwritten team snapshots.')
add('lg_pts_fb lg_opp_pts_fb lg_pts_off_tov lg_opp_pts_off_tov lg_fga lg_fg_pct lg_fg3a lg_tov lg_stl','completed_prior_season_league_aggregate','points/game, attempts/game, FG fraction, or events/game','League averages of completed-prior-season team observations. FG percentage is weighted by total makes/attempts.')
add('opp_rim_fg_pct_allowed opp_paint_fg_pct_allowed opp_midrange_fg_pct_allowed opp_corner3_fg_pct_allowed opp_above_break3_fg_pct_allowed','completed_prior_season_opponent_zones','fraction','Actual opponent makes/attempts in each shooting zone; corrected endpoint and preserved source season. Not substituted league averages.')
add('rim_shot_quality_matchup midrange_shot_quality_matchup corner3_shot_quality_matchup above_break3_shot_quality_matchup','player_and_opponent_prior_zone_measurements','fraction difference','Player zone FG fraction minus opponent allowed fraction, both completed prior season; explicit interaction useful to models without automatic interactions.')
add('opp_pnr_ppp_allowed opp_iso_ppp_allowed opp_spotup_ppp_allowed opp_transition_ppp_allowed opp_postup_ppp_allowed','completed_prior_season_defensive_synergy','points/possession','Actual NBA defensive play-type efficiency, completed prior season.')
add('pnr_matchup_advantage iso_matchup_advantage spotup_matchup_advantage transition_matchup_advantage','player_and_opponent_prior_playtype_measurements','points/possession difference','Player play-type PPP minus opponent defensive PPP, using completed prior season.')
add('rebound_contested_pct','completed_prior_season_rebounding_tracking','fraction','Contested rebounds / rebounds from actual NBA Rebounding tracking, not contested shots.')
add('post_touches_per_game front_court_touches_per_game paint_touch_to_points','completed_prior_season_possession_tracking','touches/game; paint_touch_to_points=points/touch','Actual POST_TOUCHES, FRONT_CT_TOUCHES and PTS_PER_PAINT_TOUCH tracking fields.')
add('fouls_drawn_per_game','completed_prior_season_player_misc','fouls/game','Actual PFD field from NBA Misc.')
add('fourth_quarter_usage_rate','completed_prior_season_fourth_quarter_boxscores','possessions/48min','(Q4 FGA+.44*Q4 FTA+Q4 TOV)/Q4 minutes*48; same usage units as main boxscore usage feature.')
add('crunch_time_usage','completed_prior_season_clutch_advanced','fraction','Official clutch USG_PCT, rather than a ramp based on average minutes.')
add('early_clock_shot_frequency late_clock_shot_frequency','completed_prior_season_shot_clock_dashboard','fraction','Actual FGA_FREQUENCY for24-22 plus22-18 seconds, and4-0 seconds respectively. No pace-based proxy.')
add('player_age player_height_inches player_weight','completed_prior_season_bio','years; inches; pounds','Observed completed-prior-season bio values; age is a lagged season age, not a fabricated exact birthday-age.')
add('opp_b2b_def_rating opp_b2b_pace opp_b2b_pts_allowed opp_rested_def_rating opp_rested_pace opp_def_rating_home_away_split','dated_team_history','ratings/100poss; pace/48min; points/game; rating delta','Prior opponent games classified by actual schedule rest and venue. Rested means at least2 days off. Empty category is missing.')
add('def_fg_pct_allowed','dated_opposing_boxscores','fraction','FGM/FGA by teams facing this opponent in prior games, not opponent offensive FG%.')
add('blowout_game_pct close_game_pct','dated_team_scores','fraction','Share of prior20 team games with final absolute margin>15 or<=5; not player plus-minus, and not predicted probabilities.')
add('vs_team_last_season_avg','completed_previous_season_player_game_logs','prop units/game','Player mean against this opponent in the completed previous season, not current-season head-to-head alias.')
add('team_conf_rank opp_conf_rank games_back_from_playoff','dated_conference_team_records','win-fraction rank with tied teams sharing rank; games','As-of conference win-fraction rank (not official tie-break seeding); games behind conference10th record from wins/losses differential, floored0.')
add('top_lineup_minutes_pct lineups_played_count','completed_prior_season_five_player_lineups','fraction; lineups','For lineups containing this player: largest five-player unit minutes / all such minutes; number of observed units. Actual lineups, not a minutes ramp.')
add('bench_strength','completed_prior_season_bench_boxscores','bench points/game','NBA bench-only team points/game, with the provider Bench filter.')
add('lineup_continuity new_teammate_games','dated_team_rotation_appearances','fraction; games','Rotation continuity=prior5 minutes share belonging to prior20 core players (>=10 prior appearances). New teammate games=appearances with the newest current rotation teammate inside the prior20 player-game window.')
add('dvp_gp dvp_pts_delta dvp_reb_delta dvp_ast_delta dvp_fg3m_delta dvp_stl_delta dvp_blk_delta dvp_tov_delta dvp_pts_delta_last5 dvp_pts_delta_last10 dvp_reb_delta_last5 dvp_ast_delta_last5 dvp_fg3m_delta_last5','dated_position_opponent_boxscores','games; events/team-game difference','Opponent allowed position-group totals minus league position-group mean, from strictly prior games. Position is completed-prior-season G/F/C membership; sparse/unknown membership remains missing.')
add('pts_vs_top10_defenses pts_vs_bottom10_defenses','dated_team_defensive_ranks','points/game','Player prior-game PTS split by the opponent defensive-rating rank available BEFORE each past game. Empty split remains missing.')

add('avg_shot_distance','dated_play_by_play_shots','feet','Mean measured shot distance over prior20 appearances; requires known distance for every observed FGA, never substitutes zero for an unknown distance.')
add('and_one_frequency','dated_play_by_play_linked_events','and-one opportunities/made field goal','A made field goal followed at the same clock by the same player1-of-1 free throw, linked by player/period/clock. Requires validated shot totals.')
add('typical_substitution_minute','dated_play_by_play_substitutions','elapsed game minutes','Mean first substitution-out minute across prior appearances where an explicit outgoing identity was resolved.')
add('performance_when_leading performance_when_trailing performance_when_tied performance_in_overtime','dated_play_by_play_score_states','points/game','Player points scored while own team leads/trails/is tied BEFORE scoring, or in overtime, averaged over prior observed appearances. PBP points must match NBA boxscore.')
add('performance_by_score_differential','dated_play_by_play_score_states','points/game difference','Leading-state points/game minus trailing-state points/game, from validated scoring events, not final win/loss averages.')
add('garbage_time_minutes_pct minutes_with_starting_lineup_pct','dated_validated_play_by_play_stints','fraction of tracked minutes','Actual tracked stints; every player minute total must match recorded source precision (0.12min for exact clocks,0.51min for rounded league logs). Garbage rule: Q4 margin>=20 inside6min or>=15 inside2min. Starting unit exposure requires at least4 actual starters on floor.')
add('opp_lineup_changes_last5','dated_starting_player_boxscores','starting positions changed/game','Mean5-minus-overlap of actual starting units over successive opponent games, using only prior games.')
add('national_tv_game','historical_game_broadcast_metadata','boolean','Recorded national broadcast schedule metadata, not inferred from player fame. Historical schedule publication timestamp is not independently archived.')

add('is_playoff is_play_in series_game_num team_series_wins_in opp_series_wins_in is_elimination_game playoff_home playoff_away','dated_postseason_schedule_and_prior_results','boolean; game number; wins','Game type from NBA game ID, series state from earlier series games only. Regular-season values are known0. Postseason game logs must be present; no current/future outcomes.')

add('player_vs_arena','scheduled_arena_and_dated_player_appearances','prop-unit difference','Player mean at the actual scheduled arena minus overall prior20 mean. Arena identity comes from game metadata, not home/away guessing; no prior arena appearance is missing.')
add('net_rating_with_starters','prior_season_observed_starting_units_and_official_unit_ratings','points/100poss, minute-weighted mean of unit ratings','Completed-prior-season official five-player unit net ratings, restricted to units observed as actual starters in dated boxscores, weighted by unit minutes. Missing past starters is not filled by team net rating.')

add('shot_quality_vs_expected','completed_prior_season_player_zones_and_league_zone_counts','eFG fraction above zone expectation','Observed player eFG minus league expected eFG for the same five-zone shot mix. Expectations use completed-prior-season league makes/attempts, not a TS*usage proxy or an unvalidated optical shot-quality claim.')

add('blowout_probability close_game_probability','qualified_prior_season_game_script_forecaster','probability','Logistic game-context forecasts fit/calibrated/qualified chronologically before target season. Only published if prior-period Brier beats a calibrated constant and AUC>0.5. Regular season only; failed qualification stays missing.')

add('primary_defender_rating primary_defender_age defender_size_mismatch','qualified_prior_matchup_defender_forecast_and_prior_bio','official on-court defensive rating; lagged years; defender-minus-player inches','Regular-season expected defender from prior400-day player/opponent matchup possessions, min20 possessions, share>=40%, lead>=15%; latest earlier team membership must match opponent. Enabled only if preceding regular-season assignment backtest n>=100, accuracy>=50% and beats minutes baseline by>=5 points. Rating and bio use completed previous season; identity is a forecast, not a confirmed assignment. Unknown scheduled game type and postseason remain missing.')
add('primary_defender_active','qualified_defender_forecast_and_timestamped_injuries','boolean','Availability of the historically expected defender, before adjusting assignment for injury. Out=0; Available or absent from a validated team report=1; uncertain status or missing report remains missing.')
add('def_rating_individual','qualified_defender_forecast_and_actual_matchup_points','opponent player points/100 tracked partial possessions','Expected defender actual matchup points conceded per100 tracked partial possessions, prior20 observed games within400 days, min3 games and100 partial possessions. Distinct from official on-court team defensive rating.')
add('defender_recent_form','qualified_defender_forecast_and_actual_matchup_points','points/100 partial possessions difference','Expected defender prior5 observed-game matchup scoring allowed minus prior20 observed-game rate; min3 games, strictly before prediction date.')
add('primary_defender_score01','qualified_defender_forecast_and_observed_efficiency_percentile','fraction','Fraction of prior-season measured defenders with worse scoring allowed/100 partial possessions than expected defender prior20 observed-game rate. Requires100 partial possessions per defender and>=30 comparison defenders. Descriptive efficiency percentile, not calibrated matchup probability.')

# Only justified overlap/duplicate exclusions. Distinct but unavailable sources
# are listed separately below rather than called unnecessary.
DUPLICATES={
 'minutes_fatigue_score':'Measured minutes_last3/5/7, rest and schedule density already expose its components; old score was constant near1.',
 'primary_teammate_out':'Threshold alias of team_key_players_out>=1.',
 'secondary_teammate_out':'Threshold alias of team_key_players_out>=2.',
 'lineup_stability_score':'Same intended rotation-continuity measurement as lineup_continuity.',
 'five_man_unit_net_rating':'Same intended actual unit rating as net_rating_with_starters; source recovery belongs to that canonical field.',
 'teammate_chemistry_score':'Arbitrary combination of continuity and unit rating, not an independent observed statistic.',
 'career_vs_defender':'Old implementation is team head-to-head, already represented by head_to_head_avg.',
 'recent_vs_defender':'Old implementation is the same bounded team head-to-head history, not a defender matchup.',
 'assist_percentage':'Official assist rate already represented by ast_pct_official; legacy AST/FGA is mislabeled.',
 'rebound_percentage':'Official rebound share already represented by reb_pct_official; legacy REB/100 is mislabeled.',
 'net_rating':'Actual player net rating already represented by net_rating_player; legacy value duplicates plus_minus_avg.',
 'oreb_rate':'Prior official rebound opportunity rate already covered by oreb_pct_official.',
 'dreb_rate':'Prior official rebound opportunity rate already covered by dreb_pct_official.',
 'total_reb_rate':'Prior official rebound opportunity rate already covered by reb_pct_official.',
 'years_experience':'Completed seasons already represented by seasons_in_league; draft-year proxy was incorrect.',
 'is_rookie':'Threshold alias of completed seasons; no distinct measurement.',
 'is_veteran':'Threshold alias of completed seasons; no distinct measurement.',
 'position_vs_position_dvp':'Exact legacy alias of dvp_pts_delta.',
 'current_line':'Canonical decision-time classifier input is line, not a second current_line column.',
 'implied_under_prob':'Complement of a normalized two-way implied_over_prob; preserve raw under odds for pricing.',
 'line_movement_pct':'Deterministic normalization of opening/current line and raw line movement; no new source.',
 'shot_selection_rating':'Old formula duplicates recent FG%; no independent shot-selection measurement.',
 'bad_shot_frequency':'Old formula is1-FG%; no independent bad-shot label.',
 'clutch_performance_score':'Old formula duplicates consistency_score; actual clutch stats are retained.',
 'foul_drawing_ability':'Old formula duplicates FTA volume/rate. Actual PFD and FTA rates are retained.',
 'rebound_positioning_score':'Old expression is the same rebound rate divided by itself; real rebounding tracking retained.',
 'shot_clock_management':'Undefined aggregate score; retain actual early/late shot frequencies.',
 'historical_game_script_avg':'Legacy signed blowout/close frequency is covered by actual margin-frequency components.',
 'expected_game_script':'Undefined projection of already measured margin-frequency components, not a separate observation.',
 'playoff_implications':'Old threshold of calendar phase, rank and games back; retain its measured components.',
 'playoff_seeding_impact':'Handmade score of rank and calendar phase, already covered by those inputs.',
 'tanking_indicator':'Speculative motivation label from rank and win percentage, already covered by measured components.',
 'must_win_situation':'Speculative motivation label from rank and games back, already covered by those components.',
}
# Missing source is not a decision to remove the capability.
BLOCKED={
 'defender_switching_frequency':'NBA switchesOn is unpopulated across the expanded validated matchup archive; an all-zero field does not establish real switching counts. Requires a populated tracking source.',
 'model_accuracy_player':'Requires logged, out-of-sample pregame predictions joined to later outcomes; no such historical ledger is present.',
 'avg_prediction_error_player':'Requires actual out-of-sample prediction residuals, not a multiple of volatility.',
 'calibration_score_player':'Requires actual out-of-sample probabilities and later outcomes, not a multiple of volatility.',
 'avg_shot_clock_time':'Bucket frequencies do not identify actual average shot-clock time; requires shot-level times.',
 'implied_game_total':'Requires actual pregame bookmaker game total history; pace*1.1 is not implied market total.',
 'opening_line':'Requires decision-time historical prop odds snapshots.',
 'line_movement':'Requires timestamped opening and decision-time prop lines.',
 'implied_over_prob':'Requires actual contemporaneous prices, not synthetic lines.',
 'market_consensus_std':'Requires contemporaneous multi-book prop quotes.',
 'sharp_action_score':'Requires an objective source/definition of sharp action; not ticket/money imbalance relabeled.',
 'line_velocity':'Requires timestamped prop-line changes.',
 'stale_line_flag':'Requires timestamped quote freshness and cross-book comparison.',
 'bookmaker_count':'Requires real cross-book observations at decision time.',
}

@lru_cache(maxsize=1)
def reports():return load_reports(SOURCES)


def teammate_absence_usage(history, roster_logs, star, team, date):
    """Do not count pre-transfer games as games played without a teammate.

    Membership is latest strictly earlier observed team, not a future roster.
    Same-day observations may exclude an appearance, never establish membership.
    """
    required={'PLAYER_ID','TEAM_ID','GAME_DATE','GAME_ID'}
    if not required.issubset(roster_logs) or 'TEAM_ID' not in history:return None
    star_logs=roster_logs[(roster_logs.PLAYER_ID==int(star)) &
                          (pd.to_datetime(roster_logs.GAME_DATE)<pd.Timestamp(date))].copy()
    if star_logs.empty:return None
    star_logs['GAME_DATE']=pd.to_datetime(star_logs.GAME_DATE)
    star_logs=star_logs.sort_values('GAME_DATE').drop_duplicates('GAME_DATE',keep='last')
    idcol='GAME_ID' if 'GAME_ID' in history else 'Game_ID'
    own=history[(history.TEAM_ID==int(team)) &
                (pd.to_datetime(history.GAME_DATE)<pd.Timestamp(date))].copy()
    own['GAME_DATE']=pd.to_datetime(own.GAME_DATE)
    own=own[~own[idcol].astype(str).isin(star_logs.GAME_ID.astype(str))]
    if own.empty:return None
    membership=pd.merge_asof(own.sort_values('GAME_DATE'),
        star_logs[['GAME_DATE','TEAM_ID']].rename(columns={'TEAM_ID':'star_team'}),
        on='GAME_DATE',direction='backward',allow_exact_matches=False)
    without=membership[membership.star_team==int(team)]
    if len(without)<3 or without.MIN.sum()<=0:return None
    return float((without.FGA.sum()+.44*without.FTA.sum()+without.TOV.sum())*48/without.MIN.sum())

class RecoveryContext:
    def __init__(self,inputs,logs):
        self.inputs=inputs
        path=SOURCES/f'snapshots_{inputs.source_season}.json'
        snapshot=json.loads(path.read_text()) if path.exists() else {}
        if snapshot and snapshot.get('source_season')!=inputs.source_season:raise ValueError('Recovery snapshot season mismatch')
        self.sources=snapshot.get('sources',{})
        from src.defender_evidence import load_evidence
        self.defender_evidence=load_evidence()
        metadata=SOURCES/'game_metadata.json'
        self.game_metadata=json.loads(metadata.read_text()) if metadata.exists() else {}
        self.logs=logs.copy()
        if 'GAME_DATE' in self.logs:self.logs['GAME_DATE']=pd.to_datetime(self.logs.GAME_DATE).dt.normalize()
        self.rich=pd.DataFrame();self._date_cache={};self._injury_cache={};self._dvp_cache={};self._rank_cache={}
        if {'TEAM_ID','GAME_DATE','GAME_ID','MATCHUP'}.issubset(self.logs):
            frame=self.logs[self.logs.GAME_ID.astype(str).str.startswith(('002','004','005') if inputs.audited else '002')].copy()
            from nba_api.stats.static import teams
            team_map={t['abbreviation']:t['id'] for t in teams.get_teams()}
            frame['OPP_ID']=frame.MATCHUP.str.split().str[-1].map(team_map)
            self.logs=frame
            cols=[c for c in 'MIN PTS FGA FGM FG3A FG3M FTA FTM AST OREB DREB REB STL BLK TOV PF'.split() if c in frame]
            self.rich=frame.groupby(['TEAM_ID','GAME_ID','GAME_DATE','OPP_ID','MATCHUP'],as_index=False)[cols].sum().sort_values('GAME_DATE')
        self.previous_logs=pd.DataFrame()
        prev=ROOT/'logs/points_pilot_cache'/f'players_{inputs.source_season}.json'
        if prev.exists():self.previous_logs=pd.read_json(prev,dtype={'GAME_ID':str})
        # Player identity is not selected using future performance.
        self.name_ids={}
        exact=defaultdict(set)
        if {'PLAYER_NAME','PLAYER_ID'}.issubset(self.logs):
            for _,r in self.logs[['PLAYER_NAME','PLAYER_ID']].drop_duplicates().iterrows():
                self.name_ids[normalize_name(r.PLAYER_NAME)]=int(r.PLAYER_ID)
                exact[normalize_name(r.PLAYER_NAME,keep_suffix=True)].add(int(r.PLAYER_ID))
        prior_path=ROOT/'logs/precompute_cache'/f'player_averages_{inputs.source_season}.json'
        self.prior_players={}
        if prior_path.exists():
            for r in json.loads(prior_path.read_text()):
                self.prior_players[int(r['PLAYER_ID'])]=r
                if 'PLAYER_NAME' in r:
                    self.name_ids.setdefault(normalize_name(r['PLAYER_NAME']),int(r['PLAYER_ID']))
                    exact[normalize_name(r['PLAYER_NAME'],keep_suffix=True)].add(int(r['PLAYER_ID']))
        # Stable identities may be used across seasons, but ambiguous
        # normalized names (e.g. father/son suffixes) are never guessed.
        from nba_api.stats.static import players as static_players
        identities=defaultdict(set)
        for player in static_players.get_players():
            identities[normalize_name(player['full_name'])].add(int(player['id']))
            exact[normalize_name(player['full_name'],keep_suffix=True)].add(int(player['id']))
        self.exact_ids={name:next(iter(ids)) for name,ids in exact.items() if len(ids)==1}
        for name,ids in identities.items():
            if len(ids)==1:self.name_ids.setdefault(name,next(iter(ids)))
        self.positions={}
        for pos in ('F','G','C'):
            for pid in self.sources.get('position_'+pos,{}):self.positions[int(pid)]=pos
        self.position_games=pd.DataFrame()
        if self.positions and not self.rich.empty:
            pg=self.logs.copy();pg['POS']=pg.PLAYER_ID.map(self.positions)
            cols=[c for c in 'PTS REB AST FG3M STL BLK TOV'.split() if c in pg]
            self.position_games=pg.dropna(subset=['POS']).groupby(['OPP_ID','GAME_ID','GAME_DATE','POS'],as_index=False)[cols].sum().sort_values('GAME_DATE')
        pbp_path=SOURCES/'pbp_player_games.json'
        self.pbp=pd.DataFrame(json.loads(pbp_path.read_text())) if pbp_path.exists() else pd.DataFrame()
        if len(self.pbp):self.pbp['game_date']=pd.to_datetime(self.pbp.game_date)
        self.pbp_players={int(pid):part.sort_values('game_date') for pid,part in self.pbp.groupby('player_id')} if len(self.pbp) else {}
        self.pbp_starters={int(tid):part.groupby(['game_date','game_id']).player_id.apply(set) for tid,part in self.pbp[self.pbp.starter].groupby('team_id')} if len(self.pbp) else {}
        self._pbp_cache={};self._cutoff_cache={}
        self.game_script=None
        script_path=SOURCES/'game_script'/f'{inputs.season}.joblib'
        if script_path.exists():
            import joblib
            self.game_script=joblib.load(script_path)
            if self.game_script['target_season']!=inputs.season or self.game_script['last_outcome_date']>=inputs.season[:4]+'-10-01':raise ValueError('Future or mismatched game-script model')
        self.injury_index=defaultdict(list)
        self.medical_index=defaultdict(list)
        for report in reports():
            stamp=pd.Timestamp(report['published_at'])
            grouped=defaultdict(list)
            for row in report['rows']:
                grouped[(normalize_name(row['team']),row['game_date'])].append(row)
                player=self.resolve_name(row['player'])
                if player is not None and row['status']=='Out' and 'injury' in row['reason'].lower():
                    self.medical_index[player].append((stamp,row['game_date']))
            for key,rows in grouped.items():self.injury_index[key].append((stamp,rows))
        self.lineups=defaultdict(list)
        for row in (self.sources.get('lineups',{}) if snapshot.get('lineup_team_complete')==30 else {}).values():
            for pid in re.findall(r'\d+',row['group_id']):self.lineups[int(pid)].append(row)

    def resolve_name(self,name):
        return self.exact_ids.get(normalize_name(name,keep_suffix=True),self.name_ids.get(normalize_name(name)))

    def prior(self,date):
        key=pd.Timestamp(date).strftime('%Y-%m-%d')
        if key not in self._date_cache:self._date_cache[key]=self.rich[self.rich.GAME_DATE<pd.Timestamp(date)] if not self.rich.empty else self.rich
        return self._date_cache[key]

    def role_minutes(self,pid,team,date):
        if {'PLAYER_ID','TEAM_ID','GAME_DATE'}.issubset(self.logs):
            past=self.logs[(self.logs.PLAYER_ID==pid)&(self.logs.TEAM_ID==team)&(self.logs.GAME_DATE<pd.Timestamp(date))].sort_values('GAME_DATE').tail(10)
            if len(past):return float(past.MIN.mean())
        return float(self.prior_players.get(pid,{}).get('MIN',0))

    def injuries(self,team,date):
        key=(int(team),pd.Timestamp(date).strftime('%Y-%m-%d'))
        live=getattr(self.inputs,'live_injury_context',None)
        if live and live['game_date']==key[1] and pd.Timestamp(live['published_at'])<=self.cutoff(date):
            if str(team) in live['teams']:return live['teams'][str(team)]
        if key in self._injury_cache:return self._injury_cache[key]
        from nba_api.stats.static import teams
        team_info=teams.find_team_name_by_id(int(team));norm=normalize_name(team_info['full_name'])
        aliases={norm}
        if team_info['abbreviation']=='LAC':aliases.add('laclippers')
        cutoff=self.cutoff(date)
        best=None
        for alias in aliases:
            for pub,candidates in self.injury_index.get((alias,key[1]),[]):
                if pub<cutoff and (best is None or pub>best[0]):best=(pub,candidates)
        result=None if best is None else best[1]
        self._injury_cache[key]=result
        return result

    def cutoff(self,date):
        if getattr(self.inputs,'prediction_cutoff',None) is not None:return pd.Timestamp(self.inputs.prediction_cutoff)
        gid=str(self.inputs.prediction_game_id or '')
        if gid not in self._cutoff_cache:
            cutoff=pd.Timestamp(date).tz_localize('America/New_York') if pd.Timestamp(date).tzinfo is None else pd.Timestamp(date)
            path=ROOT/'logs/referee_cache'/f'{gid}_summary.json'
            if path.exists():
                raw=json.loads(path.read_text());summary=raw.get('boxScoreSummary',raw)
                if summary.get('gameTimeUTC'):
                    scheduled=pd.Timestamp(summary['gameTimeUTC'])
                    if scheduled.tzinfo is not None:cutoff=scheduled-pd.Timedelta(hours=1)
            self._cutoff_cache[gid]=cutoff
        return self._cutoff_cache[gid]

    def records(self,date):
        key=pd.Timestamp(date).strftime('%Y-%m-%d')
        if key in self._rank_cache:return self._rank_cache[key]
        result={}
        for tid,games in self.inputs.team_games.items():
            prior=[g for g in games if self.inputs.season[:4]+'-10-01'<=str(g['game_date'])[:10]<key and g.get('pts') is not None and g.get('opp_pts') is not None]
            wins=sum(g['pts']>g['opp_pts'] for g in prior);losses=len(prior)-wins
            result[int(tid)]={'wins':wins,'losses':losses,'pct':wins/len(prior) if prior else .5,'def_rating':float(np.mean([g['def_rating'] for g in prior if g.get('def_rating') is not None])) if prior else None}
        self._rank_cache[key]=result
        return result


def apply_recovered(f,hist,target,pid,team,opp,date,inputs,missing):
    ctx=inputs.recovery;src=ctx.sources;pid=str(int(pid));team=str(int(team));opp=str(int(opp))
    missing.update(RECOVERED)
    def put(key,value):
        if value is not None and np.isfinite(float(value)):
            f[key]=float(value);missing.discard(key)
    def field(bucket,identity,column):return src.get(bucket,{}).get(str(identity),{}).get(column)
    def div(a,b):return a/b if a is not None and b is not None and b>0 else None
    if ctx.defender_evidence is not None:
        for key,value in ctx.defender_evidence.features(ctx,int(pid),int(opp),date).items():put(key,value)
    for key,bucket,identity,column in [
        ('team_pts_fb','team_misc',team,'PTS_FB'),('team_pts_off_tov','team_misc',team,'PTS_OFF_TOV'),('opp_pts_fb_allowed','team_misc',opp,'OPP_PTS_FB'),('opp_pts_off_tov_allowed','team_misc',opp,'OPP_PTS_OFF_TOV'),('opp_pts_paint','team_misc',opp,'PTS_PAINT'),
        ('rebound_contested_pct','player_rebounding',pid,'REB_CONTEST_PCT'),('post_touches_per_game','player_possessions',pid,'POST_TOUCHES'),('front_court_touches_per_game','player_possessions',pid,'FRONT_CT_TOUCHES'),('paint_touch_to_points','player_possessions',pid,'PTS_PER_PAINT_TOUCH'),('fouls_drawn_per_game','player_misc',pid,'PFD'),('crunch_time_usage','player_clutch_advanced',pid,'USG_PCT'),('late_clock_shot_frequency','clock_late',pid,'FGA_FREQUENCY'),('bench_strength','bench',team,'PTS')]:put(key,field(bucket,identity,column))
    q4=src.get('player_q4',{}).get(pid,{})
    if all(q4.get(k) is not None for k in ('FGA','FTA','TOV','MIN')):put('fourth_quarter_usage_rate',div((q4['FGA']+.44*q4['FTA']+q4['TOV'])*48,q4['MIN']))
    a=field('clock_early24',pid,'FGA_FREQUENCY');b=field('clock_very_early',pid,'FGA_FREQUENCY')
    if a is not None and b is not None:put('early_clock_shot_frequency',a+b)
    for key,column in [('player_age','age'),('player_height_inches','height_inches'),('player_weight','weight')]:put(key,inputs.precomputed.get('advanced',{}).get(int(pid),{}).get(column))
    for key,column,bucket in [('lg_pts_fb','PTS_FB','team_misc'),('lg_opp_pts_fb','OPP_PTS_FB','team_misc'),('lg_pts_off_tov','PTS_OFF_TOV','team_misc'),('lg_opp_pts_off_tov','OPP_PTS_OFF_TOV','team_misc'),('lg_fga','FGA','team_base'),('lg_fg3a','FG3A','team_base'),('lg_tov','TOV','team_base'),('lg_stl','STL','team_base')]:
        values=[r[column] for r in src.get(bucket,{}).values() if r.get(column) is not None]
        if len(values)==30:put(key,np.mean(values))
    base=list(src.get('team_base',{}).values())
    if len(base)==30:put('lg_fg_pct',div(sum(r.get('FGM',0)*r.get('GP',0) for r in base),sum(r.get('FGA',0)*r.get('GP',0) for r in base)))
    for zone in ('rim','paint','midrange','corner3','above_break3'):
        put('opp_'+zone+'_fg_pct_allowed',field('opponent_zones',opp,zone+'_fg_pct_allowed'))
    for zone in ('rim','midrange','corner3','above_break3'):
        key=zone+'_shot_quality_matchup';other='opp_'+zone+'_fg_pct_allowed';own=zone+'_fg_pct'
        if other not in missing and own not in missing:put(key,f[own]-f[other])
    for play in ('pnr','iso','spotup','transition','postup'):put('opp_'+play+'_ppp_allowed',field('opponent_synergy',opp,play+'_ppp_allowed'))
    for play,own in [('pnr','pnr_bh_ppp'),('iso','iso_ppp'),('spotup','spotup_ppp'),('transition','transition_ppp')]:
        other='opp_'+play+'_ppp_allowed'
        if own not in missing and other not in missing:put(play+'_matchup_advantage',f[own]-f[other])
    lineup_rows=ctx.lineups.get(int(pid),[])
    if lineup_rows:
        minutes=sum(r['minutes'] for r in lineup_rows)
        put('lineups_played_count',len(lineup_rows));put('top_lineup_minutes_pct',div(max(r['minutes'] for r in lineup_rows),minutes))
    # Full team schedules, including games the player missed.
    games=inputs.prior_team_games(int(opp),date)
    b2b=[];rested=[]
    for i,g in enumerate(games):
        if i==0:continue
        rest=(pd.Timestamp(g['game_date'])-pd.Timestamp(games[i-1]['game_date'])).days-1
        if rest==0:b2b.append(g)
        elif rest>=2:rested.append(g)
    for key,group,column in [('opp_b2b_def_rating',b2b,'def_rating'),('opp_b2b_pace',b2b,'pace'),('opp_b2b_pts_allowed',b2b,'opp_pts'),('opp_rested_def_rating',rested,'def_rating'),('opp_rested_pace',rested,'pace')]:
        values=[g[column] for g in group[-20:] if g.get(column) is not None]
        if values:put(key,np.mean(values))
    past=ctx.prior(date)
    if not past.empty:
        allowed=past[past.OPP_ID==int(opp)].tail(20)
        if len(allowed):put('def_fg_pct_allowed',div(allowed.FGM.sum(),allowed.FGA.sum()))
        venue=past[past.TEAM_ID==int(opp)]
        home_days=set(venue[venue.MATCHUP.str.contains('vs',regex=False)].GAME_DATE.dt.strftime('%Y-%m-%d'));away_days=set(venue[venue.MATCHUP.str.contains('@',regex=False)].GAME_DATE.dt.strftime('%Y-%m-%d'))
        home=[g['def_rating'] for g in games if str(g['game_date'])[:10] in home_days and g.get('def_rating') is not None];away=[g['def_rating'] for g in games if str(g['game_date'])[:10] in away_days and g.get('def_rating') is not None]
        if home and away:put('opp_def_rating_home_away_split',np.mean(home[-20:])-np.mean(away[-20:]))
        rotation=past[past.TEAM_ID==int(team)].tail(20)
        if len(rotation):
            appearances=ctx.logs[(ctx.logs.TEAM_ID==int(team))&ctx.logs.GAME_ID.isin(rotation.GAME_ID)]
            core=set(appearances.groupby('PLAYER_ID').GAME_ID.nunique().loc[lambda x:x>=10].index)
            recent=appearances[appearances.GAME_ID.isin(rotation.tail(5).GAME_ID)]
            if recent.MIN.sum()>0:put('lineup_continuity',recent[recent.PLAYER_ID.isin(core)].MIN.sum()/recent.MIN.sum())
            counts=appearances[appearances.PLAYER_ID!=int(pid)].groupby('PLAYER_ID').GAME_ID.nunique()
            current=set(recent.PLAYER_ID)-{int(pid)}
            values=[n for player,n in counts.items() if player in current]
            if values:put('new_teammate_games',min(values))
    own_games=inputs.prior_team_games(int(team),date)[-20:]
    margins=[abs(g['pts']-g['opp_pts']) for g in own_games if g.get('pts') is not None and g.get('opp_pts') is not None]
    if margins:put('blowout_game_pct',np.mean(np.array(margins)>15));put('close_game_pct',np.mean(np.array(margins)<=5))
    # Previous-season H2H is distinct from bounded current-season H2H.
    prev=ctx.previous_logs
    if not prev.empty:
        from nba_api.stats.static import teams
        abbrev=teams.find_team_name_by_id(int(opp))['abbreviation']
        games_prev=prev[(prev.PLAYER_ID==int(pid))&(prev.MATCHUP.str.split().str[-1]==abbrev)]
        if len(games_prev):
            if isinstance(target,(list,tuple)):put('vs_team_last_season_avg',games_prev[list(target)].sum(axis=1).mean())
            elif target in games_prev:put('vs_team_last_season_avg',games_prev[target].mean())
    # Standings are explicitly win-fraction ranks, not fabricated tie-breaks.
    east={1610612737,1610612738,1610612751,1610612766,1610612741,1610612739,1610612765,1610612754,1610612748,1610612749,1610612752,1610612753,1610612755,1610612761,1610612764}
    records=ctx.records(date)
    for tid,key in [(int(team),'team_conf_rank'),(int(opp),'opp_conf_rank')]:
        if tid not in records:continue
        conference=[r for t,r in records.items() if (t in east)==(tid in east)]
        if len(conference)==15:
            put(key,1+sum(r['pct']>records[tid]['pct'] for r in conference))
            if tid==int(team):
                cut=sorted(conference,key=lambda r:(r['pct'],r['wins']),reverse=True)[9];own=records[tid]
                put('games_back_from_playoff',max(0,(cut['wins']-own['wins']+own['losses']-cut['losses'])/2))
    # Position matchup deltas are computed from actual past opponents' boxes.
    position=ctx.positions.get(int(pid))
    if position and not ctx.position_games.empty:
        cache_key=(int(opp),position,str(pd.Timestamp(date).date()))
        if cache_key not in ctx._dvp_cache:
            pg=ctx.position_games;prior=pg[(pg.GAME_DATE<pd.Timestamp(date))&(pg.POS==position)];against=prior[prior.OPP_ID==int(opp)]
            values={}
            if len(against)>=3 and len(prior)>=30:
                values['dvp_gp']=len(against)
                for stat,name in [('PTS','pts'),('REB','reb'),('AST','ast'),('FG3M','fg3m'),('STL','stl'),('BLK','blk'),('TOV','tov')]:
                    league=prior[stat].mean();values['dvp_'+name+'_delta']=against.tail(20)[stat].mean()-league
                    if name in ('pts','reb','ast','fg3m'):values['dvp_'+name+'_delta_last5']=against.tail(5)[stat].mean()-league
                    if name=='pts':values['dvp_pts_delta_last10']=against.tail(10)[stat].mean()-league
            ctx._dvp_cache[cache_key]=values
        for key,value in ctx._dvp_cache[cache_key].items():put(key,value)
    # Deep injury/role features use explicit published reports only.
    for tid,prefix in [(int(team),'team'),(int(opp),'opp')]:
        injuries=ctx.injuries(tid,date)
        if injuries is None:continue
        outs=[];unresolved=False
        for row in injuries:
            if row['status']!='Out':continue
            player=ctx.resolve_name(row['player'])
            if player is None:unresolved=True;continue
            outs.append((player,ctx.role_minutes(player,tid,date)))
        if unresolved:continue # do not turn unresolved identities into healthy rosters
        put(prefix+'_key_players_out',sum(minutes>=20 for _,minutes in outs))
        if prefix=='opp':put('opp_injury_impact',min(1,sum(m for _,m in outs)/240))
        elif any(player!=int(pid) for player,_ in outs) and {'GAME_ID','TEAM_ID'}.issubset(ctx.logs):
            star=max((item for item in outs if item[0]!=int(pid)),key=lambda x:x[1])[0]
            roster=pd.concat([ctx.previous_logs,ctx.logs],ignore_index=True)
            usage=teammate_absence_usage(hist,roster,star,team,date)
            if usage is not None:put('usage_rate_with_star_out',usage)
    # Return trajectory is based on confirmed medical absence before a known
    # past appearance. It never infers an injury just from a calendar gap.
    cutoff=pd.Timestamp(date).tz_localize('America/New_York') if pd.Timestamp(date).tzinfo is None else pd.Timestamp(date)
    all_player=ctx.logs[(ctx.logs.PLAYER_ID==int(pid))&(ctx.logs.GAME_DATE<pd.Timestamp(date))] if 'PLAYER_ID' in ctx.logs else hist
    played_days=set(all_player.GAME_DATE.dt.strftime('%Y-%m-%d'))
    injury_days={day for published,day in ctx.medical_index.get(int(pid),[]) if published<cutoff and day<str(pd.Timestamp(date).date()) and day not in played_days}
    if injury_days and ctx.injuries(int(team),date) is not None:
        last=max(injury_days);returned=hist[hist.GAME_DATE>pd.Timestamp(last)]
        if len(returned):
            before=all_player[all_player.GAME_DATE<pd.Timestamp(last)]
            start=before.GAME_DATE.max().strftime('%Y-%m-%d') if len(before) else min(injury_days)
            spell={d for d in injury_days if start<=d<=last}
            put('games_since_return',len(returned));put('missed_games_before_return',len(spell))
    # Defensive strength classification must be before each historical game.
    buckets={'top':[],'bottom':[]}
    for _,game in hist.iterrows():
        old=ctx.records(game.GAME_DATE);op_abbrev=str(game.MATCHUP).split()[-1]
        from nba_api.stats.static import teams
        opponent=teams.find_team_by_abbreviation(op_abbrev)
        if not opponent or opponent['id'] not in old:continue
        rating=old[opponent['id']]['def_rating'];ratings=[r['def_rating'] for r in old.values() if r['def_rating'] is not None]
        if rating is None or len(ratings)<30:continue
        rank=1+sum(r<rating for r in ratings)
        if rank<=10:buckets['top'].append(game.PTS)
        if rank>=21:buckets['bottom'].append(game.PTS)
    for group,key in [('top','pts_vs_top10_defenses'),('bottom','pts_vs_bottom10_defenses')]:
        if buckets[group]:put(key,np.mean(buckets[group]))

    # Dated PBP rows are independently checked against NBA game boxscores.
    if not ctx.pbp.empty:
        key=(int(pid),str(pd.Timestamp(date).date()))
        if key not in ctx._pbp_cache:
            player_frame=ctx.pbp_players.get(int(pid),ctx.pbp.iloc[:0])
            ctx._pbp_cache[key]=player_frame[(player_frame.game_date<pd.Timestamp(date))&(player_frame.minutes>0)].tail(20)
        past_plays=ctx._pbp_cache[key]
        history_ids=set(hist['GAME_ID' if 'GAME_ID' in hist else 'Game_ID'].astype(str))
        past_plays=past_plays[past_plays.game_id.astype(str).isin(history_ids)]
        valid=past_plays[past_plays.points_valid] if len(past_plays) else past_plays
        if len(valid):
            for name,col in [('performance_when_leading','leading_points'),('performance_when_trailing','trailing_points'),('performance_when_tied','tied_points'),('performance_in_overtime','overtime_points')]:put(name,valid[col].mean())
            put('performance_by_score_differential',valid.leading_points.mean()-valid.trailing_points.mean())
        shots=past_plays[past_plays.shots_valid] if len(past_plays) else past_plays
        if len(shots):
            known=shots[shots.known_shot_distances==shots.fga]
            if len(known)>=3 and known.fga.sum()>0:put('avg_shot_distance',known.shot_distance_sum.sum()/known.fga.sum())
            put('and_one_frequency',div(shots.and_ones.sum(),shots.fgm.sum()))
        if len(past_plays):
            subs=past_plays.first_sub_minute.dropna()
            if len(subs):put('typical_substitution_minute',subs.mean())
            stints=past_plays[past_plays.stints_valid]
            if len(stints):
                put('garbage_time_minutes_pct',div(stints.garbage_seconds.sum(),stints.tracked_seconds.sum()))
                put('minutes_with_starting_lineup_pct',div(stints.starter_unit_seconds.sum(),stints.tracked_seconds.sum()))
        opp_starters=ctx.pbp_starters.get(int(opp),pd.Series(dtype=object))
        if len(opp_starters):opp_starters=opp_starters[opp_starters.index.get_level_values('game_date')<pd.Timestamp(date)].tail(6)
        units=[unit for unit in opp_starters if len(unit)==5]
        if len(units)>=2:put('opp_lineup_changes_last5',np.mean([5-len(a&b) for a,b in zip(units[:-1],units[1:])]))
    # Schedule metadata comes from the matched game's scoreboard, not PBP outcomes.
    gid=str(inputs.prediction_game_id or '')
    metadata=ctx.game_metadata.get(gid)
    if metadata is not None and 'national_tv' in metadata:put('national_tv_game',float(metadata['national_tv']))

    gid=str(inputs.prediction_game_id or '')
    if gid.startswith(('002','004','005')):
        playoff=gid.startswith('004');playin=gid.startswith('005')
        put('is_playoff',int(playoff));put('is_play_in',int(playin))
        put('playoff_home',int(playoff and f['is_home_game_num']==1));put('playoff_away',int(playoff and f['is_home_game_num']==0))
        if not playoff and not playin:
            for key in ('series_game_num','team_series_wins_in','opp_series_wins_in','is_elimination_game'):put(key,0)
        elif playoff:
            number=int(gid[-1]);put('series_game_num',number)
            if not ctx.logs.empty and 'WL' in ctx.logs:
                earlier=ctx.logs[(ctx.logs.TEAM_ID==int(team))&(ctx.logs.GAME_ID.astype(str).str.startswith(gid[:-1]))&(ctx.logs.GAME_DATE<pd.Timestamp(date))].drop_duplicates('GAME_ID')
                if len(earlier)==number-1:
                    wins=int((earlier.WL=='W').sum());losses=int((earlier.WL=='L').sum())
                    put('team_series_wins_in',wins);put('opp_series_wins_in',losses);put('is_elimination_game',int(wins==3 or losses==3))
        else:
            # Verified NBA schedule series codes:010/011=7vs8,
            #012/013=9vs10,020/021=final8th-seed contest.
            stage=gid[-4:-1];put('series_game_num',1)
            put('team_series_wins_in',0);put('opp_series_wins_in',0)
            if stage in ('010','011','012','013','020','021'):put('is_elimination_game',int(stage in ('012','013','020','021')))

    metadata=ctx.game_metadata.get(str(inputs.prediction_game_id or ''),{})
    arena=metadata.get('arena_id')
    if arena:
        idcol='GAME_ID' if 'GAME_ID' in hist else 'Game_ID'
        at=hist[hist[idcol].astype(str).map(lambda g:ctx.game_metadata.get(g,{}).get('arena_id'))==arena]
        if len(at):
            values=at[list(target)].sum(axis=1) if isinstance(target,(list,tuple)) else at[target]
            overall=hist[list(target)].sum(axis=1) if isinstance(target,(list,tuple)) else hist[target]
            put('player_vs_arena',values.mean()-overall.mean())
    if lineup_rows and not ctx.pbp.empty:
        key=('prior_starting_units',inputs.source_season)
        if key not in ctx._pbp_cache:
            start=inputs.source_season[:4]+'-10-01';end=str(int(inputs.source_season[:4])+1)+'-07-01'
            starts=ctx.pbp[ctx.pbp.starter&(ctx.pbp.game_date>=pd.Timestamp(start))&(ctx.pbp.game_date<pd.Timestamp(end))]
            ctx._pbp_cache[key]={tuple(sorted(group.player_id)) for _,group in starts.groupby(['team_id','game_id']) if len(group)==5}
        actual=ctx._pbp_cache[key]
        known=[u for u in lineup_rows if tuple(sorted(int(v) for v in re.findall(r'\d+',u['group_id']))) in actual and u['minutes']>0]
        if known:put('net_rating_with_starters',sum(u['net_rating']*u['minutes'] for u in known)/sum(u['minutes'] for u in known))

    expected=actual=0.;known=True
    for zone in ('rim','paint','midrange','corner3','above_break3'):
        share=f[zone+'_fga_pct'];eff=f[zone+'_fg_pct']
        observations=list(src.get('opponent_zones',{}).values())
        if zone+'_fga_pct' in missing or zone+'_fg_pct' in missing or len(observations)!=30 or any(r.get(zone+'_fga_allowed') is None or r.get(zone+'_fgm_allowed') is None for r in observations):known=False;break
        attempts=sum(r[zone+'_fga_allowed'] for r in observations)
        made=sum(r[zone+'_fgm_allowed'] for r in observations)
        if attempts<=0:known=False;break
        weight=1.5 if zone in ('corner3','above_break3') else 1.
        expected+=share*made/attempts*weight;actual+=share*eff*weight
    if known:put('shot_quality_vs_expected',actual-expected)

    if ctx.game_script and str(inputs.prediction_game_id or '').startswith('002') and f['is_home_game_num'] in (0,1):
        required=['team_off_rating','team_def_rating','team_pace','opp_off_rating','opp_def_rating','opp_pace']
        if not any(k in missing for k in required):
            home={'off_rating':f['team_off_rating'],'def_rating':f['team_def_rating'],'pace':f['team_pace'],'rest':inputs.team_context(int(team),date)['days_rest']}
            away={'off_rating':f['opp_off_rating'],'def_rating':f['opp_def_rating'],'pace':f['opp_pace'],'rest':inputs.team_context(int(opp),date)['days_rest']}
            if f['is_home_game_num']==0:home,away=away,home
            strength=(home['off_rating']+away['def_rating']-away['off_rating']-home['def_rating'])/2
            vector=pd.DataFrame([[strength,abs(strength),strength**2,(home['pace']+away['pace'])/2,min(7,home['rest'])-min(7,away['rest'])]],columns=ctx.game_script['columns'])
            for key,bundle in ctx.game_script['models'].items():
                if not bundle['qualified']:continue
                logit=bundle['model'].decision_function(vector).reshape(-1,1)
                put(key,bundle['calibrator'].predict_proba(logit)[0,1])
