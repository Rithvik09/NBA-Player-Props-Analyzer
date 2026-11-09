"""Serving adapter for the versioned audited feature contract.

Fail closed when the dated team cache is absent; using a latest snapshot would
silently reintroduce the training/serving mismatch this contract removes.
"""
from functools import lru_cache
from copy import copy
from pathlib import Path
import os
import time
import pandas as pd
from src.audited_features import build_audited_features, POLICY_VERSION, SPECS
from src.historical_inputs import HistoricalInputs,load_season_logs

TARGETS={'points':'PTS','assists':'AST','rebounds':'REB','steals':'STL','blocks':'BLK','turnovers':'TOV','three_pointers':'FG3M','pts_reb':['PTS','REB'],'pts_ast':['PTS','AST'],'ast_reb':['AST','REB'],'pts_ast_reb':['PTS','AST','REB'],'stl_blk':['STL','BLK'],'double_double':'PTS','triple_double':'PTS'}

@lru_cache(maxsize=2)
def _inputs(season,cache_mtime,bucket):
    from scripts.train_models import _load_precomputed_player_data
    path=Path(__file__).resolve().parents[1]/'logs/points_pilot_cache'/f'players_{season}.json'
    raw=_load_precomputed_player_data(os.environ['DATABASE_URL'])
    return HistoricalInputs(season,raw,load_season_logs(season,include_postseason=True),audited=True)


def prepare(player_stats,player_context):
    date=pd.Timestamp(player_stats.get('_prediction_date') or pd.Timestamp.now().normalize())
    year=date.year if date.month>=10 else date.year-1
    season=f'{year}-{str(year+1)[2:]}'
    inputs=player_stats.get('_audited_inputs')
    if inputs is None:
        path=Path(__file__).resolve().parents[1]/'logs/points_pilot_cache'/f'players_{season}.json'
        if not path.exists():raise ValueError('Audited serving requires a dated team-player cache for '+season)
        inputs=_inputs(season,path.stat().st_mtime,int(time.time()//300))
    if inputs.season!=season:raise ValueError('Audited input season does not match prediction date')
    hist=pd.DataFrame(player_stats.get('_boxscores') or [])
    if hist.empty:raise ValueError('Audited serving requires raw prior player boxscores')
    prop=player_stats['_prop_type'];target=TARGETS[prop]
    team_id=player_stats.get('_team_id') or player_context.get('team_id')
    opp=player_stats['_opponent_team_id'];pid=player_stats['_player_id']
    if not team_id or not opp:raise ValueError('Audited serving requires known team and opponent')
    # Historical replay may supply a validated game id. Future crews without
    # a dated tendency record use the same documented warmup defaults.
    inputs=copy(inputs)
    inputs.recovery=copy(inputs.recovery)
    inputs.recovery.inputs=inputs
    inputs.recovery._injury_cache={}
    inputs.recovery._cutoff_cache={}
    if player_stats.get('_audited_inputs') is None and date.date()==pd.Timestamp.now(tz='America/New_York').date():
        attach_live_injuries(inputs,date)
    inputs.prediction_game_id=player_stats.get('_prediction_game_id')
    features,gaps=build_audited_features(hist,target,pid,team_id,opp,date,player_stats.get('is_home'),inputs)
    features['_feature_policy']=POLICY_VERSION
    features['_missing_features']=gaps
    return features


def enforce(features,policy):
    if policy!=POLICY_VERSION:return features
    if features.get('_feature_policy')!=POLICY_VERSION:
        raise ValueError('Audited model requires audited point-in-time feature preparation')
    clean=dict(features)
    for key,spec in SPECS.items():
        if spec.status!='active':clean[key]=0.
    for key in features.get('_missing_features',[]):
        if key in SPECS and SPECS[key].status=='active':clean[key]=float('nan')
    return clean


def attach_live_injuries(inputs,date):
    """Current JSON observations share the historical availability semantics.

    Source season validation lives in the JSON collector. A failed report
    remains missing. Never scrape an unversioned HTML fallback into ML inputs.
    """
    import json,logging
    from nba_api.stats.static import teams
    from src.historical_injuries import normalize_name
    from src.live_injuries import _fetch_espn_json
    try:
        data,observed=_live_injury_snapshot(int(time.time()//300))
        names={normalize_name(t['full_name']):str(t['id']) for t in teams.get_teams()}
        names['laclippers']='1610612746'
        mapping={}
        for name,rows in data.items():
            tid=names.get(normalize_name(name))
            if tid is None:continue
            mapping[tid]=[{'player':r['player'],'status':r['status'],'reason':r.get('injury',''),'team':name,'game_date':str(date.date())} for r in rows]
        now=pd.Timestamp.now(tz='America/New_York')
        inputs.live_injury_context={'published_at':observed.isoformat(),'game_date':str(date.date()),'teams':mapping,'source':'ESPN season-validated JSON observation'}
        inputs.prediction_cutoff=now
        # Preserve observations for later replay rather than overwriting a
        # single current snapshot. Reports with no team coverage stay missing.
        path=Path(__file__).resolve().parents[1]/'logs/feature_recovery/live_injuries'
        path.mkdir(parents=True,exist_ok=True)
        output=path/(observed.strftime('%Y%m%d_%H%M%S')+'.json')
        if not output.exists():output.write_text(json.dumps(inputs.live_injury_context))
    except Exception as exc:
        logging.getLogger(__name__).warning('Audited live injury source unavailable (%s)',type(exc).__name__)


@lru_cache(maxsize=1)
def _live_injury_snapshot(bucket):
    from src.live_injuries import _fetch_espn_json
    data=_fetch_espn_json()
    return data,pd.Timestamp.now(tz='America/New_York')
