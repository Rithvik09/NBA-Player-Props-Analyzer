"""Regression tests for actual recovery, missingness, and temporal boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from src.audited_features import SPECS,INACTIVE_CANDIDATES,POLICY_VERSION,build_audited_features
from src.recovered_features import RECOVERED,DUPLICATES,BLOCKED
from src.historical_inputs import HistoricalInputs,PLAYER_SOURCES,SNAPSHOT_SOURCES
from src.ml_features import build_feature_vector,build_classifier_vector
from tests.test_feature_defaults import _canned_game_log

BOS=1610612738;LAL=1610612747


def inputs(monkeypatch,tmp_path,sources=None,reports=None):
    import src.recovered_features as recovery
    monkeypatch.setattr(recovery,'SOURCES',tmp_path)
    monkeypatch.setattr(recovery,'reports',lambda:reports or [])
    if sources is not None:(tmp_path/'snapshots_2023-24.json').write_text(json.dumps({'source_season':'2023-24','sources':sources}))
    logs=_canned_game_log().rename(columns={'Game_ID':'GAME_ID'})
    logs['GAME_DATE']=pd.to_datetime(logs.GAME_DATE);logs['PLAYER_ID']=2544;logs['PLAYER_NAME']='LeBron James';logs['TEAM_ID']=BOS
    other=logs.copy();other.PLAYER_ID=222;other.PLAYER_NAME='Player Other';other.MIN=34
    logs=pd.concat([logs,other],ignore_index=True)
    raw={key:{} for key in PLAYER_SOURCES+SNAPSHOT_SOURCES};raw.update(game_officials={},referee_history={},game_team_stats={BOS:[{'game_date':'2025-01-18','pace':100,'off_rating':110,'def_rating':108,'pts':100,'opp_pts':90}],LAL:[{'game_date':'2025-01-19','pace':103,'off_rating':112,'def_rating':109,'pts':90,'opp_pts':100}]})
    provider=HistoricalInputs('2024-25',raw,logs,audited=True)
    provider.prediction_game_id='0022400999'
    return logs,raw,provider


def build(logs,provider,date='2025-01-21'):
    return build_audited_features(logs,'PTS',2544,BOS,LAL,date,True,provider)


def test_every_original_inactive_feature_has_exactly_one_disposition():
    original={key for names in INACTIVE_CANDIDATES.values() for key in names.split()}
    assert len(original)==162
    assert set(RECOVERED)|set(DUPLICATES)|set(BLOCKED)==original
    assert not(set(RECOVERED)&set(DUPLICATES) or set(RECOVERED)&set(BLOCKED) or set(DUPLICATES)&set(BLOCKED))
    assert all(SPECS[k].status=='active' for k in RECOVERED)
    assert all(SPECS[k].status=='duplicate' for k in DUPLICATES)
    assert all(SPECS[k].status=='source_blocked' for k in BLOCKED)


def test_missing_report_survives_both_regression_and_classifier_vectors():
    from scripts.train_models import Example,_example_raw
    raw={'recent_avg':20,'season_avg':21,'team_key_players_out':0,'_feature_policy':POLICY_VERSION,'_missing_features':['team_key_players_out']}
    X=build_feature_vector(raw).X
    assert np.isnan(X.team_key_players_out.iloc[0])
    ex=Example('points',X,20,pd.Timestamp('2025-01-21'),feature_policy=POLICY_VERSION)
    Xc=build_classifier_vector(_example_raw(ex),19.5).X
    assert np.isnan(Xc.team_key_players_out.iloc[0])
    assert Xc.line.iloc[0]==19.5
    assert build_feature_vector({'team_key_players_out':0}).X.team_key_players_out.iloc[0]==0


def test_a_reported_gap_is_nan_whatever_the_policy_is_called():
    """Missingness must not depend on a policy name being on a list.

    build_feature_vector fillna(0)s the whole frame and only then restores the
    reported gaps to NaN. That restore used to be gated on a hardcoded tuple of
    policy names, so bumping POLICY_VERSION without editing that tuple silently
    turned every missing feature into 0.0 -- a healthy roster, a zero shot rate
    -- in the serving path, which is this project's signature defect rather
    than a crash it would notice.

    Only the audited producers ever set _missing_features
    (audited_runtime.prepare and the two train_models sites), so honouring the
    key whenever it is present is equivalent for every real caller and cannot
    go stale at the next bump.
    """
    raw={'recent_avg':20,'season_avg':21,'team_key_players_out':0,
         '_feature_policy':'point_in_time_v999','_missing_features':['team_key_players_out']}
    assert np.isnan(build_feature_vector(raw).X.team_key_players_out.iloc[0]), \
        'a reported gap was served as 0.0 because its policy name was unrecognised'


def test_an_absent_missing_list_still_fills_with_defaults():
    """The mirror: no reported gaps means no NaN, whatever the policy."""
    raw={'recent_avg':20,'season_avg':21,'team_key_players_out':0,
         '_feature_policy':'point_in_time_v999'}
    assert build_feature_vector(raw).X.team_key_players_out.iloc[0]==0


def test_real_opponent_zone_zero_is_not_replaced_by_league_average():
    from src.precompute_jobs import parse_opponent_zone_frame
    zones=['Restricted Area','In The Paint (Non-RA)','Mid-Range','Corner 3','Above the Break 3']
    row={('','TEAM_ID'):BOS}
    for zone in zones:row[(zone,'OPP_FGM')]=0;row[(zone,'OPP_FGA')]=10
    result=parse_opponent_zone_frame(pd.DataFrame([row]))[0]
    assert result['rim_fg_pct_allowed']==0
    assert result['corner3_fga_allowed']==10
    with pytest.raises(ValueError,match='schema'):
        parse_opponent_zone_frame(pd.DataFrame([{'TEAM_ID':BOS,'FG_PCT':.4}]))


def test_actual_source_values_and_zone_interactions(monkeypatch,tmp_path):
    sources={'opponent_zones':{str(LAL):{z+'_fg_pct_allowed':.4 for z in ('rim','paint','midrange','corner3','above_break3')}},'player_rebounding':{'2544':{'REB_CONTEST_PCT':.47}},'player_possessions':{'2544':{'POST_TOUCHES':4,'FRONT_CT_TOUCHES':29,'PTS_PER_PAINT_TOUCH':1.7}},'player_q4':{'2544':{'FGA':3,'FTA':2,'TOV':1,'MIN':8}}}
    logs,raw,provider=inputs(monkeypatch,tmp_path,sources)
    provider.precomputed['shot_zones']={2544:{'rim_fg_pct':.6},(2544,'2024-25'):{'rim_fg_pct':.6}}
    f,gaps=build(logs,provider)
    assert f['opp_rim_fg_pct_allowed']==.4
    assert f['rim_shot_quality_matchup']==pytest.approx(.2)
    assert f['rebound_contested_pct']==.47
    assert f['post_touches_per_game']==4 and f['front_court_touches_per_game']==29
    assert f['paint_touch_to_points']==1.7
    assert f['fourth_quarter_usage_rate']==pytest.approx((3+.44*2+1)/8*48)
    assert 'fourth_quarter_usage_rate' not in gaps


def test_injuries_use_publication_cutoff_not_later_upgrade(monkeypatch,tmp_path):
    row={'game_date':'2025-01-21','matchup':'LAL@BOS','team':'Boston Celtics','player':'Other, Player','status':'Out','reason':'Injury/Illness - Ankle'}
    earlier={'published_at':'2025-01-21T17:30:00-05:00','rows':[row]}
    later={'published_at':'2025-01-21T19:30:00-05:00','rows':[{**row,'status':'Available'}]}
    logs,raw,provider=inputs(monkeypatch,tmp_path,reports=[earlier,later])
    provider.prediction_cutoff=pd.Timestamp('2025-01-21T18:00:00-05:00')
    f,gaps=build(logs,provider)
    assert f['team_key_players_out']==1
    assert 'team_key_players_out' not in gaps
    assert 'opp_key_players_out' in gaps # absent report is not a healthy opponent
    provider.prediction_cutoff=pd.Timestamp('2025-01-21T20:00:00-05:00');provider.recovery._injury_cache.clear()
    after,_=build(logs,provider)
    assert after['team_key_players_out']==0


def test_current_and_future_outcomes_cannot_change_recovered_inputs(monkeypatch,tmp_path):
    logs,raw,provider=inputs(monkeypatch,tmp_path)
    before,gaps=build(logs,provider)
    changed=logs.copy();future=changed.GAME_DATE>=pd.Timestamp('2025-01-21')
    for col in ('PTS','MIN','FGA','FGM','REB','AST'):changed.loc[future,col]=9999
    poisoned=deepcopy(raw)
    for tid in (BOS,LAL):poisoned['game_team_stats'][tid].append({'game_date':'2025-01-21','pace':9999,'off_rating':9999,'def_rating':9999,'pts':9999,'opp_pts':0})
    after_source=HistoricalInputs('2024-25',poisoned,changed,audited=True);after_source.prediction_game_id=provider.prediction_game_id
    after,after_gaps=build(changed,after_source)
    assert before==after
    assert set(gaps)==set(after_gaps)


def test_postseason_series_state_excludes_target_and_future_result(monkeypatch,tmp_path):
    logs,raw,provider=inputs(monkeypatch,tmp_path)
    post=logs[(logs.PLAYER_ID==2544)].iloc[:5].copy()
    post['GAME_ID']=[f'004240011{i}' for i in range(1,6)]
    post['GAME_DATE']=pd.to_datetime(['2025-04-16','2025-04-18','2025-04-20','2025-04-22','2025-04-24'])
    post['WL']=['W','L','W','L','L']
    full=pd.concat([logs,post],ignore_index=True)
    source=HistoricalInputs('2024-25',raw,full,audited=True);source.prediction_game_id='0042400114'
    f,_=build(full,source,'2025-04-22')
    assert f['is_playoff']==f['playoff_home']==1
    assert f['series_game_num']==4 and f['team_series_wins_in']==2 and f['opp_series_wins_in']==1
    post.loc[post.GAME_DATE>=pd.Timestamp('2025-04-22'),'WL']='W'
    new=pd.concat([logs,post],ignore_index=True);source2=HistoricalInputs('2024-25',raw,new,audited=True);source2.prediction_game_id='0042400114'
    after,_=build(new,source2,'2025-04-22')
    for key in ('team_series_wins_in','opp_series_wins_in','is_elimination_game'):assert f[key]==after[key]


@pytest.mark.parametrize('code,elimination',[('010',0),('011',0),('012',1),('013',1),('020',1),('021',1)])
def test_playin_stage_codes_from_verified_historical_schedule(monkeypatch,tmp_path,code,elimination):
    logs,raw,provider=inputs(monkeypatch,tmp_path)
    # NBA game IDs have an additional zero before the series code.
    provider.prediction_game_id='005240'+code+'1'
    f,_=build(logs,provider)
    assert f['is_play_in']==1 and f['is_elimination_game']==elimination


def test_future_trained_game_script_bundle_is_rejected(monkeypatch,tmp_path):
    import joblib
    from src.recovered_features import RecoveryContext
    logs,raw,provider=inputs(monkeypatch,tmp_path)
    path=tmp_path/'game_script';path.mkdir();(path/'2024-25.joblib').write_bytes(b'test')
    monkeypatch.setattr(joblib,'load',lambda p:{'target_season':'2024-25','last_outcome_date':'2025-01-01'})
    with pytest.raises(ValueError,match='Future'):RecoveryContext(provider,logs)


def test_pdf_header_timestamp_and_header_only_footer(monkeypatch,tmp_path):
    from src.historical_injuries import parse_pdf
    def word(text,x,y):return {'text':text,'x0':x,'top':y,'bottom':y+8}
    headers=[word(t,x,60) for t,x in [('GameDate',10),('GameTime',90),('Matchup',160),('Team',240),('PlayerName',360),('CurrentStatus',500),('Reason',600)]]
    body=[word(t,x,90) for t,x in [('01/15/2023',10),('07:00',90),('LAL@BOS',160),('Boston Celtics',240),('Other, Player',360),('Out',500),('Injury/Illness',600)]]
    class Page:
        width=842;height=595
        def __init__(self,words):self.words=words
        def extract_text(self,**kw):return 'Injury Report: 01/15/23 05:30 PM'
        def extract_words(self,**kw):return self.words
    class PDF:
        pages=[Page(headers+body),Page([])]
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setitem(sys.modules,'pdfplumber',SimpleNamespace(open=lambda p:PDF()))
    result=parse_pdf(tmp_path/'Injury-Report_2023-01-15_05PM.pdf')
    assert result['published_at']=='2023-01-15T17:30:00-05:00'
    assert len(result['rows'])==1 and result['rows'][0]['status']=='Out'


def pbp_fixture():
    players=[{'personId':pid,'teamId':10 if pid<=5 else 20,'position':'G','minutes':48.,'points':25 if pid==1 else 1 if pid==6 else 0,'fieldGoalsAttempted':12 if pid==1 else 0,'fieldGoalsMade':12 if pid==1 else 0,'familyName':f'Player{pid}','nameI':f'P{pid}'} for pid in range(1,11)]
    plays=[];score=[0,0]
    def action(pid,period,clock,points=0,field_goal=False,kind='Event',subtype=''):
        tid=10 if pid<=5 else 20
        if points:score[0 if tid==10 else 1]+=points
        plays.append({'personId':pid,'teamId':tid,'period':period,'clock':clock,'location':'h' if tid==10 else 'v','isFieldGoal':int(field_goal),'shotResult':'Made' if field_goal else '', 'shotDistance':2 if field_goal else 0,'scoreHome':str(score[0]),'scoreAway':str(score[1]),'actionType':kind,'subType':subtype,'description':''})
    for period in range(1,5):
        for pid in range(1,11):action(pid,period,'PT12M00.00S')
        if period==1:
            action(1,1,'PT12M00.00S',2,True,'Made Shot')
            action(6,1,'PT12M00.00S',1,False,'Free Throw','Free Throw 1 of 2')
        if period==4:
            for _ in range(11):action(1,4,'PT06M00.00S',2,True,'Made Shot')
            action(1,4,'PT06M00.00S',1,False,'Free Throw','Free Throw 1 of 1')
        action(1,period,'PT00M00.00S')
    return {'game_id':'0022400001','players':players,'plays':plays}


def test_pbp_scoring_states_and_garbage_stints_are_actual_measurements():
    from src.pbp_features import analyze_game
    rows=analyze_game(pbp_fixture(),'2025-01-21');home=next(r for r in rows if r['player_id']==1);away=next(r for r in rows if r['player_id']==6)
    assert all(r['points_valid'] and r['shots_valid'] and r['stints_valid'] for r in rows)
    assert home['tied_points']==2 and home['leading_points']==23
    assert away['trailing_points']==1
    assert home['garbage_seconds']==360
    assert home['tracked_seconds']==2880 and home['starter_unit_seconds']==2880
    assert home['shot_distance_sum']==24 and home['and_ones']==1


def test_invalid_points_or_ambiguous_substitutions_are_not_silently_accepted():
    from src.pbp_features import analyze_game
    data=pbp_fixture();data['players'][0]['points']=100
    rows=analyze_game(data,'2025-01-21')
    assert not next(r for r in rows if r['player_id']==1)['points_valid']
    data=pbp_fixture();data['plays'][10]['actionType']='Substitution';data['plays'][10]['description']='Unknown FOR Unknown'
    rows=analyze_game(data,'2025-01-21')
    assert not any(r['stints_valid'] for r in rows)
    assert all(r['garbage_seconds'] is None for r in rows)
