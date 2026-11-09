import numpy as np
import pandas as pd
from src.defender_assignment import AssignmentState,FEATURES,state_before,choose


def boxes():
    return pd.DataFrame([dict(PLAYER_ID=p,TEAM_ID=t,GAME_DATE=pd.Timestamp(day),MIN=30,
        FGA=12,FG3A=4,REB=6,AST=3,PTS=20) for day in ['2025-01-01','2025-01-02','2025-01-03']
        for p,t in [(1,10),(2,20),(3,20)]])


def matches():
    return pd.DataFrame([dict(personIdOff=1,personIdDef=2,partialPossessions=30.,
        game_date=pd.Timestamp('2025-01-03'))])


def test_candidate_can_cover_new_pair_but_never_self_or_wrong_team():
    state=state_before(matches().iloc[:0],boxes(),'2025-01-04')
    ids,x=state.candidates(1,20,'2025-01-04')
    assert ids==[2,3] and len(x[0])==len(FEATURES)
    assert x[0][0]==0 # no fabricated direct matchup
    state.latest[1]=(20,pd.Timestamp('2025-01-03')) # old team after trade
    assert state.candidates(1,20,'2025-01-04')[0]==[2,3]
    state.latest[2]=(30,pd.Timestamp('2025-01-03'))
    assert state.candidates(1,20,'2025-01-04')[0]==[3]


def test_same_day_and_future_boxes_and_assignments_cannot_change_features():
    logs=boxes();rows=matches()
    state=state_before(rows,logs,'2025-01-04');ids,x=state.candidates(1,20,'2025-01-04')
    poison=logs.copy();poison['GAME_DATE']=pd.Timestamp('2025-01-04');poison['PTS']=99999
    future=rows.copy();future['game_date']=pd.Timestamp('2025-01-04');future['partialPossessions']=99999
    again=state_before(pd.concat([rows,future]),pd.concat([logs,poison]),'2025-01-04')
    ids2,x2=again.candidates(1,20,'2025-01-04')
    assert ids==ids2
    np.testing.assert_allclose(x,x2,equal_nan=True)


def test_threshold_abstains_without_changing_candidate_ranking():
    class Model:
        def predict_proba(self,x):return np.array([[.6,.4],[.4,.6]])
    assert choose(Model(),[2,3],[[0],[1]],.61) is None
    assert choose(Model(),[2,3],[[0],[1]],.6)==3
    assert choose(Model(),[],[],.1) is None


def test_inadequate_or_expired_history_does_not_invent_candidates():
    state=AssignmentState()
    assert state.candidates(1,20,'2025-01-04')==([],[])
    state=state_before(matches(),boxes(),'2025-01-04')
    assert state.candidates(1,20,'2027-01-04')==([],[])


def test_feature_builder_preserves_prior_season_roster_evidence():
    from types import SimpleNamespace
    from src.defender_evidence import DefenderEvidence
    rows=pd.DataFrame([dict(personIdOff=1,personIdDef=2,partialPossessions=30.,
        defTeamId=20,game_date=pd.Timestamp('2025-04-01'),game_id='0022400001',playerPoints=20)])
    evidence=DefenderEvidence({'rows':rows,'qualifications':{'2025-26':{
        'qualified':True,'last_outcome_date':'2025-04-01','rule':{'regular_season_only':True}}}})
    previous=pd.DataFrame([dict(PLAYER_ID=2,TEAM_ID=20,GAME_DATE='2025-04-01',GAME_ID='0022400001')])
    current=previous.iloc[:0].copy()
    ctx=SimpleNamespace(inputs=SimpleNamespace(season='2025-26',prediction_game_id='0022500001',
        team_games={},precomputed={'advanced':{2:{'age':29,'def_rating':105}}}),
        previous_logs=previous,logs=current,injuries=lambda *a:None)
    out=evidence.features(ctx,1,20,'2025-10-21')
    assert out['primary_defender_age']==29 and out['primary_defender_rating']==105
    # A real later transfer wins over the previous-season appearance.
    ctx.logs=pd.DataFrame([dict(PLAYER_ID=2,TEAM_ID=30,GAME_DATE='2025-10-20',GAME_ID='0022500000')])
    ctx._defender_rosters={}
    assert evidence.features(ctx,1,20,'2025-10-21')=={}
