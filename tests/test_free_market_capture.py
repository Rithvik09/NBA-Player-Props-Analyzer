from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src import free_market_capture as feed
from src.source_evidence import market_features
from src.defender_evidence import expected_defender, select_assignment
from scripts.qualify_defender_rules import qualify


def fixture():
    observed='2026-10-25T12:00:00+00:00'
    roster={'season':'2026-27','fetched_at':datetime.fromisoformat(observed).timestamp()-10,
            'players':[{'team_id':1610612747,'nba_id':1,'full_name':'Player One'}]}
    games=[{'gameId':'0022600112','gameTimeUTC':'2026-10-25T21:00:00Z',
            'homeTeam':{'teamId':1610612762},'awayTeam':{'teamId':1610612747}}]
    snapshot={'observed_at':observed,'payload':{'id':'a'*32,'home_team':'Utah Jazz',
        'away_team':'Los Angeles Lakers','commence_time':'2026-10-25T21:10:00Z',
        'bookmakers':[{'key':'a','markets':[{'key':'player_points','last_update':observed,
            'outcomes':[{'name':side,'description':'Player One','point':20.5,'price':-110}
                        for side in ['Over','Under']]}]}]}}
    return snapshot,games,roster


def test_normalization_requires_asof_roster_and_official_game():
    s,g,r=fixture()
    out=feed.normalize(s,g,r)
    assert len(out['props'])==1 and out['props'][0]['game_id']=='0022600112'
    assert out['props'][0]['is_opening'] is False
    assert feed.normalize(s,g,{**r,'fetched_at':r['fetched_at']+20})['props']==[]
    assert feed.normalize(s,g,{**r,'season':'2025-26'})['props']==[]
    with pytest.raises(ValueError,match='unique'):feed.normalize(s,g+g,r)
    late=deepcopy(s);late['observed_at']='2026-10-25T21:05:00Z'
    with pytest.raises(ValueError,match='tipoff'):feed.normalize(late,g,r)


def test_missing_pair_and_future_timestamp_are_rejected():
    s,g,r=fixture()
    s['payload']['bookmakers'][0]['markets'][0]['outcomes'][1]['point']=21.5
    assert feed.normalize(s,g,r)['props']==[]
    s,g,r=fixture();s['payload']['bookmakers'][0]['markets'][0]['last_update']='2026-10-25T12:01:00Z'
    assert feed.normalize(s,g,r)['props']==[]


def test_last_seen_timestamp_does_not_claim_stale_or_opening():
    s,g,r=fixture();quote=feed.normalize(s,g,r)['props'][0]
    result=market_features([quote,dict(quote,bookmaker='b',line=21.5)],
        cutoff=s['observed_at'],game_id=g[0]['gameId'],player_id=1,prop_type='points',decision_line=20.5)
    assert result=={'bookmaker_count':2.,'market_consensus_std':.5,'implied_over_prob':.5}


def test_credit_reservations_survive_network_failure(tmp_path,monkeypatch):
    def get(url,**kwargs):
        if url.endswith('/events'):
            return SimpleNamespace(ok=True,headers={'x-requests-remaining':'100'},json=lambda:[{
                'id':'a'*32,'commence_time':'2099-01-01T00:00:00Z'}])
        raise feed.requests.ConnectionError('SECRET URL')
    monkeypatch.setattr(feed.requests,'get',get)
    for _ in range(10):
        with pytest.raises(RuntimeError,match='^Odds request failed: ConnectionError$'):
            feed.capture('secret',tmp_path)
    with pytest.raises(RuntimeError,match='ceiling'):feed.capture('secret',tmp_path)
    assert json.loads((tmp_path/'budget.json').read_text())['reserved_credits']==20


def test_account_reserve_blocks_before_metered_call(tmp_path,monkeypatch):
    monkeypatch.setattr(feed.requests,'get',lambda *a,**k:SimpleNamespace(ok=True,
        headers={'x-requests-remaining':'51'}))
    with pytest.raises(RuntimeError,match='reserve'):feed.capture('secret',tmp_path)
    assert not (tmp_path/'budget.json').exists()


def test_decayed_regular_season_assignment_matches_shared_selector():
    rows=pd.DataFrame([{'personIdOff':1,'personIdDef':d,'defTeamId':20,'game_id':gid,
        'game_date':pd.Timestamp(day),'partialPossessions':p} for day,gid,d,p in [
        ('2025-01-01','0021',2,80),('2025-01-31','0022',3,60),
        ('2025-01-31','0041',2,9999),('2025-02-01','0023',2,9999)]])
    rule={'regular_season_only':True,'half_life_days':30,'min_share':.5,'min_margin':.15}
    counts={2:80*.5**(31/30),3:60*.5**(1/30)}
    assert select_assignment(counts,140,{2:20,3:20},20,rule)==3
    assert expected_defender(rows,player_id=1,opponent_id=20,day='2025-02-01',
                             latest_teams={2:20,3:20},rule=rule)==3


def test_rule_selection_cannot_use_qualification_outcomes():
    dev=[{'date':'2024-11-01','rule_id':i,'correct':i==0,'baseline_correct':False}
         for i in [0,1] for _ in range(100)]
    held=[{'date':'2025-02-01','rule_id':i,'correct':i==1,'baseline_correct':False}
          for i in [0,1] for _ in range(100)]
    q=qualify(dev+held)['2025-26']
    assert q['rule_id']==0 and not q['qualified']
    for row in held:row['correct']=not row['correct']
    q=qualify(dev+held)['2025-26']
    assert q['rule_id']==0 and q['qualified']
