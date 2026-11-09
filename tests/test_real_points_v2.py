import copy
import json
from pathlib import Path
import pandas as pd
import pytest
from scripts.qualify_real_points_v2 import verified_start
from scripts.build_real_points_training import validate_record


def evidence():
    s={'gameId':'1','gameTimeUTC':'2024-03-17T00:30:00Z'}
    p={'game':{'gameId':'1','actions':[{'period':1,'clock':'PT12M00.00S','actionType':'period','subType':'start','description':'Start of 1st Period (8:41 PM EST)'}]}}
    e={'plays':[{'period':{'number':1},'clock':{'displayValue':'12:00'},'type':{'text':'Jumpball'},'homeScore':0,'awayScore':0,'wallclock':'2024-03-17T00:41:08Z'}]}
    return s,p,e


def test_actual_start_uses_conservative_independently_verified_bound():
    assert verified_start(*evidence()) == pd.Timestamp('2024-03-17T00:41:00Z')


@pytest.mark.parametrize('failure',['wrong_game','later_play','conflict','score'])
def test_bad_timing_fails_closed(failure):
    s,p,e=evidence()
    if failure=='wrong_game':p['game']['gameId']='2'
    if failure=='later_play':e['plays'][0]['clock']['displayValue']='11:59'
    if failure=='conflict':e['plays'][0]['wallclock']='2024-03-17T00:43:00Z'
    if failure=='score':e['plays'][0]['homeScore']=2
    with pytest.raises(ValueError):verified_start(s,p,e)


def test_existing_scheduled_guard_stays_strict():
    s,_,_=evidence()
    q={'game_date':'2024-03-16','market':'player_points','captured_at':'2024-03-17T00:35:38Z'}
    with pytest.raises(ValueError,match='precede official'):validate_record(q,s,set())


def test_actual_bound_still_rejects_inplay_and_protected_season():
    s,p,e=evidence(); s['gameTimeUTC']=verified_start(s,p,e).isoformat()
    q={'game_date':'2024-03-16','market':'player_points','captured_at':'2024-03-17T00:41:01Z'}
    with pytest.raises(ValueError,match='precede official'):validate_record(q,s,set())
    q['game_date']='2026-10-25'
    with pytest.raises(ValueError,match='Protected'):validate_record(q,s,set())


def test_missing_side_odds_is_not_imputed():
    s,_,_=evidence()
    q={'game_date':'2024-03-16','market':'player_points','captured_at':'2024-03-17T00:20:00Z','line':3.5,'over_odds':-156,'under_odds':None}
    with pytest.raises(ValueError,match='Missing under_odds'):validate_record(q,s,set())
