import pandas as pd
import pytest
from scripts.import_public_real_points import pair_quotes


def fixture():
    base={'event_id':'a','bookmaker':'book','player_name':'Player','point':20.5,'snapshot_ts':'2025-10-24T18:00:00Z','commence_time':'2025-10-24T23:00:00Z','home_team':'Home','away_team':'Away','market':'player_points','last_update':'2025-10-24T17:59:00Z','price':-110}
    return pd.DataFrame([dict(base,outcome_name='Over'),dict(base,outcome_name='Under')])


def test_pair_retains_actual_book_and_timestamp():
    rows,errors=pair_quotes(fixture());assert len(rows)==1 and not errors
    assert rows[0]['line']==20.5 and rows[0]['bookmaker']=='book'
    assert rows[0]['captured_at']=='2025-10-24T18:00:00+00:00'


@pytest.mark.parametrize('failure',['inplay','future','stale','timezone','missing_side','duplicate'])
def test_bad_public_records_are_not_repaired_by_guessing(failure):
    d=fixture()
    if failure=='inplay':d['snapshot_ts']='2025-10-24T23:01:00Z'
    if failure=='future':d['last_update']='2025-10-24T18:01:00Z'
    if failure=='stale':d['last_update']='2025-10-24T17:00:00Z'
    if failure=='timezone':d['last_update']='2025-10-24T17:59:00'
    if failure=='missing_side':d=d.iloc[:1]
    if failure=='duplicate':d=pd.concat([d,d.iloc[:1]],ignore_index=True)
    rows,errors=pair_quotes(d);assert not rows and errors


def test_official_team_alias_and_direction():
    from scripts.build_real_points_training import provider_teams_match
    s={'homeTeam':{'teamCity':'LA','teamName':'Clippers','teamId':1610612746},'awayTeam':{'teamCity':'Los Angeles','teamName':'Lakers','teamId':1610612747}}
    q={'home_team':'Los Angeles Clippers','away_team':'Los Angeles Lakers'}
    assert provider_teams_match(q,s)
    assert not provider_teams_match({'home_team':q['away_team'],'away_team':q['home_team']},s)
