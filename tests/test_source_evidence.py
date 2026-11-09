from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from src.source_evidence import performance_features,market_features,instant
from scripts.crawl_defender_matchups import validate
from src.defender_evidence import expected_defender,DefenderEvidence
from types import SimpleNamespace


def forecasts():
    return [{'player_id':1,'prop_type':'points','model_version':'v1','game_id':str(i),
        'predicted_at':f'2025-01-{i+1:02}T12:00:00Z','game_start':f'2025-01-{i+1:02}T19:00:00Z',
        'result_recorded_at':f'2025-01-{i+1:02}T23:00:00Z',
        'training_outcomes_through':'2024-12-31T00:00:00Z','prediction':20,
        'actual':22,'line':20.5,'over_probability':.75} for i in range(20)]


def performance(rows):
    return performance_features(rows,cutoff='2025-02-01T12:00:00Z',player_id=1,prop_type='points',model_version='v1')


def test_out_of_sample_history_deduplicates_and_excludes_future_grades():
    rows=forecasts()
    assert performance(rows)=={'avg_prediction_error_player':2.,'model_accuracy_player':1.}
    assert performance(rows+deepcopy(rows))==performance(rows)
    poisoned=deepcopy(rows);poisoned[0]['result_recorded_at']='2025-02-02T00:00:00Z'
    assert performance(poisoned)=={} # only 19 known outcomes
    for key in ['training_outcomes_through','predicted_at']:
        poisoned=deepcopy(rows);poisoned[0][key]='2025-01-31T00:00:00Z'
        assert performance(poisoned)=={}
    assert performance([dict(r,model_version='other') for r in rows])=={}
    assert performance_features(rows,cutoff='2025-02-01T12:00:00Z',player_id=1,prop_type='points',model_version=None)=={}
    with pytest.raises(ValueError):instant('2025-01-01T12:00:00')


def quote(book='a',time='12:00',line=20.5,**kw):
    return {'game_id':'g','player_id':1,'prop_type':'points','bookmaker':book,'line':line,
        'published_at':f'2025-01-01T{time}:00Z','observed_at':f'2025-01-01T{time}:01Z',
        'game_start':'2025-01-01T19:00:00Z','source':'test provider',
        'over_price':-110,'under_price':-110,**kw}


def market(rows,line=20.5):
    return market_features(rows,cutoff='2025-01-01T12:20:00Z',game_id='g',player_id=1,prop_type='points',decision_line=line)


def test_quote_asof_filters_alternates_wrong_games_and_future_observations():
    good=[quote('a','12:10'),quote('b','12:11',21.5)]
    before=market(good)
    assert before['bookmaker_count']==2 and before['market_consensus_std']==.5
    assert before['implied_over_prob']==.5
    assert 'opening_line' not in before # first observed is not a certified open
    bad=[quote('c','12:30'),quote('c',game_id='other'),quote('c',is_alternate=True),
         quote('c',observed_at='2025-01-01T12:30:00Z'),quote(None)]
    assert market(good+bad)==before
    assert 'implied_over_prob' not in market(good,line=22.5)


def test_movement_requires_same_book_certified_open_and_real_elapsed_time():
    rows=[quote('a','10:00',19.5,is_opening=True),quote('a','12:00',20.5),quote('b','12:01',21.5)]
    result=market(rows)
    assert result['opening_line']==19.5 and result['line_movement']==1
    assert result['line_velocity']==.5
    assert result['stale_line_flag']==1 # old discordant a vs fresh median
    assert market([quote('a','12:00',20.5,is_opening=True)]).get('line_velocity') is None


def test_matchup_validation_rejects_target_game_and_identity_errors():
    players=pd.DataFrame({'PLAYER_ID':[1,2],'TEAM_ID':[10,20]})
    row={'gameId':'g','personIdOff':1,'personIdDef':2,'teamId':10,
         'partialPossessions':2,'switchesOn':1,'playerPoints':2,
         'matchupFieldGoalsMade':1,'matchupFieldGoalsAttempted':1,'matchupMinutesSort':30}
    assert validate(pd.DataFrame([row]),players,'g')[0]['defTeamId']==20
    with pytest.raises(ValueError,match='Wrong game'):validate(pd.DataFrame([row]),players,'other')
    with pytest.raises(ValueError,match='identity'):validate(pd.DataFrame([dict(row,personIdDef=999)]),players,'g')
    with pytest.raises(ValueError,match='count'):validate(pd.DataFrame([dict(row,partialPossessions=np.inf)]),players,'g')


def test_defender_forecast_uses_prior_assignments_and_current_team_only():
    prior=pd.DataFrame([{'personIdOff':1,'personIdDef':d,'defTeamId':20,
        'game_date':pd.Timestamp(day),'partialPossessions':poss} for day,d,poss in
        [('2025-01-01',2,30),('2025-01-01',3,10),('2025-02-01',3,1000)]])
    args=dict(player_id=1,opponent_id=20,day='2025-02-01',latest_teams={2:20,3:20})
    assert expected_defender(prior,**args)==2
    # Target-game assignment is neither a forecast input nor roster evidence.
    prior.loc[prior.game_date==pd.Timestamp('2025-02-01'),'partialPossessions']=999999
    assert expected_defender(prior,**args)==2
    assert expected_defender(prior,**dict(args,latest_teams={2:30,3:20})) is None


def test_switching_uses_measured_past_games_not_lineup_changes():
    rows=pd.DataFrame([{'personIdOff':1,'personIdDef':2,'defTeamId':20,'game_id':str(i),
        'game_date':pd.Timestamp(f'2025-01-0{i}'),'partialPossessions':10.,'switchesOn':i,
        'playerPoints':5.} for i in range(1,5)])
    evidence=DefenderEvidence({'rows':rows,'qualifications':{},'switches_available':True})
    ctx=SimpleNamespace(inputs=SimpleNamespace(season='2024-25',team_games={20:[{'game_date':f'2025-01-0{i}'} for i in range(1,5)]}))
    result=evidence.features(ctx,1,20,'2025-01-04')
    assert result=={'defender_switching_frequency':20.}
    # No qualified assignment => do not publish inferred defender traits.
    assert not evidence.qualified('2024-25')
    evidence.switches_available=False
    assert evidence.features(ctx,1,20,'2025-01-04')=={}
    evidence.qualifications={'2024-25':{'qualified':True,'last_outcome_date':'2025-01-01'}}
    with pytest.raises(ValueError,match='future'):evidence.qualified('2024-25')
    evidence.qualifications={'2024-25':{'qualified':True,'last_outcome_date':'2024-06-01'}}
    assert evidence.qualified('2024-25','0022400001')
    assert not evidence.qualified('2024-25','0042400001')
    assert not evidence.qualified('2024-25')


def test_quote_storage_preserves_publication_and_capture_time(db_url,db_conn):
    from src.odds_tracker import OddsTracker
    tracker=OddsTracker(db_path=db_url)
    published='2025-01-01T11:59:00Z'
    tracker._store_snapshot('g',[{'player_name':'Player One','prop_type':'points',
        'bookmaker':'book','line':20.5,'over_price':-110,'under_price':-110,
        'source_updated_at':published}])
    row=db_conn.execute('SELECT source_updated_at, observed_at FROM prop_line_history').fetchone()
    assert row[0]==published and instant(row[1]).tzinfo is not None
    tracker._store_game_lines([{'id':'g','bookmakers':[{'key':'book','last_update':published,
        'markets':[{'key':'totals','outcomes':[{'name':'Over','point':225.5,'price':-110}]}]}]}],'open')
    row=db_conn.execute('SELECT source_updated_at, observed_at FROM game_line_history').fetchone()
    assert row[0]==published and instant(row[1]).tzinfo is not None


def test_prediction_log_records_known_outcome_time_and_training_provenance(db_url,db_conn):
    from src.basketball_betting_helper import BasketballBettingHelper
    helper=BasketballBettingHelper.__new__(BasketballBettingHelper);helper.db_name=db_url
    helper.create_tables()
    key=helper.log_prediction(1,'Player One','points',20.5,True,True,22,.7,'OVER',
        'HIGH',.1,20,21,20,model_version='v1',training_outcomes_through='2024-12-31T00:00:00Z',
        feature_policy='test-policy', game_id='0022400001', game_start='2025-01-01T19:00:00Z',
        schedule_source='NBA', requested_at='2025-01-01T12:00:00Z')
    assert key is not None
    row=db_conn.execute('SELECT timestamp, result_recorded_at, training_outcomes_through FROM prediction_logs WHERE id=%s',(key,)).fetchone()
    assert instant(row[0]).tzinfo is not None and row[1] is None
    assert row[2]=='2024-12-31T00:00:00Z'
    schedule = db_conn.execute('SELECT game_id, game_start, schedule_source FROM prediction_logs WHERE id=%s', (key,)).fetchone()
    assert tuple(schedule) == ('0022400001', '2025-01-01T19:00:00Z', 'NBA')
    assert db_conn.execute('SELECT requested_at FROM prediction_logs WHERE id=%s', (key,)).fetchone()[0] == '2025-01-01T12:00:00Z'
    success,_=helper.update_actual_result(key,22)
    assert success
    known=db_conn.execute('SELECT result_recorded_at FROM prediction_logs WHERE id=%s',(key,)).fetchone()[0]
    assert instant(known)>=instant(row[0])


def test_game_totals_exclude_future_team_alternate_and_period_markets():
    from src.source_evidence import game_total_features
    rows = [quote('a', line=220.5, market='totals'), quote('b', line=222.5, market='totals')]
    args = dict(cutoff='2025-01-01T12:20:00Z', game_id='g')
    assert game_total_features(rows, **args) == {'implied_game_total': 221.5}
    bad = [dict(rows[0], bookmaker='c', **extra) for extra in
           [{'team':'BOS'}, {'period':'first_half'}, {'is_alternate':True},
            {'observed_at':'2025-01-01T13:00:00Z'}, {'line':float('inf')}]]
    assert game_total_features(rows + bad, **args) == game_total_features(rows, **args)


def test_schedule_rejects_wrong_date_teams_and_ambiguous_games():
    from src.prediction_schedule import select_game
    game = {'gameId':'0022400001', 'gameTimeUTC':'2025-01-02T00:00:00Z',
            'homeTeam':{'teamId':1}, 'awayTeam':{'teamId':2}}
    payload = {'scoreboard':{'gameDate':'2025-01-01', 'games':[game]}}
    assert select_game(payload, 1, 2, '2025-01-01')['game_id'] == '0022400001'
    assert select_game(payload, 1, 3, '2025-01-01') == {}
    assert select_game(payload, 1, 2, '2025-01-02') == {}
    payload['scoreboard']['games'].append(game)
    assert select_game(payload, 1, 2, '2025-01-01') == {}


def test_prop_parser_never_pairs_different_lines_or_fabricates_missing_prices(monkeypatch):
    from src.odds_tracker import OddsTracker
    tracker = OddsTracker.__new__(OddsTracker)
    tracker.api_key = 'mock'
    outcomes = [{'description':'One','name':'Over','point':20.5,'price':-115},
                {'description':'Two','name':'Over','point':21.5,'price':-110},
                {'description':'Two','name':'Under','point':22.5,'price':-110},
                {'description':'Three','name':'Over','point':20.5}]
    monkeypatch.setattr(tracker, '_get', lambda *args, **kw: {'bookmakers':[
        {'key':'book','markets':[{'key':'player_points','outcomes':outcomes}]}]})
    result = tracker.fetch_player_props('game')
    assert len(result) == 1
    assert result[0]['player_name'] == 'One'
    assert result[0]['over_price'] == -115 and result[0]['under_price'] is None


def test_history_replay_does_not_activate_missing_sources():
    from scripts.audit_source_history import replay
    decisions = [{'cutoff':'2025-01-01T12:20:00Z','game_id':'g','player_id':1,
                  'prop_type':'points','line':20.5,'model_version':'v1'}]
    assert replay(decisions)['observed_counts'] == {}
    result = replay(decisions, quotes=[quote()])
    assert result['observed_counts']['bookmaker_count'] == 1
    assert result['activation_changed'] is False


def test_schedule_fallback_keeps_provider_ids_separate(monkeypatch):
    from src import prediction_schedule as schedule
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    start = today + 'T19:00:00-04:00'
    event = {'id':'ESPN-123', 'date':start, 'competitions':[{'competitors':[
        {'team':{'abbreviation':'BOS'}}, {'team':{'abbreviation':'NY'}}]}]}
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'events':[event]}
    def get(url, **kwargs):
        if url == schedule.URL:
            raise schedule.requests.HTTPError('CDN unavailable')
        return Response()
    monkeypatch.setattr(schedule.requests, 'get', get)
    result = schedule.current_game_schedule(1610612738, 1610612752, '0022600001')
    assert result['game_id'] == '0022600001'
    assert 'ESPN-123' in result['source']
    assert schedule.current_game_schedule(1610612738, 1610612752) == {}


def test_archive_import_preserves_conflicts_and_rejects_unknown_identity(tmp_path, monkeypatch):
    from scripts.import_matchup_archive import import_files
    import gzip, json
    cache = tmp_path / 'cache'; cache.mkdir()
    pd.DataFrame({'GAME_ID':['0022400001'] * 2, 'PLAYER_ID':[1,2],
                  'TEAM_ID':[10,20], 'GAME_DATE':['2025-01-01'] * 2}).to_json(cache / 'players_test.json')
    archive = tmp_path / 'test.parquet'; archive.write_bytes(b'fixture')
    row = {'game_id':22400001, 'person_id':1, 'matchups_person_id':2,
           'team_id':10, 'partial_possessions':5, 'switches_on':0, 'player_points':2,
           'matchup_field_goals_made':1, 'matchup_field_goals_attempted':2, 'matchup_minutes_sort':30}
    monkeypatch.setattr(pd, 'read_parquet', lambda _: pd.DataFrame([row]))
    output = tmp_path / 'out'
    assert import_files([archive], cache, output)['imported'] == 1
    assert import_files([archive], cache, output)['matched_existing'] == 1
    row['player_points'] = 3
    assert 'differs' in import_files([archive], cache, output)['rejected'][0]['reason']
    with gzip.open(output / '0022400001.json.gz', 'rt') as handle:
        assert json.load(handle)['rows'][0]['playerPoints'] == 2
    row['matchups_person_id'] = 999
    assert 'identity' in import_files([archive], cache, output)['rejected'][0]['reason']


def test_unknown_report_identity_is_not_a_healthy_expected_defender():
    rows=pd.DataFrame([{'personIdOff':1,'personIdDef':2,'defTeamId':20,'game_id':str(i),
        'game_date':pd.Timestamp(f'2025-01-0{i}'),'partialPossessions':20.,'switchesOn':0.,
        'playerPoints':5.} for i in range(1,4)])
    evidence=DefenderEvidence({'rows':rows,'qualifications':{'2024-25':{'qualified':True,'last_outcome_date':'2024-06-01'}}})
    ctx=SimpleNamespace(inputs=SimpleNamespace(season='2024-25',prediction_game_id='0022400001',
        team_games={},precomputed={'advanced':{}}),logs=pd.DataFrame({'PLAYER_ID':[1,2],
        'TEAM_ID':[10,20],'GAME_DATE':pd.to_datetime(['2025-01-01']*2)}),
        injuries=lambda *args:[{'player':'Unknown','status':'Out'}],resolve_name=lambda name:None)
    assert 'primary_defender_active' not in evidence.features(ctx,1,20,'2025-01-04')
    ctx.resolve_name=lambda name:2
    assert evidence.features(ctx,1,20,'2025-01-04')['primary_defender_active']==0
    ctx.resolve_name=lambda name:3
    assert evidence.features(ctx,1,20,'2025-01-04')['primary_defender_active']==1
