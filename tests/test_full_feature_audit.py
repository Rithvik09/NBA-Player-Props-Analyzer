"""Independent arithmetic, provenance and caller tests for the audited contract."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from src.audited_features import SPECS,PLAYER_MAP,ALIASES,POLICY_VERSION,build_audited_features
from src.historical_inputs import HistoricalInputs,PLAYER_SOURCES,SNAPSHOT_SOURCES
from src.ml_features import NUMERIC_FEATURE_KEYS,build_feature_vector
from tests.test_feature_defaults import _canned_game_log

BOS=1610612738
LAL=1610612747


def fixture():
    logs=_canned_game_log().rename(columns={'Game_ID':'GAME_ID'})
    logs['GAME_DATE']=pd.to_datetime(logs.GAME_DATE)
    logs['PLAYER_ID']=2544;logs['TEAM_ID']=BOS
    raw={k:{} for k in PLAYER_SOURCES+SNAPSHOT_SOURCES}
    raw.update(game_officials={},referee_history={},game_team_stats={})
    for table,mapping in PLAYER_MAP.items():
        raw[table][(2544,'2023-24')]={col:.23+i*.001 for i,col in enumerate(mapping)}
        raw[table][(2544,'2024-25')]={col:999 for col in mapping}
    raw['game_team_stats']={BOS:[{'game_date':'2025-01-18','pace':100.,'off_rating':110.,'def_rating':108.,'pts':100,'opp_pts':90}],LAL:[{'game_date':'2025-01-19','pace':105.,'off_rating':115.,'def_rating':109.,'pts':90,'opp_pts':100}]}
    inputs=HistoricalInputs('2024-25',raw,logs,audited=True)
    return logs,raw,inputs


def build(logs,inputs,target='PTS'):
    return build_audited_features(logs,target,2544,BOS,LAL,'2025-01-21',False,inputs)


def test_all_416_features_have_explicit_specs_and_no_extra_keys():
    assert list(dict.fromkeys(NUMERIC_FEATURE_KEYS))==NUMERIC_FEATURE_KEYS
    assert set(SPECS)==set(NUMERIC_FEATURE_KEYS)
    assert all(s.source and s.units and s.definition for s in SPECS.values())
    logs,raw,inputs=fixture();features,gaps=build(logs,inputs)
    assert list(features)==NUMERIC_FEATURE_KEYS
    assert all(features[k]==0 for k,s in SPECS.items() if s.status!='active')
    assert np.isfinite(build_feature_vector(features).X).all().all()


@pytest.mark.parametrize('table',list(PLAYER_MAP))
def test_every_snapshot_field_and_alias_reads_completed_previous_season(table):
    logs,raw,inputs=fixture();features,_=build(logs,inputs)
    for col,key in PLAYER_MAP[table].items():
        assert features[key]==pytest.approx(raw[table][(2544,'2023-24')][col]),key
    for key,source in ALIASES.items():
        assert features[key]==features[source],key


def test_future_player_team_and_snapshot_values_cannot_change_any_feature():
    logs,raw,inputs=fixture();before,_=build(logs,inputs)
    changed=logs.copy();mask=changed.GAME_DATE>=pd.Timestamp('2025-01-21')
    for col in 'PTS FGA FGM FG3A FG3M FTA FTM MIN'.split():changed.loc[mask,col]=99999
    poisoned=deepcopy(raw)
    for table,mapping in PLAYER_MAP.items():
        poisoned[table][(2544,'2024-25')]={col:-99999 for col in mapping}
    for tid in (BOS,LAL):
        poisoned['game_team_stats'][tid]+=[{'game_date':'2025-01-21','pace':99999,'off_rating':99999,'def_rating':99999,'pts':9999,'opp_pts':0}]
    after,_=build(changed,HistoricalInputs('2024-25',poisoned,changed,audited=True))
    assert before==after


def test_shooting_uses_attempt_weights_and_real_zone_and_tracking_sources():
    logs,raw,inputs=fixture();logs.loc[0,'FGA']=100;logs.loc[0,'FGM']=0
    features,_=build(logs,inputs)
    prior=logs[logs.GAME_DATE<'2025-01-21'].tail(5)
    assert features['fg_pct_recent']==pytest.approx(prior.FGM.sum()/prior.FGA.sum())
    assert features['corner_3_pct']==features['corner3_fg_pct']
    assert features['touches_per_game']==features['tracking_touches_pg']
    assert features['restricted_area_fg_pct']==features['rim_fg_pct']
    assert features['wide_open_shot_pct']==features['open_shot_frequency']
    assert features['paint_touch_frequency']==pytest.approx(features['paint_touches_per_game']/features['tracking_touches_pg'])
    assert features['effective_fg_pct']!=features['efg_pct_official']


def test_away_arena_uses_opponent_and_rest_uses_complete_team_schedule():
    logs,raw,inputs=fixture();features,_=build(logs,inputs)
    assert features['arena_altitude']==300 # Lakers venue, not Boston's10ft
    assert features['rest_advantage']==1 # Boston2 days, Lakers1
    assert features['team_l10_wins']==1 # one team game, not10 player wins
    assert features['opp_l10_wins']==0


def test_serving_adapter_and_training_builder_produce_identical_all416_inputs():
    from scripts.train_models import build_training_examples
    from src.models import EnhancedMLPredictor
    logs,raw,inputs=fixture();inputs.catalog=lambda n:pd.DataFrame([{'PLAYER_ID':2544}])
    examples=build_training_examples('2024-25',1,None,None,historical_inputs=inputs,prop_types=('points',))
    ex=examples[0];predictor=EnhancedMLPredictor.__new__(EnhancedMLPredictor);predictor.feature_policy=POLICY_VERSION
    features=predictor.prepare_features({'_audited_inputs':inputs,'_boxscores':logs.to_dict('records'),'_prediction_date':ex.game_date,'_prediction_game_id':ex.game_id,'_prop_type':'points','_player_id':2544,'_team_id':BOS,'_opponent_team_id':LAL,'is_home':True},{},{},{})
    pd.testing.assert_frame_equal(ex.X,build_feature_vector(features).X)
    assert ex.feature_policy==POLICY_VERSION


def test_all14_props_have_correct_audited_targets_and_identity():
    from scripts.train_models import build_training_examples,STAT_TARGETS,COMBO_TARGETS
    logs,raw,inputs=fixture();inputs.catalog=lambda n:pd.DataFrame([{'PLAYER_ID':2544}])
    examples=build_training_examples('2024-25',1,None,None,historical_inputs=inputs)
    assert len({e.prop_type for e in examples})==14
    for ex in examples:
        target=STAT_TARGETS.get(ex.prop_type) or COMBO_TARGETS.get(ex.prop_type) or 'PTS'
        expected,gaps=build_audited_features(logs,target,2544,BOS,LAL,ex.game_date,True,inputs)
        expected['_feature_policy']=POLICY_VERSION;expected['_missing_features']=gaps
        pd.testing.assert_frame_equal(ex.X,build_feature_vector(expected).X)
        assert ex.player_id==2544 and ex.game_id.startswith('002')


def test_unsupported_live_injections_cannot_reactivate_disabled_features():
    from src.audited_runtime import enforce
    features={'_feature_policy':POLICY_VERSION,'_missing_features':['team_key_players_out'],'team_key_players_out':10,'sharp_action_score':.9,'recent_avg':20}
    result=enforce(features,POLICY_VERSION)
    assert np.isnan(result['team_key_players_out'])
    assert result['sharp_action_score']==0
    assert result['recent_avg']==20
    with pytest.raises(ValueError,match='requires audited'):enforce({},POLICY_VERSION)
    assert enforce(features,'legacy') is features


def test_scoring_collector_uses_correct_midrange_and_unassisted_columns(monkeypatch):
    from src import precompute_jobs as jobs
    row={'PLAYER_ID':2544,'PCT_PTS_3PT':.3,'PCT_PTS_PAINT':.4,'PCT_PTS_FT':.2,'PCT_PTS_2PT_MR':.1,'PCT_UAST_2PM':.9,'PCT_UAST_FGM':.35,'PCT_AST_FGM':.65}
    class Endpoint:
        def get_data_frames(self):return [pd.DataFrame([row])]
    monkeypatch.setattr(jobs.leaguedashplayerstats,'LeagueDashPlayerStats',lambda **kw:Endpoint())
    monkeypatch.setattr(jobs.time,'sleep',lambda _:None)
    result=jobs.compute_player_scoring_breakdown('2024-25')[0]
    assert result['pct_pts_midrange']==.1
    assert result['pct_uast_fgm']==.35
