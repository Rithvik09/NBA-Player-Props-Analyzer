"""Negative controls for future-data leakage in the points pilot."""
from copy import deepcopy
import numpy as np
import pandas as pd

from src.historical_inputs import HistoricalInputs,PLAYER_SOURCES,SNAPSHOT_SOURCES
from scripts.points_pilot import split_dates,classifier_rows


def raw_sources():
    data={key:{} for key in PLAYER_SOURCES+SNAPSHOT_SOURCES}
    data.update(game_officials={},referee_history={},game_team_stats={})
    return data


def log_fixture():
    from tests.test_feature_defaults import _canned_game_log
    logs=_canned_game_log().rename(columns={'Game_ID':'GAME_ID'})
    logs['PLAYER_ID']=2544
    return logs


def test_only_completed_prior_season_snapshots_are_exposed():
    raw=raw_sources();raw['advanced']={(2544,'2023-24'):{'usg_pct':.2},
        (2544,'2024-25'):{'usg_pct':999},2544:{'usg_pct':999}}
    raw['team_stats']={1:{'pace':999}};raw['injury_status']={1:{'out':9}}
    source=HistoricalInputs('2024-25',raw,log_fixture())
    assert source.precomputed['advanced'][(2544,'2024-25')]['usg_pct']==.2
    assert source.precomputed['advanced'][2544]['usg_pct']==.2
    assert all(source.precomputed[key]=={} for key in SNAPSHOT_SOURCES)


def test_team_context_excludes_same_day_and_future():
    raw=raw_sources();raw['game_team_stats']={1:[
        {'game_date':'2025-01-01','pace':98,'off_rating':111,'def_rating':109},
        {'game_date':'2025-01-03','pace':900,'off_rating':999,'def_rating':999},
        {'game_date':'2025-01-05','pace':900,'off_rating':999,'def_rating':999}]}
    source=HistoricalInputs('2024-25',raw,log_fixture())
    ctx=source.team_context(1,'2025-01-03')
    assert ctx['pace']==98 and ctx['offensive_rating']==111
    assert ctx['days_rest']==1


def test_current_target_and_future_snapshot_mutations_do_not_change_features(monkeypatch):
    from scripts.train_models import build_training_examples
    logs=log_fixture();raw=raw_sources()
    raw['advanced']={(2544,'2023-24'):{'usg_pct':.2},(2544,'2024-25'):{'usg_pct':.7}}
    def build(frame,inputs):
        source=HistoricalInputs('2024-25',inputs,frame)
        source.catalog=lambda n:pd.DataFrame([{'PLAYER_ID':2544}])
        return build_training_examples('2024-25',1,None,None,historical_inputs=source,prop_types=('points',))
    before=build(logs,raw)
    poisoned=logs.copy();poisoned.loc[poisoned.index[-1],'PTS']=999
    changed=deepcopy(raw);changed['advanced'][(2544,'2024-25')]['usg_pct']=999
    after=build(poisoned,changed)
    assert len(before)==len(after)==4
    pd.testing.assert_frame_equal(before[-1].X,after[-1].X)
    assert before[-1].y!=after[-1].y
    for left,right in zip(before[:-1],after[:-1]):
        pd.testing.assert_frame_equal(left.X,right.X)


def test_temporal_splits_never_separate_same_game_lines():
    dates=pd.to_datetime(['2024-12-31']*5+['2025-01-01']*5+['2025-10-21']*5)
    split=split_dates(dates)
    assert split['fit'].sum()==split['calibration'].sum()==split['test'].sum()==5
    assert np.all(sum(mask.astype(int) for mask in split.values())==1)


def test_synthetic_lines_have_no_pushes_or_target_dependence():
    X=pd.DataFrame([{'recent_avg':20.,'season_avg':20.}])
    a=classifier_rows(X,np.array([20]),pd.to_datetime(['2025-10-21']),np.array(['0022500001']))
    b=classifier_rows(X,np.array([99]),pd.to_datetime(['2025-10-21']),np.array(['0022500001']))
    pd.testing.assert_frame_equal(a[0],b[0])
    assert len(a[1])==5
    assert not np.array_equal(a[1],b[1])


def test_reliability_bins_partition_boundary_probabilities_once():
    from scripts.points_pilot import metrics
    y=np.array([0,1,0,1,1])
    p=np.array([.3,.4,.5,.6,1.])
    result=metrics(y,p)
    assert sum(row['n'] for row in result['calibration_bins'])==len(y)
    assert all(row['n']==1 for row in result['calibration_bins'])
