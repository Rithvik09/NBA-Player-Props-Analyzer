import numpy as np
import pandas as pd
import pytest
from src.audited_features import SPECS
from src.verified_training import (verified_columns,DEFERRED_DEFENDERS,
    classifier_frame,expand_points,chronological_masks)


def test_verified_profile_excludes_all_unqualified_and_duplicate_inputs():
    cols=verified_columns()
    assert len(cols)==362
    assert not DEFERRED_DEFENDERS.intersection(cols)
    assert all(SPECS[c].status=='active' for c in cols)
    assert not {'sharp_action_score','opening_line','defender_switching_frequency'}.intersection(cols)


def test_training_and_inference_share_line_features_and_ignore_deferred_columns():
    cols=['recent_avg','season_avg','stddev']
    frame=pd.DataFrame({'recent_avg':[20.,12.],'season_avg':[21.,11.],
                        'stddev':[0.,4.],'primary_defender_rating':[80.,140.]})
    data={'X':frame,'y':np.array([22.,10.])}
    x,y,idx,lines=expand_points(data,cols)
    expected=classifier_frame(frame.iloc[idx].reset_index(drop=True),lines,cols)
    pd.testing.assert_frame_equal(x,expected)
    changed=frame.copy();changed['primary_defender_rating']=999999
    poisoned,labels,idx2,lines2=expand_points({'X':changed,'y':1-data['y']},cols)
    pd.testing.assert_frame_equal(x,poisoned)
    np.testing.assert_array_equal(lines,lines2)
    assert not np.array_equal(y,labels)
    assert np.isfinite(x.standardized_edge).all()


def test_split_keeps_every_same_day_game_and_expanded_line_together():
    dates=pd.to_datetime(['2024-12-31']*3+['2025-01-01']*3+['2025-07-01']*3)
    masks=chronological_masks(dates)
    for name,index in [('fit',0),('calibration',3),('evaluation',6)]:
        assert masks[name].sum()==3 and masks[name][index:index+3].all()
    with pytest.raises(ValueError,match='Empty'):chronological_masks(dates[:3])


def test_invalid_decision_line_is_rejected():
    frame=pd.DataFrame({'recent_avg':[20.],'season_avg':[21.],'stddev':[4.]})
    for lines in [[np.nan],[-1],[20,21]]:
        with pytest.raises(ValueError):classifier_frame(frame,lines,list(frame.columns))
