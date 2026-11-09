import numpy as np
import pandas as pd
from src.points_target_research import fixed_threshold_expansion,FIXED_THRESHOLDS
from src.points_training_repair import outcome_weights


def test_fixed_lines_do_not_depend_on_features_or_outcomes():
    x=pd.DataFrame({'recent_avg':[1.,30.],'season_avg':[2.,20.],'stddev':[3.,4.]})
    d={'X':x,'y':np.array([2.,25.])}
    _,y,idx,lines=fixed_threshold_expansion(d,list(x))
    assert np.array_equal(lines,np.tile(FIXED_THRESHOLDS,2))
    assert np.array_equal(y,d['y'][idx]>lines)
    d['y'][:]=99;x.recent_avg=99
    assert np.array_equal(fixed_threshold_expansion(d,list(x))[3],lines)
    w=outcome_weights(idx,0)
    assert all(np.isclose(w[idx==i].sum(),1) for i in [0,1])


def test_point_target_probability_is_monotone_without_line_features():
    # Exercise conversion with a fixed regressor and minimal valid prepare input.
    from unittest.mock import patch
    from src.points_target_research import TargetCandidate
    class Regressor:
        def predict(self,x):
            assert list(x)==['recent_avg']
            return np.array([20.,20.])
    c=TargetCandidate(['recent_avg'],Regressor(),'points',{})
    x=pd.DataFrame({'recent_avg':[20.,20.]})
    with patch('src.points_target_research.prepare',lambda f:f):
        p=c.raw(x,np.array([15.,25.]))
    assert p[0]>.5>p[1]
