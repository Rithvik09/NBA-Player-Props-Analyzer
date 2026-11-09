import numpy as np
import pandas as pd
from scripts.improve_points_pilot import expand


def test_synthetic_inputs_do_not_depend_on_current_target():
    data={'X':pd.DataFrame({'recent_avg':[12.,22.], 'season_avg':[14.,20.], 'stddev':[0.,6.]}),'y':np.array([10.,25.])}
    X,y,rows=expand(data)
    other={**data,'y':np.array([100.,0.])}
    poisoned,changed,other_rows=expand(other)
    pd.testing.assert_frame_equal(X,poisoned)
    np.testing.assert_array_equal(rows,other_rows)
    assert not np.array_equal(y,changed)
    assert len(X)==10
    assert np.isfinite(X.standardized_edge).all()
    np.testing.assert_array_equal(y,(data['y'][rows]>X.synthetic_line).astype(int))


def test_floor_lines_are_unique_and_positive():
    data={'X':pd.DataFrame({'recent_avg':[0.], 'season_avg':[0.], 'stddev':[1.]}),'y':np.array([0.])}
    X,y,rows=expand(data)
    assert X.synthetic_line.tolist()==[.5,1.5,2.5]
    assert not y.any()
    assert len(rows)==3
