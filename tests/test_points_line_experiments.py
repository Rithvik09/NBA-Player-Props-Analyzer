import numpy as np
import pandas as pd
from src.points_line_experiments import expand_with_offsets,offsets_from_training_quotes,feature_groups
from src.points_training_repair import outcome_weights


def test_labels_lines_and_outcome_weights_stay_aligned():
    x=pd.DataFrame({'recent_avg':[1.,20.],'season_avg':[1.,20.],'stddev':[3.,4.]})
    f,y,i,l=expand_with_offsets({'X':x,'y':np.array([3.,25.])},list(x),[-8,-4,0,4,8])
    assert np.array_equal(y,np.array([3.,25.])[i]>l)
    assert (l>=.5).all()
    w=outcome_weights(i,0)
    assert all(np.isclose(w[i==j].sum(),1) for j in [0,1])
    assert np.array_equal(f.decision_line,l)


def test_empirical_recipe_uses_only_supplied_training_quotes():
    x=pd.DataFrame({'recent_avg':[10.,10.],'season_avg':[10.,10.]})
    assert np.allclose(offsets_from_training_quotes(x,[6.,14.]),[-3.6,-2.,0.,2.,3.6])


def test_feature_partition_is_disjoint_and_complete():
    full=['recent_avg','team_injury_unknown','fg_pct_recent','vs_team_avg','touches','trend','player_age']
    groups=feature_groups(full,['recent_avg']);flat=sum(groups.values(),[])
    assert set(flat)==set(full)-{'recent_avg'} and len(flat)==len(set(flat))


def test_narrow_control_exactly_reproduces_legacy_rounding():
    from src.verified_training import expand_points
    x=pd.DataFrame({'recent_avg':[12.999999999999998,20.],'season_avg':[12.999999999999998,20.],'stddev':[3.,4.]})
    d={'X':x,'y':np.array([13.,21.])}
    a=expand_points(d,list(x));b=expand_with_offsets(d,list(x),[-2,-1,0,1,2])
    for i in [1,2,3]:assert np.array_equal(a[i],b[i])
