import numpy as np
import pandas as pd
import pytest
from src.points_simple_correction import features,balanced_weights,fit_simple


def test_balancing_preserves_cohort_and_game_mass():
    w=balanced_weights(['a','a','b','c'],[1,2,3,20])
    assert np.allclose(w,[.125,.125,.25,.5])
    assert np.isclose(w.sum(),1)


def test_short_inputs_do_not_require_long_history_and_fail_on_missing():
    x=pd.DataFrame({'games_played':[1,20],'recent_avg':[10.,10.],'season_avg':[10.,10.],'volume_fga':[8.,8.],'rest_days':[1.,1.]})
    z=features(x,np.array([5.,5.]));assert np.isfinite(z).all() and z[0,0]<z[1,0]
    x.loc[0,'volume_fga']=np.nan
    with pytest.raises(ValueError):features(x,np.array([5.,5.]))


def test_regularization_shrinks_and_line_response_is_monotone():
    n=200;x=pd.DataFrame({'games_played':np.tile([3,20],100),'recent_avg':np.linspace(5,25,n),'season_avg':np.full(n,15.),'volume_fga':np.full(n,10.),'rest_days':np.ones(n)})
    line=np.full(n,15.);market=np.full(n,.5);y=(x.recent_avg.to_numpy()>15).astype(int);games=np.repeat(np.arange(20),10)
    a=fit_simple(x,line,market,y,games,.25);b=fit_simple(x,line,market,y,games,4)
    assert np.linalg.norm(b.beta)<np.linalg.norm(a.beta)
    assert (b.predict_over(x,line+1,market)<=b.predict_over(x,line,market)+1e-12).all()
