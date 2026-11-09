import numpy as np
import pandas as pd
import pytest
from src.neutral_volume_correction import fit_volume
from src.points_simple_correction import balanced_weights


def fixture():
    n=240
    x=pd.DataFrame({'volume_fga':np.tile(np.linspace(3,24,120),2),'games_played':np.r_[np.full(120,5),np.full(120,15)]})
    games=np.r_[np.repeat(np.arange(20),6),np.repeat(np.arange(20,40),6)]
    return x,np.full(n,20.5),np.linspace(.45,.55,n),games


def test_outcome_imbalance_cannot_learn_blanket_under_bias():
    x,lines,market,games=fixture();y=np.zeros(240);y[::20]=1
    model=fit_volume(x,lines,market,y,games)
    p=model.predict_over(x,lines,market);w=balanced_weights(games,x.games_played)
    for mask in [x.games_played.to_numpy()<10,x.games_played.to_numpy()>=10]:
        assert abs(np.sum(w[mask]*(p[mask]-market[mask]))/w[mask].sum())<1e-10
    np.testing.assert_array_equal(p,model.predict_over(x,lines+3,market))
    assert model.beta>=0


def test_missing_inputs_and_small_cohorts_fail_closed():
    x,lines,market,games=fixture();y=np.arange(240)%2
    with pytest.raises(ValueError,match='coverage'):fit_volume(x.iloc[:180],lines[:180],market[:180],y[:180],games[:180])
    model=fit_volume(x,lines,market,y,games)
    x.loc[0,'volume_fga']=np.nan
    with pytest.raises(ValueError,match='Complete'):model.predict_over(x,lines,market)
