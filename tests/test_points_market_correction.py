import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from scipy.special import expit,logit
from src.points_market_correction import fit_correction,market_offset


def frame(n):
    return pd.DataFrame({'recent_avg':np.linspace(5,25,n),'season_avg':np.full(n,15.),'stddev':np.full(n,5.)})


def test_zero_correction_reproduces_market_and_line_constraints():
    x=frame(100);market=np.linspace(.4,.6,100);lines=np.full(100,15.)
    with patch('src.points_market_correction.prepare',lambda x:x):
        c=fit_correction(x,lines,market,np.tile([0,1],50),['recent_avg'],10)
        before=c.predict_over(x,lines,market);after=c.predict_over(x,lines+1,market)
        assert (after<=before+1e-12).all()
        c.coefficients[:]=0
        assert np.allclose(c.predict_over(x,lines,market),market)
        median=c.median.copy();scale=c.scale.copy()
        c.predict_over(x*100,lines,market)
        assert np.array_equal(median,c.median) and np.array_equal(scale,c.scale)


def test_learns_correction_to_offset_on_real_labels():
    rng=np.random.default_rng(12);x=frame(2000);market=np.full(len(x),.55)
    y=rng.binomial(1,expit(logit(market)+.15*(x.recent_avg.to_numpy()-15)))
    with patch('src.points_market_correction.prepare',lambda x:x):
        c=fit_correction(x,np.full(len(x),15.),market,y,['recent_avg'],10)
        p=c.predict_over(x,np.full(len(x),15.),market)
        assert p[-200:].mean()>market.mean()>p[:200].mean()


def test_invalid_market_probabilities_rejected():
    for p in [[0,.5],[1,.5],[np.nan,.5]]:
        with pytest.raises(ValueError):market_offset(p)
