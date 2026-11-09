import numpy as np
import pandas as pd
from scipy.special import expit,logit
from src.points_correction_diagnostics import shapley_losses,fit_arm


def test_shapley_exact_sum_and_zero_unused_terms():
    y=np.array([0,1,0,1]);p=np.array([.4,.6,.55,.45]);terms=np.zeros((4,5));terms[:,0]=-.1;terms[:,1]=[.2,-.2,.3,-.3]
    values=shapley_losses(y,p,terms)
    eta=logit(p);new=eta+terms.sum(axis=1)
    np.testing.assert_allclose(values.sum(axis=1),(np.logaddexp(0,new)-y*new)-(np.logaddexp(0,eta)-y*eta),atol=1e-12)
    np.testing.assert_array_equal(values[:,2:],0)


def test_inactive_coefficients_stay_zero_and_line_is_monotone():
    n=40;x=pd.DataFrame({'games_played':[5,15]*20,'recent_avg':np.linspace(10,25,n),'season_avg':np.linspace(10,23,n),'volume_fga':np.linspace(8,18,n),'rest_days':[1,2]*20})
    market=np.full(n,.5);line=np.full(n,18.5);y=np.array([0,1]*20);games=np.repeat(np.arange(20),2)
    model=fit_arm(x,line,market,y,games,[1],1.)
    np.testing.assert_array_equal(model.beta[[0,2,3,4]],0)
    assert model.beta[1]>=0
    assert np.all(model.predict_over(x,line+1,market)<=model.predict_over(x,line,market))
