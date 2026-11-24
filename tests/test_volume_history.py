import numpy as np
import pandas as pd
import pytest
from scripts.research_volume_history import fit, predict, design, ARMS
from src.points_simple_correction import balanced_weights


def data():
    rng=np.random.default_rng(12);n=400
    x=pd.DataFrame({'volume_fga':rng.uniform(3,25,n),'games_played':np.tile([3,20],n//2)})
    lines=.8*x.volume_fga.to_numpy()+rng.normal(0,2,n)
    market=np.full(n,.52);y=(x.volume_fga.to_numpy()>14).astype(int);games=np.repeat(np.arange(40),10)
    return x,lines,market,y,games


@pytest.mark.parametrize('arm',ARMS)
def test_neutrality_and_batch_invariance(arm):
    x,l,p,y,g=data();m=fit(x,l,p,y,g,arm);pred=predict(m,x,l,p);w=balanced_weights(g,x.games_played)
    for c in (x.games_played<10,x.games_played>=10):
        assert abs(w[c]@(pred[c]-p[c])/w[c].sum())<1e-10
    assert np.allclose(pred[:3],predict(m,x.iloc[:3],l[:3],p[:3]),atol=1e-14)
    assert (m['beta']>=0).all()


def test_neutral_offsets_do_not_learn_blanket_under_bias():
    x,l,p,y,g=data();m=fit(x,l,p,np.zeros(len(x)),g,'separate')
    assert np.allclose(predict(m,x,l,p),p,atol=1e-6)


def test_coverage_fails_closed():
    x,l,p,y,g=data()
    with pytest.raises(ValueError,match='coverage'):fit(x.iloc[:100],l[:100],p[:100],y[:100],g[:100],'shared')


def test_line_residual_fit_uses_training_only_and_removes_linear_component():
    x,l,p,y,g=data();w=balanced_weights(g,x.games_played)
    z,s=design(x,l,w=w,arm='line_residual')
    assert abs(w@(x.volume_fga.to_numpy()-np.c_[np.ones(len(l)),l]@s['coef']))<1e-12
    a,_=design(x.iloc[:1],l[:1],state=s,arm='line_residual')
    assert a[0]==z[0]
