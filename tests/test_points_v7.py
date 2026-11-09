import numpy as np
import pytest
from src.points_v7 import market_probability,anchored_probability,fit_anchor,mixture_weights

def test_odds_and_market_fallback():
    assert market_probability([-110],[-110])[0]==.5
    assert market_probability([-200],[200])[0]==pytest.approx(2/3)
    p=market_probability([-115,120],[-105,-140])
    assert np.allclose(anchored_probability([.1,.9],p,0),p)
    assert np.allclose(anchored_probability([.1,.9],p,1),[.1,.9])
    with pytest.raises(ValueError):market_probability([np.nan],[-110])
    with pytest.raises(ValueError):anchored_probability([.5],[.5],-1)

def test_bad_signal_shrinks_to_market_without_intercept_bias():
    y=np.array([0,1]*20);raw=np.where(y,.1,.9);market=np.full(len(y),.5)
    assert fit_anchor(raw,market,y)==0
    assert np.allclose(anchored_probability(raw,market,0),.5)

def test_real_weight_share_is_exact_and_not_book_count():
    for share in [0,.01,.1,.5]:
        w=mixture_weights(1000,10,share)
        assert w[-10:].sum()/w.sum()==pytest.approx(share)
    assert mixture_weights(1000,10,.01)[-1]<mixture_weights(1000,10,.5)[-1]


def test_selection_success_is_not_promotion_and_market_must_be_beaten():
    from src.points_research_gate import promotion_gate
    args=dict(independent=True,player_games=200,independent_games=30,market_interval=[-.02,-.01],compact_interval=[-.01,-.001])
    assert promotion_gate(**args)['eligible_for_promotion']
    assert not promotion_gate(**dict(args,independent=False))['eligible_for_promotion']
    assert not promotion_gate(**dict(args,market_interval=[.002,.01]))['eligible_for_promotion']
    assert not promotion_gate(**dict(args,independent_games=14))['eligible_for_promotion']


def test_pooled_coverage_cannot_certify_short_history_calibration():
    from src.points_research_gate import calibration_support
    n=np.r_[np.full(200,20),[2,3]];games=np.array([str(i%20) for i in range(202)])
    fit=np.arange(202)<100;selection=(np.arange(202)>=100)&(np.arange(202)<200)
    result=calibration_support(n,games,fit,selection)
    assert result['eligible_history']['supported']
    assert not result['short_history']['supported']
