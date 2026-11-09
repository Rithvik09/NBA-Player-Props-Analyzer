"""Research-only market-anchored calibration; no frozen serving dependencies."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from src.points_v5 import prepare
from src.verified_training import classifier_frame


def market_probability(over_odds,under_odds):
    def implied(a):
        a=np.asarray(a,dtype=float)
        if not np.isfinite(a).all() or (np.abs(a)<100).any():raise ValueError('Finite American odds required')
        out=np.empty_like(a);positive=a>0
        out[positive]=100/(a[positive]+100);out[~positive]=-a[~positive]/(100-a[~positive])
        return out
    a,b=implied(over_odds),implied(under_odds)
    if a.shape!=b.shape:raise ValueError('Both odds sides must align')
    return a/(a+b)


def anchored_probability(raw,market,weight):
    a,b=np.asarray(raw,dtype=float),np.asarray(market,dtype=float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Aligned finite probabilities required')
    if not 0<=weight<=1:raise ValueError('Shrinkage must lie in [0,1]')
    a,b=np.clip(a,1e-7,1-1e-7),np.clip(b,1e-7,1-1e-7)
    return expit(logit(b)+weight*(logit(a)-logit(b)))


def fit_anchor(raw,market,y):
    objective=lambda w:log_loss(y,anchored_probability(raw,market,w),labels=[0,1])
    fit=minimize_scalar(objective,bounds=(0.,1.),method='bounded')
    return min([0.,1.,float(fit.x)],key=objective)


def mixture_weights(synthetic_count,real_count,real_share):
    if not synthetic_count or not real_count or not 0<=real_share<1:raise ValueError('Invalid fitting mixture')
    # Preserve row-average scale for controlled comparison with v6.
    real_weight=synthetic_count*real_share/(real_count*(1-real_share))
    w=np.r_[np.ones(synthetic_count),np.full(real_count,real_weight)]
    return w/w.mean()


@dataclass
class MarketAnchoredPoints:
    columns:list
    classifier:object
    anchor_weight:float
    profile:str='points_v7_market_anchor'

    def predict_over(self,frame,lines,over_odds,under_odds):
        raw=self.classifier.predict_proba(classifier_frame(prepare(frame),lines,self.columns))[:,1]
        return anchored_probability(raw,market_probability(over_odds,under_odds),self.anchor_weight)
