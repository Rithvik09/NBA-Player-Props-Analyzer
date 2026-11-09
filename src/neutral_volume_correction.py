"""Market-neutral shot-volume research; group offsets never fit outcome rates."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import brentq,minimize_scalar
from src.points_simple_correction import balanced_weights


def raw_volume(frame):
    v=frame.volume_fga.to_numpy(float);h=frame.games_played.to_numpy(float)
    if not np.isfinite(v).all() or not np.isfinite(h).all() or (h<1).any() or (v<0).any():raise ValueError('Complete prior volume and history required')
    return v,h<10


def market_logit(market,n):
    p=np.asarray(market,float)
    if p.shape!=(n,) or not np.isfinite(p).all() or ((p<=0)|(p>=1)).any():raise ValueError('Invalid paired market probabilities')
    return logit(p)


@dataclass
class NeutralVolume:
    mean:float
    scale:float
    beta:float
    offsets:np.ndarray
    penalty:float=1.

    def predict_over(self,frame,lines,market):
        v,short=raw_volume(frame);offset=market_logit(market,len(v))
        z=np.clip((v-self.mean)/self.scale,-5,5)
        return expit(offset+self.beta*z+self.offsets[short.astype(int)])


def fit_volume(frame,lines,market,y,games,penalty=1.):
    v,short=raw_volume(frame);offset=market_logit(market,len(v));y=np.asarray(y,float)
    if y.shape!=v.shape or set(np.unique(y))!={0.,1.} or penalty<=0:raise ValueError('Aligned binary targets and positive penalty required')
    for group in (short,~short):
        if group.sum()<100 or len(set(np.asarray(games)[group]))<20:raise ValueError('Insufficient history cohort fitting coverage')
    w=balanced_weights(games,frame.games_played);mean=float(np.sum(w*v));scale=float(np.sqrt(np.sum(w*(v-mean)**2)));scale=max(scale,1e-8);z=np.clip((v-mean)/scale,-5,5)
    def neutral_offsets(beta):
        result=np.zeros(2)
        for group in (0,1):
            m=short.astype(int)==group;wg=w[m]/w[m].sum();target=float(np.sum(wg*expit(offset[m])))
            result[group]=brentq(lambda a:float(np.sum(wg*expit(offset[m]+beta*z[m]+a)))-target,-30,30,xtol=1e-14)
        return result
    def loss(beta):
        a=neutral_offsets(beta);eta=offset+beta*z+a[short.astype(int)]
        return float(np.sum(w*(np.logaddexp(0,eta)-y*eta))+penalty*beta*beta/2)
    fit=minimize_scalar(loss,bounds=(0.,2.),method='bounded',options={'xatol':1e-12})
    if not fit.success or not np.isfinite(fit.fun):raise ValueError('Volume fit failed')
    beta=float(fit.x) if fit.fun<loss(0.) else 0.
    return NeutralVolume(mean,scale,beta,neutral_offsets(beta),penalty)
