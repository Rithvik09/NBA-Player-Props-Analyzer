"""Separate offline market correction using history-available inputs."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from src.points_market_correction import market_offset


def features(frame,lines):
    n=frame.games_played.to_numpy(float);line=np.asarray(lines,float)
    # Same prior-only fields in all histories. Attenuate uncertain scoring edge;
    # no unavailable five/seven/fourteen-game feature or missingness surrogate.
    values=np.column_stack([((.7*frame.recent_avg+.3*frame.season_avg).to_numpy()-line)*n/(n+10),frame.volume_fga.to_numpy(float),np.log1p(n),frame.rest_days.to_numpy(float)])
    if not np.isfinite(values).all() or (n<1).any() or len(line)!=len(frame):raise ValueError('Complete prior-history inputs required')
    return values


def balanced_weights(games,history,separate=False):
    games=np.asarray(games);short=np.asarray(history)<10;w=np.zeros(len(games))
    cohorts=[np.ones(len(games),bool)] if separate else [short,~short]
    for m in cohorts:
        if not m.any():raise ValueError('History group absent')
        _,ids,counts=np.unique(games[m],return_inverse=True,return_counts=True)
        w[m]=1/(len(cohorts)*len(counts)*counts[ids])
    return w


@dataclass
class SimpleCorrection:
    mean:np.ndarray
    scale:np.ndarray
    beta:np.ndarray
    penalty:float

    def predict_over(self,frame,lines,market):
        z=np.c_[np.ones(len(frame)),np.clip((features(frame,lines)-self.mean)/self.scale,-5,5)]
        return expit(market_offset(market)+np.einsum('ij,j->i',z,self.beta))


def fit_simple(frame,lines,market,y,games,penalty,separate=False):
    raw=features(frame,lines);w=balanced_weights(games,frame.games_played,separate)
    mean=np.sum(raw*w[:,None],axis=0);scale=np.sqrt(np.sum((raw-mean)**2*w[:,None],axis=0));scale=np.where(scale>1e-8,scale,1.)
    z=np.c_[np.ones(len(raw)),np.clip((raw-mean)/scale,-5,5)];offset=market_offset(market);y=np.asarray(y)
    def objective(beta):
        eta=offset+np.einsum('ij,j->i',z,beta)
        return np.sum(w*(np.logaddexp(0,eta)-y*eta))+penalty*np.dot(beta,beta)/2,np.einsum('ij,i->j',z,w*(expit(eta)-y))+penalty*beta
    result=minimize(objective,np.zeros(5),jac=True,method='L-BFGS-B',bounds=[(None,None),(0,None),(None,None),(None,None),(None,None)],options={'ftol':1e-13,'gtol':1e-9})
    if not result.success:raise ValueError(result.message)
    return SimpleCorrection(mean,scale,result.x,penalty)
