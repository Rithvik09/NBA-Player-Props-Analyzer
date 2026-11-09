"""Offline constrained correction ablations; original models remain unchanged."""
from dataclasses import dataclass
from itertools import combinations
from math import factorial
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from src.points_simple_correction import features,balanced_weights,SimpleCorrection

NAMES=['intercept','scoring_edge','shot_volume','history','rest']


def fit_arm(x,lines,market,y,games,active,penalty=1.,weighting='balanced'):
    raw=features(x,lines);y=np.asarray(y,float);market=np.asarray(market,float)
    if len(y)!=len(x) or set(np.unique(y))!={0.,1.} or not np.isfinite(market).all() or ((market<=0)|(market>=1)).any():raise ValueError('Invalid labels or market')
    w=balanced_weights(games,x.games_played) if weighting=='balanced' else np.full(len(x),1/len(x))
    mean=np.sum(raw*w[:,None],axis=0);scale=np.sqrt(np.sum((raw-mean)**2*w[:,None],axis=0));scale=np.where(scale>1e-8,scale,1.)
    z=np.c_[np.ones(len(x)),np.clip((raw-mean)/scale,-5,5)];offset=logit(market)
    def objective(beta):
        eta=offset+np.einsum('ij,j->i',z,beta)
        return np.sum(w*(np.logaddexp(0,eta)-y*eta))+penalty*np.dot(beta,beta)/2,np.einsum('ij,i->j',z,w*(expit(eta)-y))+penalty*beta
    bounds=[((0,None) if i==1 else (None,None)) if i in active else (0,0) for i in range(5)]
    result=minimize(objective,np.zeros(5),jac=True,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-13,'gtol':1e-9,'maxiter':3000})
    if not result.success:raise ValueError(result.message)
    return SimpleCorrection(mean,scale,result.x,penalty)


def terms(model,x,lines):
    return np.c_[np.ones(len(x)),np.clip((features(x,lines)-model.mean)/model.scale,-5,5)]*model.beta


def shapley_losses(y,market,contributions):
    """Exact per-row log-loss attribution across all32 fixed-coefficient subsets."""
    y=np.asarray(y);base=logit(market);losses={}
    for mask in range(32):
        eta=base+contributions[:,[i for i in range(5) if mask&(1<<i)]].sum(axis=1)
        losses[mask]=np.logaddexp(0,eta)-y*eta
    values=np.zeros_like(contributions)
    for i in range(5):
        for mask in range(32):
            if mask&(1<<i):continue
            k=mask.bit_count();w=factorial(k)*factorial(4-k)/factorial(5)
            values[:,i]+=w*(losses[mask|(1<<i)]-losses[mask])
    if not np.allclose(values.sum(axis=1),losses[31]-losses[0],atol=1e-12):raise ValueError('Attribution mismatch')
    return values
