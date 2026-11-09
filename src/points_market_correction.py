"""Research-only real-quote logistic corrections with a fixed market offset."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from src.points_v5 import prepare
from src.verified_training import classifier_frame
from src.points_training_repair import LINE_SIGNS


def market_offset(probabilities):
    p=np.asarray(probabilities,float)
    if not np.isfinite(p).all() or (p<=0).any() or (p>=1).any():
        raise ValueError('Finite market probabilities strictly between zero and one required')
    return logit(p)


@dataclass
class RealMarketCorrection:
    columns: list
    feature_names: list
    median: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    penalty: float
    anchored: bool

    def design(self,frame,lines):
        raw=classifier_frame(prepare(frame),lines,self.columns).loc[:,self.feature_names].to_numpy(float)
        missing=~np.isfinite(raw)
        z=np.clip((np.where(missing,self.median,raw)-self.median)/self.scale,-5,5)
        return np.c_[np.ones(len(z)),z,missing.astype(float)]

    def predict_over(self,frame,lines,market):
        offset=market_offset(market) if self.anchored else np.zeros(len(frame))
        return expit(offset+np.einsum('ij,j->i',self.design(frame,lines),self.coefficients))


def fit_correction(frame,lines,market,y,columns,penalty,anchored=True):
    if penalty<=0:raise ValueError('Positive ridge penalty required')
    y=np.asarray(y,float);offset=market_offset(market) if anchored else np.zeros(len(y))
    if len(y)!=len(frame) or set(np.unique(y))!={0.,1.}:raise ValueError('Aligned binary labels required')
    raw=classifier_frame(prepare(frame),lines,columns).replace([np.inf,-np.inf],np.nan)
    # Median and scale fitted only on supplied training rows. All-missing fields
    # get a neutral constant; no comparison distribution enters preprocessing.
    median=raw.median().fillna(0).to_numpy();values=raw.to_numpy(float);missing=~np.isfinite(values)
    filled=np.where(missing,median,values);scale=np.std(filled,axis=0);scale=np.where(scale>1e-8,scale,1.)
    candidate=RealMarketCorrection(columns,list(raw),median,scale,np.zeros(1+2*len(raw.columns)),penalty,anchored)
    z=candidate.design(frame,lines)
    def objective(beta):
        eta=offset+np.einsum('ij,j->i',z,beta)
        loss=np.sum(np.logaddexp(0,eta)-y*eta)+penalty*np.dot(beta,beta)/2
        gradient=np.einsum('ij,i->j',z,expit(eta)-y)+penalty*beta
        return loss,gradient
    bounds=[(None,None)]
    for name in raw:
        sign=LINE_SIGNS.get(name,0)
        bounds.append((0,None) if sign>0 else (None,0) if sign<0 else (None,None))
    bounds += [(None,None)]*len(raw.columns)
    fit=minimize(objective,candidate.coefficients,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-12,'gtol':1e-7})
    if not fit.success or not np.isfinite(fit.fun):raise ValueError('Correction fit failed: '+fit.message)
    candidate.coefficients=fit.x
    return candidate
