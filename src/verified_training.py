"""Frozen feature selection for candidates trained before defender qualification.

Qualification changes never mutate a saved candidate's selected columns. Adding
features requires a new manifest, retraining and another evaluation.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.audited_features import SPECS, POLICY_VERSION
from src.ml_features import NUMERIC_FEATURE_KEYS

DEFERRED_DEFENDERS=frozenset('primary_defender_score01 def_rating_individual primary_defender_rating primary_defender_age defender_size_mismatch defender_recent_form primary_defender_active'.split())
PROFILE='verified_without_defenders_v1'
LINE_FEATURES=['decision_line','recent_minus_line','season_minus_line','standardized_edge']


def verified_columns():
    return [name for name in NUMERIC_FEATURE_KEYS
            if SPECS[name].status=='active' and name not in DEFERRED_DEFENDERS]


def classifier_frame(frame,lines,columns):
    """Identical construction for training and candidate inference at a line."""
    x=frame.loc[:,columns].copy()
    line=np.asarray(lines,dtype=float)
    if line.ndim==0:line=np.full(len(x),float(line))
    if len(line)!=len(x) or not np.isfinite(line).all() or (line<0).any():
        raise ValueError('One finite nonnegative decision line required per row')
    x['decision_line']=line
    x['recent_minus_line']=frame.recent_avg.to_numpy()-line
    x['season_minus_line']=frame.season_avg.to_numpy()-line
    x['standardized_edge']=(frame.recent_avg.to_numpy()-line)/frame.stddev.clip(lower=3).to_numpy()
    return x.replace([np.inf,-np.inf],np.nan)


def expand_points(data,columns):
    frame=data['X'];center=.7*frame.recent_avg.to_numpy()+.3*frame.season_avg.to_numpy()
    indexes=[];lines=[]
    for index,value in enumerate(center):
        if not np.isfinite(value):raise ValueError('Missing prior-only line center')
        choices=sorted({max(.5,float(np.floor(value)+.5+delta)) for delta in (-2,-1,0,1,2)})
        indexes.extend([index]*len(choices));lines.extend(choices)
    indexes=np.asarray(indexes);lines=np.asarray(lines)
    return classifier_frame(frame.iloc[indexes].reset_index(drop=True),lines,columns), (data['y'][indexes]>lines).astype(int), indexes,lines


def chronological_masks(dates):
    dates=pd.to_datetime(dates)
    masks={'fit':np.asarray(dates<'2025-01-01'),
           'calibration':np.asarray((dates>='2025-01-01')&(dates<'2025-07-01')),
           'evaluation':np.asarray(dates>='2025-07-01')}
    if any(not mask.any() for mask in masks.values()):raise ValueError('Empty chronological split')
    assert dates[masks['fit']].max()<dates[masks['calibration']].min()<dates[masks['evaluation']].min()
    return masks


@dataclass
class VerifiedPointsCandidate:
    columns:list
    regressor:object
    classifier:object
    calibrator:object
    feature_policy:str=POLICY_VERSION
    feature_profile:str=PROFILE

    def predict(self,frame):
        return self.regressor.predict(frame.loc[:,self.columns])

    def predict_over(self,frame,lines):
        x=classifier_frame(frame,lines,self.columns)
        return self.calibrator.predict(self.classifier.predict_proba(x)[:,1])


@dataclass
class ProbabilityCalibration:
    """Serializable identity or sigmoid calibration with a common interface."""
    method:str='raw'
    estimator:object=None

    def predict(self, probabilities):
        p=np.clip(np.asarray(probabilities),1e-7,1-1e-7)
        if self.method=='raw':return p
        if self.method=='sigmoid':
            return self.estimator.predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
        raise ValueError('Unknown probability calibration')


@dataclass
class SelectedPointsCandidate(VerifiedPointsCandidate):
    regressor_columns:object=None

    def predict(self,frame):
        return self.regressor.predict(frame.loc[:,self.regressor_columns])
