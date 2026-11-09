"""Separate research-only outcome weighting and monotone point-line contract."""
from dataclasses import dataclass
import numpy as np
from src.points_v5 import prepare
from src.verified_training import classifier_frame
from src.points_calibration_research_v2 import probability

LINE_SIGNS={'decision_line':-1,'recent_minus_line':1,'season_minus_line':1,'standardized_edge':1}


def monotone_constraints(columns):
    if not set(LINE_SIGNS).issubset(columns):raise ValueError('All line-derived fields required')
    return tuple(LINE_SIGNS.get(c,0) for c in columns)


def unique_outcomes(games,players):
    keys=list(zip(map(str,games),map(int,players)))
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate source player-game outcomes')
    return keys


def keep_synthetic(games,players,real_games,real_players):
    synthetic=unique_outcomes(games,players);real=set(unique_outcomes(real_games,real_players))
    return np.array([k not in real for k in synthetic],dtype=bool)


def outcome_weights(expanded_indices,real_count):
    """Each original outcome totals one unit, independent of expansion count."""
    indices=np.asarray(expanded_indices,dtype=int)
    _,inverse,counts=np.unique(indices,return_inverse=True,return_counts=True)
    return np.r_[1./counts[inverse],np.ones(real_count)]


def assert_disjoint_games(*groups):
    seen=set()
    for group in groups:
        current=set(map(str,group))
        if current&seen:raise ValueError('Outcome game crosses training/calibration/evaluation')
        seen.update(current)


@dataclass
class RepairedMonotonePoints:
    columns:list
    classifier:object
    calibrations:dict

    def predict_over(self,frame,lines):
        x=prepare(frame);raw=self.classifier.predict_proba(classifier_frame(x,lines,self.columns))[:,1]
        out=np.empty(len(x));short=x.games_played.to_numpy()<10
        for cohort,m in [('short_history',short),('eligible_history',~short)]:out[m]=probability(raw[m],self.calibrations[cohort])
        return out
