"""Calibration-only development research. Never certifies historical independence."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from src.points_v7 import market_probability
from src.points_research_gate import calibration_support

KINDS = ('identity', 'intercept', 'intercept_slope')


def probability(raw, parameters):
    raw = np.asarray(raw, dtype=np.float64)
    a, b = parameters
    return np.clip(expit(a + b * logit(np.clip(raw, 1e-6, 1-1e-6))), 1e-6, 1-1e-6)


def fit(raw, y, kind):
    if kind == 'identity':
        return [0., 1.]
    if kind not in KINDS or len(np.unique(y)) != 2:
        raise ValueError('Two outcome classes and a known calibrator required')
    def params(v):
        return [float(v[0]), 1. if kind == 'intercept' else float(v[1])]
    result = minimize(lambda v: log_loss(y, probability(raw, params(v)), labels=[0,1]),
                      [0.] if kind == 'intercept' else [0.,1.],
                      bounds=[(None,None)] if kind == 'intercept' else [(None,None),(0,None)],
                      method='L-BFGS-B')
    if not result.success or not np.isfinite(result.fun):
        raise ValueError('Calibration optimizer failed')
    return params(result.x)


def select(raw, y, fitting, selection):
    options=[]
    for kind in KINDS:
        parameters=fit(raw[fitting],y[fitting],kind)
        score=log_loss(y[selection],probability(raw[selection],parameters),labels=[0,1])
        options.append({'kind':kind,'parameters':parameters,'selection_log_loss':float(score)})
    # Exact ties preserve the declared simpler-first order. Never refit selection labels.
    return min(options,key=lambda x:x['selection_log_loss']),options


def validate(bundle, boundary='2025-07-01', require_coverage=True):
    d=bundle['data'];q=bundle['quotes'];n=len(d['X'])
    if not d.get('provenance_validated'):raise ValueError('Unqualified source provenance')
    if any(len(d[k])!=n for k in ('games','players','dates','lines','points','y')) or len(q)!=n:
        raise ValueError('Row alignment')
    if len(set(zip(d['games'],d['players'])))!=n:raise ValueError('Duplicate player-game')
    days=pd.to_datetime(d['dates']);history=d['X'].games_played.to_numpy()
    if (days>='2026-10-01').any():raise ValueError('Protected prospective season')
    if not np.isfinite(history).all() or (history<1).any():raise ValueError('Zero or unknown history')
    if not np.isfinite(d['lines']).all() or not np.isfinite(d['points']).all():raise ValueError('Invalid outcomes/lines')
    if (d['points']==d['lines']).any():raise ValueError('Pushes excluded from binary calibration')
    if not np.array_equal(d['y'],(d['points']>d['lines']).astype(int)):raise ValueError('Label mismatch')
    for i,row in enumerate(q):
        if float(row['line'])!=float(d['lines'][i]) or pd.Timestamp(str(row['game_date']))!=days[i]:raise ValueError('Quote alignment')
    market=market_probability([r['over_odds'] for r in q],[r['under_odds'] for r in q])
    if not np.allclose(market,bundle['market'],rtol=0,atol=1e-12):raise ValueError('Market mismatch')
    fitting=np.asarray(days<pd.Timestamp(boundary));selection=~fitting
    if not fitting.any() or not selection.any():raise ValueError('Empty chronological split')
    if set(d['games'][fitting])&set(d['games'][selection]):raise ValueError('Game crosses splits')
    support=calibration_support(history,d['games'],fitting,selection)
    if require_coverage and not all(v['supported'] for v in support.values()):raise ValueError('Insufficient separate cohort coverage')
    return fitting,selection,support
