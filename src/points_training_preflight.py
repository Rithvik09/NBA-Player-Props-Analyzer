"""Mandatory gates for the next independent, early-season research runner."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.points_research_gate import calibration_support
from src.points_v7 import market_probability


def validate_inputs(bundle,exposed_games):
    d=bundle['data'];m=bundle['masks'];n=len(d['X']);days=pd.to_datetime(d['dates']);errors=[]
    if not d.get('provenance_validated'):errors.append('Unqualified input dataset')
    for key in ['dates','games','players','lines','points','y']:
        if len(d[key])!=n:raise ValueError('Dataset row alignment mismatch')
    if len(bundle['quotes'])!=n:raise ValueError('Quote alignment mismatch')
    if len(set(zip(d['games'],d['players'])))!=n:errors.append('Duplicate NBA player-game')
    if not np.array_equal(d['y'],(d['points']>d['lines']).astype(int)):errors.append('Outcome/line labels mismatch')
    if (days>='2026-10-01').any():errors.append('Protected prospective season')
    names=['fit','calfit','selection','evaluation'];masks=[np.asarray(m[k],dtype=bool) for k in names]
    if any(len(v)!=n for v in masks):raise ValueError('Split alignment mismatch')
    if (np.sum(masks,axis=0)>1).any():errors.append('Overlapping row splits')
    for a,b in zip(masks,masks[1:]):
        if not a.any() or not b.any() or days[a].max()>=days[b].min():errors.append('Nonchronological or empty split')
    for i,a in enumerate(masks):
        for b in masks[i+1:]:
            if set(d['games'][a])&set(d['games'][b]):errors.append('Game crosses splits')
    support=calibration_support(d['X'].games_played,d['games'],m['calfit'],m['selection'])
    if not all(v['supported'] for v in support.values()):errors.append('Insufficient separate short/eligible calibration coverage')
    if set(map(str,d['games'][m['evaluation']]))&set(exposed_games):errors.append('Evaluation outcomes already exposed; not a fresh holdout')
    for cohort in [d['X'].games_played.to_numpy()<10,d['X'].games_played.to_numpy()>=10]:
        mask=cohort&masks[-1]
        if mask.sum()<100 or len(set(d['games'][mask]))<20:errors.append('Insufficient independent evaluation coverage by cohort')
    for i,q in enumerate(bundle['quotes']):
        if float(q['line'])!=float(d['lines'][i]) or pd.Timestamp(str(q['game_date']))!=days[i]:errors.append('Quote line/date mismatch');break
    calculated=market_probability([q['over_odds'] for q in bundle['quotes']],[q['under_odds'] for q in bundle['quotes']])
    if not np.allclose(calculated,bundle['market'],rtol=0,atol=1e-12):errors.append('Market prices do not match baseline probabilities')
    if errors:raise ValueError('; '.join(sorted(set(errors))))
    return support


def historical_exposed_games(root=Path('logs/points_v5_data')):
    files=sorted(root.glob('20??-??.joblib'))
    if len(files)<4:raise ValueError('Incomplete exposure inventory')
    return set(str(g) for p in files for g in joblib.load(p)['games'])
