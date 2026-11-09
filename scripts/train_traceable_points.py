"""Offline, versioned real-line points experiment with exact replay and refit."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from src.serving_provenance import ROOT, file_hash, safe
from src.training_provenance import record_training_inputs
from src.points_calibration_research_v2 import validate
from src.points_simple_correction import fit_simple, features, balanced_weights
from src.points_market_correction import fit_correction
from src.points_training_repair import assert_disjoint_games
from scripts.points_pilot import metrics
from scripts.train_selected_points import paired_losses

INPUT=ROOT/'logs/points_evidence_representation_01/inputs.joblib'
PRIOR=ROOT/'logs/points_market_correction_02'
BOUNDARY='2025-07-01'


def write(path,value):
    with path.open('x') as f:json.dump(safe(value),f,indent=2,allow_nan=False)


def masks(bundle):
    validate(bundle)
    d=bundle['data'];train=np.asarray(pd.to_datetime(d['dates'])<BOUNDARY);test=~train
    assert_disjoint_games(d['games'][train],d['games'][test])
    features(d['X'],d['lines'])
    return train,test


def fit_models(bundle,train,columns):
    d=bundle['data'];x=d['X'].loc[train];lines=d['lines'][train];market=bundle['market'][train];y=d['y'][train]
    return {'candidate':fit_simple(x,lines,market,y,d['games'][train],penalty=1.),
            'compact_real':fit_correction(x,lines,market,y,columns,penalty=100.,anchored=False)}


def predictions(models,bundle,test):
    d=bundle['data'];return {k:m.predict_over(d['X'].loc[test],d['lines'][test],bundle['market'][test]) for k,m in models.items()}


def protected():
    old=json.loads((ROOT/'logs/real_points_paid_01/frozen_before.json').read_text())
    paths=set(old['hashes'])|set(old['serving'])
    for name in ('prospective_points_v1','prospective_points_v5'):
        p=ROOT/'logs'/name/'protocol.json';v=json.loads(p.read_text());paths.add(str(p.relative_to(ROOT)))
        paths.update(v.get('source_hashes',{}))
        if 'candidate_path' in v:paths.add(v['candidate_path'])
    for folder in ('logs/source_blocked/assignment_ranker','logs/prospective_points_v1','logs/prospective_points_v5'):
        paths.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file())
    return {p:file_hash(ROOT/p) for p in sorted(paths)}


def verify(output,refit=False):
    m=json.loads((output/'manifest.json').read_text())
    for p,h in m['files'].items():
        if file_hash(output/p)!=h:raise ValueError('Experiment file drift: '+p)
    current=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    if current!=(output/'requirements.lock').read_text():raise ValueError('Environment drift')
    b=joblib.load(output/'inputs.joblib');train,test=masks(b)
    models=joblib.load(output/'models.joblib');p=predictions(models,b,test)
    expected=json.loads((output/'predictions.json').read_text())
    for k in p:
        if not np.array_equal(p[k],np.asarray(expected[k])):raise ValueError('Prediction replay mismatch: '+k)
    if refit:
        fitted=fit_models(b,train,m['compact_columns'])
        for k,v in predictions(fitted,b,test).items():
            if not np.array_equal(v,p[k]):raise ValueError('Refit mismatch: '+k)
    return {'exact_prediction_replay':True,'exact_refit':refit,'comparison_rows':int(test.sum())}


def run(output):
    # Every source is local; make accidental network access a hard error.
    socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Network forbidden in traceable training'))
    before=protected()
    b=joblib.load(INPUT);train,test=masks(b);d=b['data'];x=d['X'];short=x.games_played.to_numpy()<10
    prior=json.loads((PRIOR/'plan.json').read_text())
    if file_hash(INPUT)!=prior['input_sha256']:raise ValueError('Cached baseline uses a different input dataset')
    construction=json.loads((ROOT/'logs/points_evidence_representation_01/construction.json').read_text())
    for group in ('raw_source_hashes','dataset_hashes'):
        for p,h in construction[group].items():
            if file_hash(ROOT/p)!=h:raise ValueError('Underlying source drift: '+p)
    columns=prior['groups']['compact']
    output.mkdir(parents=True,exist_ok=False)
    support={name:{'rows':int(mask.sum()),'games':len(set(d['games'][mask])),
             'short_rows':int((mask&short).sum()),'short_games':len(set(d['games'][mask&short])),
             'eligible_rows':int((mask&~short).sum()),'eligible_games':len(set(d['games'][mask&~short]))}
             for name,mask in [('training',train),('comparison',test)]}
    write(output/'plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),'scope':'Points probabilities conditional on a real line and paired no-vig market price; research candidate, not a 14-prop drop-in replacement.',
        'historical_independent':False,'promoted':False,'training_before':BOUNDARY,'support':support,
        'candidate':'Fixed four-feature SimpleCorrection plus intercept; market logit offset; mean-loss ridge1; no hyperparameter search or post-fit calibration.',
        'candidate_features':['history_attenuated_scoring_edge','volume_fga','log1p_games_played','rest_days'],
        'compact':'Existing 19-column compact/history/missingness contract, direct logistic prediction, sum-loss ridge100; retrained on identical real training rows. This is a control, not the frozen16 model.',
        'weighting':'Candidate: 50% mass per history cohort, equal mass per game within cohort. Compact retains established equal-player-game objective. Different weighting is part of the model comparison, not a feature-only ablation.',
        'missing':'Candidate fails on missing required features. Compact uses training-only median imputation, missing indicators and zero for all-missing columns.',
        'validation':'All outcome dates before protected2026-27 season. No synthetic expansions. No game overlap or duplicate player-games. At least100 rows and20 games in each history cohort per split.',
        'selection':'No new arm selection, tuning or promotion. July2025 onward comparison outcomes were previously exposed. Independent future validation remains necessary.',
        'input_sha256':file_hash(INPUT),'frozen_before':before})
    shutil.copy2(INPUT,output/'inputs.joblib')
    # Capture exact code, including dirty working-tree files, not only a Git SHA.
    for folder in ('src','scripts'):
        for p in (ROOT/folder).glob('*.py'):
            target=output/'source'/folder/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    (output/'requirements.lock').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    shutil.copy2(ROOT/'logs/points_evidence_representation_01/construction.json',output/'input_construction.json')
    shutil.copy2(PRIOR/'plan.json',output/'cached_baseline_plan.json');shutil.copy2(PRIOR/'predictions.csv',output/'cached_baselines.csv')
    write(output/'row_assignments.json',[{'index':i,'game':str(d['games'][i]),'player':int(d['players'][i]),'date':str(d['dates'][i]),'split':'training' if train[i] else 'comparison','line':float(d['lines'][i]),'points':float(d['points'][i]),'label':int(d['y'][i]),'quote':b['quotes'][i]} for i in range(len(x))])
    examples=[]
    for i in np.flatnonzero(train):
        frame=x.iloc[[i]].copy();frame['_real_line']=d['lines'][i];frame['_market_probability']=b['market'][i]
        examples.append(SimpleNamespace(prop_type='points',player_id=int(d['players'][i]),game_id=str(d['games'][i]),game_date=pd.Timestamp(d['dates'][i]),y=int(d['y'][i]),minutes=0.,feature_policy=d['feature_policy'],X=frame))
    write(output/'training_provenance.json',record_training_inputs(examples,{'candidate_penalty':1.,'compact_penalty':100.,'boundary':BOUNDARY,'compact_columns':columns,'random_search':False}))
    w=balanced_weights(d['games'][train],x.games_played.to_numpy()[train])
    write(output/'training_weights.json',{'candidate':w.tolist(),'compact':np.ones(train.sum()).tolist(),'training_row_indices':np.flatnonzero(train).tolist()})
    models=fit_models(b,train,columns);joblib.dump(models,output/'models.joblib')
    write(output/'parameters.json',{k:vars(v) for k,v in models.items()})
    # Fitting is finished and serialized before comparison outcomes are scored.
    probs=predictions(models,b,test)
    probs.update(market=b['market'][test],coin_flip=np.full(test.sum(),.5))
    cached=pd.read_csv(output/'cached_baselines.csv',dtype={'game':str,'player':int}).set_index(['game','player'])
    keys=pd.MultiIndex.from_arrays([d['games'][test].astype(str),d['players'][test].astype(int)],names=['game','player'])
    if cached.index.duplicated().any() or len(cached)!=test.sum() or set(cached.index)!=set(keys):raise ValueError('Cached baseline identity mismatch')
    cached=cached.reindex(keys)
    if not np.array_equal(cached.over.to_numpy(),d['y'][test]) or not np.allclose(cached.market,probs['market'],atol=1e-14,rtol=0):raise ValueError('Cached baseline label/market mismatch')
    probs['frozen_compact']=cached.frozen_compact.to_numpy()
    monotonic={}
    for name,model in models.items():
        last=None;violations=0
        for offset in np.arange(-5,5.01,.5):
            p=model.predict_over(x.loc[test],np.maximum(.5,d['lines'][test]+offset),b['market'][test])
            if last is not None:violations+=int((p>last+1e-10).sum())
            last=p
        if violations:raise ValueError('Line monotonicity failure')
        monotonic[name]={'comparisons':int(test.sum())*20,'violations':violations,'scope':'Fixed market probability'}
    report={'historical_independent':False,'promoted':False,'support':support,'monotonicity':monotonic,'cohorts':{}}
    for cohort,mask in [('all',np.ones(test.sum(),bool)),('short_history',short[test]),('eligible_history',~short[test])]:
        y=d['y'][test][mask];games=d['games'][test][mask]
        report['cohorts'][cohort]={'rows':len(y),'games':len(set(games)),'metrics':{k:metrics(y,v[mask]) for k,v in probs.items()},
            'candidate_paired':{k:paired_losses(y,probs['candidate'][mask],v[mask],games) for k,v in probs.items() if k!='candidate'}}
    write(output/'predictions.json',probs);write(output/'report.json',report)
    if before!=protected():raise ValueError('Protected evidence changed during run')
    write(output/'integrity.json',{'protected_unchanged':True,'credits_spent':0})
    files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}
    write(output/'manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),'python':sys.version,'compact_columns':columns,'files':files,'credits_spent':0,'historical_independent':False,'promoted':False})
    write(output/'reproduction.json',verify(output,refit=True))
    print(json.dumps({'support':support,'metrics':{k:{z:v[z] for z in ('log_loss','brier','auc')} for k,v in report['cohorts']['all']['metrics'].items()}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');p.add_argument('--refit',action='store_true');a=p.parse_args()
    if a.verify:print(json.dumps(verify(a.output,a.refit),indent=2))
    else:run(a.output)
