"""Chronological points pilot. Candidates never overwrite serving artifacts.

Synthetic half-point lines are derived from prior player results, not historical
bookmaker odds. Metrics therefore cannot establish betting profitability.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss,mean_squared_error

from scripts.train_models import (_load_precomputed_player_data,build_training_examples,create_classifier,create_regressor)
from src.historical_inputs import HistoricalInputs
from src.ml_features import build_classifier_vector

ROOT=Path(__file__).resolve().parents[1]
SEASONS=['2022-23','2023-24','2024-25','2025-26']
OUT=ROOT/'logs/points_pilot'


def split_dates(dates):
    dates=pd.to_datetime(dates)
    return {'fit':np.asarray(dates<'2025-01-01'),
            'calibration':np.asarray((dates>='2025-01-01')&(dates<'2025-07-01')),
            'test':np.asarray(dates>='2025-07-01')}


def classifier_rows(X,y,dates,games):
    frames=[];labels=[];row_dates=[];groups=[];sources=[]
    for i,raw in enumerate(X.to_dict('records')):
        center=.7*raw['recent_avg']+.3*raw['season_avg']
        for line in sorted({max(.5,float(np.floor(center)+.5+delta)) for delta in (-2,-1,0,1,2)}):
            frames.append(build_classifier_vector(raw,line=line).X)
            labels.append(int(y[i]>line));row_dates.append(dates[i]);groups.append(games[i]);sources.append(i)
    return pd.concat(frames,ignore_index=True),np.array(labels),pd.to_datetime(row_dates),np.array(groups),np.array(sources)


def metrics(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    bins=[]
    bucket=np.minimum((p*10).astype(int),9)
    for index in range(10):
        lo=index/10
        mask=bucket==index
        if mask.any():bins.append({'lower':round(float(lo),1),'n':int(mask.sum()),'predicted':float(p[mask].mean()),'observed':float(y[mask].mean())})
    return {'auc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p)),
            'log_loss':float(log_loss(y,p)),'ece':float(sum(b['n']*abs(b['predicted']-b['observed']) for b in bins)/len(y)),
            'calibration_bins':bins}


def bootstrap(y,a,b,groups,iterations=400):
    rng=np.random.default_rng(42);unique=np.unique(groups)
    indexes={key:np.flatnonzero(groups==key) for key in unique}
    delta=[]
    for _ in range(iterations):
        idx=np.concatenate([indexes[key] for key in rng.choice(unique,len(unique),replace=True)])
        delta.append(roc_auc_score(y[idx],a[idx])-roc_auc_score(y[idx],b[idx]))
    return {'unit':'NBA game; all players and synthetic lines stay together','iterations':iterations,
            'auc_delta_95_interval':np.quantile(delta,[.025,.975]).tolist(),
            'auc_delta':float(roc_auc_score(y,a)-roc_auc_score(y,b))}


def model_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'models').rglob('*') if p.is_file() and p.suffix in ('.joblib','.json','.txt')}


def build_dataset(raw,snapshots,players):
    digest=hashlib.sha256((Path(__file__).read_text()+ (ROOT/'scripts/train_models.py').read_text() + (ROOT/'src/historical_inputs.py').read_text() + joblib.hash(raw)).encode()).hexdigest()[:12]
    path=OUT/f"{'lagged' if snapshots else 'baseline'}_{players}_{digest}.joblib"
    if path.exists():return joblib.load(path)
    frames=[];targets=[];dates=[];games=[];player_ids=[];audit=[]
    for season in SEASONS:
        logs=pd.read_json(ROOT/f'logs/points_pilot_cache/players_{season}.json',dtype={'GAME_ID':str})
        inputs=HistoricalInputs(season,raw,logs,snapshots=snapshots)
        exs=build_training_examples(season,players,None,None,historical_inputs=inputs,prop_types=('points',))
        if not exs:raise ValueError('No examples: '+season)
        frames.extend(e.X for e in exs);targets.extend(e.y for e in exs)
        dates.extend(e.game_date for e in exs);games.extend(e.game_id for e in exs);player_ids.extend(e.player_id for e in exs)
        audit.append({'season':season,'snapshot_source_season':inputs.source_season,'snapshot_rows':inputs.source_rows,
                      'examples':len(exs),'cohort_selected_from_prior_season':True,'history_window':inputs.history_window,
                      'team_context_queries':inputs.context_calls,'last_source_game_date':inputs.max_source_date})
        print('built',snapshots,season,len(exs),flush=True)
    data={'X':pd.concat(frames,ignore_index=True),'y':np.array(targets),'dates':pd.to_datetime(dates),
          'games':np.array(games),'players':np.array(player_ids),'audit':audit}
    order=np.argsort(np.asarray(data['dates']),kind='stable')
    data['X']=data['X'].iloc[order].reset_index(drop=True)
    for key in ('y','dates','games','players'):data[key]=data[key][order]
    joblib.dump(data,path)
    return data


def run(players=50):
    OUT.mkdir(parents=True,exist_ok=True)
    before=model_hashes();report={'started_at':time.time(),'players_per_season':players,
        'protocol':{'fit':'2022-10 through 2024-12','calibration':'2025-01 through 2025-06',
                    'test':'2025-26 regular season','lines':'Five synthetic half-point lines around prior rolling average; no bookmaker odds',
                    'expansion_gate':'Paired game-cluster bootstrap AUC improvement interval above zero and calibrated Brier no worse than baseline',
                    'legacy_auc_0_571':'Different historical protocol; not a matched benchmark',
                    'serving_model_training_cutoff':'Unknown; frozen model scores are descriptive only'}}
    raw=_load_precomputed_player_data(os.environ['DATABASE_URL'])
    predictions={};reference=None
    for name,snapshots in [('baseline',False),('lagged_sources',True)]:
        data=build_dataset(raw,snapshots,players)
        if reference is None:reference=data
        else:
            assert np.array_equal(data['games'],reference['games']) and np.array_equal(data['players'],reference['players'])
            assert np.array_equal(data['y'],reference['y'])
        X,y,dates,groups,source=classifier_rows(data['X'],data['y'],data['dates'],data['games'])
        masks=split_dates(dates);reg_masks=split_dates(data['dates'])
        assert dates[masks['fit']].max()<dates[masks['calibration']].min()<dates[masks['test']].min()
        clf=create_classifier();clf.set_params(n_jobs=4)
        clf.fit(X[masks['fit']],y[masks['fit']])
        cal_raw=clf.predict_proba(X[masks['calibration']])[:,1]
        iso=IsotonicRegression(out_of_bounds='clip').fit(cal_raw,y[masks['calibration']])
        raw_test=clf.predict_proba(X[masks['test']])[:,1];cal_test=iso.predict(raw_test)
        reg=create_regressor();reg.set_params(n_jobs=4)
        reg.fit(data['X'][reg_masks['fit']],data['y'][reg_masks['fit']])
        reg_pred=reg.predict(data['X'][reg_masks['test']])
        result={'raw':metrics(y[masks['test']],raw_test),'calibrated':metrics(y[masks['test']],cal_test),
                'regression_rmse':float(np.sqrt(mean_squared_error(data['y'][reg_masks['test']],reg_pred))),
                'player_games_by_split':{k:int(v.sum()) for k,v in reg_masks.items()},
                'classifier_rows_by_split':{k:int(v.sum()) for k,v in masks.items()},'source_audit':data['audit']}
        report[name]=result;predictions[name]=cal_test
        joblib.dump({'classifier':clf,'calibrator':iso,'regressor':reg},OUT/f'{name}_candidate.joblib')
        if snapshots:
            for file,label in [('clf_raw_points.joblib','frozen_raw'),('clf_cal_points.joblib','frozen_calibrated')]:
                old=joblib.load(ROOT/'models'/file);p=old.predict_proba(X[masks['test']])[:,1]
                report[label]=metrics(y[masks['test']],p)
            pd.DataFrame({'game_id':groups[masks['test']],'date':dates[masks['test']],
                          'actual_over':y[masks['test']],'baseline':predictions['baseline'],
                          'candidate':cal_test}).to_csv(OUT/'holdout_predictions.csv',index=False)
        (OUT/'report.partial.json').write_text(json.dumps(report,indent=2,default=str))
        print(name,result['calibrated'],flush=True)
    comparison=bootstrap(y[masks['test']],predictions['lagged_sources'],predictions['baseline'],groups[masks['test']])
    report['comparison']=comparison
    report['expand_to_all_props']=bool(comparison['auc_delta_95_interval'][0]>0 and
        report['lagged_sources']['calibrated']['brier']<=report['baseline']['calibrated']['brier'])
    assert model_hashes()==before,'Serving model artifacts changed'
    report['serving_models_unchanged']=True;report['finished_at']=time.time()
    (OUT/'report.json').write_text(json.dumps(report,indent=2,default=str));print('DONE',report['comparison'],report['expand_to_all_props'],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--players',type=int,default=50)
    args=parser.parse_args();run(args.players)
