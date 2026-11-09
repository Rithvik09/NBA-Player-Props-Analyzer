"""Select points candidates on historical development, then evaluate once.

Final season was exposed in previous research and is not a fresh blind test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,mean_squared_error,mean_absolute_error
from xgboost import XGBClassifier,XGBRegressor
from scripts.investigate_points_features import load_data,folds,PARAMS
from scripts.train_verified_points import COMPACT
from scripts.points_pilot import metrics,model_hashes
from src.verified_training import (verified_columns,expand_points,LINE_FEATURES,
    SelectedPointsCandidate,ProbabilityCalibration)


def fit_calibration(method,p,y,sample_weight=None):
    if method=='isotonic':return IsotonicRegression(out_of_bounds='clip').fit(p,y,sample_weight=sample_weight)
    if method=='raw':return ProbabilityCalibration()
    p=np.clip(p,1e-7,1-1e-7)
    estimator=LogisticRegression(C=1,random_state=42).fit(np.log(p/(1-p)).reshape(-1,1),y,sample_weight=sample_weight)
    return ProbabilityCalibration('sigmoid',estimator)


def paired_losses(y,a,b,games):
    """Game-cluster paired intervals; negative means candidate is better."""
    y=np.asarray(y);a=np.clip(a,1e-7,1-1e-7);b=np.clip(b,1e-7,1-1e-7)
    deltas={'brier':(a-y)**2-(b-y)**2,
        'log_loss':-(y*np.log(a)+(1-y)*np.log(1-a))+(y*np.log(b)+(1-y)*np.log(1-b))}
    rng=np.random.default_rng(42);out={}
    for name,delta in deltas.items():
        grouped=pd.DataFrame({'game':games,'delta':delta}).groupby('game').delta.agg(['sum','count'])
        sums=grouped['sum'].to_numpy();counts=grouped['count'].to_numpy()
        draws=rng.integers(0,len(grouped),size=(1000,len(grouped)))
        samples=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        out[name]={'delta':float(delta.mean()),'interval_95':np.quantile(samples,[.025,.975]).tolist()}
    return out


def run(output,selection):
    output.mkdir(parents=True,exist_ok=False);before=model_hashes();data=load_data()
    selected=json.loads(selection.read_text());cols=verified_columns()
    X,y,idx,lines=expand_points(data,cols);dates=pd.to_datetime(data['dates']);cdates=dates[idx]
    profiles={'compact':COMPACT,'selected_classifier':selected['classifier_columns'],
              'selected_regressor':selected['regressor_columns'],'full':cols}
    settings={'original':{},'shallow':{'max_depth':2,'min_child_weight':30,'reg_lambda':30},
              'regularized':{'max_depth':3,'min_child_weight':50,'reg_lambda':50,'colsample_bytree':1}}
    protocol={'selection':'Two Jan-Feb development folds in 2024/2025, mean log loss for classifier and RMSE for regressor. Features chosen from same folds; development scores are optimistic.',
        'calibration':'Fit Mar-Apr 2025; select method on May-Jun 2025; refit chosen method Mar-Jun.',
        'evaluation':'Previously exposed 2025-26 research comparison; synthetic lines, not real bookmaker odds.',
        'profiles':profiles,'settings':settings,'parameters':PARAMS,
        'selection_sha256':hashlib.sha256(selection.read_bytes()).hexdigest(),
        'input_hashes':selected['input_hashes']}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
    for profile,columns in profiles.items():
        for setting,overrides in settings.items():
            params={**PARAMS,**overrides}
            for (fold,fit,dev),(_,cfit,cdev) in zip(folds(dates),folds(cdates)):
                c=XGBClassifier(**params,eval_metric='logloss').fit(X.loc[cfit,columns+LINE_FEATURES],y[cfit])
                r=XGBRegressor(**params,objective='reg:squarederror').fit(data['X'].loc[fit,columns],data['y'][fit])
                results.append({'profile':profile,'setting':setting,'fold':fold,
                    'log_loss':float(log_loss(y[cdev],c.predict_proba(X.loc[cdev,columns+LINE_FEATURES])[:,1])),
                    'rmse':float(np.sqrt(mean_squared_error(data['y'][dev],r.predict(data['X'].loc[dev,columns]))))})
            print(profile,setting,flush=True)
    frame=pd.DataFrame(results);frame.to_csv(output/'development.csv',index=False)
    means=frame.groupby(['profile','setting'])[['log_loss','rmse']].mean()
    cp,cs=means.log_loss.idxmin();rp,rs=means.rmse.idxmin();cc=profiles[cp];rc=profiles[rp]
    fit=np.asarray(dates<'2025-01-01');cfit=np.asarray(cdates<'2025-01-01')
    c=XGBClassifier(**{**PARAMS,**settings[cs]},eval_metric='logloss').fit(X.loc[cfit,cc+LINE_FEATURES],y[cfit])
    r=XGBRegressor(**{**PARAMS,**settings[rs]},objective='reg:squarederror').fit(data['X'].loc[fit,rc],data['y'][fit])
    calfit=np.asarray((cdates>='2025-03-01')&(cdates<'2025-05-01'))
    caldev=np.asarray((cdates>='2025-05-01')&(cdates<'2025-07-01'))
    raw=c.predict_proba(X.loc[:,cc+LINE_FEATURES])[:,1];calresults={}
    for method in ['raw','sigmoid','isotonic']:
        cal=fit_calibration(method,raw[calfit],y[calfit])
        calresults[method]=metrics(y[caldev],cal.predict(raw[caldev]))
    method=min(calresults,key=lambda k:calresults[k]['log_loss'])
    calibration=fit_calibration(method,raw[calfit|caldev],y[calfit|caldev])
    manifest={**protocol,'classifier':{'profile':cp,'setting':cs,'columns':cc},
        'regressor':{'profile':rp,'setting':rs,'columns':rc},'calibration_method':method,
        'calibration_selection':calresults,'serving_promotion':False}
    # Persist all decisions before evaluating the final season.
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    candidate=SelectedPointsCandidate(cc,r,c,calibration,feature_profile='selected_points_v2',regressor_columns=rc)
    joblib.dump(candidate,output/'candidate.joblib');candidate=joblib.load(output/'candidate.joblib')
    test=np.asarray(dates>='2025-07-01');ctest=np.asarray(cdates>='2025-07-01')
    probabilities={};report={'arms':{},'comparisons':{},'training_player_games':int(fit.sum()),'evaluation_player_games':int(test.sum())}
    for name,model in [('selected',candidate),('verified_v1',joblib.load('logs/verified_points_v1/verified.joblib')),
                       ('compact_v1',joblib.load('logs/verified_points_v1/compact_baseline.joblib'))]:
        p=model.predict_over(data['X'].iloc[idx[ctest]],lines[ctest]);pred=model.predict(data['X'].loc[test]);probabilities[name]=p
        report['arms'][name]={**metrics(y[ctest],p),'rmse':float(np.sqrt(mean_squared_error(data['y'][test],pred))),
            'mae':float(mean_absolute_error(data['y'][test],pred))}
    for name in ['verified_v1','compact_v1']:
        report['comparisons'][name]=paired_losses(y[ctest],probabilities['selected'],probabilities[name],data['games'][idx[ctest]])
    report['serving_models_unchanged']=before==model_hashes();assert report['serving_models_unchanged']
    report['candidate_sha256']=hashlib.sha256((output/'candidate.joblib').read_bytes()).hexdigest()
    report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/verified_training.py'),Path('scripts/investigate_points_features.py')]}
    pd.DataFrame({'game':data['games'][idx[ctest]],'player':data['players'][idx[ctest]],'date':data['dates'][idx[ctest]],'line':lines[ctest],'over':y[ctest],**probabilities}).to_csv(output/'predictions.csv',index=False)
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    args=p.parse_args();run(args.output,args.selection)
