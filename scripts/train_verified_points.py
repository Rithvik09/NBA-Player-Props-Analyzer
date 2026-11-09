"""Train the first verified-feature candidate without changing serving models."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from xgboost import XGBClassifier,XGBRegressor
from src.verified_training import (verified_columns,expand_points,chronological_masks,
    VerifiedPointsCandidate,PROFILE,POLICY_VERSION,DEFERRED_DEFENDERS,LINE_FEATURES)
from src.audited_features import SPECS
from scripts.points_pilot import metrics,bootstrap,model_hashes

ROOT=Path(__file__).resolve().parents[1]
COMPACT=['recent_avg','season_avg','stddev','last_3_games_trend','last_5_games_trend',
         'last_10_games_trend','minutes_last_3_games','minutes_last_5_games',
         'minutes_last_7_games','rolling_7day_avg','rolling_14day_avg','rolling_30day_avg',
         'team_pace','opp_def_rating','rest_days','is_home_game_num']


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    before=model_hashes();parts=[];hashes={}
    for season in ['2022-23','2023-24','2024-25','2025-26']:
        path=ROOT/f'logs/full_feature_audit/audited_{season}.joblib'
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        part=joblib.load(path)
        if part['feature_policy']!=POLICY_VERSION:raise ValueError('Unexpected cached feature contract')
        parts.append(part)
    data={'X':pd.concat([p['X'] for p in parts],ignore_index=True)}
    for key in ['y','dates','games','players']:data[key]=np.concatenate([p[key] for p in parts])
    order=np.argsort(data['dates'],kind='stable');data['X']=data['X'].iloc[order].reset_index(drop=True)
    for key in ['y','dates','games','players']:data[key]=data[key][order]
    columns=verified_columns();split=chronological_masks(data['dates'])
    # No information from evaluation outcomes selects features or parameters.
    empty=[c for c in columns if data['X'].loc[split['fit'],c].notna().sum()==0]
    columns=[c for c in columns if c not in empty]
    params=dict(n_estimators=250,max_depth=3,learning_rate=.04,min_child_weight=10,
        subsample=.8,colsample_bytree=.8,reg_lambda=10,random_state=42,n_jobs=4,tree_method='hist')
    report={'created_at':datetime.now(timezone.utc).isoformat(),'feature_policy':POLICY_VERSION,
        'feature_profile':PROFILE,'scope':'points candidate only; serving models not promoted',
        'columns':columns,'classifier_extra_columns':LINE_FEATURES,'fit_all_missing_excluded':empty,
        'excluded':{k:('awaiting current-season defender qualification' if k in DEFERRED_DEFENDERS else SPECS[k].status)
                    for k in SPECS if k not in columns and k not in empty},
        'feature_count':len(columns),'input_sha256':hashes,'parameters':params,
        'protocol':{'fit':'before 2025-01-01','calibration':'2025-01-01 through 2025-06-30',
            'evaluation':'2025-26 regular season and postseason; previously exposed retrospective benchmark, not a fresh blind test',
            'lines':'prior-only synthetic half-point lines; not actual bookmaker odds',
            'selection':'fixed parameters; verified and compact arms predefined, no test-set selection',
            'future_features':'New versioned manifest and full retraining after qualification; never hot-add columns'},
        'player_games_by_split':{k:int(v.sum()) for k,v in split.items()},'arms':{}}
    (output/'manifest.json').write_text(json.dumps(report,indent=2))
    probabilities={};regpred={}
    for name,cols in [('verified',columns),('compact_baseline',[c for c in COMPACT if c in columns])]:
        X,y,indexes,lines=expand_points(data,cols);masks=chronological_masks(data['dates'][indexes])
        classifier=XGBClassifier(**params,eval_metric='logloss')
        classifier.fit(X.loc[masks['fit']],y[masks['fit']])
        calibrator=IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(classifier.predict_proba(X.loc[masks['calibration']])[:,1],y[masks['calibration']])
        regressor=XGBRegressor(**params,objective='reg:squarederror')
        regressor.fit(data['X'].loc[split['fit'],cols],data['y'][split['fit']])
        candidate=VerifiedPointsCandidate(cols,regressor,classifier,calibrator)
        joblib.dump(candidate,output/f'{name}.joblib')
        # Test the saved public prediction interface, not a separate eval path.
        candidate=joblib.load(output/f'{name}.joblib')
        test=split['evaluation'];ctest=masks['evaluation']
        p=candidate.predict_over(data['X'].iloc[indexes[ctest]],lines[ctest]);pred=candidate.predict(data['X'].loc[test])
        probabilities[name]=p;regpred[name]=pred
        raw=classifier.predict_proba(X.loc[ctest])[:,1]
        report['arms'][name]={'raw':metrics(y[ctest],raw),'calibrated':metrics(y[ctest],p),
            'rmse':float(np.sqrt(mean_squared_error(data['y'][test],pred))),
            'mae':float(mean_absolute_error(data['y'][test],pred)),
            'feature_count':len(cols),'classifier_rows_by_split':{k:int(v.sum()) for k,v in masks.items()}}
        print(name,json.dumps(report['arms'][name]),flush=True)
    report['comparison']=bootstrap(y[ctest],probabilities['verified'],probabilities['compact_baseline'],data['games'][indexes[ctest]])
    report['improves_over_compact_on_this_benchmark']=bool(report['comparison']['auc_delta_95_interval'][0]>0 and
        report['arms']['verified']['calibrated']['brier']<=report['arms']['compact_baseline']['calibrated']['brier'])
    report['serving_models_unchanged']=before==model_hashes();assert report['serving_models_unchanged']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'src/verified_training.py',Path(__file__),ROOT/'src/audited_features.py',ROOT/'src/recovered_features.py']}
    report['candidate_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.joblib')}
    pd.DataFrame({'game_id':data['games'][indexes[ctest]],'player_id':data['players'][indexes[ctest]],
        'date':data['dates'][indexes[ctest]],'synthetic_line':lines[ctest],'actual_over':y[ctest],
        **probabilities}).to_csv(output/'evaluation_predictions.csv',index=False)
    report['data_manifest_sha256']=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'completed':True,'output':str(output),'feature_count':len(columns),
                     'comparison':report['comparison'],'serving_models_unchanged':True}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
