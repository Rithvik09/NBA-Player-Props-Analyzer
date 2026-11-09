"""Temporal ablation and permutation diagnostics; select before final evaluation.

The 2025-26 comparison is previously exposed research, not a new blind holdout.
Individual permutations diagnose reliance, not causal feature usefulness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,mean_squared_error
from xgboost import XGBClassifier,XGBRegressor
from src.audited_features import SPECS
from src.verified_training import verified_columns,expand_points,classifier_frame,LINE_FEATURES,POLICY_VERSION
from scripts.train_verified_points import COMPACT

ROOT=Path(__file__).resolve().parents[1]
PARAMS=dict(n_estimators=250,max_depth=3,learning_rate=.04,min_child_weight=10,
    subsample=.8,colsample_bytree=.8,reg_lambda=10,random_state=42,n_jobs=4,tree_method='hist')


def group_for(column):
    if column in COMPACT:return 'compact_core'
    source=SPECS[column].source
    for group,tokens in [
        ('injuries',['injur']),('shooting',['zone','shot_prof','shot_clock','play_by_play_shots']),
        ('play_types',['playtype','play_types','synergy']),('tracking_hustle',['tracking','hustle']),
        ('rotation',['lineup','rotation','stints','substitutions','starting','bench']),
        ('clutch_quarters',['clutch','quarter','q_splits']),
        ('schedule_context',['referee','venue','schedule','calendar','broadcast','game_script']),
        ('team_matchup',['team','opposing','position']),
        ('lagged_player',['advanced','on_off','yoy','bio','profile']),
        ('scoring_pbp',['play_by_play','scoring']),('extra_boxscores',['boxscores'])]:
        if any(token in source for token in tokens):return group
    return 'other_priors'


def load_data():
    parts=[joblib.load(ROOT/f'logs/full_feature_audit/audited_{s}.joblib') for s in ['2022-23','2023-24','2024-25','2025-26']]
    if any(p['feature_policy']!=POLICY_VERSION for p in parts):
        raise ValueError('Unexpected cached feature contract')
    data={'X':pd.concat([p['X'] for p in parts],ignore_index=True)}
    for k in ['y','dates','games','players']:data[k]=np.concatenate([p[k] for p in parts])
    order=np.argsort(data['dates'],kind='stable');data['X']=data['X'].iloc[order].reset_index(drop=True)
    for k in ['y','dates','games','players']:data[k]=data[k][order]
    return data


def folds(dates):
    dates=pd.to_datetime(dates)
    return [(f'{year}_jan_feb',np.asarray(dates<f'{year}-01-01'),
             np.asarray((dates>=f'{year}-01-01')&(dates<f'{year}-03-01'))) for year in [2024,2025]]


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    data=load_data();cols=verified_columns();groups={g:[c for c in cols if group_for(c)==g] for g in sorted({group_for(c) for c in cols})}
    X,y,idx,lines=expand_points(data,cols);dates=data['dates'][idx]
    arms={'compact':COMPACT,'full':cols}
    for group,members in groups.items():
        if group=='compact_core':continue
        arms['add_'+group]=[c for c in cols if c in COMPACT or c in members]
        arms['drop_'+group]=[c for c in cols if c not in members]
    report={'protocol':'Fit before Jan 1, develop Jan-Feb in 2024 and 2025; no calibration or final evaluation rows used for feature selection.',
            'groups':groups,'parameters':PARAMS,'folds':[],'input_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'logs/full_feature_audit').glob('audited_*.joblib')}}
    (output/'protocol.json').write_text(json.dumps(report,indent=2))
    results=[];importance=[]
    for (fold,fit,dev),(_,cfit,cdev) in zip(folds(data['dates']),folds(dates)):
        for name,selected in arms.items():
            classifier=XGBClassifier(**PARAMS,eval_metric='logloss').fit(X.loc[cfit,selected+LINE_FEATURES],y[cfit])
            regressor=XGBRegressor(**PARAMS,objective='reg:squarederror').fit(data['X'].loc[fit,selected],data['y'][fit])
            p=classifier.predict_proba(X.loc[cdev,selected+LINE_FEATURES])[:,1]
            pred=regressor.predict(data['X'].loc[dev,selected])
            loss=float(log_loss(y[cdev],p));rmse=float(np.sqrt(mean_squared_error(data['y'][dev],pred)))
            results.append({'fold':fold,'arm':name,'features':len(selected),'log_loss':loss,'rmse':rmse,'player_games':int(dev.sum())})
            if name=='full':
                base=data['X'].iloc[idx[cdev]].reset_index(drop=True)
                game_rows=idx[cdev];unique=np.unique(game_rows)
                # Permute at player-game level: all five synthetic lines retain
                # the same source row. Derived line edges are recomputed.
                for column in cols:
                    changes=[]
                    for seed in [17,41]:
                        rng=np.random.default_rng(seed)
                        shuffled=unique.copy()
                        # Keep the source season/month distribution intact.
                        months=pd.DatetimeIndex(data['dates'][unique]).to_period('M')
                        for month in months.unique():
                            where=np.flatnonzero(months==month);shuffled[where]=rng.permutation(unique[where])
                        mapping=dict(zip(unique,shuffled))
                        altered=base.copy();altered[column]=data['X'][column].to_numpy()[[mapping[i] for i in game_rows]]
                        xp=classifier_frame(altered,lines[cdev],cols)
                        perm=classifier.predict_proba(xp)[:,1]
                        changes.append(float(log_loss(y[cdev],perm)-loss))
                    importance.append({'fold':fold,'feature':column,'group':group_for(column),
                        'permutation_log_loss_increase':float(np.mean(changes)),'repeat_spread':float(np.ptp(changes)),
                        'fit_missing_rate':float(data['X'].loc[fit,column].isna().mean()),
                        'development_missing_rate':float(data['X'].loc[dev,column].isna().mean())})
            pd.DataFrame(results).to_csv(output/'ablations.csv',index=False)
            print(fold,name,round(loss,5),round(rmse,4),flush=True)
        pd.DataFrame(importance).to_csv(output/'feature_diagnostics.csv',index=False)
    frame=pd.DataFrame(results);summary=[]
    for group in groups:
        if group=='compact_core':continue
        rows=[]
        for fold in frame.fold.unique():
            f=frame[frame.fold==fold].set_index('arm')
            rows.append({'fold':fold,'add_log_loss_gain':float(f.loc['compact','log_loss']-f.loc['add_'+group,'log_loss']),
                'drop_log_loss_gain':float(f.loc['full','log_loss']-f.loc['drop_'+group,'log_loss']),
                'add_rmse_gain':float(f.loc['compact','rmse']-f.loc['add_'+group,'rmse'])})
        summary.append({'group':group,'features':len(groups[group]),'folds':rows})
    report['group_results']=summary
    # Positive incremental utility in BOTH development folds, not a single lucky window.
    helpful=[r['group'] for r in summary if all(f['add_log_loss_gain']>0 for f in r['folds'])]
    reg_helpful=[r['group'] for r in summary if all(f['add_rmse_gain']>0 for f in r['folds'])]
    report['classifier_helpful_groups']=helpful;report['regressor_helpful_groups']=reg_helpful
    report['classifier_columns']=[c for c in cols if group_for(c)=='compact_core' or group_for(c) in helpful]
    report['regressor_columns']=[c for c in cols if group_for(c)=='compact_core' or group_for(c) in reg_helpful]
    (output/'selection.json').write_text(json.dumps(report,indent=2))
    print('selected groups',helpful,reg_helpful,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
