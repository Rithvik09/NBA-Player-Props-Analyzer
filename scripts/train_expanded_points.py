"""Fair expanded-cohort comparison with development-only feature selection.

No serving promotion. Evaluation season is previously exposed research.
"""
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss,mean_squared_error,mean_absolute_error
from xgboost import XGBClassifier,XGBRegressor
from src.verified_training import verified_columns,expand_points,LINE_FEATURES,SelectedPointsCandidate
from src.audited_features import POLICY_VERSION
from src.points_candidate import ExpandedPointsCandidate,normalize_points_inputs,NORMALIZATION
from scripts.investigate_points_features import PARAMS,folds
from scripts.train_verified_points import COMPACT
from scripts.train_selected_points import fit_calibration,paired_losses
from scripts.points_pilot import metrics,bootstrap,model_hashes


def load_expanded(directory):
    parts=[joblib.load(directory/f'audited_{s}.joblib') for s in ['2022-23','2023-24','2024-25','2025-26']]
    if any(p['feature_policy']!=POLICY_VERSION for p in parts):raise ValueError('Mixed feature policies')
    data={'X':pd.concat([p['X'] for p in parts],ignore_index=True)}
    for k in ['y','dates','games','players','original_cohort']:data[k]=np.concatenate([p[k] for p in parts])
    order=np.argsort(data['dates'],kind='stable');data['X']=data['X'].iloc[order].reset_index(drop=True)
    for k in ['y','dates','games','players','original_cohort']:data[k]=data[k][order]
    return data


def profiles(importance=None,previous_columns=None):
    if importance is None:importance=pd.read_csv('logs/points_feature_investigation_v1/feature_diagnostics.csv')
    bad=set(importance.groupby('feature').permutation_log_loss_increase.max().loc[lambda s:s<0].index)
    old=previous_columns if previous_columns is not None else json.loads(Path('logs/selected_points_v2/manifest.json').read_text())['classifier']['columns']
    cols=verified_columns()
    return {'compact':COMPACT,'full_repaired':cols,'selected183':old,
            'pruned': [c for c in cols if c not in bad or c in COMPACT],
            'selected_pruned':[c for c in old if c not in bad or c in COMPACT]},sorted(bad)


def make_protocol(directory):
    groups,bad=profiles()
    return {'feature_policy':POLICY_VERSION,'input_normalization':NORMALIZATION,'profiles':groups,'suspect_features':bad,
        'settings':{'original':{},'shallow':{'max_depth':2,'min_child_weight':30,'reg_lambda':30},
                    'regularized':{'max_depth':3,'min_child_weight':50,'reg_lambda':50,'colsample_bytree':1}},
        'parameters':PARAMS,
        'training_cohorts':{'expanded':'All eligible rows, uniform weight','expanded_weighted':'Original cohort weight1; added players weight0.25; fit weights normalized to mean1','original':'Original cohort only'},
        'fit':'Before January 2024/2025 for the two Jan-Feb development folds; final fit before January 2025.',
        'selection':'Compare equal regularization budgets for each profile. Select by average development log loss / RMSE independently. Compact and rich profiles receive identical per-profile tuning budgets. The overall recommended classifier/regressor minimize development loss/RMSE across all profiles and cohorts, including compact controls.',
        'calibration':'Fit Mar-Apr 2025; choose raw/sigmoid/isotonic by May-Jun log loss; refit selected method Mar-Jun. Fit/refit on the respective eligible cohort; select calibration on the common original cohort.',
        'evaluation':'2025-26 previously exposed retrospective benchmark; primary matched original cohort, secondary expanded population. No current season outcomes used.',
        'meaningful_improvement_rule':'On original cohort: log-loss gain >=0.002 vs equally tuned expanded compact with game-cluster 95% CI excluding zero; Brier and RMSE no worse. Expanded-population Brier and log loss must also be no worse. This is a research threshold, not a deployment or profit claim.',
        'line_policy':'Existing prior-only synthetic half-point lines. All arms use identical quotes and outcome rows; not actual bookmaker odds.',
        'dataset_manifest_sha256':hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
        'prior_diagnostics_sha256':hashlib.sha256(Path('logs/points_feature_investigation_v1/feature_diagnostics.csv').read_bytes()).hexdigest()}


def cohort_weights(original,cohort):
    original=np.asarray(original,dtype=bool)
    if cohort=='original':return original.astype(float)
    if cohort=='expanded':return np.ones(len(original))
    if cohort=='expanded_weighted':return np.where(original,1.,.25)
    raise ValueError('Unknown cohort')


def normalized_weights(weights,mask):
    w=weights[mask]
    if not len(w) or w.mean()<=0:raise ValueError('Empty or zero-weight fit')
    return w/w.mean()


def run(directory,output):
    before=model_hashes();protocol=json.loads((output/'protocol.json').read_text())
    if protocol!=make_protocol(directory):raise ValueError('Protocol or data manifest changed')
    if not (directory/'report.json').exists():raise ValueError('Dataset build is incomplete')
    data=load_expanded(directory)
    # XGBoost uses float32 internally; avoid duplicating a large float64 matrix.
    data['X']=normalize_points_inputs(data['X']).astype(np.float32)
    X,y,idx,lines=expand_points(data,verified_columns());X=X.astype(np.float32)
    dates=pd.to_datetime(data['dates']);cdates=dates[idx];core=data['original_cohort'].astype(bool);ccore=core[idx]
    groups=protocol['profiles'];settings=protocol['settings'];results=[]
    arms=[(name,cohort) for name in groups for cohort in ['expanded','expanded_weighted']]+[('compact','original')]
    for name,cohort in arms:
        cols=groups[name];weights=cohort_weights(core,cohort);cweights=weights[idx];eligible=weights>0;celigible=eligible[idx]
        for setting,override in settings.items():
            for (fold,fit,dev),(_,cfit,cdev) in zip(folds(dates),folds(cdates)):
                checkpoint=output/f'dev_{name}_{cohort}_{setting}_{fold}.json'
                if checkpoint.exists():results.append(json.loads(checkpoint.read_text()));continue
                fit=fit&eligible;cfit=cfit&celigible
                # Always select on the SAME original cohort: more bench outcomes
                # cannot improve the score merely by changing the evaluation mix.
                dev=dev&core;cdev=cdev&ccore
                params={**PARAMS,**override}
                c=XGBClassifier(**params,eval_metric='logloss').fit(X.loc[cfit,cols+LINE_FEATURES],y[cfit],sample_weight=normalized_weights(cweights,cfit))
                r=XGBRegressor(**params,objective='reg:squarederror').fit(data['X'].loc[fit,cols],data['y'][fit],sample_weight=normalized_weights(weights,fit))
                row={'profile':name,'cohort':cohort,'setting':setting,'fold':fold,'fit_player_games':int(fit.sum()),
                    'log_loss':float(log_loss(y[cdev],c.predict_proba(X.loc[cdev,cols+LINE_FEATURES])[:,1])),
                    'rmse':float(np.sqrt(mean_squared_error(data['y'][dev],r.predict(data['X'].loc[dev,cols]))))}
                checkpoint.write_text(json.dumps(row,indent=2));results.append(row)
                print(name,cohort,setting,fold,round(row['log_loss'],6),round(row['rmse'],4),flush=True)
    frame=pd.DataFrame(results);frame.to_csv(output/'development.csv',index=False)
    means=frame.groupby(['profile','cohort','setting'])[['log_loss','rmse']].mean()
    selections={}
    for name,subset in [('candidate',means.loc[means.index.get_level_values('profile')!='compact']),
        ('compact_expanded',means.loc[(means.index.get_level_values('profile')=='compact')&(means.index.get_level_values('cohort')!='original')]),
        ('compact_unweighted',means.loc[(means.index.get_level_values('profile')=='compact')&(means.index.get_level_values('cohort')=='expanded')]),
        ('compact_original',means.loc[(means.index.get_level_values('profile')=='compact')&(means.index.get_level_values('cohort')=='original')])]:
        selections[name]={'classifier':list(subset.log_loss.idxmin()),'regressor':list(subset.rmse.idxmin())}
    selections['development_selected']={'classifier':list(means.log_loss.idxmin()),'regressor':list(means.rmse.idxmin())}
    (output/'selection.json').write_text(json.dumps(selections,indent=2));models={};calibrations={}
    for name,selection in selections.items():
        cp,cohort,cs=selection['classifier'];rp,rcohort,rs=selection['regressor'];cols=groups[cp];rc=groups[rp]
        weights=cohort_weights(core,rcohort);cweights=cohort_weights(core,cohort)[idx]
        fit=np.asarray(dates<'2025-01-01') & (core if rcohort=='original' else True)
        cfit=np.asarray(cdates<'2025-01-01') & (ccore if cohort=='original' else True)
        c=XGBClassifier(**{**PARAMS,**settings[cs]},eval_metric='logloss').fit(X.loc[cfit,cols+LINE_FEATURES],y[cfit],sample_weight=normalized_weights(cweights,cfit))
        r=XGBRegressor(**{**PARAMS,**settings[rs]},objective='reg:squarederror').fit(data['X'].loc[fit,rc],data['y'][fit],sample_weight=normalized_weights(weights,fit))
        calfit=np.asarray((cdates>='2025-03-01')&(cdates<'2025-05-01')) & (ccore if cohort=='original' else True)
        # Select calibration on common original cohort too, to isolate training-size effects.
        caldev=np.asarray((cdates>='2025-05-01')&(cdates<'2025-07-01')) & ccore
        calall=np.asarray((cdates>='2025-03-01')&(cdates<'2025-07-01')) & (ccore if cohort=='original' else True)
        raw=c.predict_proba(X.loc[:,cols+LINE_FEATURES])[:,1];scores={}
        for method in ['raw','sigmoid','isotonic']:
            cal=fit_calibration(method,raw[calfit],y[calfit],normalized_weights(cweights,calfit));scores[method]=metrics(y[caldev],cal.predict(raw[caldev]))
        method=min(scores,key=lambda k:scores[k]['log_loss']);cal=fit_calibration(method,raw[calall],y[calall],normalized_weights(cweights,calall))
        calibrations[name]={'method':method,'selection_scores':scores,'classifier_columns':cols,'regressor_columns':rc}
        model=ExpandedPointsCandidate(cols,r,c,cal,feature_profile='expanded_points_v4_'+name,regressor_columns=rc)
        path=output/f'{name}.joblib';joblib.dump(model,path);models[name]=joblib.load(path)
    manifest={**protocol,'selections':selections,'calibrations':calibrations,
        'dataset_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('audited_*.joblib')},
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('scripts/train_selected_points.py'),Path('src/verified_training.py'),Path('src/points_candidate.py'),Path('src/recovered_features.py'),Path('src/audited_features.py')]}}
    # Final outcomes accessed for scoring only after every selection is frozen.
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    report={'populations':{},'serving_models_unchanged':None,
            'fitting_player_games':int(np.sum(dates<'2025-01-01')),
            'original_fitting_player_games':int(np.sum((dates<'2025-01-01')&core))};probs={};preds={};rawprobs={}
    test=np.asarray(dates>='2025-07-01');ctest=np.asarray(cdates>='2025-07-01')
    for name,model in models.items():
        probs[name]=model.predict_over(data['X'].iloc[idx[ctest]],lines[ctest]);preds[name]=model.predict(data['X'].loc[test])
        rawprobs[name]=model.classifier.predict_proba(X.loc[ctest,model.columns+LINE_FEATURES])[:,1]
    for population in ['original_cohort','expanded']:
        subset=core[test] if population=='original_cohort' else np.ones(test.sum(),dtype=bool)
        csubset=ccore[ctest] if population=='original_cohort' else np.ones(ctest.sum(),dtype=bool)
        outcomes=y[ctest][csubset];games=data['games'][idx[ctest]][csubset];armscores={}
        for name in models:
            armscores[name]={**metrics(outcomes,probs[name][csubset]),
                'rmse':float(np.sqrt(mean_squared_error(data['y'][test][subset],preds[name][subset]))),
                'mae':float(mean_absolute_error(data['y'][test][subset],preds[name][subset]))}
        comparison=paired_losses(outcomes,probs['candidate'][csubset],probs['compact_expanded'][csubset],games)
        report['populations'][population]={'player_games':int(subset.sum()),'arms':armscores,
            'raw_probability_metrics':{name:metrics(outcomes,p[csubset]) for name,p in rawprobs.items()},
            'candidate_minus_compact':comparison,
            'auc_comparison':bootstrap(outcomes,probs['candidate'][csubset],probs['compact_expanded'][csubset],games),
            'expanded_minus_original_compact':paired_losses(outcomes,probs['compact_expanded'][csubset],probs['compact_original'][csubset],games)}
    primary=report['populations']['original_cohort'];secondary=report['populations']['expanded']
    report['role_subgroups']={}
    prior_minutes=data['X'].mins_season.to_numpy()[idx[ctest]]
    for label,low,high in [('below_20',0,20),('20_to_30',20,30),('30_plus',30,np.inf)]:
        mask=(prior_minutes>=low)&(prior_minutes<high)
        if not mask.any():continue
        report['role_subgroups'][label]={'classifier_rows':int(mask.sum()),
            'arms':{name:metrics(y[ctest][mask],p[mask]) for name,p in probs.items()}}
    report['meets_predeclared_research_threshold']=bool(primary['candidate_minus_compact']['log_loss']['delta']<=-.002 and
        primary['candidate_minus_compact']['log_loss']['interval_95'][1]<0 and
        all(primary['arms']['candidate'][k]<=primary['arms']['compact_expanded'][k] for k in ['brier','rmse']) and
        all(secondary['arms']['candidate'][k]<=secondary['arms']['compact_expanded'][k] for k in ['brier','log_loss']))
    report['serving_models_unchanged']=before==model_hashes();assert report['serving_models_unchanged']
    report['candidate_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.joblib')}
    pd.DataFrame({'game':data['games'][idx[ctest]],'player':data['players'][idx[ctest]],'date':data['dates'][idx[ctest]],
        'line':lines[ctest],'over':y[ctest],'original_cohort':ccore[ctest],**probs}).to_csv(output/'predictions.csv',index=False)
    pd.DataFrame({'game':data['games'][test],'player':data['players'][test],'actual_points':data['y'][test],
        'original_cohort':core[test],**preds}).to_csv(output/'point_predictions.csv',index=False)
    (output/'report.json').write_text(json.dumps(report,indent=2));print('COMPLETE',report['meets_predeclared_research_threshold'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    if args.prepare:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'protocol.json').write_text(json.dumps(make_protocol(args.dataset),indent=2))
    else:run(args.dataset,args.output)
