"""Predeclared offline attribution and chronological ablations, no promotion."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,socket
import joblib,numpy as np,pandas as pd
from scipy.special import expit,logit
from src.serving_provenance import safe,file_hash
from src.points_correction_diagnostics import fit_arm,terms,shapley_losses,NAMES
from scripts.train_traceable_points import protected,masks,write
from scripts.points_pilot import metrics
from scripts.train_selected_points import paired_losses

OUT=Path('logs/points_traceable_diagnosis_03');BASE=Path('logs/points_traceable_replacement_01')
ARMS={'full':([0,1,2,3,4],1.,'balanced'),'no_intercept':([1,2,3,4],1.,'balanced'),
'edge_only':([1],1.,'balanced'),'edge_rest':([1,4],1.,'balanced'),'intercept_only':([0],1.,'balanced'),
'uniform_full':([0,1,2,3,4],1.,'uniform'),'strong_ridge':([0,1,2,3,4],4.,'balanced')}


def run():
    socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Network forbidden'))
    before=protected();b=joblib.load(BASE/'inputs.joblib');train,test=masks(b);d=b['data'];x=d['X'];dates=pd.to_datetime(d['dates']);short=x.games_played.to_numpy()<10
    OUT.mkdir(exist_ok=False)
    write(OUT/'plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),'historical_independent':False,'promoted':False,'arms':ARMS,'input_hash':file_hash(BASE/'inputs.joblib'),
    'attribution':'Exact Shapley log-loss contributions of five fixed fitted terms across32 subsets; diagnostic, not independent feature causality.',
    'folds':[['before2024-10-01','2024-10-01 through2024-12-31'],['before2025-01-01','2025-01-01 through2025-06-30']],
    'unsupported_folds':'Keep20-game/100-row gates; report unsupported folds without fitting. No short-history qualification without sufficient chronological validation.',
    'selection':'Among fixed arms and market, smallest pooled pre-July2025 OOF loss. Accept only if paired game-cluster95% upper loss delta versus market below0 AND each history OOF cohort has100 rows/20 games; otherwise retain market. No tuning on later comparison.',
    'source_hashes':{p:file_hash(p) for p in [__file__,'src/points_correction_diagnostics.py','src/points_simple_correction.py']},'frozen_before':before})
    original=joblib.load(BASE/'models.joblib')['candidate']
    oof={k:np.full(len(x),np.nan) for k in ARMS};oof['market']=np.full(len(x),np.nan);folds=[]
    for start,end in [('2024-10-01','2025-01-01'),('2025-01-01','2025-07-01')]:
        fit=np.asarray(dates<start);val=np.asarray((dates>=start)&(dates<end))
        if fit.sum()<100 or len(set(d['games'][fit]))<20 or val.sum()<100 or len(set(d['games'][val]))<20:
            folds.append({'start':start,'end':end,'status':'unsupported_not_fitted','fit_rows':int(fit.sum()),'fit_games':len(set(d['games'][fit])),'validation_rows':int(val.sum()),'validation_games':len(set(d['games'][val]))});continue
        fold={'start':start,'end':end,'fit_rows':int(fit.sum()),'fit_games':len(set(d['games'][fit])),'validation_rows':int(val.sum()),'validation_games':len(set(d['games'][val])),'short_validation_rows':int((val&short).sum()),'coefficients':{}}
        for name,(active,penalty,weighting) in ARMS.items():
            model=fit_arm(x.loc[fit],d['lines'][fit],b['market'][fit],d['y'][fit],d['games'][fit],active,penalty,weighting)
            oof[name][val]=model.predict_over(x.loc[val],d['lines'][val],b['market'][val]);fold['coefficients'][name]=model.beta.tolist()
        oof['market'][val]=b['market'][val];folds.append(fold)
    valid=np.isfinite(oof['market']);scores={k:metrics(d['y'][valid],p[valid]) for k,p in oof.items()};best=min(scores,key=lambda k:scores[k]['log_loss'])
    support={c:{'rows':int((valid&m).sum()),'games':len(set(d['games'][valid&m]))} for c,m in [('short',short),('eligible',~short)]}
    ci=paired_losses(d['y'][valid],oof[best][valid],b['market'][valid],d['games'][valid])
    qualified=best!='market' and ci['log_loss']['interval_95'][1]<0 and all(v['rows']>=100 and v['games']>=20 for v in support.values())
    selected=best if qualified else 'market'
    write(OUT/'selection.json',{'best_diagnostic':best,'retained':selected,'support':support,'metrics':scores,'paired_best_market':ci,'folds':folds,'comparison_used':False})
    models={name:fit_arm(x.loc[train],d['lines'][train],b['market'][train],d['y'][train],d['games'][train],*config) for name,config in ARMS.items()}
    joblib.dump(models,OUT/'models.joblib')
    probs={k:m.predict_over(x.loc[test],d['lines'][test],b['market'][test]) for k,m in models.items()};probs.update(market=b['market'][test],coin_flip=np.full(test.sum(),.5),original=original.predict_over(x.loc[test],d['lines'][test],b['market'][test]))
    attributes=shapley_losses(d['y'][test],b['market'][test],terms(original,x.loc[test],d['lines'][test]))
    report={'historical_independent':False,'promoted':False,'retained':selected,'cohorts':{},'distribution':{}}
    for name,mask in [('training',train),('comparison',test)]:
        from src.points_simple_correction import balanced_weights
        w=balanced_weights(d['games'][mask],x.games_played.to_numpy()[mask]);report['distribution'][name]={'actual_over_rate':float(d['y'][mask].mean()),'market_over_probability':float(b['market'][mask].mean()),'weighted_actual':float(np.sum(w*d['y'][mask])),'weighted_market':float(np.sum(w*b['market'][mask])),'short_share':float(short[mask].mean())}
    for c,m in [('all',np.ones(test.sum(),bool)),('short',short[test]),('eligible',~short[test])]:
        y=d['y'][test][m];g=d['games'][test][m]
        report['cohorts'][c]={'rows':len(y),'games':len(set(g)),'metrics':{k:metrics(y,p[m]) for k,p in probs.items()},'attribution_log_loss_delta':dict(zip(NAMES,attributes[m].mean(axis=0))),
        'paired_market':{k:paired_losses(y,p[m],probs['market'][m],g) for k,p in probs.items() if k!='market'}}
    # Distribution of harm across games, avoiding a single average hiding outliers.
    pd.DataFrame({'game':d['games'][test],'player':d['players'][test],**{n:attributes[:,i] for i,n in enumerate(NAMES)}}).to_csv(OUT/'attribution.csv',index=False)
    write(OUT/'report.json',report);write(OUT/'predictions.json',probs)
    for p in [Path(__file__),Path('src/points_correction_diagnostics.py')]:shutil.copy2(p,OUT/p.name)
    if before!=protected():raise ValueError('Frozen drift')
    write(OUT/'integrity.json',{'protected_unchanged':True,'credits_spent':0})
    print(json.dumps({'retained':selected,'best_oof':best,'attribution':report['cohorts']['all']['attribution_log_loss_delta'],'metrics':{k:v['log_loss'] for k,v in report['cohorts']['all']['metrics'].items()},'distribution':report['distribution']},indent=2))

if __name__=='__main__':run()
