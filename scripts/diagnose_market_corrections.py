"""Offline diagnostics for real-quote correction variance and transport."""
from pathlib import Path
from datetime import datetime,timezone
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit
from scipy.stats import spearmanr
from sklearn.metrics import log_loss
from src.prospective_points import save_new,digest
from src.points_market_correction import fit_correction
from src.points_v5 import prepare
from src.verified_training import classifier_frame

ROOT=Path('logs/points_market_diagnosis_01')
SOURCE=Path('logs/points_market_correction_02')
INPUT=Path('logs/points_evidence_representation_01/inputs.joblib')


def loss(y,p):return float(log_loss(y,p,labels=[0,1]))


def intercept_fit(y,p,penalty=100):
    offset=logit(p)
    def objective(a):
        eta=offset+a
        return np.sum(np.logaddexp(0,eta)-y*eta)+penalty*a*a/2
    result=minimize_scalar(objective,bounds=(-3,3),method='bounded')
    if not result.success:raise ValueError('Intercept optimization failed')
    return float(result.x)


def run():
    b=joblib.load(INPUT);d=b['data'];x=d['X'];dt=pd.to_datetime(d['dates']);market=b['market'];y=d['y'];games=d['games'];lines=d['lines']
    masks={'fit':np.asarray(dt<'2025-01-01'),'selection':np.asarray((dt>='2025-01-01')&(dt<'2025-07-01')),'development':np.asarray(dt<'2025-07-01'),'comparison':np.asarray(dt>='2025-07-01')}
    train=masks['development'];test=masks['comparison'];idx=np.flatnonzero(train);ti=np.flatnonzero(test)
    plan=json.loads((SOURCE/'plan.json').read_text());columns=plan['groups']['compact'];shooting=plan['groups']['compact_shooting']
    ROOT.mkdir(exist_ok=False)
    save_new(ROOT/'plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),'input_sha256':digest(INPUT),'source_hashes':{p:digest(p) for p in [__file__,'src/points_market_correction.py']},'models_sha256':{str(p):digest(p) for p in SOURCE.glob('*.joblib')},'diagnostics':'Existing train/generalization gaps; feature missingness/distribution/correlation shifts; fixed100-penalty low-dimensional controls; chronological prefix learning curve26/40/52/66 games;30 NBA-game bootstrap fits;30 within-game label permutations. No selection or promotion on comparison. All outcomes previously exposed.','seed':42})
    report={'promoted':False,'historical_independent':False,'splits':{},'generalization':{},'controls':{},'learning_curve':[]}
    for n,m in masks.items():
        report['splits'][n]={'rows':int(m.sum()),'games':len(set(games[m])),'short_fraction':float((x.games_played.to_numpy()[m]<10).mean()),'over_rate':float(y[m].mean()),'market_mean':float(market[m].mean()),'market_loss':loss(y[m],market[m])}
    for name in ['offset_compact_ridge100','offset_compact_ridge10','offset_compact_ridge1','offset_compact_shooting_ridge100']:
        report['generalization'][name]={}
        for suffix,phases in [('_development',['fit','selection']),('', ['development','comparison'])]:
            c=joblib.load(SOURCE/(name+suffix+'.joblib'))
            for phase in phases:
                m=masks[phase];p=c.predict_over(x.loc[m],lines[m],market[m])
                report['generalization'][name][phase]={'loss':loss(y[m],p),'delta_market':loss(y[m],p)-loss(y[m],market[m]),'mean_abs_probability_adjustment':float(np.mean(abs(p-market[m]))),'coefficients':len(c.coefficients),'nonzero_coefficients':int((abs(c.coefficients)>1e-8).sum())}
    # Feature audit uses saved train preprocessing and reports comparison shifts only.
    c=joblib.load(SOURCE/'offset_compact_shooting_ridge100.joblib');raw=classifier_frame(prepare(x),lines,shooting);features=[]
    for k in raw:
        a=raw.loc[train,k].replace([np.inf,-np.inf],np.nan);z=raw.loc[test,k].replace([np.inf,-np.inf],np.nan)
        residuals={}
        for phase,m in [('development',train),('comparison',test)]:
            v=raw.loc[m,k].to_numpy();valid=np.isfinite(v)
            residuals[phase]=float(spearmanr(v[valid],(y-market)[m][valid]).statistic) if valid.sum()>2 and np.std(v[valid])>0 else None
        sd=float(a.std());shift=float((z.mean()-a.mean())/sd) if sd>1e-8 else None
        features.append({'name':k,'missing_development':float(a.isna().mean()),'missing_comparison':float(z.isna().mean()),'mean_shift_training_sd':shift,'training_unique':int(a.nunique()),'residual_spearman':residuals})
    report['features']=features
    z=c.design(x.loc[train],lines[train]);vary=np.std(z,axis=0)>1e-8
    report['design']={'coefficient_count':len(c.coefficients),'varying_columns':int(vary.sum()),'matrix_rank':int(np.linalg.matrix_rank(z)),'independent_training_games':len(set(games[train]))}
    for group,cols in [('line_only',[]),('small',['recent_avg','season_avg','minutes_last_5_games','rest_days','is_home_game_num']),('compact',columns),('compact_shooting',shooting)]:
        model=fit_correction(x.loc[train],lines[train],market[train],y[train],cols,100)
        report['controls'][group]={phase:loss(y[m],model.predict_over(x.loc[m],lines[m],market[m])) for phase,m in [('development',train),('comparison',test)]}
        joblib.dump(model,ROOT/(group+'.joblib'))
    intercept=intercept_fit(y[train],market[train]);report['controls']['intercept_only']={'intercept':intercept,**{phase:loss(y[m],expit(logit(market[m])+intercept)) for phase,m in [('development',train),('comparison',test)]}}
    ordered=pd.DataFrame({'game':games[train],'date':dt[train]}).groupby('game').date.min().sort_values(kind='stable').index.to_numpy()
    for n in [26,40,52,66]:
        fitting=train & np.isin(games,ordered[:n]);model=fit_correction(x.loc[fitting],lines[fitting],market[fitting],y[fitting],columns,100)
        report['learning_curve'].append({'games':n,'rows':int(fitting.sum()),'short_fraction':float((x.games_played.to_numpy()[fitting]<10).mean()),'fit_loss':loss(y[fitting],model.predict_over(x.loc[fitting],lines[fitting],market[fitting])),'comparison_loss':loss(y[test],model.predict_over(x.loc[test],lines[test],market[test]))})
    rng=np.random.default_rng(42);game_rows={g:idx[games[idx]==g] for g in ordered};boot=[];null=[]
    actual=joblib.load(SOURCE/'offset_compact_ridge100.joblib');actual_p=actual.predict_over(x.loc[test],lines[test],market[test])
    for iteration in range(30):
        sample=np.concatenate([game_rows[g] for g in rng.choice(ordered,len(ordered),replace=True)])
        model=fit_correction(x.iloc[sample],lines[sample],market[sample],y[sample],columns,100)
        boot.append(model.predict_over(x.loc[test],lines[test],market[test]))
        perm=y.copy()
        for rows in game_rows.values():perm[rows]=rng.permutation(y[rows])
        model=fit_correction(x.loc[train],lines[train],market[train],perm[train],columns,100)
        null.append({'training_gain':loss(perm[train],market[train])-loss(perm[train],model.predict_over(x.loc[train],lines[train],market[train])),'comparison_delta_market':loss(y[test],model.predict_over(x.loc[test],lines[test],market[test]))-loss(y[test],market[test])})
        if iteration%10==9:print('completed',iteration+1,'bootstrap and permutation fits',flush=True)
    boot=np.asarray(boot);low,high=np.quantile(boot,[.025,.975],axis=0)
    report['bootstrap']={'draws':30,'unit':'whole NBA games, refit preprocessing each draw','median_prediction_sd':float(np.median(boot.std(axis=0))),'mean_prediction_sd':float(np.mean(boot.std(axis=0))),'fraction_correction_interval_crosses_zero':float(((low<=market[test])&(high>=market[test])).mean()),'median_absolute_original_correction':float(np.median(abs(actual_p-market[test]))),'warning':'Small descriptive stability ensemble, not a confidence interval for a selected strategy'}
    gain=loss(y[train],market[train])-loss(y[train],actual.predict_over(x.loc[train],lines[train],market[train]))
    report['permutation']={'draws':30,'scheme':'Shuffle labels within training game, preserving each game over rate; not a complete market-conditional null','actual_training_gain':gain,'null_training_gain_quantiles':np.quantile([v['training_gain'] for v in null],[.025,.5,.975]).tolist(),'fraction_null_training_gain_at_least_actual':float(np.mean([v['training_gain']>=gain for v in null])),'null_comparison_delta_quantiles':np.quantile([v['comparison_delta_market'] for v in null],[.025,.5,.975]).tolist(),'draw_results':null}
    save_new(ROOT/'report.json',report);np.savez_compressed(ROOT/'bootstrap_predictions.npz',predictions=boot,game=games[test],player=d['players'][test]);print(json.dumps({k:v for k,v in report.items() if k!='features'},indent=2))

if __name__=='__main__':run()
