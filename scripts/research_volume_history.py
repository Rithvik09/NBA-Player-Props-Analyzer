"""Offline, exposed-data diagnosis of history-dependent volume corrections."""
from pathlib import Path
from datetime import datetime, timezone
import json, socket, shutil, sys, subprocess
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import brentq, minimize_scalar
from scripts.train_traceable_points import masks, protected, write
from scripts.train_selected_points import paired_losses
from scripts.points_pilot import metrics
from src.points_simple_correction import balanced_weights
from src.serving_provenance import file_hash

OUT=Path('logs/points_volume_history_02')
INPUT=Path('logs/points_neutral_volume_01/inputs.joblib')
ARMS=('shared','reliability','separate','line_residual','line_residual_separate')


def design(x,lines,w=None,state=None,arm='shared'):
    v=x.volume_fga.to_numpy(float);h=x.games_played.to_numpy(float)
    if not np.isfinite(v).all() or not np.isfinite(h).all() or (h<1).any():raise ValueError('Invalid history/volume')
    a=np.column_stack([np.ones(len(v)),np.asarray(lines,float)])
    if state is None:
        mean=float(np.sum(w*v))
        coef=np.linalg.lstsq(a*np.sqrt(w[:,None]),v*np.sqrt(w),rcond=None)[0] if 'line_residual' in arm else np.zeros(2)
        raw=v-np.einsum('ij,j->i',a,coef)
        if arm=='reliability':raw=mean+(raw-mean)*h/(h+10.)
        center=float(np.sum(w*raw));scale=max(float(np.sqrt(np.sum(w*((raw-center)**2)))),1e-8)
        state=dict(mean=mean,coef=coef,center=center,scale=scale)
    raw=v-np.einsum('ij,j->i',a,state['coef'])
    if arm=='reliability':raw=state['mean']+(raw-state['mean'])*h/(h+10.)
    return np.clip((raw-state['center'])/state['scale'],-5,5),state


def predict(model,x,lines,market):
    z,_=design(x,lines,state=model['state'],arm=model['arm']);g=(x.games_played.to_numpy()<10).astype(int)
    return expit(logit(market)+model['beta'][g]*z+model['offsets'][g])


def fit(x,lines,market,y,games,arm):
    g=(x.games_played.to_numpy()<10).astype(int);w=balanced_weights(games,x.games_played)
    for k in (0,1):
        m=g==k
        if m.sum()<100 or len(set(np.asarray(games)[m]))<20:raise ValueError('Insufficient cohort coverage')
    z,state=design(x,lines,w=w,arm=arm);o=logit(market)
    def offsets(beta):
        return np.array([brentq(lambda a:float(np.sum(w[g==k]*(expit(o[g==k]+beta[k]*z[g==k]+a)-market[g==k]))),-30,30,xtol=1e-14) for k in (0,1)])
    def loss(beta):
        eta=o+beta[g]*z+offsets(beta)[g]
        return float(np.sum(w*(np.logaddexp(0,eta)-y*eta))+np.sum(beta**2)/4)
    beta=np.zeros(2)
    if 'separate' in arm:
        for k in (0,1):
            def obj(b):
                v=beta.copy();v[k]=b;return loss(v)
            r=minimize_scalar(obj,bounds=(0.,2.),method='bounded',options={'xatol':1e-12})
            if not r.success:raise ValueError('Fit failed')
            if r.fun<obj(0.):beta[k]=r.x
    else:
        r=minimize_scalar(lambda b:loss(np.full(2,b)),bounds=(0.,2.),method='bounded',options={'xatol':1e-12})
        if not r.success:raise ValueError('Fit failed')
        if r.fun<loss(beta):beta[:]=r.x
    model=dict(arm=arm,state=state,beta=beta,offsets=offsets(beta))
    p=predict(model,x,lines,market)
    for k in (0,1):
        assert abs(np.sum(w[g==k]*(p[g==k]-market[g==k]))/w[g==k].sum())<1e-10
    return model


def run():
    socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Network forbidden'))
    before=protected();b=joblib.load(INPUT);train,test=masks(b);d=b['data'];x=d['X'];dates=pd.to_datetime(d['dates']);short=x.games_played.to_numpy()<10
    OUT.mkdir(exist_ok=False)
    write(OUT/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),input_sha256=file_hash(INPUT),arms=list(ARMS),
        rationale={'reliability':'Shrink volume toward fitting mean by n/(n+10), fixed strength10; then standardize.', 'separate':'Two nonnegative slopes, same effective ridge1 per cohort, no fitted outcome intercept.', 'line_residual':'Training-only weighted linear regression of volume on quoted line; correct using residual volume.'},
        fitting='Fixed pre-July2025 and walk-forward strictly earlier calendar dates. No same-day outcomes. Every fitting cohort >=100 rows/20 games.',
        evaluation='All July2025 onward rows, already exposed. No tuning, selection or promotion; no independent holdout claim. Subgroups descriptive.',
        regularization='Nonnegative slopes bounded0..2; ridge1; same group market-neutral offset equation. No penalty search.',
        diagnostic='Compare prior volume to actual target FGA as diagnostic only. Residual score uses y-market; report game-cluster intervals. Training and comparison separately.',
        protected_before=before))
    shutil.copy2(INPUT,OUT/'inputs.joblib')
    preds={'market':b['market'][test]};models={};folds=[]
    for arm in ARMS:
        model=fit(x.loc[train],d['lines'][train],b['market'][train],d['y'][train],d['games'][train],arm)
        models['fixed_'+arm]=model;preds['fixed_'+arm]=predict(model,x.loc[test],d['lines'][test],b['market'][test])
        walk=np.full(len(x),np.nan)
        for day in sorted(set(dates[test])):
            fm=np.asarray(dates<day);vm=np.asarray(dates==day)&test
            assert not set(d['games'][fm])&set(d['games'][vm])
            model=fit(x.loc[fm],d['lines'][fm],b['market'][fm],d['y'][fm],d['games'][fm],arm)
            models[arm+'_'+str(day)]=model;walk[vm]=predict(model,x.loc[vm],d['lines'][vm],b['market'][vm])
            folds.append(dict(arm=arm,date=str(day),fit_indices=np.flatnonzero(fm),validation_indices=np.flatnonzero(vm),beta=model['beta']))
        assert np.isfinite(walk[test]).all();preds['walk_'+arm]=walk[test]
    # Reproduce the previous experiment before interpreting alternatives.
    old=json.loads(Path('logs/points_neutral_volume_01/predictions.json').read_text())
    for new,prior in [('fixed_shared','fixed_volume'),('walk_shared','walk_forward_volume')]:
        assert np.allclose(preds[new],old[prior],atol=1e-8,rtol=0)
    report=dict(independent=False,promoted=False,cohorts={})
    for name,m in [('all',np.ones(test.sum(),bool)),('short',short[test]),('eligible',~short[test])]:
        y=d['y'][test][m];games=d['games'][test][m]
        assert len(y)>=100 and len(set(games))>=20
        report['cohorts'][name]=dict(rows=len(y),games=len(set(games)),metrics={k:metrics(y,p[m]) for k,p in preds.items()},
            vs_market={k:paired_losses(y,p[m],preds['market'][m],games) for k,p in preds.items() if k!='market'},
            vs_shared={k:paired_losses(y,p[m],preds[('walk_' if k.startswith('walk') else 'fixed_')+'shared'][m],games) for k,p in preds.items() if k!='market'},
            mean_correction={k:float(np.mean(p[m]-preds['market'][m])) for k,p in preds.items()})
    # Local official boxscores are outcomes for diagnostics, never predictor inputs.
    paths=[Path(f'logs/points_pilot_cache/{kind}_{s}.json') for s in ['2022-23','2023-24','2024-25','2025-26'] for kind in ['players','playoffs','playin']]
    raw=pd.concat([pd.read_json(p,dtype={'GAME_ID':str}) for p in paths]);assert not raw.duplicated(['GAME_ID','PLAYER_ID']).any()
    target=raw.set_index(['GAME_ID','PLAYER_ID']).reindex(list(zip(d['games'].astype(str),d['players'].astype(int))))
    assert np.array_equal(target.PTS.to_numpy(),d['points'])
    diagnostics={}
    for split,sm in [('train',train),('comparison',test)]:
        for cohort,cm in [('short',short),('eligible',~short)]:
            m=sm&cm;v=x.volume_fga.to_numpy()[m];y=d['y'][m];market=b['market'][m];games=d['games'][m]
            z,_=design(x.loc[m],d['lines'][m],state=models['fixed_shared']['state'])
            score=z*(y-market)
            # Bootstrap row-weighted residual association by resampling entire games.
            grouped=pd.DataFrame({'game':games,'score':score}).groupby('game').score.agg(['sum','count'])
            rng=np.random.default_rng(42);ix=rng.integers(len(grouped),size=(2000,len(grouped)))
            samples=grouped['sum'].to_numpy()[ix].sum(1)/grouped['count'].to_numpy()[ix].sum(1)
            diagnostics[split+'_'+cohort]=dict(rows=int(m.sum()),games=len(set(games)),history_mean=float(x.games_played[m].mean()),volume_mean=float(v.mean()),volume_sd=float(v.std()),line_volume_corr=float(np.corrcoef(v,d['lines'][m])[0,1]),
                actual_fga_rmse=float(np.sqrt(np.mean((v-target.FGA.to_numpy()[m])**2))),actual_fga_mae=float(np.mean(np.abs(v-target.FGA.to_numpy()[m]))),
                volume_residual_score=float(score.mean()),score_game_cluster_ci=np.quantile(samples,[.025,.975]),market_residual_mean=float(np.mean(y-market)),
                months=pd.Series(dates[m].strftime('%Y-%m')).value_counts().to_dict())
    write(OUT/'diagnostics.json',diagnostics);write(OUT/'report.json',report);write(OUT/'predictions.json',preds);write(OUT/'folds.json',folds)
    joblib.dump(models,OUT/'models.joblib')
    loaded=joblib.load(OUT/'models.joblib')
    for arm in ARMS:assert np.array_equal(predict(loaded['fixed_'+arm],x.loc[test],d['lines'][test],b['market'][test]),preds['fixed_'+arm])
    write(OUT/'parameters.json',{k:v for k,v in models.items() if k.startswith('fixed')})
    assert before==protected()
    write(OUT/'integrity.json',dict(protected_unchanged=True,network_disabled=True,credits_spent=0,original_predictions_reproduced=True,serialization_replay=True,raw_hashes={str(p):file_hash(p) for p in paths}))
    for folder in ['src','scripts']:
        for p in Path(folder).glob('*.py'):
            dest=OUT/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    (OUT/'requirements.lock').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    write(OUT/'manifest.json',dict(files={str(p.relative_to(OUT)):file_hash(p) for p in OUT.rglob('*') if p.is_file()}))
    print(json.dumps({c:{k:round(v['log_loss'],9) for k,v in s['metrics'].items()} for c,s in report['cohorts'].items()},indent=2))

if __name__=='__main__':run()
