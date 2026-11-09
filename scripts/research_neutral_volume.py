"""Fixed volume-only comparison with strictly earlier-date fitting; offline."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,socket,subprocess,sys
import joblib,numpy as np,pandas as pd
from src.neutral_volume_correction import fit_volume
from src.points_simple_correction import balanced_weights
from src.serving_provenance import file_hash
from scripts.train_traceable_points import masks,protected,write
from scripts.points_pilot import metrics
from scripts.train_selected_points import paired_losses

BASE=Path('logs/points_traceable_replacement_01');OUT=Path('logs/points_neutral_volume_01')


def check_neutral(model,b,mask):
    d=b['data'];x=d['X'].loc[mask];w=balanced_weights(d['games'][mask],x.games_played);p=model.predict_over(x,d['lines'][mask],b['market'][mask]);short=x.games_played.to_numpy()<10
    delta={}
    for name,g in [('short',short),('eligible',~short)]:
        delta[name]=float(np.sum(w[g]*(p[g]-b['market'][mask][g]))/w[g].sum())
        if abs(delta[name])>1e-10:raise ValueError('Training market neutrality failed')
    return delta


def run():
    socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('External network forbidden'))
    before=protected();b=joblib.load(BASE/'inputs.joblib');train,test=masks(b);d=b['data'];x=d['X'];dates=pd.to_datetime(d['dates']);short=x.games_played.to_numpy()<10
    OUT.mkdir(exist_ok=False)
    write(OUT/'plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),'historical_independent':False,'promotion':False,'input_sha256':file_hash(BASE/'inputs.joblib'),
        'model':'One nonnegative shot-volume coefficient, ridge1, bounded0..2. Standardize with training-only history/game weights; clip z to[-5,5]. Two cohort offsets solved from training market/feature inputs only, never from outcome frequencies. Weighted mean predicted probability must equal weighted market within each training cohort.',
        'fits':['fixed before2025-07-01','walk-forward: refit using all strictly earlier calendar dates before each validation day'],
        'validation':'All existing real-line rows dated2025-07-01 onward, all previously exposed. Whole calendar-day blocks: no same-day outcome enters fit.',
        'gates':'At least100 rows/20 games per history cohort in every fit and aggregate validation. No threshold reduction.',
        'interpretation':'Fills diagnostic short-history validation coverage, not fresh holdout independence. Fixed vs adaptive training are predefined alternatives, not selected on these results. Neutrality on training distribution does not guarantee neutrality after distribution shift.',
        'source_hashes':{p:file_hash(p) for p in [__file__,'src/neutral_volume_correction.py']},'protected_before':before})
    shutil.copy2(BASE/'inputs.joblib',OUT/'inputs.joblib')
    old=joblib.load(BASE/'models.joblib')['candidate'];frozen=fit_volume(x.loc[train],d['lines'][train],b['market'][train],d['y'][train],d['games'][train]);joblib.dump(frozen,OUT/'fixed.joblib')
    neutrality={'fixed':check_neutral(frozen,b,train)}
    p_walk=np.full(len(x),np.nan);folds=[];saved={}
    for day in sorted(set(dates[test])):
        fit=np.asarray(dates<day);val=np.asarray(dates==day)&test
        assert not set(d['games'][fit])&set(d['games'][val])
        model=fit_volume(x.loc[fit],d['lines'][fit],b['market'][fit],d['y'][fit],d['games'][fit]);saved[str(day)]=model
        p_walk[val]=model.predict_over(x.loc[val],d['lines'][val],b['market'][val])
        folds.append({'date':str(day),'fit_indices':np.flatnonzero(fit).tolist(),'validation_indices':np.flatnonzero(val).tolist(),'fit_max_date':str(dates[fit].max()),'beta':model.beta,'offsets':model.offsets.tolist(),'neutrality':check_neutral(model,b,fit)})
    if not np.isfinite(p_walk[test]).all():raise ValueError('Missing validation forecasts')
    joblib.dump(saved,OUT/'walk_forward.joblib');write(OUT/'folds.json',folds)
    probs={'market':b['market'][test],'fixed_volume':frozen.predict_over(x.loc[test],d['lines'][test],b['market'][test]),'walk_forward_volume':p_walk[test],'original':old.predict_over(x.loc[test],d['lines'][test],b['market'][test]),'coin_flip':np.full(test.sum(),.5)}
    report={'historical_independent':False,'promoted':False,'fixed_parameters':vars(frozen),'neutrality':neutrality,'cohorts':{}}
    for name,m in [('all',np.ones(test.sum(),bool)),('short',short[test]),('eligible',~short[test])]:
        y=d['y'][test][m];games=d['games'][test][m]
        if len(y)<100 or len(set(games))<20:raise ValueError('Validation cohort insufficient')
        report['cohorts'][name]={'rows':len(y),'games':len(set(games)),'metrics':{k:metrics(y,p[m]) for k,p in probs.items()},'mean_probability_correction':{k:float(np.mean(p[m]-probs['market'][m])) for k,p in probs.items()},'paired_market':{k:paired_losses(y,p[m],probs['market'][m],games) for k,p in probs.items() if k!='market'}}
    write(OUT/'report.json',report);write(OUT/'predictions.json',probs)
    # Round-trip each model and require exact predictions on its own validation day.
    loaded=joblib.load(OUT/'walk_forward.joblib')
    for day,model in loaded.items():
        m=np.asarray(dates==pd.Timestamp(day))&test
        if not np.array_equal(model.predict_over(x.loc[m],d['lines'][m],b['market'][m]),p_walk[m]):raise ValueError('Replay mismatch')
    (OUT/'requirements.lock').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    for folder in ('src','scripts'):
        for p in Path(folder).glob('*.py'):
            dest=OUT/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    if before!=protected():raise ValueError('Protected artifact drift')
    write(OUT/'integrity.json',{'protected_unchanged':True,'credits_spent':0,'replay_exact':True})
    write(OUT/'manifest.json',{'python':sys.version,'files':{str(p.relative_to(OUT)):file_hash(p) for p in OUT.rglob('*') if p.is_file()}})
    print(json.dumps({'parameters':{'beta':frozen.beta,'offsets':frozen.offsets.tolist()},'cohorts':{c:{'rows':v['rows'],'games':v['games'],'loss':{k:m['log_loss'] for k,m in v['metrics'].items()},'paired':v['paired_market']} for c,v in report['cohorts'].items()}},indent=2))

if __name__=='__main__':run()
