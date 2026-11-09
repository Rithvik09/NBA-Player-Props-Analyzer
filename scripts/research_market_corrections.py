"""Real quotes only; chronological correction selection then exposed comparison."""
from pathlib import Path
from datetime import datetime,timezone
import json
import joblib
import numpy as np
import pandas as pd
from src.prospective_points import digest,save_new
from src.points_calibration_research_v2 import validate
from src.points_training_repair import assert_disjoint_games
from src.points_market_correction import fit_correction
from scripts.points_pilot import metrics
from scripts.train_selected_points import paired_losses

ROOT=Path('logs/points_market_correction_02')
INPUT=Path('logs/points_evidence_representation_01/inputs.joblib')


def run():
    b=joblib.load(INPUT);validate(b);d=b['data'];x=d['X'];dt=pd.to_datetime(d['dates']);h=x.games_played.to_numpy()
    fitting=np.asarray(dt<'2025-01-01');selection=np.asarray((dt>='2025-01-01')&(dt<'2025-07-01'));comparison=np.asarray(dt>='2025-07-01');development=fitting|selection
    assert_disjoint_games(d['games'][fitting],d['games'][selection],d['games'][comparison])
    support={}
    for name,m in [('fitting',fitting),('selection',selection),('development_refit',development),('comparison',comparison)]:
        support[name]={'rows':int(m.sum()),'games':len(set(d['games'][m])),'cohorts':{c:{'rows':int((m&k).sum()),'games':len(set(d['games'][m&k]))} for c,k in [('short_history',h<10),('eligible_history',h>=10)]}}
        if m.sum()<100 or len(set(d['games'][m]))<20:raise ValueError('Pooled split insufficient')
    previous=json.loads(Path('logs/points_target_experiment_01/plan.json').read_text())
    compact=joblib.load('logs/points_training_repair_01/compact_retrained.joblib').columns
    groups={'compact':compact,'compact_shooting':list(dict.fromkeys(compact+previous['groups']['shooting']))}
    ROOT.mkdir(exist_ok=False)
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'historical_independent':False,'input_sha256':digest(INPUT),'groups':groups,'support':support,'penalties':[1.,10.,100.], 'objective':'Sum binary log loss + lambda/2 times squared coefficients, including intercept; anchored models fix coefficient of no-vig market logit to1. One real quote and weight1 per player-game. No synthetic labels.', 'selection':'Pooled Jan-Jun2025 log loss, zero-correction market included; exact ties prefer market then stronger regularization. Select before accessing comparison metrics; refit chosen and diagnostic arms on all pre-July2025 real rows.', 'limitations':'Only2 short-history selection cases; cannot tune/qualify a separate short-history model. All later outcomes previously exposed in earlier research. Pooled fitting/selection gates do not establish separate short-history selection sufficiency.', 'monotonicity':'Line-derived coefficients constrained; test at fixed market probability. Does not validate arbitrary incoherent across-line market quotes.', 'source_hashes':{p:digest(p) for p in [__file__,'src/points_market_correction.py','src/points_v5.py','src/verified_training.py']},'comparison_model_hashes':{p:digest(p) for p in ['logs/points_evidence_feature_experiment_01/legacy_compact.joblib','logs/expanded_points_experiment_v4/compact_expanded.joblib']}}
    save_new(ROOT/'plan.json',plan)
    configs={f'{kind}_{group}_ridge{int(penalty)}':(cols,penalty,anchored) for kind,anchored in [('offset',True),('direct',False)] for group,cols in groups.items() if anchored or group=='compact' for penalty in [100.,10.,1.]}
    scores={'market':metrics(d['y'][selection],b['market'][selection])['log_loss']}
    for name,(cols,penalty,anchor) in configs.items():
        model=fit_correction(x.loc[fitting],d['lines'][fitting],b['market'][fitting],d['y'][fitting],cols,penalty,anchor)
        scores[name]=metrics(d['y'][selection],model.predict_over(x.loc[selection],d['lines'][selection],b['market'][selection]))['log_loss']
        joblib.dump(model,ROOT/(name+'_development.joblib'))
    selected=min([n for n in scores if n=='market' or n.startswith('offset')],key=lambda n:scores[n])
    compact_selected=min([n for n in scores if n.startswith('direct')],key=lambda n:scores[n])
    save_new(ROOT/'selection.json',{'pooled_selection_log_loss':scores,'chosen_correction':selected,'chosen_real_compact':compact_selected,'comparison_metrics_not_used':True})
    models={}
    for name,(cols,penalty,anchor) in configs.items():
        model=fit_correction(x.loc[development],d['lines'][development],b['market'][development],d['y'][development],cols,penalty,anchor)
        joblib.dump(model,ROOT/(name+'.joblib'));models[name]=model
    # All configuration selection, preprocessing and model fitting saved before scoring comparison.
    xx=x.loc[comparison];lines=d['lines'][comparison];market=b['market'][comparison];y=d['y'][comparison];games=d['games'][comparison]
    probs={n:c.predict_over(xx,lines,market) for n,c in models.items()};probs['market']=market;probs['coin_flip']=np.full(len(xx),.5)
    for name,path in [('legacy_research_compact','logs/points_evidence_feature_experiment_01/legacy_compact.joblib'),('frozen_compact','logs/expanded_points_experiment_v4/compact_expanded.joblib')]:
        probs[name]=joblib.load(path).predict_over(xx,lines)
    probs['selected_correction']=probs[selected];probs['selected_real_compact']=probs[compact_selected]
    monotonicity={}
    for name,c in models.items():
        previous=None;violations=0
        for offset in np.arange(-5,5.01,.5):
            p=c.predict_over(xx,np.maximum(.5,lines+offset),market)
            if previous is not None:violations+=int((p>previous+1e-10).sum())
            previous=p
        if violations:raise ValueError('Line monotonicity violation')
        monotonicity[name]={'comparisons':len(xx)*20,'violations':violations}
    report={'historical_independent':False,'promoted':False,'selection':selected,'real_compact_selection':compact_selected,'monotonicity':monotonicity,'cohorts':{}}
    for cohort,m in [('all',np.ones(len(xx),bool)),('short_history',h[comparison]<10),('eligible_history',h[comparison]>=10)]:
        report['cohorts'][cohort]={'rows':int(m.sum()),'games':len(set(games[m])),'metrics':{n:metrics(y[m],p[m]) for n,p in probs.items()},'selected_paired':{n:paired_losses(y[m],probs['selected_correction'][m],probs[n][m],games[m]) for n in ['market','selected_real_compact','legacy_research_compact','frozen_compact','coin_flip']}}
    save_new(ROOT/'report.json',report)
    pd.DataFrame({'game':games,'player':d['players'][comparison],'history':h[comparison],'over':y,**probs}).to_csv(ROOT/'predictions.csv',index=False)
    print(json.dumps({'selection':selected,'real_compact':compact_selected,'scores':scores,'support':support,'results':{c:{n:{k:v[k] for k in ['log_loss','auc','brier']} for n,v in r['metrics'].items() if n in ['selected_correction','selected_real_compact','market','legacy_research_compact','frozen_compact','coin_flip']} for c,r in report['cohorts'].items()}},indent=2))

if __name__=='__main__':run()
