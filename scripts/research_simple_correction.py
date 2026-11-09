"""Predeclared small, strongly shrunk corrections; no prospective mutations."""
from pathlib import Path
from datetime import datetime,timezone
import json
import joblib
import numpy as np
import pandas as pd
from src.prospective_points import digest,save_new
from src.points_calibration_research_v2 import validate
from src.points_training_repair import assert_disjoint_games
from src.points_simple_correction import features,fit_simple
from scripts.points_pilot import metrics
from scripts.train_selected_points import paired_losses

ROOT=Path('logs/points_simple_correction_01')
INPUT=Path('logs/points_evidence_representation_01/inputs.joblib')


def run():
    b=joblib.load(INPUT);validate(b);d=b['data'];x=d['X'];dates=pd.to_datetime(d['dates']);train=np.asarray(dates<'2025-07-01');test=~train;short=x.games_played.to_numpy()<10
    assert_disjoint_games(d['games'][train],d['games'][test]);features(x,d['lines'])
    for m in [short,~short]:
        for s in [train,test]:
            if (m&s).sum()<100 or len(set(d['games'][m&s]))<20:raise ValueError('Insufficient fitting/comparison cohort support')
    ROOT.mkdir(exist_ok=False)
    save_new(ROOT/'plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),'historical_independent':False,'input_sha256':digest(INPUT),'source_hashes':{p:digest(p) for p in [__file__,'src/points_simple_correction.py','src/points_market_correction.py']},'columns':['history_attenuated_scoring_edge','prior_fga_per_appearance','log_history','rest_days'],'parameters':5,'penalties_mean_loss':[.25,1.,4.],'training_before':'2025-07-01','selection':'None; six fixed diagnostic arms, no tuning or promotion on exposed comparison. Only2 earlier short-history selection cases remain insufficient.','weighting':'Balanced pooled:50% each history group, equal weight to games within each group. Separate: equal weight per game within that cohort. Weights sum1; preprocessing uses training weights only.','history_edge':'(.7*recent_avg+.3*season_avg-line)*n/(n+10), n is available prior appearances capped20; factor fixed, not learned from comparison.','missing':'Fail closed if any required input absent; no imputation. No unavailable long-window fields.','limitation':'Prior-based feature estimates are available, not necessarily precise. Reweighting cannot create independent games. All prior-history source contracts unchanged.'})
    models={}
    for penalty in [.25,1.,4.]:
        for mode in ['balanced','separate']:
            name=f'{mode}_{penalty:g}';group={}
            for cohort,m in ([('all',train)] if mode=='balanced' else [('short',train&short),('eligible',train&~short)]):
                group[cohort]=fit_simple(x.loc[m],d['lines'][m],b['market'][m],d['y'][m],d['games'][m],penalty,separate=mode=='separate')
            joblib.dump(group,ROOT/(name+'.joblib'));models[name]=group
    xx=x.loc[test];lines=d['lines'][test];market=b['market'][test];y=d['y'][test];games=d['games'][test];sh=short[test]
    def predict(group,line):
        if 'all' in group:return group['all'].predict_over(xx,line,market)
        p=np.empty(len(xx))
        for c,m in [('short',sh),('eligible',~sh)]:p[m]=group[c].predict_over(xx.loc[m],line[m],market[m])
        return p
    predictions={n:predict(g,lines) for n,g in models.items()};predictions.update(market=market,coin_flip=np.full(len(xx),.5))
    old=joblib.load('logs/points_market_correction_02/offset_compact_ridge100.joblib');predictions['previous_correction']=old.predict_over(xx,lines,market)
    compact=joblib.load('logs/points_evidence_feature_experiment_01/legacy_compact.joblib');predictions['compact']=compact.predict_over(xx,lines)
    checks={}
    for n,g in models.items():
        last=None;violations=0
        for delta in np.arange(-5,5.01,.5):
            p=predict(g,np.maximum(.5,lines+delta))
            if last is not None:violations+=int((p>last+1e-10).sum())
            last=p
        if violations:raise ValueError('Nonmonotone correction')
        checks[n]={'comparisons':len(xx)*20,'violations':violations}
    report={'promoted':False,'historical_independent':False,'selection':'No arm selected; retain previously selected market fallback','monotonicity':checks,'coverage':{'all_required_inputs':1.0,'training_short_rows':int((train&short).sum()),'training_short_games':len(set(d['games'][train&short]))},'cohorts':{}}
    for cohort,m in [('all',np.ones(len(xx),bool)),('short_history',sh),('eligible_history',~sh)]:
        report['cohorts'][cohort]={'rows':int(m.sum()),'games':len(set(games[m])),'metrics':{n:metrics(y[m],p[m]) for n,p in predictions.items()},'mean_abs_correction':{n:float(np.mean(abs(p[m]-market[m]))) for n,p in predictions.items()},'paired':{n:{ref:paired_losses(y[m],p[m],predictions[ref][m],games[m]) for ref in ['market','previous_correction','compact']} for n,p in predictions.items() if n in models}}
    save_new(ROOT/'report.json',report)
    pd.DataFrame({'game':games,'player':d['players'][test],'history':x.games_played.to_numpy()[test],'over':y,**predictions}).to_csv(ROOT/'predictions.csv',index=False)
    print(json.dumps({c:{n:{k:v[k] for k in ['log_loss','auc']} for n,v in r['metrics'].items()} for c,r in report['cohorts'].items()},indent=2))

if __name__=='__main__':run()
