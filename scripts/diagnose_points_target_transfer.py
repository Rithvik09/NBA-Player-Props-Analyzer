"""Matched-row transfer diagnostics, using existing artifacts only."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.prospective_points import save_new

ROOT=Path('logs/points_target_experiment_01')


def run():
    d=joblib.load('logs/points_evidence_representation_01/inputs.joblib')['data']
    mask=np.asarray(pd.to_datetime(d['dates'])>='2025-07-01');x=d['X'].loc[mask];pts=d['points'][mask];lines=d['lines'][mask];y=d['y'][mask]
    centers=np.maximum(.5,np.floor(.7*x.recent_avg.to_numpy()+.3*x.season_avg.to_numpy())+.5);synthetic_y=(pts>centers).astype(int)
    models={n:joblib.load(ROOT/(n+'.joblib')) for t in ['centered','fixed','points'] for n in [t+'_core',t+'_shooting']}
    real={n:c.raw(x,lines) for n,c in models.items()};synthetic={n:c.raw(x,centers) for n,c in models.items()}
    from src.points_v5 import prepare
    means={n:c.model.predict(prepare(x)[c.columns]) for n,c in models.items() if n.startswith('points')}
    report={'interpretation':'Paired same rows, different thresholds/labels. Descriptive unadjusted game-cluster intervals; previously exposed outcomes. Positive AUC delta means shooting helps; negative MAE delta means shooting helps. Raw AUC avoids a fitted zero calibration slope hiding ranking.', 'cohorts':{}}
    for cohort,m in [('short_history',x.games_played.to_numpy()<10),('eligible_history',x.games_played.to_numpy()>=10)]:
        games,ids=np.unique(d['games'][mask][m],return_inverse=True);rng=np.random.default_rng(42)
        draws={t:[] for t in ['centered','fixed','points']};errors=[]
        for _ in range(500):
            w=np.bincount(rng.integers(0,len(games),len(games)),minlength=len(games))[ids]
            if any(len(set(labels[m][w>0]))<2 for labels in [y,synthetic_y]):continue
            for t in draws:
                a=t+'_shooting';b=t+'_core'
                actual=roc_auc_score(y[m],real[a][m],sample_weight=w)-roc_auc_score(y[m],real[b][m],sample_weight=w)
                synth=roc_auc_score(synthetic_y[m],synthetic[a][m],sample_weight=w)-roc_auc_score(synthetic_y[m],synthetic[b][m],sample_weight=w)
                draws[t].append([actual,synth,actual-synth])
            errors.append(np.average(abs(means['points_shooting'][m]-pts[m])-abs(means['points_core'][m]-pts[m]),weights=w))
        entry={'rows':int(m.sum()),'games':len(games),'targets':{}}
        for t in draws:
            a=t+'_shooting';b=t+'_core';actual=roc_auc_score(y[m],real[a][m])-roc_auc_score(y[m],real[b][m]);synth=roc_auc_score(synthetic_y[m],synthetic[a][m])-roc_auc_score(synthetic_y[m],synthetic[b][m])
            entry['targets'][t]={'real_raw_auc':{n:roc_auc_score(y[m],real[n][m]) for n in [a,b]},'real_auc_delta':actual,'synthetic_auc_delta':synth,'transfer_delta':actual-synth,'intervals_95_actual_synthetic_transfer':np.quantile(draws[t],[.025,.975],axis=0).T.tolist()}
        entry['points_shooting_minus_core_mae']={'delta':float(np.mean(abs(means['points_shooting'][m]-pts[m])-abs(means['points_core'][m]-pts[m]))),'interval_95':np.quantile(errors,[.025,.975]).tolist()}
        report['cohorts'][cohort]=entry
    save_new(ROOT/'transfer_diagnostics.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':run()
