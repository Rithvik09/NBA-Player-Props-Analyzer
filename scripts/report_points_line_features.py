"""Descriptive game-cluster AUC differences for the predeclared ablations."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from src.prospective_points import save_new
ROOT=Path('logs/points_line_feature_experiment_02')


def run():
 p=pd.read_csv(ROOT/'predictions.csv');plan=json.loads((ROOT/'plan.json').read_text());report={'limitation':'Exposed retrospective comparison; many probes, unadjusted descriptive95% intervals. No feature declared independently validated.','cohorts':{}}
 for cohort,mask in [('short_history',p.history<10),('eligible_history',p.history>=10)]:
  d=p.loc[mask];y=d.over.to_numpy();games,ids=np.unique(d.game.to_numpy(),return_inverse=True);rng=np.random.default_rng(42)
  pairs=[('wide_full','narrow_full'),('wide_compact','narrow_compact'),('empirical_full','narrow_full'),('empirical_compact','narrow_compact')]
  for group in plan['feature_partitions']:pairs.extend([('wide_add_'+group,'wide_compact'),('wide_drop_'+group,'wide_full')])
  names=sorted({n for pair in pairs for n in pair});values={n:d['raw_'+n].to_numpy() for n in names};point={n:roc_auc_score(y,v) for n,v in values.items()};draws={n:[] for n in names}
  for _ in range(500):
   counts=np.bincount(rng.integers(0,len(games),len(games)),minlength=len(games));w=counts[ids]
   if len(set(y[w>0]))<2:continue
   for n,v in values.items():draws[n].append(roc_auc_score(y,v,sample_weight=w))
  report['cohorts'][cohort]={'rows':len(d),'games':len(games),'comparisons':[{ 'arm':a,'reference':b,'auc_delta':float(point[a]-point[b]),'interval_95':np.quantile(np.array(draws[a])-draws[b],[.025,.975]).tolist()} for a,b in pairs]}
 save_new(ROOT/'ranking_uncertainty.json',report)
 for cohort,v in report['cohorts'].items():
  print(cohort)
  for x in v['comparisons']:print(x)

if __name__=='__main__':run()
