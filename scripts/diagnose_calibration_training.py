"""Offline forensic probes of exposed research data; never changes a frozen study."""
import json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.metrics import log_loss,roc_auc_score
from src.prospective_points import save_new,digest
from src.points_v5 import prepare
from src.verified_training import classifier_frame
from src.points_calibration_research_v2 import validate,probability

ROOT=Path('logs/points_training_diagnosis_20261026')

def score(y,p):
 return {'n':len(y),'log_loss':float(log_loss(y,p,labels=[0,1])),'brier':float(np.mean((p-y)**2)),'auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,'mean_probability':float(np.mean(p)),'over_rate':float(np.mean(y))}

def run():
 ROOT.mkdir(exist_ok=False)
 b=joblib.load('logs/real_points_paid_calibration_01/calibration_inputs.joblib');d=b['data'];x=prepare(d['X']);y=d['y'];dates=pd.to_datetime(d['dates']);fit,sel,support=validate(b);hist=x.games_played.to_numpy()
 parts=[joblib.load(f'logs/points_v5_data/{s}.joblib') for s in ['2022-23','2023-24','2024-25','2025-26']]
 trainkeys=set((str(g),int(p)) for a in parts for g,p,t in zip(a['games'],a['players'],pd.to_datetime(a['dates'])) if t<pd.Timestamp('2025-01-01'))
 overlap=np.array([(str(g),int(p)) in trainkeys for g,p in zip(d['games'],d['players'])])
 masks={'calfit_short':fit&(hist<10),'calfit_eligible':fit&(hist>=10),'selection_short':sel&(hist<10),'selection_eligible':sel&(hist>=10),'base_seen':fit&overlap,'base_unseen_calfit':fit&~overlap}
 report={'historical_independent':False,'support':support,'base_training_overlap':{k:{'n':int(m.sum()),'in_base_training':int((m&overlap).sum()),'games':len(set(d['games'][m]))} for k,m in masks.items()},'models':{},'probes':{}}
 for name in ['repaired','compact_retrained']:
  model=joblib.load(f'logs/points_v6_real_01/{name}.joblib');cf=classifier_frame(x,d['lines'],model.columns);raw=model.classifier.predict_proba(cf)[:,1]
  center=.7*x.recent_avg.to_numpy()+.3*x.season_avg.to_numpy();synthetic=np.maximum(.5,np.floor(center)+.5)
  synpred=model.classifier.predict_proba(classifier_frame(x,synthetic,model.columns))[:,1]
  lower=model.classifier.predict_proba(classifier_frame(x,np.maximum(.5,d['lines']-1),model.columns))[:,1];upper=model.classifier.predict_proba(classifier_frame(x,d['lines']+1,model.columns))[:,1]
  report['models'][name]={'metrics':{k:score(y[m],raw[m]) for k,m in masks.items() if m.any()},'synthetic_line_counterfactual':score((d['points'][sel]>synthetic[sel]).astype(int),synpred[sel]),'monotonicity':{'tested':int(sel.sum()),'probability_increases_when_line_rises':int(((upper>lower+1e-7)&sel).sum())},'line_support':{'outside_training_center_plusminus2_5':int((np.abs(d['lines']-center)[sel]>2.5).sum()),'selection_rows':int(sel.sum()),'median_abs_market_minus_center':float(np.median(np.abs(d['lines']-center)[sel]))},'importance':dict(sorted(zip(cf.columns,map(float,model.classifier.feature_importances_)),key=lambda a:-a[1]))}
  if name=='repaired':
   report['feature_audit']=[]
   for col in cf:
    a=cf.loc[fit,col].to_numpy(float);z=cf.loc[sel,col].to_numpy(float);finite=a[np.isfinite(a)];test=z[np.isfinite(z)]
    report['feature_audit'].append({'feature':col,'missing_fit':float(np.mean(~np.isfinite(a))),'missing_selection':float(np.mean(~np.isfinite(z))),'unique_fit':len(set(finite)),'unique_selection':len(set(test)),'outside_fit_range':float(np.mean((test<finite.min())|(test>finite.max()))) if len(finite) and len(test) else None})
   groups={'injury':[c for c in cf if any(k in c for k in ['injury','players_out','out_count','out_unresolved','out_unknown'])],'matchup':[c for c in cf if any(k in c for k in ['vs_','matchup','h2h'])],'shooting':[c for c in cf if any(k in c for k in ['fg','shoot','shot','ts_pct','scoring_efficiency'])],'history':[c for c in cf if any(k in c for k in ['history','games_played','short_history'])]}
   base=log_loss(y[sel],raw[sel]);abl={}
   for group,cols in groups.items():
    xx=cf.loc[sel].copy();xx[cols]=np.nan;p=model.classifier.predict_proba(xx)[:,1];abl[group]={'columns':cols,'masking_log_loss':float(log_loss(y[sel],p)),'delta_vs_raw':float(log_loss(y[sel],p)-base),'interpretation':'Masking sensitivity only, not retrained feature ablation'}
   report['masking_probes']=abl
 # Real-line-only, fixed regularized low-dimensional market residual probe.
 # Fitting labels only; no synthetic outcomes, no grid or selection-label tuning.
 edge=(x.recent_avg.to_numpy()-d['lines'])/np.maximum(x.stddev.to_numpy(),3)
 z=np.c_[np.nan_to_num(edge,nan=0),np.nan_to_num(x.season_avg.to_numpy()-d['lines'],nan=0)/10,(hist<10).astype(float)]
 z=np.clip(z,-5,5);offset=logit(np.clip(b['market'],1e-6,1-1e-6))
 for strength in [1.]:
  def objective(w):
   v=offset[fit]+z[fit]@w;return np.mean(np.logaddexp(0,v)-y[fit]*v)+strength*np.sum(w*w)/2
  opt=minimize(objective,np.zeros(z.shape[1]),method='BFGS');assert opt.success
  p=expit(offset+z@opt.x)
  report['probes']['fixed_real_only_market_residual']={'coefficients':opt.x.tolist(),'regularization':strength,'metrics':{k:score(y[m],p[m]) for k,m in masks.items() if m.any()}}
 report['baselines']={k:{name:score(y[m],p[m]) for name,p in [('market',b['market']),('coin_flip',np.full(len(y),.5))]} for k,m in masks.items() if m.any()}
 report['source_hashes']={p:digest(p) for p in [__file__,'logs/real_points_paid_calibration_01/calibration_inputs.joblib','logs/points_v6_real_01/repaired.joblib','logs/points_v6_real_01/compact_retrained.joblib']}
 save_new(ROOT/'report.json',report)
 print(json.dumps({k:v for k,v in report.items() if k not in ['feature_audit','models']},indent=2))
 print('MODELS',json.dumps({k:{a:b for a,b in v.items() if a!='importance'} for k,v in report['models'].items()},indent=2))

if __name__=='__main__':run()
