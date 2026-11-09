"""No-network chronological research comparison with fixed repaired training rules."""
from pathlib import Path
from datetime import datetime,timezone
import json
import joblib,numpy as np,pandas as pd
from xgboost import XGBClassifier
from src.prospective_points import save_new,digest
from src.points_v5 import prepare
from src.verified_training import expand_points,classifier_frame
from src.points_calibration_research_v2 import validate,fit as calibrate
from src.points_training_repair import keep_synthetic,outcome_weights,assert_disjoint_games,monotone_constraints,RepairedMonotonePoints,unique_outcomes
from scripts.points_pilot import metrics
from scripts.train_selected_points import paired_losses
from scripts.investigate_points_features import PARAMS
from src.points_evidence_representation import RAW_SHOOTING,INJURY
from src.points_line_experiments import expand_with_offsets

ROOT=Path('logs/points_evidence_feature_experiment_01')
INPUT=Path('logs/points_evidence_representation_01/inputs.joblib')


def run():
 b=joblib.load(INPUT);d=b['data'];validate(b)
 dates=pd.to_datetime(d['dates']);base=np.asarray(dates<'2024-10-01');cal=np.asarray((dates>='2024-10-01')&(dates<'2025-07-01'));evaluation=np.asarray(dates>='2025-07-01');x=prepare(d['X']);h=x.games_played.to_numpy()
 assert_disjoint_games(d['games'][base],d['games'][cal],d['games'][evaluation])
 support={}
 for name,mask in [('short_history',h<10),('eligible_history',h>=10)]:
  support[name]={}
  for split,m in [('calibration',cal&mask),('evaluation',evaluation&mask)]:
   support[name][split]={'rows':int(m.sum()),'games':len(set(d['games'][m]))}
   if m.sum()<100 or len(set(d['games'][m]))<20:raise ValueError('Insufficient separate cohort support')
 parts=[];training_sources={}
 for season in ['2022-23','2023-24','2024-25','2025-26']:
  path=Path(f'logs/points_evidence_representation_01/{season}.joblib');a=joblib.load(path);m=pd.to_datetime(a['dates'])<'2024-10-01';parts.append({'X':a['X'].loc[m],**{k:a[k][m] for k in ['y','games','players','dates']}});training_sources[str(path)]=digest(path)
 sx=pd.concat([a['X'] for a in parts],ignore_index=True);sy=np.concatenate([a['y'] for a in parts]);sg=np.concatenate([a['games'] for a in parts]);sp=np.concatenate([a['players'] for a in parts])
 keep=keep_synthetic(sg,sp,d['games'][base],d['players'][base]);sg=sg[keep];sp=sp[keep];sx=prepare(sx.loc[keep].reset_index(drop=True));sy=sy[keep]
 assert_disjoint_games(sg,d['games'][cal],d['games'][evaluation]);unique_outcomes(np.r_[sg,d['games'][base]],np.r_[sp,d['players'][base]])
 if len(sg)+base.sum()<100 or len(set(sg)|set(d['games'][base]))<20:raise ValueError('Insufficient base training')
 full=joblib.load('logs/points_training_repair_01/repaired.joblib').columns
 compact=joblib.load('logs/points_training_repair_01/compact_retrained.joblib').columns
 core=[c for c in compact if c not in INJURY]
 shooting=[c for c in x if c.startswith(('reliable_','support_','volume_'))]
 injury=[c for c in x if c.startswith('qualified_')]
 groups={'core':core,'legacy_compact':compact,'core_raw_shooting':core+RAW_SHOOTING,'core_reliable_shooting':core+shooting,'core_legacy_injury':list(dict.fromkeys(core+INJURY)),'core_qualified_injury':core+injury,'core_combined':core+shooting+injury,'legacy_full':full,'full_represented':[c for c in full if c not in RAW_SHOOTING+INJURY+['ft_attempts_per_game']]+shooting+injury}
 ROOT.mkdir(exist_ok=False)
 plan={'created_at':datetime.now(timezone.utc).isoformat(),'base_before':'2024-10-01','calibration_before':'2025-07-01','evaluation_after':'2025-07-01','historical_independent':False,'calibration':'Fixed intercept+nonnegative slope per cohort, fit on calibration only; no method selection or refit on evaluation.','weighting':'Each original player-game outcome totals1; real quote replaces all synthetic expansions for the same player-game. No source block reweighting.','synthetic_lines':'Fixed wide[-8,-4,0,4,8];same in all nine arms','representation_evidence_sha256':digest('logs/points_evidence_representation_01/construction.json'),'groups':groups,'parameters':{**PARAMS,'max_depth':3,'min_child_weight':50,'reg_lambda':50,'colsample_bytree':1},'input_sha256':digest(INPUT),'training_sources':training_sources,'support':support,'removed_synthetic_outcomes':int((~keep).sum()),'base_real_outcomes':int(base.sum()),'base_synthetic_outcomes':len(sg),'base_games':len(set(sg)|set(d['games'][base])),'source_hashes':{p:digest(p) for p in [__file__,'src/points_evidence_representation.py','src/points_line_experiments.py','src/points_training_repair.py','src/points_calibration_research_v2.py','src/points_v5.py','src/verified_training.py']}}
 save_new(ROOT/'plan.json',plan)
 saved={};raws={}
 for name,cols in groups.items():
  xx,yy,idx,_=expand_with_offsets({'X':sx,'y':sy},cols,[-8,-4,0,4,8]);rx=classifier_frame(x,d['lines'],cols);xx=pd.concat([xx,rx.loc[base]],ignore_index=True).astype(np.float32);yy=np.r_[yy,d['y'][base]];weights=outcome_weights(idx,int(base.sum()))
  model=XGBClassifier(**plan['parameters'],eval_metric='logloss',monotone_constraints=monotone_constraints(list(xx.columns)))
  model.fit(xx,yy,sample_weight=weights)
  raw=model.predict_proba(rx)[:,1];calibrations={cohort:calibrate(raw[cal&m],d['y'][cal&m],'intercept_slope') for cohort,m in [('short_history',h<10),('eligible_history',h>=10)]}
  candidate=RepairedMonotonePoints(cols,model,calibrations);joblib.dump(candidate,ROOT/(name+'.joblib'));saved[name]=candidate;raws[name]=raw
  save_new(ROOT/(name+'_fit.json'),{'calibrations':calibrations,'weight_min':float(weights.min()),'weight_max':float(weights.max()),'total_weight':float(weights.sum()),'real_total_weight':float(weights[-int(base.sum()):].sum()),'constraints':list(monotone_constraints(list(xx.columns)))})
  print(name,'fit and calibration saved',flush=True)
 # Both candidates and their calibration parameters saved before scoring outcomes.
 predictions={name:model.predict_over(d['X'],d['lines']) for name,model in saved.items()};predictions.update(market=b['market'],coin_flip=np.full(len(h),.5))
 checks={}
 for name,model in saved.items():
  last=None;violations=0
  for offset in np.arange(-5,5.01,.5):
   p=model.predict_over(d['X'].loc[evaluation],np.maximum(.5,d['lines'][evaluation]+offset))
   if last is not None:violations+=int((p>last+1e-10).sum())
   last=p
  checks[name]={'rows':int(evaluation.sum()),'adjacent_line_comparisons':int(evaluation.sum())*20,'violations':violations}
  if violations:raise ValueError('Monotonicity failed')
 report={'historical_independent':False,'promoted':False,'support':support,'monotonicity':checks,'cohorts':{}}
 for cohort,m in [('short_history',evaluation&(h<10)),('eligible_history',evaluation&(h>=10))]:
  report['cohorts'][cohort]={'rows':int(m.sum()),'games':len(set(d['games'][m])),'metrics':{k:metrics(d['y'][m],p[m]) for k,p in predictions.items()},'raw_model_metrics':{k:metrics(d['y'][m],p[m]) for k,p in raws.items()},'paired':{k:paired_losses(d['y'][m],predictions['core_combined'][m],predictions[k][m],d['games'][m]) for k in ['legacy_compact','core','legacy_full','market','coin_flip']}}
 save_new(ROOT/'report.json',report)
 pd.DataFrame({'game':d['games'][evaluation],'player':d['players'][evaluation],'over':d['y'][evaluation],'history':h[evaluation],**{k:p[evaluation] for k,p in predictions.items()},**{'raw_'+k:p[evaluation] for k,p in raws.items()}}).to_csv(ROOT/'predictions.csv',index=False)
 print(json.dumps({k:{a:{metric:values[metric] for metric in ['log_loss','brier','auc']} for a,values in v['metrics'].items()} for k,v in report['cohorts'].items()},indent=2))

if __name__=='__main__':run()
