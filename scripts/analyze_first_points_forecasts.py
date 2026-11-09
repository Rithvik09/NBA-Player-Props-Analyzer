"""Post-outcome diagnosis only; never changes the frozen prospective study."""
import json,joblib,numpy as np,pandas as pd,xgboost as xgb
from pathlib import Path
from src.verified_training import classifier_frame
root=Path('logs/prospective_points_v1')
models={n:joblib.load('logs/expanded_points_experiment_v4/'+p+'.joblib') for n,p in [('candidate','development_selected'),('compact','compact_expanded')]}
rows=[]
for path in sorted((root/'forecasts').glob('*.json')):
 f=json.loads(path.read_text())
 if f['quote']['game_id']!='0022600112' or f['horizon']!='request_time':continue
 o=json.loads((root/'settlements'/path.name).read_text()); y=o['over'];line=f['quote']['line']
 raw=pd.DataFrame([f['features']],dtype=float); contributions={};probs={};losses={}
 for n,m in models.items():
  frame=classifier_frame(m._normalize(raw),[line],m.columns)
  dm=xgb.DMatrix(frame,feature_names=list(frame.columns))
  vals=m.classifier.get_booster().predict(dm,pred_contribs=True)[0]
  a,b=(1.,0.) if m.calibrator.method=='raw' else (float(m.calibrator.estimator.coef_[0,0]),float(m.calibrator.estimator.intercept_[0]))
  con=dict(zip([*frame.columns,'BASE'],map(float,vals*a)));con['BASE']+=b
  contributions[n]=con
  p=float(m.predict_over(raw,[line])[0]);probs[n]=p
  assert abs(p-f['predictions'][n]['over_probability'])<1e-9
  assert abs(1/(1+np.exp(-sum(con.values())))-p)<1e-5
  losses[n]=float(-(y*np.log(p)+(1-y)*np.log(1-p)))
  pr=float(m.classifier.predict_proba(frame)[0,1]);probs[n+'_uncalibrated']=pr
 delta={k:contributions['candidate'].get(k,0)-contributions['compact'].get(k,0) for k in set(contributions['candidate'])|set(contributions['compact'])}
 ranked=sorted(delta,key=lambda k:-abs(delta[k]))
 values=classifier_frame(models['candidate']._normalize(raw),[line],models['candidate'].columns).iloc[0].to_dict()
 row={'player':f['quote']['player_name'],'id':f['quote']['player_id'],'line':line,'points':o['points'],'over':y,'probabilities':probs,'log_losses':losses,'loss_delta':losses['candidate']-losses['compact'],'point_predictions':{n:f['predictions'][n]['points'] for n in models},'prior_appearances':f['prior_appearances'],'missing_selected':{n:sum(pd.isna(raw[c].iloc[0]) for c in m.columns) for n,m in models.items()},'values':f['features'],'contributions':contributions,'contribution_differences':delta,'top_differences':[{ 'feature':k,'value':float(values[k]) if k in values and pd.notna(values[k]) else None,'log_odds_delta':delta[k]} for k in ranked[:8]]}
 rows.append(row)
assert len(rows)==14
training=[]
for season in ['2022-23','2023-24','2024-25']:
 d=joblib.load('logs/expanded_points_v4/audited_'+season+'.joblib')
 training.append(d['X'].loc[pd.to_datetime(d['dates'])<'2025-01-01'])
training=pd.concat(training)
outside=[]
for r in rows:
 center=.7*r['values']['recent_avg']+.3*r['values']['season_avg']
 low,high=max(.5,float(np.floor(center)+.5-2)),float(np.floor(center)+.5+2)
 r['synthetic_line_band_at_same_inputs']=[low,high]
 if not low<=r['line']<=high:outside.append(r['player'])
summary={'n':14,'independent_games':1,'candidate_better_probability_cases':sum(r['loss_delta']<0 for r in rows),
 'mean_log_loss_delta':float(np.mean([r['loss_delta'] for r in rows])),
 'mean_log_loss':{n:float(np.mean([-(r['over']*np.log(r['probabilities'][n])+(1-r['over'])*np.log(1-r['probabilities'][n])) for r in rows])) for n in ['candidate','compact','compact_uncalibrated']},
 'missing_team_injury_count':sum(r['values']['team_key_players_out'] is None for r in rows),
 'training_missing_team_injury_fraction':float(training.team_key_players_out.isna().mean()),
 'real_lines_outside_synthetic_band':outside,
 'training_feature_quantiles':{c:{str(q):float(v) for q,v in training[c].quantile([.01,.5,.99]).items()} for c in ['trend_slope_5games','minutes_last_3_games','minutes_last_5_games','effective_fg_pct','true_shooting_pct']}}
result={'method':'XGBoost exact tree contributions on frozen archived inputs; compact contributions adjusted by frozen sigmoid slope/intercept. Contributions explain model behavior, not causal effects or feature value on independent data.','summary':summary,'rows':rows}
(root/'diagnostics'/'first14_attribution.json').write_text(json.dumps(result,indent=2,allow_nan=False))
for r in sorted(rows,key=lambda r:-r['loss_delta']):
 print(r['player'],'line',r['line'],'actual',r['points'],'P',[round(r['probabilities'][n],4) for n in models],'delta',round(r['loss_delta'],4),'history',r['prior_appearances'],'missing',r['missing_selected'])
 print(' TOP',[(a['feature'],a['value'],round(a['log_odds_delta'],3)) for a in r['top_differences'][:5]])
print('wins',sum(r['loss_delta']<0 for r in rows),'totaldelta',sum(r['loss_delta'] for r in rows))
