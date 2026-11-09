import json,joblib,numpy as np,pandas as pd
from pathlib import Path
from src.historical_inputs import HistoricalInputs,load_season_logs
from src.verified_training import expand_points
from scripts.train_models import _parse_matchup,_team_id
from scripts.points_pilot import metrics
from scripts.train_selected_points import paired_losses
season='2025-26';data=joblib.load('logs/points_v5_data/'+season+'.joblib');mask=data['X'].games_played.to_numpy()<10
short={'X':data['X'].loc[mask].reset_index(drop=True),'y':data['y'][mask]};oldx=short['X'].copy()
inputs=HistoricalInputs(season,joblib.load('logs/expanded_points_v4/source_snapshot.joblib'),load_season_logs(season,include_postseason=True),audited=True)
logs=inputs.recovery.logs;lookup={(str(r.GAME_ID),int(r.PLAYER_ID)):r for r in logs.itertuples()};cache={}
def priorinj(tid,date,gid):
 key=(gid,tid)
 if key in cache:return cache[key]
 inputs.prediction_game_id=gid;report=inputs.recovery.injuries(tid,date)
 roles=[];valid=report is not None
 for r in report or []:
  if r['status']!='Out':continue
  pid=inputs.recovery.resolve_name(r['player'])
  if pid is None:valid=False;continue
  roles.append(inputs.recovery.role_minutes(pid,tid,date))
 result=(float(sum(m>=20 for m in roles)),min(1.,sum(roles)/240)) if valid else (np.nan,np.nan)
 cache[key]=result;return result
for i,(gid,pid) in enumerate(zip(data['games'][mask],data['players'][mask])):
 r=lookup[(str(gid),int(pid))];own,opp,_=_parse_matchup(r.MATCHUP);day=pd.Timestamp(r.GAME_DATE)
 oldx.loc[i,'team_key_players_out']=priorinj(_team_id(own),day,str(gid))[0]
 count,impact=priorinj(_team_id(opp),day,str(gid));oldx.loc[i,'opp_key_players_out']=count;oldx.loc[i,'opp_injury_impact']=impact
m1=joblib.load('logs/expanded_points_experiment_v4/development_selected.joblib');m2=joblib.load('logs/points_v5_synthetic/repaired.joblib');mc=joblib.load('logs/expanded_points_experiment_v4/compact_expanded.joblib')
_,y,idx,lines=expand_points(short,m1.columns);a=m1.predict_over(oldx.iloc[idx],lines);b=m2.predict_over(short['X'].iloc[idx],lines);c=mc.predict_over(oldx.iloc[idx],lines)
r={'note':'Original injury semantics reconstructed exactly (three fields changed by enrich_injuries); original short-window inputs unchanged for frozen references. Synthetic lines; exposed research season.', 'player_games':len(oldx),'frozen161':metrics(y,a),'repaired':metrics(y,b),'compact':metrics(y,c),'paired':paired_losses(y,b,a,data['games'][mask][idx])}
Path('logs/points_v5_synthetic/true_short_input_comparison.json').write_text(json.dumps(r,indent=2));print({k:r[k]['log_loss'] for k in ['frozen161','repaired','compact']});print(r['paired']['log_loss'])
