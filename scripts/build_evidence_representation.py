"""Build offline feature sidecars; all rolling windows strictly precede target day."""
from pathlib import Path
import json
import joblib,pandas as pd,numpy as np
from src.historical_inputs import load_season_logs
from src.prospective_points import save_new,digest
from src.points_evidence_representation import prior_counts,shooting_features,complete_injury
ROOT=Path('logs/points_evidence_representation_01')


def run():
 ROOT.mkdir(exist_ok=False);frames=[];hashes={}
 for season in ['2022-23','2023-24','2024-25','2025-26']:
  frames.append(load_season_logs(season,include_postseason=True))
  for kind in ['players','playoffs','playin']:
   p=Path(f'logs/points_pilot_cache/{kind}_{season}.json');hashes[str(p)]=digest(p)
 before=pd.concat(frames,ignore_index=True);before=before[pd.to_datetime(before.GAME_DATE)<'2024-10-01'];tot=before[['FGM','FGA','FG3M','FG3A','FTM','FTA','PTS']].sum()
 priors={'fg':float(tot.FGM/tot.FGA),'fg3':float(tot.FG3M/tot.FG3A),'ft':float(tot.FTM/tot.FTA),'ts':float(tot.PTS/(2*(tot.FGA+.44*tot.FTA))),'efg':float((tot.FGM+.5*tot.FG3M)/tot.FGA)}
 table=pd.concat([prior_counts(f) for f in frames],ignore_index=True).set_index(['game','player']);assert not table.index.duplicated().any()
 def enrich(d):
  idx=pd.MultiIndex.from_arrays([d['games'].astype(str),d['players'].astype(int)],names=['game','player']);c=table.reindex(idx).reset_index(drop=True)
  if c.date.isna().any() or not np.array_equal(pd.to_datetime(c.date).to_numpy(),pd.to_datetime(d['dates']).to_numpy()):raise ValueError('Identity/date mismatch')
  if not np.array_equal(c.n_20.to_numpy(),d['X'].games_played.to_numpy()):raise ValueError('History contract mismatch')
  x=d['X'].copy().reset_index(drop=True);extra=shooting_features(c,priors)
  for col in extra:x[col]=extra[col]
  for col,val in complete_injury(x).items():x[col]=val
  return {**d,'X':x},c
 sources={};audit={}
 for season in ['2022-23','2023-24','2024-25','2025-26']:
  p=Path(f'logs/points_v5_data/{season}.joblib');d,c=enrich(joblib.load(p));joblib.dump(d,ROOT/(season+'.joblib'));sources[str(p)]=digest(p)
 b=joblib.load('logs/real_points_paid_calibration_01/calibration_inputs.joblib');b['data'],c=enrich(b['data']);joblib.dump(b,ROOT/'inputs.joblib')
 x=b['data']['X'];dates=pd.to_datetime(b['data']['dates']);audit['zero_attempt_5game_rows']={k:int((c[k+'_5']==0).sum()) for k in ['FGA','FG3A','FTA']};audit['injury_partial_zero']={side:int(((x[side+'_known_key_players_out']==0)&x['qualified_'+side+'_known_key_players_out'].isna()).sum()) for side in ['team','opp']}
 audit['duplicate_fta_fields_identical']=bool(np.allclose(x.fta_per_game,x.ft_attempts_per_game,equal_nan=True));audit['coverage']={split:{col:float(x.loc[mask,col].notna().mean()) for col in x if col.startswith(('reliable_','qualified_'))} for split,mask in [('base',dates<'2024-10-01'),('calibration',(dates>='2024-10-01')&(dates<'2025-07-01')),('comparison',dates>='2025-07-01')]}
 save_new(ROOT/'construction.json',{'priors':priors,'prior_fit_before':'2024-10-01','raw_source_hashes':hashes,'dataset_hashes':sources,'audit':audit,'shooting_policy':'Prior5 aggregate rates shrunk by50 FG/20 FG3/20 FT attempts; prior20 TS/EFG by100 denominator units/50FGA. Hyperparameters fixed before model fitting. No target outcomes used in features.','injury_policy':'Both counts missing unless published report and no unresolved identities or unknown roles; genuine complete zero retained.'});print(json.dumps(audit,indent=2))

if __name__=='__main__':run()
