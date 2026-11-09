"""Prior-season-only game-script probabilities with chronological qualification.

These are feature-source forecasters, never player-prop serving replacements.
For each target season, fit/calibrate/check only on earlier seasons. A failed
prior-period Brier/AUC gate leaves that feature missing for that target season.
"""
from pathlib import Path
from bisect import bisect_left
import json,os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,roc_auc_score
from scripts.train_models import _load_precomputed_player_data

OUT=Path('logs/feature_recovery/game_script');OUT.mkdir(exist_ok=True)
COLUMNS=['strength','abs_strength','strength_squared','pace','rest_difference']


def run():
    raw=_load_precomputed_player_data(os.environ['DATABASE_URL'])
    games=[]
    first=pd.read_json('logs/feature_recovery/team_games_2021-22.json',dtype={'GAME_ID':str})
    for gid,group in first.groupby('GAME_ID'):
        if len(group)!=2:continue
        for _,row in group.iterrows():
            other=group[group.TEAM_ID!=row.TEAM_ID].iloc[0]
            poss=.5*sum(float(r.FGA)+.44*float(r.FTA)-float(r.OREB)+float(r.TOV) for r in (row,other));duration=float(row.MIN)/5
            games.append({'game_id':gid,'team_id':int(row.TEAM_ID),'game_date':str(pd.Timestamp(row.GAME_DATE).date()),'pace':48*poss/duration,'off_rating':100*float(row.PTS)/poss,'def_rating':100*float(other.PTS)/poss,'pts':float(row.PTS),'opp_pts':float(other.PTS),'home':'vs' in row.MATCHUP})
    for season in ('2022-23','2023-24','2024-25','2025-26'):
        logs=pd.read_json(f'logs/points_pilot_cache/players_{season}.json',dtype={'GAME_ID':str})
        dates={str(pd.Timestamp(r['game_date']).date()):r for rows in raw['game_team_stats'].values() for r in rows} # rates joined by team below
        rates={(int(tid),str(r['game_date'])[:10]):r for tid,rows in raw['game_team_stats'].items() for r in rows}
        for (gid,tid),part in logs.groupby(['GAME_ID','TEAM_ID']):
            day=str(pd.Timestamp(part.GAME_DATE.iloc[0]).date());rate=rates.get((int(tid),day))
            if rate and all(rate.get(k) is not None for k in ('pace','off_rating','def_rating','pts','opp_pts')):games.append({**rate,'game_id':str(gid),'team_id':int(tid),'game_date':day,'home':'vs' in part.MATCHUP.iloc[0]})
    teams={}
    for g in sorted(games,key=lambda r:r['game_date']):teams.setdefault(g['team_id'],[]).append(g)
    dates={tid:[g['game_date'] for g in rows] for tid,rows in teams.items()}
    groups={}
    for g in games:groups.setdefault(g['game_id'],[]).append(g)
    rows=[]
    for gid,pair in groups.items():
        if len(pair)!=2:continue
        home=next((r for r in pair if r['home']),None);away=next((r for r in pair if not r['home']),None)
        if home is None or away is None:continue
        day=home['game_date'];context=[]
        for row in (home,away):
            tid=row['team_id'];end=bisect_left(dates[tid],day);prior=teams[tid][max(0,end-10):end]
            if len(prior)<5:break
            context.append({k:float(np.mean([g[k] for g in prior])) for k in ('pace','off_rating','def_rating')}|{'rest':min(7,max(0,(pd.Timestamp(day)-pd.Timestamp(prior[-1]['game_date'])).days-1))})
        if len(context)!=2:continue
        h,a=context;strength=(h['off_rating']+a['def_rating']-a['off_rating']-h['def_rating'])/2
        rows.append({'date':day,'game_id':gid,'strength':strength,'abs_strength':abs(strength),'strength_squared':strength**2,'pace':(h['pace']+a['pace'])/2,'rest_difference':h['rest']-a['rest'],'margin':abs(home['pts']-away['pts'])})
    frame=pd.DataFrame(rows).sort_values('date');report={}
    for year in range(2022,2027):
        season=f'{year}-{str(year+1)[2:]}';prior=frame[frame.date<f'{year}-10-01'];unique=sorted(prior.date.unique());a=unique[int(len(unique)*.70)];b=unique[int(len(unique)*.85)]
        fit=prior[prior.date<a];cal=prior[(prior.date>=a)&(prior.date<b)];check=prior[prior.date>=b]
        bundle={'target_season':season,'columns':COLUMNS,'last_outcome_date':prior.date.max(),'protocol':{'fit_end':fit.date.max(),'calibration_end':cal.date.max(),'qualification_start':check.date.min(),'qualification_end':check.date.max()},'models':{}}
        for key,label in [('blowout_probability',lambda x:(x.margin>15).astype(int)),('close_game_probability',lambda x:(x.margin<=5).astype(int))]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000)).fit(fit[COLUMNS],label(fit))
            logit=lambda df:model.decision_function(df[COLUMNS]).reshape(-1,1)
            calibrator=LogisticRegression(C=1,max_iter=1000).fit(logit(cal),label(cal))
            pred=calibrator.predict_proba(logit(check))[:,1];truth=label(check)
            brier=float(brier_score_loss(truth,pred));auc=float(roc_auc_score(truth,pred));constant=float(brier_score_loss(truth,np.full(len(truth),label(cal).mean())))
            qualified=bool(len(check)>=100 and brier<=constant and auc>.5)
            bundle['models'][key]={'model':model,'calibrator':calibrator,'qualified':qualified,'qualification_metrics':{'n':len(check),'brier':brier,'constant_brier':constant,'auc':auc}}
        joblib.dump(bundle,OUT/f'{season}.joblib')
        report[season]={'last_outcome_date':bundle['last_outcome_date'],'protocol':bundle['protocol'],'models':{k:{'qualified':v['qualified'],**v['qualification_metrics']} for k,v in bundle['models'].items()}}
        print(season,report[season],flush=True)
    (OUT/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':run()
