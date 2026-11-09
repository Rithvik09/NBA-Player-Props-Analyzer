"""Select simpler points models chronologically; never modify serving artifacts."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
from scripts.points_pilot import metrics, bootstrap, model_hashes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/points_improved'
COMPACT=['recent_avg','season_avg','stddev','last_3_games_trend','last_5_games_trend','last_10_games_trend','minutes_last_3_games','minutes_last_5_games','minutes_last_7_games','rolling_7day_avg','rolling_14day_avg','rolling_30day_avg','team_pace','opp_def_rating','days_rest','is_home']


def expand(data):
    X=data['X'].replace([np.inf,-np.inf],np.nan)
    center=.7*X.recent_avg.to_numpy()+.3*X.season_avg.to_numpy()
    rows=[];lines=[]
    for i,c in enumerate(center):
        ls=sorted({max(.5,float(np.floor(c)+.5+d)) for d in (-2,-1,0,1,2)})
        rows.extend([i]*len(ls));lines.extend(ls)
    rows=np.array(rows);lines=np.array(lines)
    frame=X.iloc[rows].reset_index(drop=True).copy()
    frame['synthetic_line']=lines
    frame['recent_minus_line']=frame.recent_avg-lines
    frame['season_minus_line']=frame.season_avg-lines
    frame['standardized_edge']=(frame.recent_avg-lines)/frame.stddev.clip(lower=3)
    return frame,(data['y'][rows]>lines).astype(int),rows


def run():
    OUT.mkdir(exist_ok=True,parents=True);before=model_hashes()
    data=joblib.load(ROOT/'logs/points_pilot/lagged_50_e64a3bb92a05.joblib')
    X,y,rows=expand(data);dates=pd.to_datetime(data['dates'])[rows]
    fit=np.asarray(dates<'2025-01-01')
    dev=np.asarray((dates>='2025-01-01')&(dates<'2025-03-01'))
    cal=np.asarray((dates>='2025-03-01')&(dates<'2025-07-01'))
    test=np.asarray(dates>='2025-07-01')
    assert dates[fit].max()<dates[dev].min()<dates[cal].min()<dates[test].min()
    compact=[c for c in COMPACT if c in X]+['synthetic_line','recent_minus_line','season_minus_line','standardized_edge']
    candidates=[]
    for c in (.01,.1,1):
        candidates.append((f'compact_logistic_{c}',compact,make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=c,max_iter=1000))))
    for depth in (2,3):
        candidates.append((f'compact_boost_depth_{depth}',compact,HistGradientBoostingClassifier(max_iter=150,max_depth=depth,learning_rate=.04,l2_regularization=20,min_samples_leaf=100,random_state=42)))
    results=[];trained={}
    for name,cols,model in candidates:
        model.fit(X.loc[fit,cols],y[fit]);p=model.predict_proba(X.loc[dev,cols])[:,1]
        result={'name':name,'development':metrics(y[dev],p)};results.append(result);trained[name]=(cols,model)
        print(name,result['development'],flush=True)
    best=min(results,key=lambda r:r['development']['log_loss'])['name']
    cols,model=trained[best]
    # The selected model remains fitted only before development; calibrator sees a later disjoint period.
    calibrator=IsotonicRegression(out_of_bounds='clip').fit(model.predict_proba(X.loc[cal,cols])[:,1],y[cal])
    p=calibrator.predict(model.predict_proba(X.loc[test,cols])[:,1])
    previous=pd.read_csv(ROOT/'logs/points_pilot/holdout_predictions.csv',dtype={'game_id':str})
    assert np.array_equal(previous.actual_over.to_numpy(),y[test])
    assert np.array_equal(previous.game_id.to_numpy(),data['games'][rows[test]])
    report={'protocol':{'fit':'before 2025-01-01','selection':'2025-01-01 through 2025-02-28; minimum raw log loss','calibration':'2025-03-01 through 2025-06-30','followup':'2025-26; previously exposed, not an independent blind test','lines':'synthetic half-point lines, not bookmaker odds','refit_after_selection':False},'candidates':results,'selected':best,'counts':{k:int(v.sum()) for k,v in [('fit',fit),('development',dev),('calibration',cal),('followup',test)]},'followup':metrics(y[test],p),'previous_candidate':metrics(y[test],previous.candidate),'comparison':bootstrap(y[test],p,previous.candidate.to_numpy(),data['games'][rows[test]])}
    # A compact ridge regression is selected on the same development period.
    rd=pd.to_datetime(data['dates']);rf=np.asarray(rd<'2025-01-01');rv=np.asarray((rd>='2025-01-01')&(rd<'2025-03-01'));rt=np.asarray(rd>='2025-07-01')
    regcols=[c for c in COMPACT if c in data['X']];regs=[]
    for alpha in (10,100,1000):
        reg=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha))
        reg.fit(data['X'].loc[rf,regcols],data['y'][rf]);score=float(np.sqrt(mean_squared_error(data['y'][rv],reg.predict(data['X'].loc[rv,regcols]))));regs.append((score,alpha,reg))
    score,alpha,reg=min(regs,key=lambda r:r[0]);report['regression']={'selected_alpha':alpha,'development_rmse':score,'followup_rmse':float(np.sqrt(mean_squared_error(data['y'][rt],reg.predict(data['X'].loc[rt,regcols])))),'selection_scores':[{'alpha':a,'rmse':s} for s,a,_ in regs]}
    joblib.dump({'classifier':model,'columns':cols,'calibrator':calibrator,'regressor':reg,'regression_columns':regcols},OUT/'candidate.joblib')
    assert before==model_hashes();report['serving_models_unchanged']=True
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':run()
