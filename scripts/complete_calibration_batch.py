"""Qualify the authorized fixed batch, then gate offline calibration research."""
import json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from src.prospective_points import digest,save_new
from src.points_calibration_research import validate
from src.points_v7 import market_probability
from scripts.qualify_real_points_v2 import run as qualify
from scripts.build_real_points_training import run as build
from scripts.research_points_calibration import run as calibrate

ROOT=Path('logs/real_points_paid_calibration_01')
DESIGN=Path('logs/early_season_calibration_plan_v1')


def run():
    plan=json.loads((ROOT/'plan.json').read_text())
    if digest(ROOT/'plan.json')!=json.loads((ROOT/'plan_lock.json').read_text())['sha256']:raise ValueError('Plan drift')
    state=json.loads((ROOT/'state.json').read_text())
    if state['reserved']>120 or len(plan['candidates'])!=12:raise ValueError('Budget/plan mismatch')
    quotes=[];paths={}
    for e in plan['candidates']:
        a=state['attempts'][e['id']]
        if a['status']!='collected':raise ValueError('Incomplete collection')
        path=Path(a['raw_path'])
        if digest(path)!=a['raw_sha256']:raise ValueError('Raw evidence drift')
        quotes.extend(json.loads(path.read_text())['payload']['data']['props']);paths[str(path)]=digest(path)
    combined=ROOT/'combined.json'
    if not combined.exists():save_new(combined,{'props':quotes,'raw_sources':paths})
    if not (ROOT/'qualified').exists():qualify(combined,ROOT/'qualified',ROOT)
    if not (ROOT/'training').exists():build(combined,ROOT/'training',ROOT/'qualified/qualified.json')
    old=joblib.load('logs/points_v7_research/inputs.joblib');new=joblib.load(ROOT/'training/real_points.joblib')
    seen=set(zip(old['data']['games'],old['data']['players']));idx=[i for i,k in enumerate(zip(new['games'],new['players'])) if k not in seen]
    d={'X':pd.concat([old['data']['X'],new['X'].iloc[idx]],ignore_index=True),'feature_policy':new['feature_policy'],'provenance_validated':True}
    if old['data']['feature_policy']!=new['feature_policy']:raise ValueError('Feature policy mismatch')
    for k in ['games','players','dates','lines','points','y']:d[k]=np.concatenate([old['data'][k],new[k][idx]])
    nq=json.loads((ROOT/'training/report.json').read_text())['quotes'];q=old['quotes']+[nq[i] for i in idx]
    bundle={'data':d,'quotes':q,'market':market_probability([v['over_odds'] for v in q],[v['under_odds'] for v in q])}
    fitting,selection,support=validate(bundle,require_coverage=False)
    history=new['X'].games_played.to_numpy();coverage=[]
    for e in plan['candidates']:
        m=new['games']==e['nba_game_id'];cohort=(history>=1)&(history<10) if e['target_cohort']=='short_history' else history>=10
        coverage.append({'game_id':e['nba_game_id'],'date':e['date'],'target_cohort':e['target_cohort'],'usable_player_games':int(m.sum()),'intended_cohort_player_games':int((m&cohort).sum())})
    ready=all(v['supported'] for v in support.values()) and all(v['intended_cohort_player_games']>0 for v in coverage)
    report={'ready':ready,'support':support,'planned_game_coverage':coverage,'new_player_games':len(idx),'total_player_games':len(d['games']),'total_games':len(set(d['games'])),'reserved_credits':state['reserved'],'historical_independent':False,'raw_sources':paths}
    save_new(ROOT/'coverage.json',report)
    joblib.dump(bundle,ROOT/'calibration_inputs.joblib')
    if ready:calibrate(ROOT/'calibration_inputs.joblib',DESIGN/'execution_plan.json',Path('logs/points_calibration_research_01'))
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()
