"""Offline dated candidate shortlist. No provider transport or budget mutation."""
import json
from pathlib import Path
from datetime import datetime,timezone
import joblib,numpy as np,pandas as pd
from src.prospective_points import digest,save_new
from src.points_calibration_research import validate

ROOT=Path('logs/early_season_calibration_plan_v1')

def run():
    ROOT.mkdir(exist_ok=False)
    bundle=joblib.load('logs/points_v7_research/inputs.joblib');d=bundle['data']
    fitting,selection,support=validate(bundle,require_coverage=False)
    existing=set(map(str,d['games'])); attempted=set()
    for p in Path('logs').glob('real_points_*/plan.json'):
        obj=json.loads(p.read_text())
        if isinstance(obj,dict):attempted.update(str(c['nba_game_id']) for c in obj.get('candidates',[]) if c.get('nba_game_id'))
    candidates=[]
    for season,split,cohort in [('2024-25','calibration_fit','short_history'),('2025-26','calibration_selection','eligible_history')]:
        a=joblib.load(f'logs/points_v5_data/{season}.joblib');h=a['X'].games_played.to_numpy()
        m=(h>=1)&(h<10) if cohort=='short_history' else h>=10
        frame=pd.DataFrame({'game':a['games'][m],'date':a['dates'][m]})
        games=frame.groupby(['date','game']).size().reset_index(name='potential_player_games').sort_values(['date','game'])
        # Prior-history support only; no outcome, points, or model score selection.
        games=games[~games.game.astype(str).isin(existing|attempted)]
        for row in games.head(6).itertuples():
            candidates.append({'nba_game_id':str(row.game),'date':pd.Timestamp(row.date).strftime('%Y-%m-%d'),'split':split,'target_cohort':cohort,'potential_player_games':int(row.potential_player_games),'id':None})
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'spending_paused':True,'status':'proposal_identity_mapping_pending','experiment_protocol':'logs/early_season_calibration_design_v1/protocol.json','experiment_protocol_sha256':digest('logs/early_season_calibration_design_v1/protocol.json'),'split_boundary':'2025-07-01','credit_ceiling':len(candidates)*10,'cost_basis':'Archived 10-credit per-game estimate; recheck before any future transport. No live balance checked. No guarantee of usable quotes.','selection':'First six unattempted, unquoted games ordered by date/NBA ID with prior-history cohort support in each deficient split. No outcomes or performance used. No automatic replacements.','candidates':candidates,'reuse':support,'reuse_total_player_games':len(d['games']),'reuse_total_games':len(existing),'shortfall_games':{'calibration_fit_short_history':6,'calibration_selection_eligible_history':6}}
    save_new(ROOT/'plan.json',plan)
    models={k:f'logs/points_v6_real_01/{k}.joblib' for k in ['repaired','compact_retrained']}
    paths=list(models.values())+['src/points_calibration_research.py','scripts/research_points_calibration.py','src/points_v5.py','src/verified_training.py','src/points_v7.py','src/points_research_gate.py','logs/early_season_calibration_design_v1/protocol.json']
    execution={'purpose':'calibration_research','split_boundary':'2025-07-01','models':models,'hashes':{p:digest(p) for p in paths},'selection_data':'All historical data exposed; development only; no prospective outcomes','spending_paused':True}
    save_new(ROOT/'execution_plan.json',execution);save_new(ROOT/'execution_lock.json',{'sha256':digest(ROOT/'execution_plan.json')})
    print(json.dumps(plan,indent=2))

if __name__=='__main__':run()
