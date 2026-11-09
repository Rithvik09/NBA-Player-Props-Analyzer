"""User-authorized one-time24-game batch, separate240-credit reservation ledger."""
import json,os,time,fcntl
from pathlib import Path
from datetime import datetime,timezone
import requests
import joblib
import pandas as pd
from src.historical_inputs import load_season_logs
from src.prospective_points import save_new,digest
from src.points_collection_preflight import require_preflight,validate_response

OUT=Path('logs/real_points_paid_01')
RAW=Path('logs/real_points_sources/dunkapi')
LIMIT=240

def plan():
    OUT.mkdir(exist_ok=True)
    if (OUT/'plan.json').exists():return json.loads((OUT/'plan.json').read_text())
    existing=joblib.load('logs/real_points_merged_04/real_points.joblib')
    excluded=set(map(str,existing['games']))
    attempted=set()
    for p in RAW.glob('budget_*.json'):
        attempted.update(json.loads(x).get('game_id') for x in json.loads(p.read_text()).get('requests',[]))
    logs=load_season_logs('2024-25',include_postseason=True)
    aliases={'WSH':'WAS','NY':'NYK','GS':'GSW','SA':'SAS','NO':'NOP','UTAH':'UTA'}
    candidates=[]
    for split,start,end in [('fit','2024-11-01','2024-12-31'),('calibration','2025-03-01','2025-04-30'),('selection','2025-05-01','2025-06-30')]:
        chosen=0
        for day in pd.date_range(start,end):
            if chosen==8:break
            local=logs[pd.to_datetime(logs.GAME_DATE)==day]
            if local.empty:continue
            p=OUT/f'schedule_{day:%Y%m%d}.json'
            if not p.exists():
                r=requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',params={'dates':day.strftime('%Y%m%d'),'limit':100},timeout=25);r.raise_for_status()
                save_new(p,{'observed_at':datetime.now(timezone.utc).isoformat(),'payload':r.json()})
            for e in sorted(json.loads(p.read_text())['payload']['events'],key=lambda e:e['id']):
                if e['id'] in attempted:continue
                teams={aliases.get(c['team']['abbreviation'],c['team']['abbreviation']) for c in e['competitions'][0]['competitors']}
                matched=local[local.MATCHUP.map(lambda m:set([m.split()[0],m.split()[-1]])==teams)]
                gids=set(matched.GAME_ID.astype(str))
                if len(gids)!=1 or next(iter(gids)) in excluded:continue
                gid=next(iter(gids));excluded.add(gid)
                candidates.append({'split':split,'date':str(day.date()),'id':e['id'],'nba_game_id':gid,'name':e['name']})
                chosen+=1;break
        if chosen!=8:raise ValueError('Insufficient prespecified candidates '+split)
    result={'created_at':datetime.now(timezone.utc).isoformat(),'authorization':'User confirmed $5 top-up and requested24 remaining games; no purchase performed by collector.',
       'credit_ceiling':LIMIT,'selection':'Eight games per deficient split; chronological dates, first unattempted ESPN event ID per date that maps uniquely to an NBA game absent from existing dataset. No outcome or model-performance selection. No automatic expansion.',
       'existing_dataset_sha256':digest('logs/real_points_merged_04/real_points.joblib'),'candidates':candidates}
    save_new(OUT/'plan.json',result);return result

def reserve(state,identity,cost,limit=LIMIT):
    if identity in state['attempts']:raise ValueError('Already attempted; inspect evidence, never charge twice')
    if not 0<limit<=LIMIT or cost!=10 or state['reserved']+cost>limit:raise ValueError('Paid batch estimate or ceiling violation')
    state['reserved']+=cost;state['attempts'][identity]={'reserved':cost,'status':'reserved_before_transport'}

def run():
    proposal=plan()
    state_path=OUT/'state.json'
    prior=json.loads(state_path.read_text()) if state_path.exists() else {'attempts':{}}
    pending=[e for e in proposal['candidates'] if e['id'] not in prior['attempts']]
    if not pending:
        print('Batch closed; no unattempted games.');return
    require_preflight(proposal,pending)
    values=dict(line.split('=',1) for line in Path('.env').read_text().splitlines() if '=' in line and not line.startswith('#'))
    key=os.environ.get('DUNKAPI_API_KEY') or values.get('DUNKAPI_API_KEY','').strip().strip('"').strip("'")
    if not key:raise ValueError('Missing local key')
    headers={'x-api-key':key}
    # Shared with the free collector, preventing cross-batch charge races.
    RAW.mkdir(parents=True,exist_ok=True)
    with (RAW/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        require_preflight(proposal,pending)
        p=OUT/'state.json';state=json.loads(p.read_text()) if p.exists() else {'reserved':0,'attempts':{}}
        def persist():
            tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
        for e in proposal['candidates']:
            identity=e['id']
            if identity in state['attempts']:continue
            args={'game_id':identity,'market':'player_points','snapshot_type':'closing'}
            r=requests.post('https://mcp.dunkapi.com/v1/get_player_props',json={**args,'estimate_only':True},headers=headers,timeout=(5,20));r.raise_for_status()
            estimate=r.json().get('data',r.json())
            if not estimate.get('estimate_only'):raise ValueError('Invalid estimate response')
            cost=int(estimate['credits']);reserve(state,identity,cost,proposal['credit_ceiling']);persist()
            try:
                r=requests.post('https://mcp.dunkapi.com/v1/get_player_props',json=args,headers=headers,timeout=(5,30));r.raise_for_status();payload=r.json()
                validate_response(payload,e)
                target=RAW/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'_'+identity+'.json')
                save_new(target,{'observed_at':datetime.now(timezone.utc).isoformat(),'provider':'DunkAPI','estimated_credits':cost,'request':args,'payload':payload,'paid_batch':str(OUT)})
                state['attempts'][identity].update(status='collected',raw_path=str(target),raw_sha256=digest(target),quotes=len(payload['data']['props']))
                persist();print(e['split'],e['date'],identity,'collected',len(payload['data']['props']),flush=True)
            except Exception as exc:
                state['attempts'][identity]['status']='uncertain_failure_'+type(exc).__name__;persist()
                raise RuntimeError('Paid collection stopped; reservation retained; '+type(exc).__name__) from None
            time.sleep(13)

if __name__=='__main__':run()
