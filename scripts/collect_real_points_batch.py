"""Deterministic, outcome-independent historical free-tier batch. Resumable."""
import json
from pathlib import Path
from datetime import datetime, timezone
import requests
import time
from scripts.fetch_free_historical_props import fetch, MONTHLY_LIMIT
from src.prospective_points import save_new

OUT=Path('logs/real_points_batch_01')
DATES={'fit':['20241101','20241115','20241201','20241215'],
       'calibration':['20250305','20250315','20250401','20250415'],
       'selection':['20250505','20250515','20250525'],
       'evaluation':['20251115','20251201','20251215']}


def run(out=None, dates=None):
    global OUT, DATES
    OUT = out or OUT
    DATES = dates or DATES
    OUT.mkdir(exist_ok=True)
    plan=OUT/'plan.json'
    if not plan.exists():
        candidates=[]
        for split,days in DATES.items():
            for day in days:
                p=OUT/f'schedule_{day}.json'
                if not p.exists():
                    r=requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',params={'dates':day,'limit':100},timeout=25);r.raise_for_status()
                    save_new(p,{'observed_at':datetime.now(timezone.utc).isoformat(),'payload':r.json()})
                events=json.loads(p.read_text())['payload']['events']
                for e in sorted(events,key=lambda e:e['id']):
                    candidates.append({'split':split,'date':day,'id':e['id'],'name':e['name']})
        save_new(plan,{'selection':'First available provider game ID per prespecified date; no outcome/model-performance selection. One new game per listed date; shared500-credit monthly free allowance. Only one game per date.', 'candidates':candidates})
    statepath=OUT/'state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {'attempts':[],'completed_dates':[]}
    for e in json.loads(plan.read_text())['candidates']:
        if e['date'] in state['completed_dates'] or e['id'] in {a['id'] for a in state['attempts']}:continue
        budget=json.loads(Path('logs/real_points_sources/dunkapi/budget_'+datetime.now(timezone.utc).strftime('%Y-%m')+'.json').read_text())
        if budget['reserved']>=MONTHLY_LIMIT:break
        try:
            if budget['reserved'] >= 400:
                time.sleep(13)  # Provider lowers rate limit below100 remaining credits.
            fetch(game_id=e['id'])
            state['completed_dates'].append(e['date']); status='collected'
        except (ValueError,RuntimeError,requests.RequestException) as exc:
            # Never print credentials or requests; errors from transport only by type.
            status=str(exc) if isinstance(exc,(ValueError,RuntimeError)) else (type(exc).__name__+' status='+str(exc.response.status_code) if getattr(exc,'response',None) is not None else type(exc).__name__)
        state['attempts'].append({**e,'status':status})
        tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(statepath)
        print(e['split'],e['date'],e['id'],status,flush=True)
        if 'ceiling' in status or status.startswith(('HTTPError','ConnectionError')):break

if __name__=='__main__':run()
