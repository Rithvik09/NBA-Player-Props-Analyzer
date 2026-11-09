"""Archive independent timing evidence for the fixed paid batch, then qualify."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import requests
from src.nba_officials import nba_json,game_summary
from src.prospective_points import save_new,digest
from scripts.collect_paid_points_batch import OUT
from scripts.qualify_real_points_v2 import run as qualify
from scripts.build_real_points_training import run as build
ROOT=Path('logs/real_points_sources')

def run(evidence_only=False):
    plan=json.loads((OUT/'plan.json').read_text())
    for e in plan['candidates']:
        ep=ROOT/f"espn_summary_{e['id']}.json"
        if not ep.exists():
            r=requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary',params={'event':e['id']},timeout=25);r.raise_for_status()
            save_new(ep,{'observed_at':datetime.now(timezone.utc).isoformat(),'payload':r.json()})
        sp=OUT/f"{e['nba_game_id']}_summary.json"
        if not sp.exists():save_new(sp,game_summary(e['nba_game_id']))
        pp=ROOT/f"official_timing/stats_{e['nba_game_id']}.json"
        if not pp.exists():save_new(pp,nba_json('playbyplayv3',GameID=e['nba_game_id'],StartPeriod=1,EndPeriod=1))
        print(e['nba_game_id'],'independent evidence archived',flush=True)
    if evidence_only:return
    state=json.loads((OUT/'state.json').read_text());quotes=[];paths=[]
    for e in plan['candidates']:
        attempt=state['attempts'][e['id']]
        if attempt['status']!='collected':raise ValueError('Incomplete paid batch')
        p=Path(attempt['raw_path'])
        if digest(p)!=attempt['raw_sha256']:raise ValueError('Raw response drift')
        quotes.extend(json.loads(p.read_text())['payload']['data']['props']);paths.append(str(p))
    combined=OUT/'combined.json'
    if not combined.exists():save_new(combined,{'props':quotes,'raw_sources':paths})
    if not (OUT/'qualified').exists():qualify(combined,OUT/'qualified',OUT)
    if not (OUT/'training').exists():build(combined,OUT/'training',OUT/'qualified/qualified.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence-only',action='store_true');run(p.parse_args().evidence_only)
