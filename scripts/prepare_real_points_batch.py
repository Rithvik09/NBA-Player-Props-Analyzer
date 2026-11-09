"""Archive independent free event/identity/timing evidence for collected quotes."""
import json
from pathlib import Path
from datetime import datetime,timezone
import requests
import pandas as pd
from src.historical_inputs import load_season_logs
from src.historical_injuries import normalize_name
from src.nba_officials import game_summary,nba_json
from src.prospective_points import save_new
from scripts.qualify_real_points_v2 import run as qualify
from scripts.build_real_points_training import run as build
ROOT=Path('logs/real_points_sources')
OUT=Path('logs/real_points_batch_01')


def run(out=None):
    global OUT
    OUT = out or OUT
    combined=OUT/'combined.json'
    if not combined.exists():
        quotes=[]; sources=[]
        for p in sorted((ROOT/'dunkapi').glob('*.json')):
            d=json.loads(p.read_text())
            if d.get('provider')!='DunkAPI':continue
            rows=d.get('payload',{}).get('data',{}).get('props',[])
            quotes.extend(rows);sources.append(str(p))
        save_new(combined,{'props':quotes,'raw_sources':sources})
    quotes=json.loads(combined.read_text())['props'];contexts={};failures=[]
    for provider in sorted({str(q['game_id']) for q in quotes}):
        try:
            ep=ROOT/f'espn_summary_{provider}.json'
            if not ep.exists():
                r=requests.get('https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary',params={'event':provider},timeout=25);r.raise_for_status()
                save_new(ep,{'observed_at':datetime.now(timezone.utc).isoformat(),'payload':r.json()})
            rows=[q for q in quotes if str(q['game_id'])==provider];day=pd.Timestamp(rows[0]['game_date'])
            year=day.year if day.month>=10 else day.year-1;season=f'{year}-{str(year+1)[2:]}'
            if season not in contexts:contexts[season]=load_season_logs(season,include_postseason=True)
            logs=contexts[season]; names={normalize_name(q['player_name']) for q in rows}
            candidates=logs[(pd.to_datetime(logs.GAME_DATE)==day)&logs.PLAYER_NAME.map(normalize_name).isin(names)]
            gids=set(candidates.GAME_ID.astype(str))
            if len(gids)!=1:raise ValueError('Ambiguous event mapping')
            gid=gids.pop();sp=OUT/f'{gid}_summary.json';pp=ROOT/f'official_timing/stats_{gid}.json'
            if not sp.exists():
                old=Path('logs/real_points_batch_01')/f'{gid}_summary.json'
                save_new(sp,json.loads(old.read_text()) if old.exists() else game_summary(gid))
            if not pp.exists():save_new(pp,nba_json('playbyplayv3',GameID=gid,StartPeriod=1,EndPeriod=1))
            print(provider,gid,'evidence archived',flush=True)
        except Exception as exc:
            failures.append({'provider':provider,'error':type(exc).__name__});print(provider,type(exc).__name__,flush=True)
    (OUT/'evidence_failures.json').write_text(json.dumps(failures,indent=2))
    if not (OUT/'qualified').exists():qualify(combined,OUT/'qualified',OUT)
    if not (OUT/'training').exists():build(combined,OUT/'training',OUT/'qualified/qualified.json')

if __name__=='__main__':run()
