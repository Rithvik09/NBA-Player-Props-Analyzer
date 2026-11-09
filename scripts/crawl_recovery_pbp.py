"""Cache real dated play-by-play and starters; failures remain explicit."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import gzip,json,time
import pandas as pd
from nba_api.stats.endpoints import playbyplayv3,boxscoretraditionalv3

OUT=Path('logs/feature_recovery');DATA=OUT/'pbp';DATA.mkdir(exist_ok=True)


def fetch(gid):
    path=DATA/(gid+'.json.gz')
    if path.exists():return {'game_id':gid,'status':'cached'}
    try:
        plays=playbyplayv3.PlayByPlayV3(game_id=gid,timeout=15).get_data_frames()[0]
        players=boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid,timeout=15).get_data_frames()[0]
        if len(plays)<100 or players.empty:raise ValueError('Incomplete game source')
        obj={'game_id':gid,'plays':json.loads(plays.to_json(orient='records')),'players':json.loads(players.to_json(orient='records'))}
        with gzip.open(path,'wt') as f:json.dump(obj,f)
        return {'game_id':gid,'status':'complete','plays':len(plays)}
    except Exception as e:return {'game_id':gid,'status':'failed','error':type(e).__name__}


def run():
    games=set()
    for path in Path('logs/points_pilot_cache').glob('players_*.json'):
        frame=pd.read_json(path,dtype={'GAME_ID':str});games.update(frame.GAME_ID)
    results=[];start=time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch,gid) for gid in sorted(games)]):
            results.append(future.result())
            if len(results)%100==0 or len(results)==len(games):
                counts={s:sum(r['status']==s for r in results) for s in ('complete','cached','failed')}
                report={'requested':len(games),'processed':len(results),'counts':counts,'seconds':time.time()-start,'games':results}
                (OUT/'pbp_crawl.json').write_text(json.dumps(report,indent=2));print(len(results),len(games),counts,flush=True)

if __name__=='__main__':run()
