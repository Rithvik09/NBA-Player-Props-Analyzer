"""Collect official per-game matchup possessions, identities and switches.

NBA identities/teams are checked against dated boxscores. No target-game row
may subsequently be used as a pregame feature for that game.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse, gzip, json, time, hashlib, math
import pandas as pd
from nba_api.stats.endpoints import boxscorematchupsv3

OUT=Path('logs/source_blocked'); DATA=OUT/'matchups'
COLUMNS=['personIdOff','personIdDef','teamId','partialPossessions','switchesOn',
         'playerPoints','matchupFieldGoalsMade','matchupFieldGoalsAttempted',
         'matchupMinutesSort']


def validate(frame, players, gid):
    required=set(COLUMNS+['gameId'])
    if not required.issubset(frame) or frame.empty:raise ValueError('Missing matchup schema')
    teams={int(r.PLAYER_ID):int(r.TEAM_ID) for r in players.itertuples()}
    rows=[]
    for row in frame.to_dict('records'):
        if str(row['gameId'])!=gid:raise ValueError('Wrong game')
        a,b=int(row['personIdOff']),int(row['personIdDef'])
        if float(row['partialPossessions'])==0:continue
        if a not in teams or b not in teams or teams[a]==teams[b] or teams[a]!=int(row['teamId']):
            raise ValueError('Unvalidated player/team identity')
        for col in COLUMNS[3:]:
            value=float(row[col])
            if not math.isfinite(value) or value<0:raise ValueError('Invalid matchup count')
        rows.append({k:row[k] for k in COLUMNS}|{'defTeamId':teams[b]})
    if not rows:raise ValueError('No measured possessions')
    return rows


def fetch(item):
    gid,players=item;path=DATA/(gid+'.json.gz')
    if path.exists():return {'game_id':gid,'status':'cached'}
    try:
        frame=boxscorematchupsv3.BoxScoreMatchupsV3(game_id=gid,timeout=5).get_data_frames()[0]
        rows=validate(frame,players,gid)
        obj={'game_id':gid,'game_date':str(pd.Timestamp(players.GAME_DATE.iloc[0]).date()),
             'source':'NBA BoxScoreMatchupsV3','rows':rows}
        temp=path.with_suffix('.tmp')
        with gzip.open(temp,'wt') as f:json.dump(obj,f)
        temp.replace(path);time.sleep(.1)
        return {'game_id':gid,'status':'complete','rows':len(rows)}
    except Exception as e:return {'game_id':gid,'status':'failed','error':type(e).__name__+': '+str(e)[:100]}


def run(limit=None,workers=3):
    DATA.mkdir(parents=True,exist_ok=True)
    paths=[p for p in Path('logs/points_pilot_cache').glob('*.json') if p.name.startswith(('players_','playoffs_','playin_'))]
    frames=[pd.read_json(p,dtype={'GAME_ID':str}) for p in paths]
    games=list(pd.concat(frames).groupby('GAME_ID'))
    # Spread endpoint gaps across seasons; do not let a missing date range
    # consume all attempts before any older-season evidence is collected.
    games.sort(key=lambda item:hashlib.sha256(item[0].encode()).hexdigest())
    if limit:games=games[:limit]
    results=[];start=time.time()
    # Bounded batches stop a failing endpoint rather than queuing thousands of timeouts.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0,len(games),60):
            batch=[f.result() for f in as_completed([pool.submit(fetch,x) for x in games[offset:offset+60]])]
            results.extend(batch)
            counts={s:sum(x['status']==s for x in results) for s in ('complete','cached','failed')}
            (OUT/'matchup_crawl.json').write_text(json.dumps({'processed':len(results),'requested':len(games),
                'counts':counts,'seconds':time.time()-start,'games':results},indent=2))
            print(len(results),len(games),counts,flush=True)
            if sum(x['status']=='failed' for x in batch)>len(batch)*.8:break

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();run(a.limit,a.workers)
