"""Score/date/team-validated ESPN fallback for historical game events.

NBA player IDs and boxscore outcomes are validated against already-cached NBA
logs. ESPN IDs are never treated as NBA identities. No odds are fabricated.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from pathlib import Path
from threading import local
from zoneinfo import ZoneInfo
import gzip,json,re,time
import pandas as pd
import requests
from nba_api.stats.static import teams
from src.historical_injuries import normalize_name

OUT=Path('logs/feature_recovery');DATA=OUT/'pbp';BOARDS=OUT/'espn_boards';BOARDS.mkdir(exist_ok=True)
BASE='https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
TLS=local()
TEAM={t['abbreviation']:t['id'] for t in teams.get_teams()}
for alias,canonical in [('GS','GSW'),('NY','NYK'),('NO','NOP'),('SA','SAS'),('UTAH','UTA'),('WSH','WAS')]:TEAM[alias]=TEAM[canonical]

def get(url):
    if not hasattr(TLS,'session'):TLS.session=requests.Session()
    response=TLS.session.get(url,timeout=(5,10));response.raise_for_status();return response.json()


def fetch_day(item):
    day,games=item;results=[]
    path=BOARDS/(day+'.json')
    try:
        board=json.loads(path.read_text()) if path.exists() else get(BASE+'/scoreboard?dates='+day.replace('-','')+'&limit=100')
        if not path.exists():path.write_text(json.dumps(board))
        for gid,frame in games:
            try:
                output=DATA/(gid+'.json.gz')
                if output.exists():results.append({'game_id':gid,'status':'cached'});continue
                expected={int(tid):float(part.PTS.sum()) for tid,part in frame.groupby('TEAM_ID')}
                matches=[]
                for event in board.get('events',[]):
                    comp=event['competitions'][0]
                    actual={TEAM.get(c['team']['abbreviation']):float(c['score']) for c in comp['competitors']}
                    event_day=datetime.fromisoformat(event['date'].replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date().isoformat()
                    if actual==expected and event_day==day and comp['status']['type']['completed']:matches.append(event)
                if len(matches)!=1:results.append({'game_id':gid,'status':'failed','error':'Ambiguous/missing exact teams/date/scores'});continue
                event=matches[0];summary=get(BASE+'/summary?event='+event['id']);comp=event['competitions'][0]
                if str(summary['header']['id'])!=str(event['id']):raise ValueError('Wrong ESPN event')
                ids={str(c['id']):TEAM[c['team']['abbreviation']] for c in comp['competitors']}
                home=next(ids[str(c['id'])] for c in comp['competitors'] if c['homeAway']=='home')
                players=[];identity={}
                for group in summary['boxscore']['players']:
                    tid=TEAM[group['team']['abbreviation']];own=frame[frame.TEAM_ID==tid]
                    catalog={normalize_name(r['PLAYER_NAME']):r for r in own.to_dict('records')}
                    for athlete in group['statistics'][0]['athletes']:
                        name=athlete['athlete'].get('displayName',athlete['athlete'].get('fullName',''))
                        labels=group['statistics'][0]['labels'];check=dict(zip(labels,athlete.get('stats',[])))
                        if not name and (athlete.get('didNotPlay') or check.get('MIN') in (None,'0','--')):continue
                        nba=catalog.get(normalize_name(name))
                        if nba is None:
                            if athlete.get('didNotPlay') or (check.get('MIN') in ('0','--') and check.get('PTS') in ('0','--',None)):continue
                            raise ValueError('Cannot validate played athlete identity: '+name)
                        pid=int(nba['PLAYER_ID']);identity[str(athlete['athlete']['id'])]=pid
                        labels=group['statistics'][0]['labels'];stats=dict(zip(labels,athlete['stats']))
                        if float(stats['PTS'])!=float(nba['PTS']):raise ValueError('Player points mismatch')
                        if str(stats['FG'])!=f"{int(nba['FGM'])}-{int(nba['FGA'])}":raise ValueError('Player shooting mismatch')
                        players.append({'personId':pid,'teamId':int(tid),'position':'STARTER' if athlete.get('starter') else '', 'minutes':float(nba['MIN']),'points':float(nba['PTS']),'fieldGoalsAttempted':int(nba['FGA']),'fieldGoalsMade':int(nba['FGM']),'firstName':name.split()[0],'familyName':' '.join(name.split()[1:]),'nameI':athlete['athlete'].get('shortName',name)})
                player_team={p['personId']:p['teamId'] for p in players};plays=[]
                for i,play in enumerate(summary.get('plays',[])):
                    tid=ids.get(str(play.get('team',{}).get('id')),0)
                    participants=[identity.get(str(p['athlete']['id'])) for p in play.get('participants',[])]
                    participants=[p for p in participants if p is not None]
                    pid=next((p for p in participants if player_team[p]==tid),0)
                    kind=play['type']['text'];text=play.get('text','');clock=play['clock']['displayValue'].split(':')
                    if len(clock)==1:clock=['0',clock[0]]
                    free='free throw' in kind.lower();sub='substitution' in kind.lower() or 'enters the game for' in text
                    if sub and len(participants)>=2:pid=participants[0]
                    distance=re.search(r'(\d+)[- ]foot',text)
                    action={'period':int(play['period']['number']),'clock':f'PT{clock[0]}M{clock[1]}S','actionNumber':i,'teamId':tid,'personId':pid,'location':'h' if tid==home else 'v' if tid else '', 'scoreHome':str(play['homeScore']),'scoreAway':str(play['awayScore']),'isFieldGoal':int(bool(play.get('shootingPlay')) and not free),'shotResult':'Made' if play.get('scoringPlay') else 'Missed','shotDistance':int(distance[1]) if distance else None,'shotValue':int(play.get('pointsAttempted',0)),'actionType':'Substitution' if sub else 'Free Throw' if free else kind,'subType':kind,'description':text}
                    if free and '1 of 1' in text:action['subType']+=' 1 of 1'
                    if sub and len(participants)>=2:action['outgoingPersonId']=participants[1]
                    plays.append(action)
                if len(plays)<100:raise ValueError('Incomplete ESPN events')
                data={'game_id':gid,'source':'ESPN validated against NBA logs','source_event_id':event['id'],'game_start_time':event['date'],'broadcasts':comp.get('broadcasts',[]),'players':players,'plays':plays}
                temp=output.with_suffix('.tmp')
                with gzip.open(temp,'wt') as f:json.dump(data,f)
                temp.replace(output);results.append({'game_id':gid,'status':'complete','plays':len(plays)});time.sleep(.08)
            except Exception as exc:
                results.append({'game_id':gid,'status':'failed','error':str(exc)[:160]})
    except Exception as exc:results.append({'day':day,'status':'failed','error':str(exc)[:160]})
    return results


def run():
    all_frames=[]
    for path in Path('logs/points_pilot_cache').glob('players_*.json'):
        frame=pd.read_json(path,dtype={'GAME_ID':str});frame['DAY']=pd.to_datetime(frame.GAME_DATE).dt.strftime('%Y-%m-%d');all_frames.append(frame)
    frame=pd.concat(all_frames);days=[(day,list(part.groupby('GAME_ID'))) for day,part in frame.groupby('DAY')]
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,future in enumerate(as_completed([pool.submit(fetch_day,item) for item in days]),1):
            results.extend(future.result())
            if i%20==0 or i==len(days):
                counts={s:sum(r['status']==s for r in results) for s in ('complete','cached','failed')}
                (OUT/'espn_pbp_crawl.json').write_text(json.dumps({'days_processed':i,'days_requested':len(days),'counts':counts,'games':results},indent=2));print(i,len(days),counts,flush=True)

if __name__=='__main__':run()
