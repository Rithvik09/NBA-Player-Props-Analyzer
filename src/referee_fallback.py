"""Score-validated ESPN officials for gaps in NBA summary coverage."""
from datetime import datetime
import re
from zoneinfo import ZoneInfo

import requests
from nba_api.stats.static import teams

BASE='https://site.api.espn.com/apis/site/v2/sports/basketball/nba'


def get_json(url):
    response=requests.get(url,timeout=(5,15))
    response.raise_for_status()
    return response.json()


def resolve_summary(game_id, rows, identities):
    """Map providers through date + exact team IDs + scores, never by ID number."""
    day=str(rows[0]['GAME_DATE'])[:10]
    team_map={t['abbreviation']:int(t['id']) for t in teams.get_teams()}
    for alias,canonical in [('GS','GSW'),('NY','NYK'),('NO','NOP'),('SA','SAS'),('UTAH','UTA'),('WSH','WAS')]:
        team_map[alias]=team_map[canonical]
    expected={int(r['TEAM_ID']):float(r['PTS']) for r in rows}
    board=get_json(BASE+'/scoreboard?dates='+day.replace('-',''))
    matches=[]
    for event in board.get('events',[]):
        event_day=datetime.fromisoformat(event['date'].replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date().isoformat()
        comp=event['competitions'][0]
        if event_day!=day or not comp['status']['type']['completed']:
            continue
        try: actual={team_map[c['team']['abbreviation']]:float(c['score']) for c in comp['competitors']}
        except (KeyError,ValueError):continue
        if actual==expected:matches.append(event)
    if len(matches)!=1:
        raise ValueError('No unique ESPN game matching NBA teams/date/scores')
    event=matches[0];summary=get_json(BASE+'/summary?event='+str(event['id']))
    if str(summary['header']['id'])!=str(event['id']):
        raise ValueError('ESPN returned another event')
    refs=[]
    for row in summary.get('gameInfo',{}).get('officials',[]):
        name=row['fullName'];key=re.sub('[^a-z]','',name.lower())
        canonical=identities.get(key)
        if canonical is None:
            raise ValueError('Cannot verify NBA identity of ESPN referee: '+name)
        refs.append({'name':canonical['name'],'personId':canonical['id']})
    if len(refs) not in (3,4):
        raise ValueError('Incomplete ESPN crew')
    comp=event['competitions'][0]
    home=next(c for c in comp['competitors'] if c['homeAway']=='home')
    away=next(c for c in comp['competitors'] if c['homeAway']=='away')
    return {'gameId':str(game_id),'gameStatus':3,'gameEt':day,
            'homeTeamId':team_map[home['team']['abbreviation']],
            'awayTeamId':team_map[away['team']['abbreviation']],
            'homeTeam':{'teamId':team_map[home['team']['abbreviation']],'score':float(home['score'])},
            'awayTeam':{'teamId':team_map[away['team']['abbreviation']],'score':float(away['score'])},
            'officials':refs,'source':'ESPN','sourceEventId':str(event['id'])}
