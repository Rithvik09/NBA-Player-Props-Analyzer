"""Bounded free-tier sampling with durable reservations and validated identities."""
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import requests
from nba_api.stats.static import teams
from .current_rosters import name_key
from .source_evidence import instant, finite, implied

BASE='https://api.the-odds-api.com/v4/sports/basketball_nba'
MARKETS='player_points,totals'


def capture(api_key, directory=Path('logs/source_blocked/free_market'), event_id=None):
    """One event, one region, two markets: reserve at most two credits per call.

    This pilot has a fixed 20-credit monthly ceiling and a 50-credit account
    reserve. Network failures consume the local reservation conservatively.
    It never enables the existing broad poller or upgrades a subscription.
    """
    directory.mkdir(parents=True,exist_ok=True)
    def get(url, params):
        try:
            response=requests.get(url,params={**params,'apiKey':api_key},timeout=20)
        except requests.RequestException as exc:
            raise RuntimeError('Odds request failed: '+type(exc).__name__) from None
        if not response.ok:
            raise RuntimeError('Odds HTTP status '+str(response.status_code))
        return response
    with (directory/'budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        now=datetime.now(timezone.utc);month=now.strftime('%Y-%m')
        state_path=directory/'budget.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('month')!=month:state={'month':month,'reserved_credits':0,'calls':[]}
        if state.get('unexpected_cost') or state['reserved_credits']+2>20:
            raise RuntimeError('Free pilot credit ceiling reached')
        events=get(BASE+'/events',{})
        remaining=int(events.headers['x-requests-remaining'])
        if remaining<52:raise RuntimeError('Account credit reserve would be crossed')
        choices=[e for e in events.json() if instant(e['commence_time'])>now]
        if event_id:choices=[e for e in choices if e['id']==event_id]
        if not choices:raise RuntimeError('No eligible pregame event')
        event=min(choices,key=lambda e:instant(e['commence_time']))
        if not re.fullmatch('[a-fA-F0-9]{32}',event['id']):raise ValueError('Invalid provider event ID')
        state['reserved_credits']+=2
        state_path.write_text(json.dumps(state,indent=2))
        response=get(BASE+'/events/'+event['id']+'/odds',
                     {'regions':'us','markets':MARKETS,'oddsFormat':'american'})
        observed=datetime.now(timezone.utc).isoformat();payload=response.json()
        if payload.get('id')!=event['id']:raise ValueError('Provider returned wrong event')
        cost=int(response.headers.get('x-requests-last','2'))
        state['unexpected_cost']=cost>2
        path=directory/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
        result={'observed_at':observed,'provider':'The Odds API','markets':MARKETS,
                'credits_used':cost,'credits_remaining':int(response.headers['x-requests-remaining']),
                'payload':payload}
        path.write_text(json.dumps(result,indent=2))
        state['calls'].append({'observed_at':observed,'event_id':event['id'],'cost':cost,'path':str(path)})
        state_path.write_text(json.dumps(state,indent=2))
        return result


def normalize(snapshot, nba_games, roster):
    """Join exact home/away identities and nearby same-day scheduled tipoffs.

    Use official NBA tipoff (often ten minutes earlier than bookmaker time).
    Player identities additionally require a fresh, season-matched roster.
    Provider last_update is when it last SAW the market, not proof of a price
    change or the original market opening. Keep that semantic in every row.
    """
    data=snapshot['payload'];observed=instant(snapshot['observed_at'])
    ids={name_key(t['full_name']):t['id'] for t in teams.get_teams()}
    ids['laclippers']=1610612746
    home,away=(ids.get(name_key(data[key])) for key in ['home_team','away_team'])
    if home is None or away is None:raise ValueError('Unrecognized NBA team')
    matches=[g for g in nba_games if int(g['homeTeam']['teamId'])==home and int(g['awayTeam']['teamId'])==away
             and abs((instant(g['gameTimeUTC'])-instant(data['commence_time'])).total_seconds())<=1200]
    if len(matches)!=1:raise ValueError('No unique official NBA event match')
    game=matches[0];start=instant(game['gameTimeUTC'])
    if observed>=start:raise ValueError('Observed after official tipoff')
    from zoneinfo import ZoneInfo
    local=start.astimezone(ZoneInfo('America/New_York'))
    year=local.year if local.month>=10 else local.year-1
    season=f'{year}-{str(year+1)[2:]}'
    fresh=roster.get('season')==season and 0<=observed.timestamp()-roster.get('fetched_at',0)<=21600
    players=defaultdict(list)
    if fresh:
        for player in roster.get('players',[]):
            if int(player['team_id']) in (home,away):players[name_key(player['full_name'])].append(player)
    props=[];totals=[];rejected=defaultdict(int)
    for book in data.get('bookmakers',[]):
        for market in book.get('markets',[]):
            if market.get('key') not in ['player_points','totals']:continue
            try:
                published=instant(market['last_update'])
                if published>observed:raise ValueError()
            except (KeyError,TypeError,ValueError):
                rejected['unknown_or_future_market_timestamp']+=1;continue
            groups=defaultdict(dict)
            for outcome in market.get('outcomes',[]):
                try:
                    side=outcome['name'].lower();line=finite(outcome['point']);price=finite(outcome['price']);implied(price)
                    if side not in ('over','under') or line<0:raise ValueError()
                    groups[(outcome.get('description',''),line)][side]=price
                except (KeyError,TypeError,ValueError):rejected['invalid_outcome']+=1
            for (name,line),prices in groups.items():
                if set(prices)!= {'over','under'}:rejected['unpaired_line']+=1;continue
                if sum(n==name for n,_ in groups)>1:rejected['ambiguous_main_line']+=1;continue
                row={'game_id':str(game['gameId']),'provider_event_id':data['id'],
                     'bookmaker':book['key'],'line':line,'over_price':prices['over'],'under_price':prices['under'],
                     'game_start':start.isoformat(),'observed_at':observed.isoformat(),'published_at':published.isoformat(),
                     'timestamp_semantics':'provider_last_seen_market','source':'The Odds API',
                     'is_opening':False,'is_alternate':False,'scope':'game_total' if not name else 'prop'}
                if market['key']=='totals':
                    totals.append({**row,'market':'totals'})
                else:
                    matches=players.get(name_key(name),[])
                    if len(matches)!=1:rejected['unresolved_or_stale_player_roster']+=1;continue
                    props.append({**row,'player_id':int(matches[0]['nba_id']),'player_name':name,'prop_type':'points'})
    return {'props':props,'totals':totals,'rejected':dict(rejected),'nba_game_id':str(game['gameId']),
            'roster_fresh':fresh,'observed_at':observed.isoformat()}
