"""Run one bounded free-plan snapshot; no recurring job or paid requests."""
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from src.free_market_capture import capture,normalize
from src.current_rosters import load_snapshot,refresh_rosters
from src.nba_officials import nba_json
from scripts.audit_source_history import replay


if __name__=='__main__':
    # Never emit credentials or request URLs containing them.
    if not os.environ.get('ODDS_API_KEY'):
        values=dict(l.split('=',1) for l in Path('.env').read_text().splitlines() if '=' in l and not l.startswith('#'))
        os.environ['ODDS_API_KEY']=values['ODDS_API_KEY'].strip().strip('"').strip("'")
    # Establish roster evidence before observing the quote, not afterwards.
    try:roster=load_snapshot() or refresh_rosters()
    except Exception:roster={}
    snapshot=capture(os.environ['ODDS_API_KEY'])
    day=datetime.fromisoformat(snapshot['payload']['commence_time'].replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date()
    board=nba_json('scoreboardv3',GameDate=str(day),LeagueID='00')['scoreboard']['games']
    result=normalize(snapshot,board,roster)
    out=Path('logs/source_blocked/free_market')
    stamp=snapshot['observed_at'].replace(':','').replace('+','_')
    (out/('canonical_'+stamp+'.json')).write_text(json.dumps(result,indent=2))
    decisions=[{'cutoff':result['observed_at'],'game_id':r['game_id'],
                'player_id':r['player_id'],'prop_type':r['prop_type'],'line':r['line'],
                'evaluation_kind':'capture_probe'} for r in result['props']]
    decisions=list({(r['player_id'],r['line']):r for r in decisions}.values())
    probe=replay(decisions,quotes=result['props'],totals=result['totals'])
    (out/('replay_'+stamp+'.json')).write_text(json.dumps(probe,indent=2))
    print({'credits_used':snapshot['credits_used'],'credits_remaining':snapshot['credits_remaining'],
           'nba_game_id':result['nba_game_id'],'props':len(result['props']),'totals':len(result['totals']),
           'rejected':result['rejected'],'roster_fresh':result['roster_fresh']})
