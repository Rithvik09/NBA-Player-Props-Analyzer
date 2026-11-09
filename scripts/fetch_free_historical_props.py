"""Bounded DunkAPI free-tier history capture; requires a user-configured key.

No subscription, wallet purchase or Odds API calls. Each request reserves its no-charge estimated cost before transport;
maximum500 reserved per calendar month.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import requests
from src.prospective_points import save_new

OUT = Path('logs/real_points_sources/dunkapi')
MONTHLY_LIMIT = 500  # User authorized the full free allowance; never buy credits.


def fetch(player_id=None, game_id=None, from_date='2023-05-03', to_date='2026-06-30'):
    key = os.environ.get('DUNKAPI_API_KEY')
    if not key:
        values = dict(line.split('=', 1) for line in Path('.env').read_text().splitlines() if '=' in line and not line.startswith('#'))
        key = values.get('DUNKAPI_API_KEY', '').strip().strip('"').strip("'")
    if not key:
        raise ValueError('Set DUNKAPI_API_KEY locally; no request or credit use occurred')
    if not str(player_id or game_id).isdigit() or bool(player_id) == bool(game_id):
        raise ValueError('Expected provider player ID, not a name')
    arguments = {'market': 'player_points', 'snapshot_type': 'closing'}
    if game_id:
        arguments['game_id'] = str(game_id)
    else:
        arguments.update(player_id=str(player_id), from_date=from_date, to_date=to_date)
    # The live API charges per game, despite the advertised per-call headline.
    # Always obtain its no-charge estimate before reserving or fetching.
    estimate = requests.post('https://mcp.dunkapi.com/v1/get_player_props', json={**arguments, 'estimate_only': True},
                             headers={'x-api-key': key}, timeout=(5, 20))
    estimate.raise_for_status()
    payload = estimate.json(); payload = payload.get('data', payload)
    cost = int(payload['credits'])
    if not payload.get('estimate_only') or not 0 < cost <= MONTHLY_LIMIT:
        raise ValueError('Estimated cost exceeds bounded free research allowance: '+str(cost))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        month = datetime.now(timezone.utc).strftime('%Y-%m')
        path = OUT/f'budget_{month}.json'
        state = json.loads(path.read_text()) if path.exists() else {'reserved': 0, 'players': []}
        identity = json.dumps(arguments, sort_keys=True)
        if identity in state.setdefault('requests', []):
            raise ValueError('Player already attempted this month; inspect archived response')
        if state['reserved']+cost > MONTHLY_LIMIT:
            raise ValueError('Local500-credit monthly research ceiling reached')
        state['reserved'] += cost; state['requests'].append(identity)
        temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(state, indent=2)); temp.replace(path)
        try:
            response = requests.post('https://mcp.dunkapi.com/v1/get_player_props', json=arguments,
                                     headers={'x-api-key': key}, timeout=(5, 30))
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError('Historical quote request failed: '+type(exc).__name__) from None
        filename = OUT/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'_'+str(player_id or game_id)+'.json')
        save_new(filename, {'observed_at': datetime.now(timezone.utc).isoformat(),
                           'provider': 'DunkAPI', 'estimated_credits': cost, 'request': arguments, 'payload': payload})
        print(str(filename))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True); g.add_argument('--player-id'); g.add_argument('--game-id')
    p.add_argument('--from-date',default='2023-05-03');p.add_argument('--to-date',default='2026-06-30')
    a=p.parse_args();fetch(a.player_id,a.game_id,a.from_date,a.to_date)
