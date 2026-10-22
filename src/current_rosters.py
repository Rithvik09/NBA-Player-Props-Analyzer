"""Current roster snapshots. ESPN IDs are never substituted for NBA player IDs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import time
import unicodedata

import requests
from nba_api.stats.static import teams
from .api_cache import ttl_cache

BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
SNAPSHOT = Path(__file__).resolve().parents[1] / '.cache/current_rosters.json'


def season_now():
    now = datetime.now(timezone.utc)
    year = now.year if now.month >= 7 else now.year - 1
    return f'{year}-{str(year + 1)[2:]}'


def name_key(name):
    text = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    text = re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', text)
    return re.sub('[^a-z]', '', text)


def get_json(url):
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


def load_snapshot(max_age=21600):
    try:
        data = json.loads(SNAPSHOT.read_text())
        if data['season'] == season_now() and 0 <= time.time() - data['fetched_at'] < max_age:
            return data
    except (OSError, ValueError, KeyError):
        pass
    return None


def refresh_rosters():
    catalog = get_json(BASE + '/teams?limit=30')['sports'][0]['leagues'][0]
    season = season_now()
    if catalog['season']['displayName'] != season:
        raise ValueError('Roster source is not serving the current season')
    nba_teams = {name_key(t['full_name']): t for t in teams.get_teams()}
    from nba_api.stats.endpoints import commonallplayers
    nba_rows = commonallplayers.CommonAllPlayers(
        is_only_current_season=1, season=season, timeout=20
    ).get_data_frames()[0].to_dict('records')
    nba_players = {}
    for player in nba_rows:
        if int(player['ROSTERSTATUS']) == 1:
            nba_players.setdefault(name_key(player['DISPLAY_FIRST_LAST']), []).append(player)
    if len({int(p['TEAM_ID']) for p in nba_rows if int(p['ROSTERSTATUS']) == 1}) != 30:
        raise ValueError('NBA roster feed does not cover all 30 teams')
    source_teams = [item['team'] for item in catalog['teams']]
    if len(source_teams) != 30:
        raise ValueError('Expected all 30 teams; preserving previous snapshot')

    def fetch(team):
        roster = get_json(BASE + '/teams/' + str(team['id']) + '/roster')
        if roster.get('season', {}).get('displayName') != season or len(roster.get('athletes', [])) < 10:
            raise ValueError('Incomplete or wrong-season roster: ' + team['displayName'])
        nba_team = nba_teams.get(name_key(team['displayName']))
        if nba_team is None:
            # ESPN abbreviates Los Angeles Clippers as LA Clippers.
            nba_team = next((t for t in teams.get_teams() if t['nickname'] == team['name']), None)
        if nba_team is None:
            raise ValueError('Unknown team: ' + team['displayName'])
        rows = []
        for athlete in roster['athletes']:
            ids = nba_players.get(name_key(athlete['displayName']), [])
            rows.append({'espn_id': str(athlete['id']), 'nba_id': int(ids[0]['PERSON_ID']) if len(ids) == 1 else None,
                         'full_name': athlete['displayName'], 'team_id': nba_team['id'],
                         'team': nba_team['full_name'], 'position': athlete.get('position', {}).get('abbreviation', '')})
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        rosters = list(pool.map(fetch, source_teams))
    rows = [row for roster in rosters for row in roster]
    if len({r['espn_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate player on multiple current rosters')
    espn_by_nba = {row['nba_id']: row for row in rows if row['nba_id'] is not None}
    disagreements = [r for r in rows if r['nba_id'] is None]
    canonical = []
    by_team_id = {t['id']: t for t in teams.get_teams()}
    for p in nba_rows:
        if int(p['ROSTERSTATUS']) != 1:
            continue
        pid, tid = int(p['PERSON_ID']), int(p['TEAM_ID'])
        extra = espn_by_nba.get(pid, {})
        if not extra or extra['team_id'] != tid:
            disagreements.append({'full_name': p['DISPLAY_FIRST_LAST'], 'nba_team_id': tid,
                                  'espn_team_id': extra.get('team_id')})
            extra = {}
        canonical.append({'nba_id': pid, 'espn_id': extra.get('espn_id'),
                          'full_name': p['DISPLAY_FIRST_LAST'], 'team_id': tid,
                          'team': by_team_id[tid]['full_name'], 'position': extra.get('position', '')})
    data = {'season': season, 'fetched_at': time.time(), 'source': 'NBA CommonAllPlayers',
            'secondary_source': BASE, 'team_count': 30, 'players': canonical,
            'source_disagreements': disagreements, 'unmapped_players': []}
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = SNAPSHOT.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(SNAPSHOT)
    return data


@ttl_cache(ttl_seconds=900, name='current_rosters')
def get_rosters():
    # Test/offline mode must not turn player search into live network calls.
    if os.environ.get('NBA_CACHE_DISABLE', '').lower() in ('1', 'true', 'yes'):
        return None
    snapshot = load_snapshot()
    if snapshot:
        return snapshot
    try:
        return refresh_rosters()
    except Exception as exc:
        logging.getLogger(__name__).warning('Current roster refresh failed (%s)', type(exc).__name__)
        return None


def current_player(player_id):
    snapshot = get_rosters()
    if snapshot:
        return next((r for r in snapshot['players'] if r['nba_id'] == int(player_id)), None)
    return None
