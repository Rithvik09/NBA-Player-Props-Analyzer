"""Game-specific officials from ESPN summaries, with separate provider IDs."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from nba_api.stats.static import teams
from .api_cache import ttl_cache
from .current_rosters import BASE, get_json

SNAPSHOT = Path(__file__).resolve().parents[1] / '.cache/current_officials.json'


def parse_game(event, summary, team_ids):
    competition = event['competitions'][0]
    competitors = competition['competitors']
    refs = [r['fullName'] for r in summary.get('gameInfo', {}).get('officials', []) if r.get('fullName')]
    mapped = [team_ids.get(c['team']['abbreviation']) for c in competitors]
    if None in mapped or len(mapped) != 2:
        raise ValueError('Unrecognized NBA game teams')
    foul_counts = [float(s['displayValue']) for t in summary.get('boxscore', {}).get('teams', [])
                   for s in t.get('statistics', []) if s['name'] == 'fouls']
    home = next(c for c in competitors if c['homeAway'] == 'home')
    return {'espn_game_id': str(event['id']), 'date': event['date'], 'team_ids': mapped,
            'officials': refs, 'completed': bool(competition['status']['type']['completed']),
            'total_fouls': sum(foul_counts) if len(foul_counts) == 2 else None,
            'home_won': home.get('winner'), 'source': BASE + '/summary?event=' + str(event['id'])}


def refresh_officials(days=3):
    today = datetime.now(ZoneInfo('America/New_York')).date()
    team_ids = {t['abbreviation']: t['id'] for t in teams.get_teams()}
    team_ids.update({'GS': team_ids['GSW'], 'NY': team_ids['NYK'], 'NO': team_ids['NOP'],
                     'SA': team_ids['SAS'], 'UTAH': team_ids['UTA'], 'WSH': team_ids['WAS']})
    games = []
    source_errors = []
    try:
        for offset in range(days):
            day = today - timedelta(days=offset)
            board = get_json(BASE + '/scoreboard?dates=' + day.strftime('%Y%m%d'))
            for event in board.get('events', []):
                summary = get_json(BASE + '/summary?event=' + str(event['id']))
                games.append(parse_game(event, summary, team_ids))
    except Exception as exc:
        source_errors.append('ESPN:' + type(exc).__name__)
    # Prefer official NBA assignments when the feed has published a crew.
    from .nba_officials import nba_json, game_summary, summary_officials
    nba_errors = []
    try:
        board = nba_json('scoreboardv3', GameDate=str(today), LeagueID='00')['scoreboard']
        for event in board.get('games', []):
            summary = game_summary(event['gameId'])
            refs, winner, _, _ = summary_officials(summary)
            tids = {int(summary['homeTeamId']), int(summary['awayTeamId'])}
            existing = next((g for g in games if set(g['team_ids']) == tids and
                             datetime.fromisoformat(g['date'].replace('Z', '+00:00')).astimezone(ZoneInfo('America/New_York')).date() == today), None)
            if existing is not None:
                existing['nba_game_id'] = str(event['gameId'])
                if refs:
                    existing['officials'] = [r['name'] for r in refs]
                    existing['source'] = 'NBA BoxScoreSummaryV3'
            elif refs:
                games.append({'nba_game_id': str(event['gameId']), 'date': summary['gameTimeUTC'],
                              'team_ids': list(tids), 'officials': [r['name'] for r in refs],
                              'completed': summary['gameStatus'] == 3, 'home_won': winner,
                              'total_fouls': None, 'source': 'NBA BoxScoreSummaryV3'})
    except Exception as exc:
        nba_errors.append(type(exc).__name__)
    data = {'source_errors': source_errors, 'nba_errors': nba_errors, 'fetched_at': time.time(), 'date': str(today), 'source': BASE, 'games': games}
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    temporary = SNAPSHOT.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(SNAPSHOT)
    return data


@ttl_cache(ttl_seconds=900, name='live_game_officials')
def current_officials():
    try:
        data = json.loads(SNAPSHOT.read_text())
        if 0 <= time.time() - data['fetched_at'] < 900:
            return data
    except (OSError, ValueError, KeyError):
        pass
    return refresh_officials(days=1)


def matchup_officials(team_id, opponent_team_id):
    today = datetime.now(ZoneInfo('America/New_York')).date()
    for game in current_officials()['games']:
        game_day = datetime.fromisoformat(game['date'].replace('Z', '+00:00')).astimezone(ZoneInfo('America/New_York')).date()
        if game_day == today and set(game['team_ids']) == {int(team_id), int(opponent_team_id)}:
            return game['officials']
    return []
