"""Capture source-identified NBA tipoff at prediction time, fail closed."""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from .api_cache import ttl_cache
from .source_evidence import instant

URL = 'https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json'


def select_game(payload, team_id, opponent_id, today):
    if not team_id or not opponent_id:
        return {}
    board = payload['scoreboard']
    if board['gameDate'] != today:
        return {}
    matches = []
    for game in board['games']:
        if {int(game['homeTeam']['teamId']), int(game['awayTeam']['teamId'])} != {int(team_id), int(opponent_id)}:
            continue
        start = instant(game['gameTimeUTC'])
        gid = str(game['gameId'])
        if len(gid) != 10 or not gid.isdigit():
            continue
        if start.astimezone(ZoneInfo('America/New_York')).date().isoformat() != today:
            continue
        matches.append({'game_id': gid, 'game_start': start.isoformat(), 'source': URL})
    return matches[0] if len(matches) == 1 else {}


@ttl_cache(ttl_seconds=300, name='prediction_schedule')
def current_game_schedule(team_id, opponent_id, nba_game_id=None):
    if not team_id or not opponent_id:
        return {}
    today = datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    try:
        response = requests.get(URL, timeout=(3, 5))
        response.raise_for_status()
        result = select_game(response.json(), team_id, opponent_id, today)
        if result:
            return result
    except (requests.RequestException, ValueError, KeyError, TypeError):
        pass
    # The caller's official scoreboard supplies the NBA ID. ESPN supplies
    # tipoff only after exact date/team matching; its event ID is never reused.
    if nba_game_id and len(str(nba_game_id)) == 10 and str(nba_game_id).isdigit():
        try:
            from nba_api.stats.static import teams
            ids = {t['abbreviation']: t['id'] for t in teams.get_teams()}
            ids.update({alias: ids[name] for alias, name in
                        [('GS','GSW'), ('NY','NYK'), ('NO','NOP'), ('SA','SAS'), ('UTAH','UTA'), ('WSH','WAS')]})
            url = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=' + today.replace('-', '')
            response = requests.get(url, timeout=(3, 5))
            response.raise_for_status()
            matches = []
            for event in response.json().get('events', []):
                start = instant(event['date'])
                competitors = event['competitions'][0]['competitors']
                pair = {ids.get(c['team']['abbreviation']) for c in competitors}
                if pair == {int(team_id), int(opponent_id)} and start.astimezone(ZoneInfo('America/New_York')).date().isoformat() == today:
                    matches.append({'game_id': str(nba_game_id), 'game_start': start.isoformat(),
                                    'source': url + '#event=' + str(event['id'])})
            if len(matches) == 1:
                return matches[0]
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError):
            pass
    logging.getLogger(__name__).warning('Prediction schedule unavailable; pregame provenance remains unknown')
    return {}
