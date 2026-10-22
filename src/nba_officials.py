"""Modern NBA box-score summaries, using one HTTP session per worker."""
import threading
import requests
from nba_api.stats.library.http import NBAStatsHTTP

_sessions = threading.local()


def nba_json(endpoint, **parameters):
    if getattr(_sessions, 'calls', 100) >= 100 or not hasattr(_sessions, 'session'):
        previous = getattr(_sessions, 'session', None)
        if previous is not None:
            previous.close()
        _sessions.session = requests.Session()
        _sessions.session.headers.update(NBAStatsHTTP.headers)
        _sessions.calls = 0
    try:
        _sessions.calls += 1
        response = _sessions.session.get('https://stats.nba.com/stats/' + endpoint,
                                         params=parameters, timeout=(5, 10))
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        _sessions.session.close()
        _sessions.calls = 100  # Retry with a fresh connection, never a stuck pool.
        raise


def game_summary(game_id):
    data = nba_json('boxscoresummaryv3', GameID=str(game_id))['boxScoreSummary']
    if str(data['gameId']) != str(game_id):
        raise ValueError('NBA returned another game')
    return data


def summary_officials(data):
    refs = [{'name': r.get('name') or (r.get('firstName', '') + ' ' + r.get('familyName', '')).strip(),
             'id': int(r['personId']) if r.get('personId') else None}
            for r in data.get('officials', [])]
    home, away = data['homeTeam'], data['awayTeam']
    completed = int(data.get('gameStatus', 0)) == 3
    winner = float(home['score']) > float(away['score']) if completed else None
    scores = [(int(t['teamId']), float(t['score'])) for t in (home, away)] if completed else []
    return [r for r in refs if r['name']], winner, str(data['gameEt'])[:10], scores
