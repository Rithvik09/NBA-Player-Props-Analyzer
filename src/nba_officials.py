"""Modern NBA officials endpoints (v3 is not wrapped by older nba_api releases)."""
from nba_api.stats.library.http import NBAStatsHTTP


def nba_json(endpoint, **parameters):
    return NBAStatsHTTP().send_api_request(endpoint=endpoint, parameters=parameters, timeout=20).get_dict()


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
