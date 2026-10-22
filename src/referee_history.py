"""Validated per-game referee observations and strictly prior-day tendencies."""
from __future__ import annotations

import math
import time

from .nba_officials import summary_officials


def ensure_history(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS referee_game_stats (
        game_id TEXT PRIMARY KEY, season TEXT NOT NULL, game_date DATE NOT NULL,
        home_team_id BIGINT NOT NULL, away_team_id BIGINT NOT NULL,
        total_fouls DOUBLE PRECISION NOT NULL, pace DOUBLE PRECISION NOT NULL,
        home_win BOOLEAN NOT NULL, updated_at BIGINT NOT NULL, source TEXT NOT NULL DEFAULT 'NBA')''')
    conn.execute("ALTER TABLE referee_game_stats ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'NBA'")
    conn.execute('''CREATE TABLE IF NOT EXISTS referee_game_features (
        game_id TEXT NOT NULL, ref_name TEXT NOT NULL, games INTEGER NOT NULL,
        foul_rate DOUBLE PRECISION, home_win_pct DOUBLE PRECISION, pace DOUBLE PRECISION,
        PRIMARY KEY(game_id,ref_name))''')
    conn.commit()


def validate_game(game_id, base_rows, summary, advanced):
    refs, winner, day, scores = summary_officials(summary)
    if str(summary['gameId']) != str(game_id) or int(summary.get('gameStatus', 0)) != 3:
        raise ValueError('Wrong or incomplete game')
    if len(refs) not in (3, 4) or len({r['name'] for r in refs}) != len(refs) or len({r['id'] for r in refs}) != len(refs):
        raise ValueError('Expected three distinct game officials or a documented four-person crew')
    home, away = int(summary['homeTeamId']), int(summary['awayTeamId'])
    if len(base_rows) != 2 or {int(r['TEAM_ID']) for r in base_rows} != {home, away}:
        raise ValueError('Team identities disagree')
    score_by_team = dict(scores)
    for row in base_rows:
        if float(row['PTS']) != score_by_team[int(row['TEAM_ID'])] or str(row['GAME_DATE'])[:10] != day:
            raise ValueError('Score or game date disagrees with league log')
    by_team = {int(r['teamId']): r for r in advanced}
    if set(by_team) != {home, away}:
        raise ValueError('Missing advanced team totals')
    foul_total = sum(float(r['PF']) for r in base_rows)
    pace = sum(float(r['pace']) for r in advanced) / 2
    if not math.isfinite(foul_total) or not 0 < foul_total < 150 or not math.isfinite(pace) or not 50 < pace < 180:
        raise ValueError('Invalid fouls or pace')
    return {'game_id': str(game_id), 'game_date': day, 'home_team_id': home, 'away_team_id': away,
            'home_win': winner, 'total_fouls': foul_total, 'pace': pace, 'officials': refs,
            'source': summary.get('source', 'NBA'),
            'teams': [{'team_id': tid, 'pts': score_by_team[tid],
                       'opp_pts': score_by_team[away if tid == home else home],
                       'pace': float(by_team[tid]['pace']),
                       'off_rating': float(by_team[tid]['offensiveRating']),
                       'def_rating': float(by_team[tid]['defensiveRating'])} for tid in (home, away)]}


def save_game(conn, season, game):
    """Replace one observation atomically. Replaying it never adds sample weight."""
    now = int(time.time())
    with conn.transaction():
        conn.execute('''INSERT INTO referee_game_stats
            (game_id,season,game_date,home_team_id,away_team_id,total_fouls,pace,home_win,updated_at,source)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(game_id) DO UPDATE SET
            season=excluded.season,game_date=excluded.game_date,home_team_id=excluded.home_team_id,
            away_team_id=excluded.away_team_id,total_fouls=excluded.total_fouls,
            pace=excluded.pace,home_win=excluded.home_win,updated_at=excluded.updated_at,source=excluded.source''',
            (game['game_id'], season, game['game_date'], game['home_team_id'],game['away_team_id'],
             game['total_fouls'],game['pace'],game['home_win'],now,game.get('source','NBA')))
        conn.execute('DELETE FROM game_officials WHERE game_id=%s', (game['game_id'],))
        for ref in game['officials']:
            conn.execute('INSERT INTO game_officials(game_id,ref_name,ref_id,updated_at) VALUES (%s,%s,%s,%s)',
                         (game['game_id'],ref['name'],ref['id'],now))
        for team in game['teams']:
            conn.execute('''INSERT INTO game_team_stats
                (game_id,team_id,game_date,pace,off_rating,def_rating,pts,opp_pts,updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(game_id,team_id) DO UPDATE SET
                game_date=excluded.game_date,pace=excluded.pace,off_rating=excluded.off_rating,
                def_rating=excluded.def_rating,pts=excluded.pts,opp_pts=excluded.opp_pts,updated_at=excluded.updated_at''',
                (game['game_id'],team['team_id'],game['game_date'],team['pace'],team['off_rating'],
                 team['def_rating'],team['pts'],team['opp_pts'],now))


def rebuild_aggregates(conn):
    """Compute from unique game rows, rather than adding averages on each rerun."""
    with conn.transaction():
        conn.execute('DELETE FROM referee_stats')
        conn.execute('''INSERT INTO referee_stats(ref_name,games,foul_rate,home_win_pct,pace,updated_at)
            SELECT o.ref_name,COUNT(*),AVG(g.total_fouls),AVG(g.home_win::int),AVG(g.pace),%s
            FROM game_officials o JOIN referee_game_stats g USING(game_id)
            GROUP BY o.ref_name HAVING COUNT(*) >= 3''', (int(time.time()),))
        conn.execute('DELETE FROM referee_game_features')
        conn.execute('''INSERT INTO referee_game_features(game_id,ref_name,games,foul_rate,home_win_pct,pace)
            WITH daily AS (
                SELECT o.ref_name,g.game_date,COUNT(*) AS n,SUM(g.total_fouls) AS fouls,
                    SUM(g.home_win::int) AS wins,SUM(g.pace) AS pace
                FROM game_officials o JOIN referee_game_stats g USING(game_id)
                GROUP BY o.ref_name,g.game_date
            ), prior AS (
                SELECT *,SUM(n) OVER w AS prior_n,SUM(fouls) OVER w AS prior_fouls,
                    SUM(wins) OVER w AS prior_wins,SUM(pace) OVER w AS prior_pace
                FROM daily WINDOW w AS (PARTITION BY ref_name ORDER BY game_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            )
            SELECT g.game_id,o.ref_name,COALESCE(p.prior_n,0)::int,
                p.prior_fouls/NULLIF(p.prior_n,0),p.prior_wins::float/NULLIF(p.prior_n,0),
                p.prior_pace/NULLIF(p.prior_n,0)
            FROM game_officials o JOIN referee_game_stats g USING(game_id)
            JOIN prior p ON p.ref_name=o.ref_name AND p.game_date=g.game_date''')


def historical_features(precomputed, game_id):
    """Crew means from prior dates; never fall back to full-history averages."""
    names = precomputed.get('game_officials', {}).get(str(game_id), [])
    history = precomputed.get('referee_history', {})
    rows = [history[(str(game_id), name.lower())] for name in names
            if (str(game_id), name.lower()) in history]
    rows = [r for r in rows if int(r.get('games', 0)) >= 3]
    if not rows:
        return 0.0, 0.5, 100.0
    return tuple(sum(float(r[key]) for r in rows) / len(rows)
                 for key in ('foul_rate', 'home_win_pct', 'pace'))
