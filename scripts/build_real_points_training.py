"""Import timestamped historical DunkAPI points quotes, never prospective games.

Input: an archived get_player_props response. ESPN/provider IDs are not NBA IDs.
NBA identities are joined uniquely by exact normalized player name and game date,
then checked against the official NBA game summary and its tipoff.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.historical_inputs import HistoricalInputs, load_season_logs
from src.historical_injuries import normalize_name
from src.audited_features import build_audited_features, SPECS
from src.points_v5 import enrich_injuries, POLICY
from src.prospective_points import digest, save_new
from src.nba_officials import game_summary
from scripts.train_models import _parse_matchup, _team_id


def provider_teams_match(row, summary):
    if not row.get('home_team'):return True
    from nba_api.stats.static.teams import get_teams
    names={t['full_name']:int(t['id']) for t in get_teams()}
    # Official summaries sometimes use LA instead of Los Angeles; IDs stay canonical.
    for side in ('homeTeam','awayTeam'):
        t=summary[side];names[t['teamCity']+' '+t['teamName']]=int(t['teamId'])
    return names.get(row['home_team'])==int(summary['homeTeam']['teamId']) and names.get(row['away_team'])==int(summary['awayTeam']['teamId'])


def validate_record(row, summary, protected_games):
    gid = str(summary['gameId'])
    if gid in protected_games or str(row['game_date']) >= '2026-10-01':
        raise ValueError('Protected prospective game/season')
    if row['market'] != 'player_points':
        raise ValueError('Not a points market')
    captured, tip = pd.Timestamp(row['captured_at']), pd.Timestamp(summary['gameTimeUTC'])
    if captured.tzinfo is None or tip.tzinfo is None or not captured < tip:
        raise ValueError('Quote must have a timezone and precede official tipoff')
    if tip.tz_convert('America/New_York').date().isoformat() != row['game_date']:
        raise ValueError('Wrong game date')
    if not 0 <= (tip-captured).total_seconds() <= 7*86400:
        raise ValueError('Implausible quote age')
    for key in ('line', 'over_odds', 'under_odds'):
        if row.get(key) is None:
            raise ValueError('Missing '+key)
        if not np.isfinite(float(row[key])):
            raise ValueError('Nonfinite quote')
    if row['line'] < 0 or any(abs(float(row[k])) < 100 for k in ('over_odds', 'under_odds')):
        raise ValueError('Invalid line/odds')
    if not row.get('bookmaker') or not row.get('source'):
        raise ValueError('Missing quote provenance')
    return captured


def run(path, output, qualification=None):
    output.mkdir(parents=True, exist_ok=False)
    response = json.loads(path.read_text())
    response = response.get('payload', response)
    response = response.get('data', response)
    records = response.get('props', [])
    if qualification is not None:
        qualified = json.loads(qualification.read_text())
        if qualified['input_sha256'] != digest(path): raise ValueError('Qualification input drift')
        records = [dict(r['raw_quote'], player_name=r['nba_player_name']) for r in qualified['accepted']]
        for ev in qualified['evidence'].values():
            if digest(Path(ev['nba_pbp_path'])) != ev['nba_pbp_sha256'] or digest(Path(ev['espn_path'])) != ev['espn_sha256']:
                raise ValueError('Qualification evidence drift')
    protected = {json.loads(p.read_text())['quote']['game_id'] for p in Path('logs/prospective_points_v1/forecasts').glob('*.json')}
    raw = joblib.load('logs/expanded_points_v4/source_snapshot.joblib')
    contexts, boards, seen_provider, selected, rejected = {}, {}, {}, {}, []
    identity_indices = {}
    selected_keys = set()
    for q in sorted(records, key=lambda q: (str(q.get('captured_at')), str(q.get('bookmaker')))):
        try:
            day = pd.Timestamp(q['game_date'])
            if day >= pd.Timestamp('2026-10-01'):
                raise ValueError('Protected prospective season')
            year = day.year if day.month >= 10 else day.year-1
            season = f'{year}-{str(year+1)[2:]}'
            if season not in contexts:
                logs = load_season_logs(season, include_postseason=True)
                contexts[season] = HistoricalInputs(season, raw, logs, audited=True)
                identity_indices[season] = {}
                for idx,r in contexts[season].recovery.logs.iterrows():
                    key=(pd.Timestamp(r.GAME_DATE),normalize_name(r.PLAYER_NAME,keep_suffix=True))
                    identity_indices[season].setdefault(key,[]).append(idx)
            inputs = contexts[season]
            # Exact full normalized names; collisions rejected, never fuzzy.
            matches = inputs.recovery.logs
            matches = matches.loc[identity_indices[season].get((day,normalize_name(q['player_name'],keep_suffix=True)),[])]
            if len(matches) != 1:
                raise ValueError('Ambiguous or missing NBA player/date identity')
            actual = matches.iloc[0]; gid = str(actual.GAME_ID); pid = int(actual.PLAYER_ID)
            if q['game_id'] in seen_provider and seen_provider[q['game_id']] != gid:
                raise ValueError('Provider event maps to multiple NBA games')
            seen_provider[q['game_id']] = gid
            if gid not in boards:
                boards[gid] = (json.loads((Path(qualified.get('summaries_dir','logs/real_points_pilot_02'))/f'{gid}_summary.json').read_text()) if qualification else game_summary(gid))
                save_new(output/f'{gid}_summary.json', boards[gid])
            summary = boards[gid]
            if not provider_teams_match(q,summary):
                raise ValueError('Provider home/away identity mismatch')
            own, opp, home = _parse_matchup(actual.MATCHUP)
            if {int(summary[s]['teamId']) for s in ('homeTeam', 'awayTeam')} != {_team_id(own), _team_id(opp)}:
                raise ValueError('NBA team identity mismatch')
            validation_summary = summary
            if qualification is not None:
                validation_summary = dict(summary, gameTimeUTC=qualified['evidence'][gid]['actual_start_lower_bound'])
            cutoff = validate_record(q, validation_summary, protected)
            key = (gid, pid)
            if key in selected_keys:
                continue  # earliest quote, lexicographic book at that timestamp
            selected_keys.add(key)  # outcome/void checks cannot select a later book
            if int(summary.get('gameStatus', 0)) != 3 or float(actual.MIN) <= 0:
                raise ValueError('Game/player not completed with minutes')
            if float(actual.PTS) == float(q['line']):
                raise ValueError('Push')
            prior = inputs.player_log(pid)
            prior = prior[pd.to_datetime(prior.GAME_DATE) < day]
            if prior.empty:
                raise ValueError('No prior appearances')
            inputs.prediction_game_id = gid; inputs.prediction_cutoff = cutoff
            inputs.recovery._cutoff_cache = {}; inputs.recovery._injury_cache = {}
            features, gaps = build_audited_features(prior, 'PTS', pid, _team_id(own), _team_id(opp), day, home, inputs)
            for k in gaps:
                if SPECS[k].status == 'active':
                    features[k] = np.nan
            features = enrich_injuries(features, inputs.recovery, _team_id(own), _team_id(opp), day)
            selected[key] = {'X': features, 'y': int(actual.PTS > q['line']), 'points': float(actual.PTS),
                             'line': float(q['line']), 'date': day, 'game': gid, 'player': pid, 'quote': q}
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append({'provider_game': q.get('game_id'), 'player': q.get('player_name'), 'reason': str(exc)})
    rows = list(selected.values())
    result = {'X': pd.DataFrame([r['X'] for r in rows]), 'feature_policy': POLICY, 'provenance_validated': True}
    for key, field in [('y', 'y'), ('dates', 'date'), ('games', 'game'), ('players', 'player'), ('lines', 'line'), ('points', 'points')]:
        result[key] = np.array([r[field] for r in rows])
    joblib.dump(result, output/'real_points.joblib')
    save_new(output/'report.json', {'qualification_path': str(qualification) if qualification else None, 'input_sha256': digest(path), 'accepted': len(rows), 'rejected': rejected,
        'protected_games': sorted(protected), 'selection': 'earliest timestamp then lexicographic book, once per NBA player/game',
        'quotes': [r['quote'] for r in rows]})
    print('accepted', len(rows), 'rejected', len(rejected))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--quotes', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qualification', type=Path)
    a = p.parse_args(); run(a.quotes, a.output, a.qualification)
