"""Add actual opening-season histories and injury evidence; preserve v4 datasets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from src.historical_inputs import HistoricalInputs, load_season_logs
from src.audited_features import build_audited_features, SPECS
from src.points_v5 import enrich_injuries, POLICY
from scripts.train_models import _parse_matchup, _team_id

OUT = Path('logs/points_v5_data')


def build(season):
    destination = OUT/f'{season}.joblib'
    if destination.exists():
        return {'season': season, 'status': 'cached'}
    data = joblib.load(f'logs/expanded_points_v4/audited_{season}.joblib')
    raw = joblib.load('logs/expanded_points_v4/source_snapshot.joblib')
    logs = load_season_logs(season, include_postseason=True)
    inputs = HistoricalInputs(season, raw, logs, audited=True)
    original = set(map(int, inputs.catalog(50).PLAYER_ID))
    team_lookup = {(str(r.GAME_ID), int(r.PLAYER_ID)): _parse_matchup(r.MATCHUP) for r in logs.itertuples()}
    injury_cache = {}
    def enrich(f, gid, pid, date):
        own, opp, _ = team_lookup[(str(gid), int(pid))]
        inputs.prediction_game_id = str(gid)
        key = (str(gid), own)
        if key not in injury_cache:
            injury_cache[key] = enrich_injuries({}, inputs.recovery, _team_id(own), _team_id(opp), pd.Timestamp(date))
        return {**f, **injury_cache[key]}
    rows = []
    for i, f in enumerate(data['X'].to_dict('records')):
        rows.append(enrich(f, data['games'][i], data['players'][i], data['dates'][i]))
    extra = {'y': [], 'dates': [], 'games': [], 'players': [], 'original_cohort': []}
    checkpoint = OUT/season
    checkpoint.mkdir(exist_ok=True)
    for pid in sorted(inputs.logs):
        path = checkpoint/f'{pid}.joblib'
        if path.exists():
            part = joblib.load(path)
        else:
            hist = inputs.player_log(pid).sort_values('GAME_DATE').copy()
            hist['GAME_DATE'] = pd.to_datetime(hist.GAME_DATE)
            part = {'rows': [], **{k: [] for k in extra}}
            for i in range(1, min(10, len(hist))):
                prior, target = hist.iloc[:i], hist.iloc[i]
                if pid not in original and prior.MIN.mean() < 12:
                    continue
                own, opp, home = _parse_matchup(target.MATCHUP)
                inputs.prediction_game_id = str(target.GAME_ID)
                f, gaps = build_audited_features(prior, 'PTS', pid, _team_id(own), _team_id(opp), target.GAME_DATE, home, inputs)
                for k in gaps:
                    if SPECS[k].status == 'active':
                        f[k] = np.nan
                part['rows'].append(enrich(f, target.GAME_ID, pid, target.GAME_DATE))
                for k, value in [('y', float(target.PTS)), ('dates', target.GAME_DATE), ('games', str(target.GAME_ID)),
                                 ('players', pid), ('original_cohort', pid in original)]:
                    part[k].append(value)
            joblib.dump(part, path)
        rows.extend(part['rows'])
        for k in extra:
            extra[k].extend(part[k])
    result = {'X': pd.DataFrame(rows), 'feature_policy': POLICY}
    for k in extra:
        result[k] = np.concatenate([data[k], np.asarray(extra[k])])
    assert not pd.DataFrame({'game': result['games'], 'player': result['players']}).duplicated().any()
    joblib.dump(result, destination)
    report = {'season': season, 'total_rows': len(rows), 'short_history_rows': len(extra['y'])}
    print(report, flush=True)
    return report


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(build, ['2022-23', '2023-24', '2024-25', '2025-26']))
    (OUT/'report.json').write_text(json.dumps(result, indent=2))
