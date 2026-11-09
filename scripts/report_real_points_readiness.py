"""Report independent real-line coverage without touching frozen studies."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from src.prospective_points import digest
from scripts.validate_prospective_points import protocol


def coverage(data):
    days=pd.to_datetime(data['dates']); games=data['games']
    masks={'fit':days<'2025-01-01','calibration':(days>='2025-03-01')&(days<'2025-05-01'),
           'selection':(days>='2025-05-01')&(days<'2025-07-01'),'evaluation':days>='2025-07-01'}
    if (days>='2026-10-01').any():raise ValueError('Protected prospective season')
    return {k:{'player_games':int(m.sum()),'independent_games':len(set(games[m])),
               'ready':bool(m.sum()>=100 and len(set(games[m]))>=20)} for k,m in masks.items()}


def run(path):
    protocol()
    p=json.loads(Path('logs/prospective_points_v5/protocol.json').read_text())
    if not all(digest(k)==v for k,v in p['source_hashes'].items()) or digest(p['candidate_path'])!=p['candidate_sha256'] or digest('logs/prospective_points_v1/protocol.json')!=p['v1_protocol_sha256']:
        raise ValueError('Frozen v5 drift')
    d=joblib.load(path); result=coverage(d)
    report={'splits':result,'ready':all(x['ready'] for x in result.values()),'frozen_studies_verified':True,
            'minimum_additional_games':sum(max(0,20-x['independent_games']) for x in result.values()),
            'dataset_sha256':digest(path),'minimum_rows_per_split':100,'minimum_games_per_split':20}
    (path.parent/'readiness.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);run(p.parse_args().dataset)
