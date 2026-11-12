"""Report independent real-line coverage without touching frozen studies."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from src.prospective_points import digest
from src.study_provenance import verify_closed


PROSPECTIVE_START='2026-10-01'  # first day of the live season the old guard refused outright
HOLDOUT_DAYS=14  # pre-registered: games newer than this stay unseen, so a forward test survives


def holdout_boundary(today,days=HOLDOUT_DAYS):
    """The first held-back date, for a day the caller names.

    The day is an argument rather than a clock read on purpose. coverage() must
    answer the same for one dataset however often it is re-run -- readiness.json
    and every digest built on it depend on that -- and a now()-derived cutoff
    would silently admit one more day of games with every passing day.
    """
    return str((pd.Timestamp(today).normalize()-pd.Timedelta(days=days)).date())


def coverage(data,holdout_from=PROSPECTIVE_START):
    """Split coverage, refusing any game on or after the holdout boundary.

    The boundary defaults to the whole live season, which is what the guard used
    to hardcode, so no caller reaches live data without naming a date. Note the
    masks are unchanged: games past 2025-07-01 fall in `evaluation`, so moving
    the boundary lets the live season be *evaluated*, never fitted. Teaching the
    model the live season is a change to the mask dates, not to this boundary.
    """
    days=pd.to_datetime(data['dates']); games=data['games']
    masks={'fit':days<'2025-01-01','calibration':(days>='2025-03-01')&(days<'2025-05-01'),
           'selection':(days>='2025-05-01')&(days<'2025-07-01'),'evaluation':days>='2025-07-01'}
    if (days>=holdout_from).any():
        raise ValueError(f'Games on or after the holdout boundary {holdout_from} must stay unseen')
    return {k:{'player_games':int(m.sum()),'independent_games':len(set(games[m])),
               'ready':bool(m.sum()>=100 and len(set(games[m]))>=20)} for k,m in masks.items()}


def run(path,holdout_from=PROSPECTIVE_START):
    # logs/prospective_points_v2 is a separate live study with its own gate; this
    # report reads none of its models or sources, so it does not stand on them.
    study=Path('logs/prospective_points_v5')
    p=json.loads((study/'protocol.json').read_text())
    if digest(p['candidate_path'])!=p['candidate_sha256'] or digest('logs/prospective_points_v1/protocol.json')!=p['v1_protocol_sha256']:
        raise ValueError('Frozen v5 drift')
    # v5 is closed: its artifacts stay live-verified above, and so does every
    # pinned source except the ones closed.json explicitly released -- only those
    # may move on. src/points_v5.py is not released and has 24 live importers.
    verify_closed(study,p['source_hashes'])
    d=joblib.load(path); result=coverage(d,holdout_from)
    # Recorded, not recomputed: a reader of this report must be able to tell which
    # games were held back without knowing the day it was written.
    report={'splits':result,'ready':all(x['ready'] for x in result.values()),'frozen_studies_verified':True,
            'holdout_from':holdout_from,
            'minimum_additional_games':sum(max(0,20-x['independent_games']) for x in result.values()),
            'dataset_sha256':digest(path),'minimum_rows_per_split':100,'minimum_games_per_split':20}
    (path.parent/'readiness.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path)
    g=p.add_mutually_exclusive_group()
    g.add_argument('--holdout-from',help='first held-back date, e.g. 2026-10-29')
    g.add_argument('--holdout-days',type=int,help=f'derive it as today minus N days (pre-registered N={HOLDOUT_DAYS})')
    a=p.parse_args()
    boundary=a.holdout_from or (holdout_boundary(pd.Timestamp.now(tz='America/New_York').date(),a.holdout_days)
                                if a.holdout_days is not None else PROSPECTIVE_START)
    run(a.dataset,boundary)
