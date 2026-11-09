"""Read-only comparison audit and versioned feature coverage before fitting."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.audited_features import build_audited_features
from src.historical_inputs import HistoricalInputs,load_season_logs
from src.points_reference_inputs import REFERENCE_FIELDS,aligned_reference
from src.points_v5 import prepare,EXTRA_COLUMNS,MINIMUM
from src.prospective_points import digest,save_new
from scripts.train_models import _parse_matchup,_team_id


def run():
    root=Path('logs/points_pretraining_v6')
    path=Path('logs/real_points_merged_04/real_points.joblib')
    reference_path=root/'reference_merged04_v2.joblib'
    data=joblib.load(path);side=joblib.load(reference_path)
    reference=aligned_reference(data,side,digest(path))
    source=Path('logs/expanded_points_v4/source_snapshot.joblib')
    if digest(source)!=side['source_snapshot_sha256']:raise ValueError('Snapshot drift')
    original=joblib.load('logs/expanded_points_experiment_v4/development_selected.joblib')
    columns=[k for k in dict.fromkeys(original.columns+EXTRA_COLUMNS) if k!='rolling_30day_avg']
    assert len(columns)==174
    prepared=prepare(data['X'])[columns]
    days=pd.to_datetime(data['dates'])
    masks={'fit':days<'2025-01-01','calibration':(days>='2025-03-01')&(days<'2025-05-01'),
           'selection':(days>='2025-05-01')&(days<'2025-07-01'),'evaluation':days>='2025-07-01'}
    coverage={k:{s:{'observed':int(prepared.loc[m,k].notna().sum()),'rows':int(m.sum())} for s,m in masks.items()} for k in columns}
    # Outcome-independent sample spans chronology and both changed/unchanged inputs.
    changed=~(reference[list(REFERENCE_FIELDS)].eq(data['X'][list(REFERENCE_FIELDS)])|
              (reference[list(REFERENCE_FIELDS)].isna()&data['X'][list(REFERENCE_FIELDS)].isna())).all(axis=1)
    selected=sorted(set(np.linspace(0,len(reference)-1,24,dtype=int).tolist()+np.flatnonzero(changed)[:3].tolist()))
    contexts={};checks=[]
    for i in selected:
        day=pd.Timestamp(data['dates'][i]);year=day.year if day.month>=10 else day.year-1;season=f'{year}-{str(year+1)[2:]}'
        if season not in contexts:
            logs=load_season_logs(season,include_postseason=True)
            contexts[season]=(HistoricalInputs(season,joblib.load(source),logs,audited=True),{(str(r.GAME_ID),int(r.PLAYER_ID)):r for r in logs.itertuples()})
        inputs,lookup=contexts[season];gid=str(data['games'][i]);pid=int(data['players'][i]);row=lookup[(gid,pid)]
        own,opp,home=_parse_matchup(row.MATCHUP)
        inputs.prediction_game_id=gid;inputs.prediction_cutoff=pd.Timestamp(side['quote_cutoffs'][i])
        inputs.recovery._cutoff_cache={};inputs.recovery._injury_cache={}
        prior=inputs.player_log(pid);prior=prior[pd.to_datetime(prior.GAME_DATE)<day]
        features,gaps=build_audited_features(prior,'PTS',pid,_team_id(own),_team_id(opp),day,home,inputs)
        for key in REFERENCE_FIELDS:
            expected=np.nan if key in gaps else features[key]
            actual=reference.iloc[i][key]
            if not (pd.isna(expected) and pd.isna(actual)) and not np.isclose(expected,actual):raise ValueError(f'Original builder mismatch {gid} {pid} {key}')
        checks.append({'row':int(i),'game_id':gid,'player_id':pid,'cutoff':side['quote_cutoffs'][i]})
    from scripts.train_points_v6 import run as train
    blocked=root/'must_not_train'
    try:train(blocked,path,reference_path)
    except ValueError as e:
        if 'Insufficient real-line' not in str(e):raise
    else:raise AssertionError('Training incorrectly allowed')
    assert not blocked.exists()
    save_new(root/'manifest.json',{'version':'points_v6_174','columns':columns,'minimum_prior_appearances':MINIMUM,
        'excluded':{'rolling_30day_avg':'Requires30 appearances, existing history cap20'},'coverage':coverage,
        'dataset_sha256':digest(path),'reference_sha256':digest(reference_path),'source_hashes':{p:digest(p) for p in ['scripts/train_points_v6.py','src/points_reference_inputs.py','src/points_v5.py',__file__]}})
    save_new(root/'comparison_check.json',{'all_rows_aligned':len(reference),'non_injury_inputs_unchanged':True,
        'original_builder_checks':checks,'training_blocked_before_fitting':True,'feature_count':len(columns),
        'entirely_missing_columns':[k for k in columns if prepared[k].isna().all()],
        'infinite_values':int(np.isinf(prepared.to_numpy(dtype=float)).sum())})
    print('PASS',len(reference),'aligned rows;',len(checks),'original-builder replays; training correctly blocked')

if __name__=='__main__':run()
