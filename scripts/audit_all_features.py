"""Reproducible one-row-per-feature audit on dated real NBA observations.

This profiles inputs, not accuracy. It does not read existing model predictions
or select features using target correlations. Unsupported columns remain named
but explicitly inactive in the versioned new-training contract.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.audited_features import SPECS, POLICY_VERSION, build_audited_features
from src.historical_inputs import HistoricalInputs,load_season_logs
from src.ml_features import NUMERIC_FEATURE_KEYS
from scripts.audit_feature_parity import feature_definitions
from scripts.points_pilot import model_hashes
from scripts.train_models import _load_precomputed_player_data, _parse_matchup, _team_id

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/full_feature_audit'


def input_hashes():
    paths=set(ROOT.glob('src/*.py')) | set(ROOT.glob('scripts/*.py'))
    for pattern in ('logs/points_pilot_cache/*.json','logs/feature_recovery/*.json',
                    'logs/feature_recovery/injury_reports/*.json',
                    'logs/feature_recovery/game_script/*.joblib'):
        paths.update(ROOT.glob(pattern))
    paths.update(ROOT.glob('logs/source_blocked/*.joblib'))
    paths.update(ROOT.glob('logs/source_blocked/*qualification.json'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def run(max_players=50):
    OUT.mkdir(parents=True,exist_ok=True)
    old=OUT/'report.json'
    if old.exists() and not (OUT/'report_v1.json').exists():
        previous=json.loads(old.read_text())
        if previous.get('policy')=='point_in_time_v1':(OUT/'report_v1.json').write_text(old.read_text())
    if old.exists():
        previous=json.loads(old.read_text())
        if previous.get('policy')=='point_in_time_v2' and not (OUT/'report_v2.json').exists():
            (OUT/'report_v2.json').write_text(old.read_text())
    before=model_hashes()
    provenance=input_hashes()
    raw=_load_precomputed_player_data(os.environ['DATABASE_URL'])
    frames=[];coverage={};datasets=[]
    for season in ('2022-23','2023-24','2024-25','2025-26'):
        logs=load_season_logs(season,include_postseason=True)
        inputs=HistoricalInputs(season,raw,logs,audited=True)
        missing=Counter();rows=[];dates=[];games=[];players=[];targets=[]
        for pid in inputs.catalog(max_players).PLAYER_ID:
            hist=inputs.player_log(pid)
            if hist.empty:continue
            hist=hist.sort_values('GAME_DATE').copy()
            hist['GAME_DATE']=pd.to_datetime(hist.GAME_DATE)
            for idx in range(10,len(hist)):
                row=hist.iloc[idx];own,opp,home=_parse_matchup(row.MATCHUP)
                inputs.prediction_game_id=str(row.GAME_ID)
                features,gaps=build_audited_features(hist.iloc[:idx],'PTS',pid,_team_id(own),_team_id(opp),row.GAME_DATE,home,inputs)
                missing.update(gaps)
                for key in gaps:
                    if SPECS[key].status=='active':features[key]=np.nan
                rows.append(features);dates.append(row.GAME_DATE);games.append(str(row.GAME_ID));players.append(int(pid));targets.append(float(row.PTS))
        X=pd.DataFrame(rows,columns=NUMERIC_FEATURE_KEYS)
        coverage[season]={'player_games':len(X),'source_gaps':dict(missing),'missing_rate':{k:missing[k]/len(X) for k in NUMERIC_FEATURE_KEYS},'prior_snapshot_season':inputs.source_season,'prior_snapshot_rows':inputs.source_rows}
        frames.append(X);datasets.append({'X':X,'y':np.array(targets),'dates':pd.to_datetime(dates),'games':np.array(games),'players':np.array(players),'feature_policy':POLICY_VERSION})
        print(season,len(X),'active_source_gaps',sum(missing.values()),flush=True)
        joblib.dump(datasets[-1],OUT/f'audited_{season}.joblib')
    matrix=pd.concat(frames,ignore_index=True);definitions=feature_definitions((ROOT/'scripts/train_models.py').read_text(),'make_features')
    rows=[]
    for index,key in enumerate(NUMERIC_FEATURE_KEYS):
        values=matrix[key];spec=SPECS[key]
        rows.append({'index':index,'feature':key,**asdict(spec),'historical_policy':'strictly earlier day; prior completed season for snapshots; current snapshots excluded' if spec.status=='active' else 'inactive in train and serve',
                     'legacy_training_definitions':definitions.get(key,[]),'audited_definition_file':'src/audited_features.py','samples':len(values),'finite':bool(np.isfinite(values.dropna()).all()),'observed_samples':int(values.notna().sum()),'native_missing':int(values.isna().sum()),'unique':int(values.nunique()),'zero_rate':float((values==0).mean()),'observed_source_gap_count':sum(c['source_gaps'].get(key,0) for c in coverage.values()),'min':float(values.min()) if values.notna().any() else None,'p01':float(values.quantile(.01)) if values.notna().any() else None,'median':float(values.median()) if values.notna().any() else None,'p99':float(values.quantile(.99)) if values.notna().any() else None,'max':float(values.max()) if values.notna().any() else None,'constant':bool(values.nunique()==1),'source_gap_rate_by_season':{s:cov['missing_rate'][key] if spec.status=='active' else None for s,cov in coverage.items()},'parity':'shared audited builder; separate caller integration covered by tests'})
    report={'policy':POLICY_VERSION,'feature_count':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),'source_coverage':coverage,'rows':rows,'serving_models_unchanged':before==model_hashes(),'limits':['Regular season plus postseason replay. Postseason team ratings use explicitly estimated possessions from complete team boxes when advanced ratings are unavailable.','Top prior-season minute cohort, not every player. Source gaps include missing prior-season rows; they are not real zeros.','Active input definitions are not proof of statistically useful or independently validated signals.','No historical bookmaker or injury archive: these columns are inactive, not backfilled. Live injuries remain visible outside this ML contract.','Prior-season snapshot use is a modeling prior; source revision/collection timestamps are not historical publication timestamps.','Source availability is measured; zero-rate alone is not a missingness estimate.','Existing serving models remain on legacy features until explicit retraining and promotion. Activated source capability does not imply full row coverage or measured predictive gain.'],
            'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/audited_features.py','src/historical_inputs.py','scripts/train_models.py','src/audited_runtime.py','src/models.py','src/precompute_jobs.py','src/basketball_betting_helper.py','scripts/audit_all_features.py','src/recovered_features.py','src/historical_injuries.py','src/pbp_features.py']}}
    report['limits'][3]='Official timestamped injury reports are used where available; missing reports and unresolved identities stay missing. Historical bookmaker inputs remain blocked.'
    report['input_hashes']=provenance
    report['database_snapshot_hash']=joblib.hash(raw)
    report['inputs_unchanged_during_audit']=provenance==input_hashes()
    assert report['inputs_unchanged_during_audit'], 'Sources changed during audit; rerun on stable inputs'
    report['fraction_range_violations']=[r['feature'] for r in rows if r['status']=='active' and r['units']=='fraction' and r['min'] is not None and (r['min']<0 or r['max']>1)]
    report['active_all_missing']=[r['feature'] for r in rows if r['status']=='active' and r['observed_samples']==0]
    report['active_constants']=[r['feature'] for r in rows if r['status']=='active' and r['constant']]
    assert not report['fraction_range_violations']
    assert len(rows)==416 and all(r['finite'] for r in rows) and report['serving_models_unchanged']
    (OUT/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    pd.DataFrame([{k:v for k,v in row.items() if k not in ('legacy_training_definitions','source_gap_rate_by_season')} for row in rows]).to_csv(OUT/'features.csv',index=False)
    lines=['# Full feature audit', '',f'Policy: `{POLICY_VERSION}`. {len(matrix):,} real player-games; all416 schema columns reviewed.', '', 'Inactive means excluded from new model inputs, not a recovered measurement.', '', '| # | Feature | Decision | Source | Units | Definition |','|---|---|---|---|---|---|']
    for row in rows:lines.append(f"| {row['index']} | {row['feature']} | {row['status']} | {row['source']} | {row['units']} | {row['definition']} |")
    (OUT/'features.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'features':len(rows),'decisions':report['status_counts'],'player_games':len(matrix),'serving_models_unchanged':report['serving_models_unchanged']}),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--max-players',type=int,default=50)
    run(parser.parse_args().max_players)
