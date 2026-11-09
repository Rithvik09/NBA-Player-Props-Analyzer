"""Build a checkpointed, prior-only expanded points cohort from free cached data."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,os,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.audited_features import build_audited_features,SPECS,POLICY_VERSION
from src.historical_inputs import HistoricalInputs,load_season_logs
from src.ml_features import NUMERIC_FEATURE_KEYS
from scripts.train_models import _load_precomputed_player_data,_parse_matchup,_team_id
from scripts.points_pilot import model_hashes

SEASONS=['2022-23','2023-24','2024-25','2025-26']


def eligible(history, original_cohort=False):
    # Current game's minutes, points, and future season totals never select a row.
    return len(history)>=10 and (original_cohort or history.tail(20).MIN.mean()>=12)


def build_season(season,output):
    output=Path(output);raw=joblib.load(output/'source_snapshot.joblib')
    logs=load_season_logs(season,include_postseason=True)
    inputs=HistoricalInputs(season,raw,logs,audited=True)
    original=set(map(int,inputs.catalog(50).PLAYER_ID))
    checkpoint=output/season;checkpoint.mkdir(exist_ok=True)
    start=time.monotonic();parts=[];count=0
    for pid in sorted(inputs.logs):
        path=checkpoint/f'{pid}.joblib'
        if path.exists():part=joblib.load(path)
        else:
            hist=inputs.player_log(pid).sort_values('GAME_DATE').copy()
            hist['GAME_DATE']=pd.to_datetime(hist.GAME_DATE)
            rows=[];targets=[];dates=[];games=[]
            for i in range(10,len(hist)):
                prior=hist.iloc[:i];row=hist.iloc[i]
                if not eligible(prior,pid in original):continue
                own,opp,home=_parse_matchup(row.MATCHUP);inputs.prediction_game_id=str(row.GAME_ID)
                features,gaps=build_audited_features(prior,'PTS',pid,_team_id(own),_team_id(opp),row.GAME_DATE,home,inputs)
                for key in gaps:
                    if SPECS[key].status=='active':features[key]=np.nan
                rows.append(features);targets.append(float(row.PTS));dates.append(row.GAME_DATE);games.append(str(row.GAME_ID))
            part={'X':pd.DataFrame(rows,columns=NUMERIC_FEATURE_KEYS),'y':np.asarray(targets),
                'dates':pd.to_datetime(dates),'games':np.asarray(games),'players':np.full(len(rows),pid),
                'original_cohort':np.full(len(rows),pid in original),'feature_policy':POLICY_VERSION}
            temp=path.with_suffix('.tmp');joblib.dump(part,temp);temp.replace(path)
        if len(part['X']):parts.append(part);count+=len(part['X'])
        if len(parts) and len(parts)%20==0:print(season,len(parts),'players',count,'rows',round(time.monotonic()-start),'seconds',flush=True)
    data={'X':pd.concat([p['X'] for p in parts],ignore_index=True),'feature_policy':POLICY_VERSION}
    for k in ['y','dates','games','players','original_cohort']:data[k]=np.concatenate([p[k] for p in parts])
    assert not pd.DataFrame({'player':data['players'],'game':data['games']}).duplicated().any()
    assert np.isfinite(data['X'].to_numpy(dtype=float)[~data['X'].isna().to_numpy()]).all()
    joblib.dump(data,output/f'audited_{season}.joblib')
    print('COMPLETE',season,count,flush=True)
    return {'season':season,'player_games':count,'players':len(parts),'original_cohort_rows':int(data['original_cohort'].sum())}


def fingerprints():
    paths=[Path(p) for p in ['src/audited_features.py','src/recovered_features.py','src/historical_inputs.py','scripts/build_expanded_points.py']]
    for pattern in ['logs/points_pilot_cache/*.json','logs/feature_recovery/**/*.json','logs/feature_recovery/**/*.joblib','logs/precompute_cache/player_averages_*.json','logs/source_blocked/defender_evidence.joblib','logs/source_blocked/*qualification.json']:
        paths.extend(Path('.').glob(pattern))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def run(output,workers):
    before=model_hashes();hashes=fingerprints()
    if output.exists():
        manifest=json.loads((output/'manifest.json').read_text())
        if manifest['input_hashes']!=hashes or manifest['feature_policy']!=POLICY_VERSION:raise ValueError('Cannot resume with changed inputs or policy')
    else:
        output.mkdir(parents=True)
        raw=_load_precomputed_player_data(os.environ['DATABASE_URL'])
        joblib.dump(raw,output/'source_snapshot.joblib')
        manifest={'feature_policy':POLICY_VERSION,'input_hashes':hashes,'database_snapshot_hash':joblib.hash(raw),
            'eligibility':'At least10 prior season appearances and prior20 mean minutes>=12; retain original top50 prior-season-minute cohort for matched evaluation. No current/future outcomes select rows.',
            'seasons':SEASONS,'source':'Existing free NBA cached data; no paid calls.'}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(build_season,SEASONS,[str(output)]*len(SEASONS)))
    assert before==model_hashes()
    assert hashes==fingerprints(),'Source inputs changed during build'
    (output/'report.json').write_text(json.dumps({'seasons':results,'serving_models_unchanged':True},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();run(args.output,args.workers)
