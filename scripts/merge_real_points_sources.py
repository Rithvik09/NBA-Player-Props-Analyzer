"""Merge qualified research datasets without duplicating player-games or changing models."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.prospective_points import digest,save_new
from scripts.report_real_points_readiness import run as readiness


def run(public_path=Path('logs/real_points_public_01/training/real_points.joblib'),out=Path('logs/real_points_merged_03'), sources=None):
    sources=sources or [Path('logs/real_points_batch_02/training/real_points.joblib'),public_path]
    out.mkdir(exist_ok=False)
    seen=set();parts=[];provenance=[];policy=None;duplicates=0
    for source in sources:
        d=joblib.load(source)
        if not d.get('provenance_validated'):raise ValueError('Unqualified source')
        if policy is not None and policy!=d['feature_policy']:raise ValueError('Feature policy mismatch')
        policy=d['feature_policy'];keep=[]
        for i,(g,p) in enumerate(zip(d['games'],d['players'])):
            key=(str(g),int(p))
            if key in seen:duplicates+=1;continue
            seen.add(key);keep.append(i);provenance.append(str(source))
        parts.append({k:(v.iloc[keep].reset_index(drop=True) if k=='X' else v[keep]) for k,v in d.items() if k in ['X','y','dates','games','players','lines','points']})
    merged={'X':pd.concat([d['X'] for d in parts],ignore_index=True),'feature_policy':policy,'provenance_validated':True}
    for k in ['y','dates','games','players','lines','points']:merged[k]=np.concatenate([d[k] for d in parts])
    joblib.dump(merged,out/'real_points.joblib')
    save_new(out/'provenance.json',{'sources':{str(p):digest(p) for p in sources},'row_sources':provenance,'duplicates_skipped':duplicates,'selection':'Preserve existing archive for overlapping NBA player-games; never use outcome to select source.'})
    readiness(out/'real_points.joblib')

if __name__=='__main__':run()
