"""Offline fresh-process refit from preserved neutral-volume experiment sources."""
import argparse
from pathlib import Path
import subprocess,sys
from scripts.replay_traceable_points import check_inventory


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=check_inventory(args.output)
    code='''import sys,json,socket,subprocess
from pathlib import Path
socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Network forbidden'))
sys.path.insert(0,sys.argv[1]);out=Path(sys.argv[2])
if subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)!=(out/'requirements.lock').read_text():raise ValueError('Environment drift')
import joblib,numpy as np,pandas as pd
from src.neutral_volume_correction import fit_volume
b=joblib.load(out/'inputs.joblib');d=b['data'];x=d['X'];expected=json.loads((out/'predictions.json').read_text());dates=pd.to_datetime(d['dates']);test=np.asarray(dates>='2025-07-01');train=~test
m=fit_volume(x.loc[train],d['lines'][train],b['market'][train],d['y'][train],d['games'][train])
if not np.array_equal(m.predict_over(x.loc[test],d['lines'][test],b['market'][test]),expected['fixed_volume']):raise ValueError('Fixed refit mismatch')
pred=np.full(len(x),np.nan);folds=json.loads((out/'folds.json').read_text())
for fold in folds:
 fit=np.array(fold['fit_indices']);val=np.array(fold['validation_indices'])
 if not (dates[fit].max()<dates[val].min()):raise ValueError('Temporal leakage')
 model=fit_volume(x.iloc[fit],d['lines'][fit],b['market'][fit],d['y'][fit],d['games'][fit])
 pred[val]=model.predict_over(x.iloc[val],d['lines'][val],b['market'][val])
if not np.array_equal(pred[test],expected['walk_forward_volume']):raise ValueError('Walk-forward refit mismatch')
print(json.dumps({'exact_fixed_refit':True,'exact_walk_forward_refit':True,'folds':len(folds),'rows':int(test.sum()),'strictly_earlier_dates':True},indent=2))
'''
    subprocess.run([sys.executable,'-W','error::RuntimeWarning','-c',code,str(out/'source'),str(out)],cwd=out,check=True)

if __name__=='__main__':main()
