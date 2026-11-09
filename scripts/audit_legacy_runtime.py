"""Offline inventory and inference parity evidence for trusted local legacy artifacts."""
import argparse
import json
import re
import subprocess
import warnings
from pathlib import Path
import joblib
import xgboost
from src.serving_provenance import ROOT, file_hash, runtime, safe


def audit(output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    artifacts=[]
    original=xgboost.Booster.__setstate__
    versions=[]
    def capture(self,state):
        # XGBoost's embedded UBJSON Config/Model version array; no inference
        # from timestamps or the currently installed library.
        blob=bytes(state['handle'])
        for m in re.finditer(rb'version\[#L\x00\x00\x00\x00\x00\x00\x00\x03i(.)i(.)i(.)',blob,re.DOTALL):
            versions.append('.'.join(str(v[0]) for v in m.groups()))
        return original(self,state)
    xgboost.Booster.__setstate__=capture
    try:
        for p in sorted((ROOT/'models').glob('*.joblib')):
            versions.clear()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                obj=joblib.load(p)
            commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(p.relative_to(ROOT))],cwd=ROOT,text=True).strip()
            historical=subprocess.run(['git','show',f'{commit}:{p.relative_to(ROOT)}'],cwd=ROOT,capture_output=True)
            import hashlib
            artifacts.append({'path':str(p.relative_to(ROOT)),'sha256':file_hash(p),'object_type':type(obj).__module__+'.'+type(obj).__name__,'embedded_xgboost_versions':sorted(set(versions)),
                'sklearn_version_mismatches':sorted(set(getattr(w.message,'original_sklearn_version','') for w in caught)-{''}),
                'load_warnings':[str(w.message) for w in caught], 'last_artifact_commit':commit,
                'matches_commit_bytes':historical.returncode==0 and hashlib.sha256(historical.stdout).hexdigest()==file_hash(p)})
    finally: xgboost.Booster.__setstate__=original
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        from src.models import EnhancedMLPredictor
        model=EnhancedMLPredictor()
        predictions=[]
        for prop in sorted(model.prop_models):
            for mean in (5.,20.,35.):
                features={'recent_avg':mean,'season_avg':mean*.9,'stddev':max(2.,mean*.3),'games_played':20.,'mins_last5':30.,'mins_season':29.}
                for line in (mean-.5,mean+.5):
                    result=model.predict(features,line,prop)
                    if not result['model_inference_succeeded']:raise AssertionError((prop,result))
                    predictions.append({'prop':prop,'line':line,'features':features,'result':safe(result)})
    report={'runtime':runtime(),'artifacts':artifacts,'predictions':predictions,'inference_warnings':[str(w.message) for w in caught],
      'training_provenance':{'status':'partial_recovery_only','dataset_hash':None,'exact_training_rows':None,'training_cutoff':None,'original_python_version':None,'note':'Artifact commit is custody evidence, not a training timestamp or data cutoff. Embedded versions identify serialization libraries only.'}}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(f'{len(artifacts)} artifacts; {len(predictions)} successful inference checks; report {output}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);audit(p.parse_args().output)
