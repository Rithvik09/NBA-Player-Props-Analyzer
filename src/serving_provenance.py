"""Content-addressed serving identity and replay evidence, separate from studies."""
from pathlib import Path
import hashlib,json,platform,math
from importlib.metadata import version,PackageNotFoundError
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['src/app.py','src/db.py','src/prediction_schedule.py','src/models.py','src/ml_features.py','src/feature_context.py','src/audited_runtime.py','src/basketball_betting_helper.py','src/prediction_contract.py','src/serving_provenance.py','src/ml_quantile.py','src/ml_stacking.py']


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [safe(v) for v in value]
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if hasattr(value,'isoformat'):return value.isoformat()
    raise TypeError('Unserializable provenance value: '+type(value).__name__)


def fingerprint(value):return hashlib.sha256(json.dumps(safe(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime():
    out={'python':platform.python_version()}
    for name in ['numpy','pandas','scipy','scikit-learn','xgboost','joblib']:
        try:out[name]=version(name)
        except PackageNotFoundError:out[name]=None
    return out


def build_manifest(directory):
    directory=Path(directory).resolve();meta_path=directory/'model_metadata.json'
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    body={'schema_version':1,'artifact_hashes':{p.name:file_hash(p) for p in sorted(directory.iterdir()) if p.is_file() and p.suffix in ['.joblib','.json','.txt']},'source_hashes':{p:file_hash(ROOT/p) for p in SOURCES},'runtime':runtime(),'feature_policy':meta.get('feature_policy','legacy'),'training_outcomes_through':{p:v.get('training_outcomes_through') for p,v in meta.get('props',{}).items()},'training_environment':'unknown unless recorded by training metadata','provenance_status':'observed_existing_artifacts_not_retrospective_training_certification','prediction_contract':'EnhancedMLPredictor.predict:25% statistical regression anchor;75% regressor; probability25% regressor-distribution and75% classifier; fallback explicitly marked'}
    return {'manifest_id':fingerprint(body),**body}


def verify_manifest(manifest,directory):
    body={k:v for k,v in manifest.items() if k!='manifest_id'}
    if fingerprint(body)!=manifest.get('manifest_id'):raise ValueError('Serving manifest content drift')
    current=build_manifest(directory)
    if current!=manifest:raise ValueError('Serving artifact, source or runtime drift; restart with an explicitly rebuilt manifest')


def evidence(manifest,features,line,prop_type,result):
    payload={'manifest_id':manifest['manifest_id'],'features':safe(features),'line':float(line),'prop_type':prop_type,'result':safe(result)}
    return {'evidence_id':fingerprint(payload),**payload}


def verify_evidence(record):
    if fingerprint({k:v for k,v in record.items() if k!='evidence_id'})!=record.get('evidence_id'):raise ValueError('Prediction evidence drift')
