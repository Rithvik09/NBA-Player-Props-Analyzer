"""Read-only replay of a stored serving feature vector and model identity."""
import argparse,json
from src import db
from src.models import EnhancedMLPredictor
from src.serving_provenance import verify_manifest,verify_evidence,fingerprint


def replay(prediction_id,dsn=None):
    with db.connect(dsn) as c:
        row=c.execute('SELECT e.evidence,m.manifest FROM prediction_evidence e JOIN serving_manifests m USING(manifest_id) WHERE e.prediction_id=%s',(prediction_id,)).fetchone()
    if row is None:raise ValueError('Prediction has no reproducible serving evidence')
    record,manifest=row;verify_evidence(record)
    if record['manifest_id'] != manifest['manifest_id']:
        raise ValueError('Prediction evidence references a different serving manifest')
    predictor=EnhancedMLPredictor();verify_manifest(manifest,predictor.model_dir)
    result=predictor.predict(record['features'],record['line'],record['prop_type'])
    if fingerprint(result)!=fingerprint(record['result']):raise ValueError('Replayed result differs')
    return {'prediction_id':prediction_id,'manifest_id':manifest['manifest_id'],'evidence_id':record['evidence_id'],'exact_match':True,'result':result}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prediction_id',type=int);a=p.parse_args();print(json.dumps(replay(a.prediction_id),indent=2))
