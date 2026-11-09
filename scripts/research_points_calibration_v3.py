"""Offline, separately versioned calibration research; no collection or base fitting."""
import argparse,json
from pathlib import Path
import joblib,numpy as np
from src.prospective_points import digest,save_new
from src.points_calibration_research_v2 import validate,select,probability
from src.calibration_independence import validate_disjoint_calibration
from src.points_v5 import prepare
from src.verified_training import classifier_frame
from scripts.points_pilot import metrics


def run(inputs,plan_path,output):
    plan=json.loads(plan_path.read_text())
    lock=json.loads(plan_path.with_name('execution_lock.json').read_text())
    if digest(plan_path)!=lock['sha256']:raise ValueError('Execution plan drift')
    if plan.get('purpose') != 'calibration_research' or set(plan['models']) != {'repaired','compact_retrained'}:
        raise ValueError('Calibration-only paired study required')
    if not set(plan['models'].values()).issubset(plan['hashes']):
        raise ValueError('Both base model hashes required')
    for path,h in plan['hashes'].items():
        if digest(path)!=h:raise ValueError('Locked source/model drift: '+path)
    bundle=joblib.load(inputs)
    # Coverage and data gates run before output creation or loading base classifiers.
    inventory=plan.get('base_training_games')
    if not inventory or inventory not in plan['hashes']:
        raise ValueError('Hash-locked base-training game inventory required')
    fitting,selection,support=validate_disjoint_calibration(bundle,json.loads(Path(inventory).read_text())['games'])
    d=bundle['data'];history=d['X'].games_played.to_numpy();y=d['y']
    output.mkdir(parents=True,exist_ok=False)
    save_new(output/'provenance.json',{'inputs_sha256':digest(inputs),'plan_sha256':digest(plan_path),'support':support,'historical_independent':False})
    report={'historical_independent':False,'promoted':False,'cohorts':{}}
    chosen={}
    for name,path in plan['models'].items():
        model=joblib.load(path)
        raw=model.classifier.predict_proba(classifier_frame(prepare(d['X']),d['lines'],model.columns))[:,1]
        if not np.isfinite(raw).all() or ((raw<0)|(raw>1)).any():raise ValueError('Invalid base probabilities')
        chosen[name]={}
        for cohort,mask in [('short_history',history<10),('eligible_history',history>=10)]:
            choice,options=select(raw,y,fitting&mask,selection&mask)
            chosen[name][cohort]=choice
            report['cohorts'].setdefault(cohort,{})[name]={'selection':choice,'options':options,'development_metrics':metrics(y[selection&mask],probability(raw[selection&mask],choice['parameters']))}
    for cohort,mask in [('short_history',history<10),('eligible_history',history>=10)]:
        m=selection&mask
        report['cohorts'][cohort]['baselines']={k:metrics(y[m],v[m]) for k,v in [('market',bundle['market']),('coin_flip',np.full(len(y),.5))]}
    save_new(output/'calibrators.json',chosen)
    save_new(output/'report.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path);p.add_argument('--plan',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.inputs,a.plan,a.output)
