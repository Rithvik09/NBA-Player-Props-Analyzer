"""Freeze a broader defender ranker before opening current-season labels.

Fit through 2024-09-30; select coverage cutoff on 2024-25 and 2025-26.
These historical periods are development, not a new untouched holdout.
Qualification must use subsequently collected 2026-27 assignments, once.
"""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.defender_assignment import AssignmentState,FEATURES,load_logs
from scripts.qualify_defender_rules import summarize

OUT=Path('logs/source_blocked/assignment_ranker')


def dataset():
    bundle=joblib.load('logs/source_blocked/defender_evidence.joblib')
    frame=bundle['rows'];frame=frame[frame.game_id.str.startswith('002') & (frame.game_date<pd.Timestamp('2026-10-01'))]
    logs=load_logs();logs=logs[logs.GAME_DATE<pd.Timestamp('2026-10-01')]
    boxes=dict(tuple(logs.groupby('GAME_DATE')));matches=dict(tuple(frame.groupby('game_date')))
    state=AssignmentState();features=[];labels=[];groups=[];n=0
    for day in sorted(set(boxes)|set(matches)):
        if day in matches:
            part=matches[day]
            for (gid,pid,opp),actual in part.groupby(['game_id','personIdOff','defTeamId']):
                ids,x=state.candidates(pid,opp,day)
                if not ids:continue
                target=int(actual.groupby('personIdDef').partialPossessions.sum().idxmax())
                groups.append({'date':str(day.date()),'game_id':gid,'player_id':int(pid),
                    'opponent_id':int(opp),'ids':ids,'target':target,'start':n,'end':n+len(ids)})
                features.extend(x);labels.extend([int(d==target) for d in ids]);n+=len(ids)
            state.observe_matchups(part,day)
        if day in boxes:state.observe_boxes(boxes[day],day)
        if day.month==4 and day.day==15:print(str(day.date()),len(groups),flush=True)
    data={'X':np.asarray(features,dtype=np.float32),'y':np.asarray(labels,dtype=np.uint8),'groups':groups}
    joblib.dump(data,OUT/'development.joblib',compress=3)
    return data


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen.json').exists():raise RuntimeError('Candidate already frozen; do not tune against its holdout')
    data=joblib.load(OUT/'development.joblib') if (OUT/'development.joblib').exists() else dataset()
    train=np.zeros(len(data['y']),dtype=bool)
    for g in data['groups']:
        if g['date']<'2024-10-01':train[g['start']:g['end']]=True
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=10,
        learning_rate=.08,min_samples_leaf=100,random_state=42,early_stopping=False)
    model.fit(data['X'][train],data['y'][train]);scores=model.predict_proba(data['X'])[:,1]
    rows=[]
    for g in data['groups']:
        if g['date']<'2024-10-01':continue
        score=scores[g['start']:g['end']];i=int(np.argmax(score))
        rows.append({**g,'prediction':g['ids'][i],'score':float(score[i]),
            'correct':g['ids'][i]==g['target'],'baseline_correct':g['ids'][0]==g['target']})
    candidates=[]
    for cutoff in np.arange(.1,.91,.025):
        kept=[r for r in rows if r['score']>=cutoff]
        stats=summarize(kept)
        candidates.append({'threshold':float(cutoff),**stats})
    # Extra development buffer; final qualification still uses unchanged gates.
    eligible=[c for c in candidates if c['n']>=100 and c['accuracy']>=.55
              and c['accuracy']>=c['minutes_baseline_accuracy']+.05]
    chosen=max(eligible,key=lambda c:c['n']) if eligible else None
    joblib.dump(model,OUT/'model.joblib')
    result={'frozen_at':datetime.now(timezone.utc).isoformat(),'features':FEATURES,
        'fit_outcomes_before':'2024-10-01','selection_outcomes_before':'2026-10-01',
        'selection':chosen,'development_candidates':candidates,
        'model_sha256':hashlib.sha256((OUT/'model.joblib').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path('src/defender_assignment.py').read_bytes()).hexdigest(),
        'qualification_gates':{'min_forecasts':100,'min_accuracy':.5,'min_baseline_margin':.05},
        'qualified':False,'holdout':'Unopened 2026-27 regular-season assignments; no re-selection after evaluation.'}
    (OUT/'frozen.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='development_candidates'},indent=2),flush=True)


if __name__=='__main__':run()
