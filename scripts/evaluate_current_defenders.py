"""One-shot current-season qualification of the frozen assignment ranker."""
import gzip
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.defender_assignment import state_before,load_logs
from src.defender_evidence import expected_defender
from scripts.qualify_defender_rules import summarize

OUT=Path('logs/source_blocked/assignment_ranker')


def run():
    if (OUT/'holdout.json').exists():raise RuntimeError('Holdout already opened; do not retune and reuse it')
    frozen=json.loads((OUT/'frozen.json').read_text())
    assert hashlib.sha256((OUT/'model.joblib').read_bytes()).hexdigest()==frozen['model_sha256']
    assert hashlib.sha256(Path('src/defender_assignment.py').read_bytes()).hexdigest()==frozen['source_sha256']
    if frozen['selection'] is None:raise RuntimeError('No development-qualified candidate')
    model=joblib.load(OUT/'model.joblib');threshold=frozen['selection']['threshold']
    current=[]
    for path in sorted(Path('logs/source_blocked/current_matchups_holdout').glob('*.gz')):
        obj=json.load(gzip.open(path,'rt'))
        assert obj['game_id'].startswith('00226')
        current.extend({**r,'game_id':obj['game_id'],'game_date':pd.Timestamp(obj['game_date'])} for r in obj['rows'])
    frame=pd.DataFrame(current)
    assert not frame.empty and frame.game_date.min()>=pd.Timestamp('2026-10-01')
    raw=json.loads(Path('logs/source_blocked/current_player_logs_raw.json').read_text())['resultSets'][0]
    boxes=pd.DataFrame(raw['rowSet'],columns=raw['headers']);boxes['GAME_DATE']=pd.to_datetime(boxes.GAME_DATE)
    boxes['GAME_ID']=boxes.GAME_ID.astype(str).str.zfill(10)
    historical=joblib.load('logs/source_blocked/defender_evidence.joblib')['rows']
    # The canonical bundle may already contain the imported holdout. Start
    # before it so its games are ingested exactly once during replay.
    historical=historical[historical.game_id.str.startswith('002') &
                          (historical.game_date<frame.game_date.min())]
    state=state_before(historical,load_logs(),frame.game_date.min())
    records=[];all_cases=0;old_cases=[]
    old_rule=json.loads(Path('logs/source_blocked/defender_qualification.json').read_text())['qualifications']['2026-27']['rule']
    prior=historical
    matchdays=dict(tuple(frame.groupby('game_date')));boxdays=dict(tuple(boxes.groupby('GAME_DATE')))
    for day in sorted(set(matchdays)|set(boxdays)):
        if day in matchdays:
            part=matchdays[day];matrix=[];groups=[]
            for (gid,pid,opp),actual in part.groupby(['game_id','personIdOff','defTeamId']):
                all_cases+=1
                target=int(actual.groupby('personIdDef').partialPossessions.sum().idxmax())
                ids,x=state.candidates(pid,opp,day)
                baseline=ids[0] if ids else None
                old=expected_defender(prior,player_id=pid,opponent_id=opp,day=day,
                    latest_teams={p:t for p,(t,d) in state.latest.items()},rule=old_rule)
                if old is not None:old_cases.append({'correct':old==target,'baseline_correct':baseline==target})
                if not ids:continue
                groups.append((len(matrix),len(matrix)+len(x),ids,int(pid),gid,target))
                matrix.extend(x)
            scores=model.predict_proba(np.asarray(matrix,dtype=np.float32))[:,1] if matrix else []
            for start,end,ids,pid,gid,target in groups:
                values=scores[start:end];i=int(np.argmax(values))
                if values[i]<threshold:continue
                records.append({'date':str(day.date()),'game_id':gid,'player_id':pid,
                    'prediction':ids[i],'target':target,'score':float(values[i]),
                    'correct':ids[i]==target,'baseline_correct':ids[0]==target})
            state.observe_matchups(part,day)
            prior=pd.concat([prior,part],ignore_index=True)
        if day in boxdays:state.observe_boxes(boxdays[day],day)
    stats=summarize(records)
    qualified=bool(stats['n']>=100 and stats['accuracy']>=.5 and stats['accuracy']>=stats['minutes_baseline_accuracy']+.05)
    result={**stats,'qualified':qualified,'threshold':threshold,'total_player_matchups':all_cases,
        'coverage':len(records)/all_cases,'games':int(frame.game_id.nunique()),
        'first_outcome_date':str(frame.game_date.min().date()),'last_outcome_date':str(frame.game_date.max().date()),
        'evaluated_at':datetime.now(timezone.utc).isoformat(),'model_sha256':frozen['model_sha256'],
        'old_rule_same_games':{**summarize(old_cases),'coverage':len(old_cases)/all_cases},'predictions':records}
    (OUT/'holdout.json').write_text(json.dumps(result,indent=2))
    # Plain containers avoid pickling the state's local deque factory.
    joblib.dump({'boxes':dict(state.boxes),'latest':state.latest,'pairs':state.pairs,
                 'guards':state.guards,'through':str(boxes.GAME_DATE.max().date())},OUT/'state.joblib')
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2),flush=True)


if __name__=='__main__':run()
