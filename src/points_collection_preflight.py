"""Fail-closed spending checks for future historical research batches."""
import json,re
from pathlib import Path
import pandas as pd
from src.prospective_points import digest


def attempted_games(root=Path('logs')):
    """Retain attempts across months/batches, including uncertain failures."""
    identities=set()
    for p in (root/'real_points_sources/dunkapi').glob('budget_*.json'):
        for request in json.loads(p.read_text()).get('requests',[]):
            gid=json.loads(request).get('game_id')
            if gid:identities.add(str(gid))
    for p in root.glob('real_points_paid*/state.json'):
        identities.update(map(str,json.loads(p.read_text()).get('attempts',{})))
    for p in (root/'real_points_sources/dunkapi').glob('*.json'):
        gid=json.loads(p.read_text()).get('request',{}).get('game_id')
        if gid:identities.add(str(gid))
    return identities


def validate_plan(plan,protocol,exposed,attempted,pending=None):
    errors=[];candidates=plan.get('candidates',[]);pending=pending if pending is not None else candidates
    if protocol.get('purpose') not in ('calibration_research','fresh_holdout'):errors.append('Declare research versus fresh-holdout purpose')
    if not protocol.get('created_at') or pd.Timestamp(protocol['created_at']).tzinfo is None:errors.append('Timestamped protocol required')
    if protocol.get('quote_horizon')!='historical_closing':errors.append('This collector only supports historical closing quotes')
    if not protocol.get('required_cohorts') or 'short_history' not in protocol['required_cohorts']:errors.append('Explicit short-history calibration target required')
    if protocol.get('model_selection_rule') is None:errors.append('Lock model/calibration selection before collecting')
    budget=plan.get('credit_ceiling',0)
    if not isinstance(budget,int) or not 0<budget<=240:errors.append('Invalid batch credit ceiling')
    if len(candidates)*10>budget:errors.append('Plan cannot fit the ten-credit per-game budget')
    ids=[str(c.get('id','')) for c in candidates];games=[str(c.get('nba_game_id','')) for c in candidates]
    if len(set(ids))!=len(ids) or len(set(games))!=len(games):errors.append('Duplicate game within plan')
    dates={}
    for c in candidates:
        gid=str(c.get('nba_game_id',''));pid=str(c.get('id',''));split=c.get('split')
        if not re.fullmatch(r'(002|004|005)\d{7}',gid) or not pid.isdigit():errors.append('Canonical NBA/provider identity required')
        if split not in ('fit','calibration_fit','calibration_selection','holdout'):errors.append('Explicit chronological split required');continue
        day=pd.Timestamp(c['date'])
        if day>=pd.Timestamp('2026-10-01'):errors.append('Protected prospective season cannot enter historical research')
        dates.setdefault(split,[]).append(day)
        if split=='holdout' and gid in exposed:errors.append('Already exposed NBA game cannot be a fresh holdout: '+gid)
    if protocol.get('purpose')=='fresh_holdout' and 'holdout' not in dates:errors.append('Fresh-holdout plan has no holdout games')
    present=[k for k in ('fit','calibration_fit','calibration_selection','holdout') if k in dates]
    for a,b in zip(present,present[1:]):
        if max(dates[a])>=min(dates[b]):errors.append('Chronological split overlap')
    for c in pending:
        if str(c['id']) in attempted:errors.append('Provider game already attempted in another batch: '+str(c['id']))
    if errors:raise ValueError('; '.join(sorted(set(errors))))
    return True


def require_preflight(plan,pending,root=Path('logs')):
    if plan.get('spending_paused'):
        raise ValueError('Spending explicitly paused for this plan')
    path=plan.get('experiment_protocol')
    if not path or not plan.get('experiment_protocol_sha256'):raise ValueError('New spending blocked: locked experiment protocol and preflight required')
    if digest(path)!=plan['experiment_protocol_sha256']:raise ValueError('Experiment protocol drift')
    protocol=json.loads(Path(path).read_text());sources=protocol.get('exposure_sources',{})
    required={str(p) for p in (root/'points_v5_data').glob('20??-??.joblib')}
    if not required or not required.issubset(sources):raise ValueError('Complete research exposure inventory required')
    import joblib
    exposed=set()
    for path,expected in sources.items():
        if digest(path)!=expected:raise ValueError('Exposure source drift')
        exposed.update(map(str,joblib.load(path)['games']))
    return validate_plan(plan,protocol,exposed,attempted_games(root),pending)


def validate_response(payload,event):
    rows=payload.get('data',{}).get('props')
    if not isinstance(rows,list):raise ValueError('Missing props payload')
    for q in rows:
        if str(q.get('game_id'))!=str(event['id']) or str(q.get('game_date'))!=event['date']:
            raise ValueError('Paid response does not match planned game/date')
    return rows
