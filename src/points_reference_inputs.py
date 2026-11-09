"""Original injury semantics for immutable reference models, never serving mutations."""
import numpy as np

REFERENCE_FIELDS=('team_key_players_out','opp_key_players_out','opp_injury_impact')


def legacy_injuries(ctx, team, opponent, date):
    result={k:np.nan for k in REFERENCE_FIELDS}
    for prefix,tid in [('team',team),('opp',opponent)]:
        report=ctx.injuries(int(tid),date)
        if report is None:continue
        roles=[]; unresolved=False
        for row in report:
            if row['status']!='Out':continue
            pid=ctx.resolve_name(row['player'])
            if pid is None:unresolved=True;continue
            roles.append(ctx.role_minutes(pid,int(tid),date))
        if unresolved:continue
        result[prefix+'_key_players_out']=float(sum(m>=20 for m in roles))
        if prefix=='opp':result['opp_injury_impact']=min(1.,sum(roles)/240)
    return result


def aligned_reference(data, sidecar, dataset_sha256):
    if sidecar['dataset_sha256']!=dataset_sha256:raise ValueError('Reference dataset hash mismatch')
    for key in ('games','players','dates'):
        if not np.array_equal(data[key],sidecar[key]):raise ValueError('Reference row alignment mismatch: '+key)
    if len(sidecar['X'])!=len(data['X']):raise ValueError('Reference length mismatch')
    other=[k for k in data['X'].columns if k not in REFERENCE_FIELDS]
    if list(sidecar['X'].columns)!=list(data['X'].columns) or not sidecar['X'][other].equals(data['X'][other]):
        raise ValueError('Reference changed non-injury inputs')
    from src.prospective_points import digest
    for path,expected in sidecar.get('source_hashes',{}).items():
        if digest(path)!=expected:raise ValueError('Reference source hash mismatch: '+path)
    return sidecar['X']


def predict_comparison(models, candidate_frame, reference_frame, lines, reference_names):
    if len(candidate_frame)!=len(reference_frame) or len(lines)!=len(candidate_frame):
        raise ValueError('Matched comparison length mismatch')
    return {name:model.predict_over(reference_frame if name in reference_names else candidate_frame,lines)
            for name,model in models.items()}
