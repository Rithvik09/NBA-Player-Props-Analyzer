import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from src.points_reference_inputs import legacy_injuries,aligned_reference


def test_original_unknown_role_zero_retained_only_for_reference():
    ctx=SimpleNamespace(injuries=lambda t,d:[{'player':'rookie','status':'Out'}],resolve_name=lambda n:1,role_minutes=lambda p,t,d:0.)
    assert legacy_injuries(ctx,1,2,'2025-01-01')=={'team_key_players_out':0.,'opp_key_players_out':0.,'opp_injury_impact':0.}
    ctx.resolve_name=lambda n:None
    assert all(np.isnan(v) for v in legacy_injuries(ctx,1,2,'2025-01-01').values())


def test_reference_alignment_and_hash_fail_closed():
    d={'X':pd.DataFrame({'a':[1,2]}),'games':np.array(['a','b']),'players':np.array([1,2]),'dates':np.array(['2025-01-01','2025-01-02'])}
    ref={**d,'dataset_sha256':'abc'}
    assert aligned_reference(d,ref,'abc').equals(d['X'])
    with pytest.raises(ValueError,match='hash'):aligned_reference(d,ref,'other')
    with pytest.raises(ValueError,match='alignment'):aligned_reference(d,{**ref,'players':np.array([2,1])},'abc')


def test_comparison_routes_frozen_and_repaired_inputs_separately():
    from src.points_reference_inputs import predict_comparison
    class Model:
        def predict_over(self,frame,lines):return frame.team_key_players_out.to_numpy()
    candidate=pd.DataFrame({'team_key_players_out':[np.nan,2.]})
    reference=pd.DataFrame({'team_key_players_out':[0.,1.]})
    result=predict_comparison({'frozen':Model(),'new':Model()},candidate,reference,[10.5,20.5],{'frozen'})
    assert result['frozen'].tolist()==[0.,1.]
    assert np.isnan(result['new'][0]) and result['new'][1]==2.
    assert candidate.iloc[1,0]==2. and reference.iloc[1,0]==1.


def test_reference_noninjury_changes_rejected():
    d={'X':pd.DataFrame({'a':[1.]}),'games':np.array(['a']),'players':np.array([1]),'dates':np.array(['2025-01-01'])}
    ref={**d,'X':pd.DataFrame({'a':[2.]}),'dataset_sha256':'abc'}
    with pytest.raises(ValueError,match='non-injury'):aligned_reference(d,ref,'abc')


def test_retrained_compact_uses_available_candidate_matrix_columns():
    from scripts.train_points_v6 import candidate_groups
    old={'frozen161':SimpleNamespace(columns=['recent_avg','rolling_30day_avg','extra']),
         'frozen_compact':SimpleNamespace(columns=['recent_avg','rolling_30day_avg'])}
    groups=candidate_groups(old)
    assert set(groups['compact_retrained']).issubset(groups['repaired'])
    assert all('rolling_30day_avg' not in cols for cols in groups.values())
    assert 'rolling_30day_avg' in old['frozen_compact'].columns
