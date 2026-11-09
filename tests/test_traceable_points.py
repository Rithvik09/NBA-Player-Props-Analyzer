import copy
import json
from pathlib import Path
import sys
import joblib
import pytest
from scripts.replay_traceable_points import check_inventory
from scripts.train_traceable_points import masks
from src.serving_provenance import file_hash


def test_inventory_rejects_model_changes_and_runtime_drift(tmp_path):
    model=tmp_path/'model.json';model.write_text('{}')
    manifest={'python':sys.version,'files':{'model.json':file_hash(model)}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest));check_inventory(tmp_path)
    model.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='file drift'):check_inventory(tmp_path)
    manifest['python']='different';(tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='runtime drift'):check_inventory(tmp_path)


@pytest.fixture
def bundle():
    p=Path('logs/points_evidence_representation_01/inputs.joblib')
    if not p.exists():pytest.skip('Local research data required')
    return joblib.load(p)


def test_split_rejects_duplicate_outcomes_and_protected_season(bundle):
    train,test=masks(bundle)
    assert not (train&test).any()
    bad=copy.deepcopy(bundle)
    for k in ('games','players'):bad['data'][k][1]=bad['data'][k][0]
    with pytest.raises(ValueError,match='Duplicate'):masks(bad)
    bad=copy.deepcopy(bundle);bad['data']['dates'][0]='2026-10-15'
    with pytest.raises(ValueError,match='Protected'):masks(bad)


def test_split_rejects_quote_market_or_target_mismatch(bundle):
    bad=copy.deepcopy(bundle);bad['market'][0]=.999
    with pytest.raises(ValueError,match='Market mismatch'):masks(bad)
    bad=copy.deepcopy(bundle);bad['data']['y'][0]=1-bad['data']['y'][0]
    with pytest.raises(ValueError,match='Label mismatch'):masks(bad)
