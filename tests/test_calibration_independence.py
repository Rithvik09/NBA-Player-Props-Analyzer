import pytest
from tests.test_calibration_research import bundle
from src.calibration_independence import validate_disjoint_calibration


def test_same_game_different_player_still_blocks():
    b=bundle()
    with pytest.raises(ValueError,match='overlap base training'):
        validate_disjoint_calibration(b,{'g0'})


def test_selection_overlap_blocks_too():
    with pytest.raises(ValueError,match='overlap base training'):
        validate_disjoint_calibration(bundle(),{'g70'})


def test_disjoint_calibration_passes_unchanged_support_gate():
    f,s,_=validate_disjoint_calibration(bundle(),{'earlier_game'})
    assert f.sum()==s.sum()==200
