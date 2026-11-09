import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier
from src.points_training_repair import keep_synthetic,outcome_weights,assert_disjoint_games,monotone_constraints


def test_real_outcome_replaces_every_synthetic_expansion():
    keep=keep_synthetic(['a','a','b'],[1,2,1],['a'],[1])
    assert keep.tolist()==[False,True,True]
    with pytest.raises(ValueError,match='Duplicate'):keep_synthetic(['a','a'],[1,1],[],[])


def test_total_weights_are_per_outcome_not_source_block():
    indices=np.array([0,0,0,1,1]);w=outcome_weights(indices,2)
    assert np.allclose([w[:3].sum(),w[3:5].sum(),*w[5:]],[1,1,1,1])
    assert w.max()==1


def test_game_overlap_blocks_even_for_different_players():
    with pytest.raises(ValueError,match='crosses'):assert_disjoint_games(['a'],['a'],['b'])
    assert_disjoint_games(['a'],['b'],['c'])


def test_tree_remains_monotone_across_line_and_all_derived_edges():
    rng=np.random.default_rng(7);lines=rng.uniform(1,35,300)
    def frame(lines):return pd.DataFrame({'decision_line':lines,'recent_minus_line':20-lines,'season_minus_line':19-lines,'standardized_edge':(20-lines)/3,'context':np.ones(len(lines))})
    x=frame(lines);y=rng.integers(0,2,len(x))
    m=XGBClassifier(n_estimators=20,max_depth=3,n_jobs=1,monotone_constraints=monotone_constraints(x.columns)).fit(x,y)
    p=m.predict_proba(frame(np.linspace(1,35,300)))[:,1]
    assert (np.diff(p)<=1e-7).all()
    with pytest.raises(ValueError):monotone_constraints(['decision_line'])
