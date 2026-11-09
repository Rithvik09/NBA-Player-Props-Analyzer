import joblib
import numpy as np
import pandas as pd
from scripts.investigate_points_features import folds,group_for
from scripts.train_selected_points import fit_calibration,paired_losses
from src.verified_training import verified_columns,DEFERRED_DEFENDERS,SelectedPointsCandidate


def test_temporal_development_never_uses_calibration_or_final_season():
    dates=pd.date_range('2022-10-01','2026-06-30',freq='D')
    for _,fit,dev in folds(dates):
        assert dates[fit].max()<dates[dev].min()
        assert dates[dev].max()<pd.Timestamp('2025-03-01')
        assert not np.any(fit&dev)
    assert not set(verified_columns())&DEFERRED_DEFENDERS
    assert all(group_for(c) for c in verified_columns())


def test_sigmoid_calibration_reload_and_extreme_inputs(tmp_path):
    p=np.linspace(.01,.99,100);y=(np.arange(100)%3==0).astype(int)
    cal=fit_calibration('sigmoid',p,y)
    path=tmp_path/'cal.joblib';joblib.dump(cal,path)
    expected=cal.predict([0,.3,1]);actual=joblib.load(path).predict([0,.3,1])
    np.testing.assert_allclose(actual,expected)
    assert np.isfinite(actual).all() and (actual>0).all() and (actual<1).all()


def test_game_cluster_comparison_identity_and_direction():
    y=np.array([0,1,0,1]);groups=np.array(['a','a','b','b'])
    same=paired_losses(y,np.full(4,.5),np.full(4,.5),groups)
    assert same['brier']=={'delta':0.,'interval_95':[0.,0.]}
    better=paired_losses(y,np.array([.1,.9,.1,.9]),np.full(4,.5),groups)
    assert better['log_loss']['interval_95'][1]<0


class ColumnPredictor:
    def predict(self,frame):
        assert list(frame.columns)==['reg_only']
        return frame.reg_only.to_numpy()


def test_regressor_uses_own_frozen_columns():
    model=SelectedPointsCandidate(['class_only'],ColumnPredictor(),None,None,regressor_columns=['reg_only'])
    np.testing.assert_array_equal(model.predict(pd.DataFrame({'class_only':[9],'reg_only':[2]})),[2])
