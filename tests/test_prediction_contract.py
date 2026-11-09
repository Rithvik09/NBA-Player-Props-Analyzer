import json
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from src.models import EnhancedMLPredictor


class Classifier:
    def __init__(self,value=.6,fail=False):self.value=value;self.fail=fail
    def predict_proba(self,x):
        if self.fail:raise RuntimeError('sensitive provider error text')
        return np.array([[1-self.value,self.value]])


class Regressor:
    def predict(self,x):return np.array([20.])


def predictor(fail=False,value=.6):
    p=EnhancedMLPredictor.__new__(EnhancedMLPredictor);p.feature_policy='legacy';p.prop_models={'points':{'calibrated_clf':Classifier(value,fail),'regression_model':Regressor(),'scaler':FunctionTransformer()}};p._needs_recal={};p._global_needs_recal=False;p.conf_high_threshold=.08;p.conf_med_threshold=.04
    return p


def test_nonfinite_core_is_unavailable_not_one_hundred_percent():
    for k in ['stddev','recent_avg','season_avg']:
        f={'recent_avg':20.,'season_avg':18.,'stddev':6.};f[k]=np.nan
        r=predictor().predict(f,20.5,'points')
        assert r['over_probability']==.5 and not r['prediction_available'] and r['recommendation']=='PASS'
        json.dumps(r,allow_nan=False)


def test_classifier_failure_is_explicit_and_cannot_recommend():
    r=predictor(fail=True).predict({'recent_avg':20.,'season_avg':18.,'stddev':6.},20.5,'points')
    assert r['prediction_mode']=='statistical_fallback' and r['fallback_reason']=='RuntimeError'
    assert r['recommendation']=='PASS' and r['confidence']=='LOW' and not r['model_inference_succeeded']
    assert 'sensitive' not in json.dumps(r)


def test_valid_blend_still_succeeds_and_invalid_model_does_not():
    f={'recent_avg':20.,'season_avg':18.,'stddev':6.}
    good=predictor().predict(f,20.5,'points');bad=predictor(value=np.nan).predict(f,20.5,'points')
    assert good['prediction_mode']=='per_prop_ml_blend' and good['model_inference_succeeded']
    assert bad['prediction_mode']=='statistical_fallback' and bad['recommendation']=='PASS'


def test_nightly_training_does_not_start_without_explicit_opt_in(monkeypatch):
    import ast
    import os
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock
    tree=ast.parse(Path('src/app.py').read_text())
    nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_nightly_retrain_enabled' for t in n.targets)) or (isinstance(n,ast.If) and '_nightly_retrain_enabled' in ast.unparse(n.test))]
    assert len(nodes)==2
    module=compile(ast.Module(body=nodes,type_ignores=[]),'scheduler_guard','exec')
    thread=Mock();namespace={'_background_tasks_enabled':True,'os':os,'threading':SimpleNamespace(Thread=thread),'_nightly_scheduler':lambda:None,'app':SimpleNamespace(logger=Mock())}
    monkeypatch.delenv('NIGHTLY_RETRAIN',raising=False);exec(module,namespace);thread.assert_not_called()
    monkeypatch.setenv('NIGHTLY_RETRAIN','1');exec(module,namespace);thread.return_value.start.assert_called_once()
