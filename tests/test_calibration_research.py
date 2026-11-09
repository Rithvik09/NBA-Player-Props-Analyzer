import json
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
import pytest
from src.points_calibration_research import validate,select,fit,probability
from src.prospective_points import digest


def bundle():
    n=400
    dates=np.array(['2024-11-01']*200+['2025-11-01']*200)
    history=np.tile(np.repeat([3,12],100),2)
    y=np.tile([0,1],200)
    d={'X':pd.DataFrame({'games_played':history}),'games':np.array([f'g{i//5}' for i in range(n)]),'players':np.arange(n),'dates':dates,'lines':np.full(n,20.5),'points':np.where(y,25,15),'y':y,'provenance_validated':True}
    q=[{'line':20.5,'game_date':day,'over_odds':-110,'under_odds':-110} for day in dates]
    return {'data':d,'quotes':q,'market':np.full(n,.5)}


def test_supported_development_does_not_require_unexposed_outcomes():
    f,s,support=validate(bundle())
    assert f.sum()==s.sum()==200
    assert all(v['supported'] for v in support.values())


@pytest.mark.parametrize('defect',['protected','duplicate','price','label','history','quote'])
def test_invalid_data_rejected(defect):
    b=bundle();d=b['data']
    if defect=='protected':d['dates'][0]='2026-10-26'
    if defect=='duplicate':d['players'][1]=d['players'][0]
    if defect=='price':b['quotes'][0]['over_odds']=0
    if defect=='label':d['y'][0]=1
    if defect=='history':d['X'].loc[0,'games_played']=0
    if defect=='quote':b['quotes'][0]['line']=19.5
    with pytest.raises(ValueError):validate(b)


def test_selection_does_not_refit_and_slope_cannot_invert():
    raw=np.tile([.2,.8],100);y=np.tile([0,1],100);f=np.arange(200)<100;s=~f
    choice,options=select(raw,y,f,s)
    for o in options:assert o['parameters']==fit(raw[f],y[f],o['kind'])
    assert fit(raw,1-y,'intercept_slope')[1]>=0
    assert np.isfinite(probability(np.array([0.,1.]),[0,1])).all()


class Classifier:
    def predict_proba(self,x):
        p=np.full(len(x),.5);return np.c_[1-p,p]


def setup(tmp_path,b):
    source=tmp_path/'input.joblib';joblib.dump(b,source)
    model=tmp_path/'model.joblib';joblib.dump(SimpleNamespace(classifier=Classifier(),columns=[]),model)
    plan=tmp_path/'execution_plan.json';plan.write_text(json.dumps({'purpose':'calibration_research','split_boundary':'2025-07-01','models':{'repaired':str(model),'compact_retrained':str(model)},'hashes':{str(model):digest(model)}}))
    (tmp_path/'execution_lock.json').write_text(json.dumps({'sha256':digest(plan)}))
    return source,plan


def test_runner_blocks_before_output_and_model_load(tmp_path,monkeypatch):
    import scripts.research_points_calibration as runner
    b=bundle();b['data']['X']['games_played']=3
    source,plan=setup(tmp_path,b);original=joblib.load
    def guarded(path):
        assert str(path)==str(source),'Base model loaded before gate'
        return original(path)
    monkeypatch.setattr(runner.joblib,'load',guarded)
    with pytest.raises(ValueError,match='coverage'):runner.run(source,plan,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_runner_success_is_research_only_and_never_base_fits(tmp_path,monkeypatch):
    import scripts.research_points_calibration as runner
    source,plan=setup(tmp_path,bundle())
    monkeypatch.setattr(runner,'prepare',lambda x:x)
    monkeypatch.setattr(runner,'classifier_frame',lambda x,*args:x)
    report=runner.run(source,plan,tmp_path/'out')
    assert report['historical_independent'] is False and report['promoted'] is False
    assert (tmp_path/'out/calibrators.json').exists()
    with pytest.raises(FileExistsError):runner.run(source,plan,tmp_path/'out')


def test_paused_collection_is_not_executable():
    from src.points_collection_preflight import require_preflight
    with pytest.raises(ValueError,match='explicitly paused'):
        require_preflight({'spending_paused':True},[])


def test_model_drift_rejected_before_output(tmp_path):
    from scripts.research_points_calibration import run
    source,plan=setup(tmp_path,bundle())
    (tmp_path/'model.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError,match='drift'):run(source,plan,tmp_path/'out')
    assert not (tmp_path/'out').exists()
