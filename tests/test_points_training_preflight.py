import numpy as np
import pandas as pd
import pytest
from src.points_training_preflight import validate_inputs


def fixture():
    n=800;days=np.repeat(['2023-11-01','2024-01-01','2024-10-24','2025-10-24'],200)
    games=np.array([f'{i//10:010d}' for i in range(n)])
    frame=pd.DataFrame({'games_played':np.tile(np.r_[np.full(100,5),np.full(100,20)],4)})
    # Give each cohort20 independent games within each split.
    games=np.array([f'{i//200:02d}{i%100//5:08d}' for i in range(n)])
    d={'X':frame,'dates':days,'games':games,'players':np.arange(n),'lines':np.full(n,10.5),'points':np.full(n,11.),'y':np.ones(n),'provenance_validated':True}
    return {'data':d,'market':np.full(n,.5),'quotes':[{'game_date':day,'line':10.5,'over_odds':-110,'under_odds':-110} for day in days],
            'masks':{k:np.arange(n)//200==i for i,k in enumerate(['fit','calfit','selection','evaluation'])}}

def test_complete_disjoint_cohorts_pass_but_exposed_holdout_fails():
    b=fixture();assert validate_inputs(b,set())['short_history']['supported']
    with pytest.raises(ValueError,match='already exposed'):validate_inputs(b,{b['data']['games'][-1]})

def test_market_baseline_and_cohort_gates_are_mandatory():
    b=fixture();b['market'][0]=.6
    with pytest.raises(ValueError,match='Market prices'):validate_inputs(b,set())
    b=fixture();b['data']['X']['games_played']=20
    with pytest.raises(ValueError,match='calibration coverage'):validate_inputs(b,set())

def test_runner_stops_before_training_or_output(tmp_path,monkeypatch):
    import joblib
    from scripts import research_points_v8 as runner
    b=fixture();b['data']['X']['games_played']=20;path=tmp_path/'inputs.joblib';joblib.dump(b,path)
    monkeypatch.setattr(runner,'historical_exposed_games',lambda:set())
    monkeypatch.setattr(runner,'XGBClassifier',lambda **kw:pytest.fail('Fitted before preflight'))
    with pytest.raises(ValueError,match='coverage'):runner.run(path,tmp_path/'output')
    assert not (tmp_path/'output').exists()
