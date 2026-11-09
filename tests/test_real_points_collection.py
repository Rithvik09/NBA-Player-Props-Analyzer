import json
import numpy as np
from scripts.report_real_points_readiness import coverage
from scripts import collect_real_points_batch as collector
from datetime import datetime,timezone


def test_book_rows_cannot_replace_independent_games():
    d={'dates':np.array(['2024-12-01']*120),'games':np.array(['1']*120)}
    assert not coverage(d)['fit']['ready']
    assert coverage(d)['fit']['independent_games']==1


def test_budget_exhaustion_never_calls_provider(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(collector,'OUT',tmp_path/'batch')
    collector.OUT.mkdir()
    (collector.OUT/'plan.json').write_text(json.dumps({'candidates':[{'date':'20241101','id':'1','split':'fit'}]}))
    budget=tmp_path/'logs/real_points_sources/dunkapi';budget.mkdir(parents=True)
    (budget/('budget_'+datetime.now(timezone.utc).strftime('%Y-%m')+'.json')).write_text('{"reserved":500}')
    def forbidden(**kwargs):raise AssertionError('Budget bypass')
    monkeypatch.setattr(collector,'fetch',forbidden);collector.run()


def test_full_free_allowance_stops_at_500(tmp_path,monkeypatch):
    from scripts import fetch_free_historical_props as feed
    from types import SimpleNamespace
    import pytest
    monkeypatch.setattr(feed,'OUT',tmp_path)
    monkeypatch.setenv('DUNKAPI_API_KEY','test-only')
    budget=tmp_path/('budget_'+datetime.now(timezone.utc).strftime('%Y-%m')+'.json')
    budget.write_text(json.dumps({'reserved':490,'requests':[]}))
    calls=[]
    def post(url,*,json,**kwargs):
        calls.append(json)
        payload={'data':{'estimate_only':True,'credits':10}} if json.get('estimate_only') else {'data':{'props':[]}}
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:payload)
    monkeypatch.setattr(feed.requests,'post',post)
    feed.fetch(game_id='1')
    assert json.loads(budget.read_text())['reserved']==500
    with pytest.raises(ValueError,match='ceiling'):feed.fetch(game_id='2')
    assert len([c for c in calls if not c.get('estimate_only')])==1


def test_paid_batch_separate_bounded_and_no_retry():
    from scripts.collect_paid_points_batch import reserve
    import pytest
    state={'reserved':230,'attempts':{}}
    reserve(state,'last',10)
    assert state['reserved']==240
    with pytest.raises(ValueError,match='Already'):reserve(state,'last',10)
    with pytest.raises(ValueError,match='ceiling'):reserve(state,'extra',10)
    with pytest.raises(ValueError,match='estimate'):reserve({'reserved':0,'attempts':{}},'expensive',20)


def test_replacement_plan_cannot_use_full_batch_allowance():
    from scripts.collect_paid_points_batch import reserve
    import pytest
    state={'reserved':20,'attempts':{}}
    reserve(state,'third',10,limit=30)
    with pytest.raises(ValueError,match='ceiling'):reserve(state,'fourth',10,limit=30)
    assert state['reserved']==30
