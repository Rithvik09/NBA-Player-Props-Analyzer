import json
import numpy as np
from scripts.report_real_points_readiness import coverage
from scripts import collect_real_points_batch as collector
from datetime import datetime,timezone


def test_book_rows_cannot_replace_independent_games():
    d={'dates':np.array(['2024-12-01']*120),'games':np.array(['1']*120)}
    assert not coverage(d)['fit']['ready']
    assert coverage(d)['fit']['independent_games']==1


def test_the_live_season_stays_refused_by_default():
    """The default is still the protected season, so no caller gains live data by accident."""
    import pytest
    from scripts.report_real_points_readiness import PROSPECTIVE_START
    d={'dates':np.array(['2026-10-20']*120),'games':np.array([str(i) for i in range(120)])}
    with pytest.raises(ValueError,match=PROSPECTIVE_START):coverage(d)


def test_an_explicit_boundary_admits_live_games_before_it_and_refuses_the_rest():
    import pytest
    before=np.array(['2026-10-20']*60+['2026-10-28']*60)
    after=np.concatenate([before[:-1],np.array(['2026-11-05'])])
    games=np.array([str(i) for i in range(120)])
    got=coverage({'dates':before,'games':games},holdout_from='2026-10-29')
    assert got['evaluation']['player_games']==120
    with pytest.raises(ValueError,match='2026-10-29'):
        coverage({'dates':after,'games':games},holdout_from='2026-10-29')


def test_a_game_exactly_on_the_boundary_is_held_back():
    """The boundary is the first held-back day, so `>` instead of `>=` leaks one day."""
    import pytest
    d={'dates':np.array(['2026-10-29']*10),'games':np.array([str(i) for i in range(10)])}
    with pytest.raises(ValueError,match='2026-10-29'):coverage(d,holdout_from='2026-10-29')


def test_the_report_records_the_boundary_it_enforced(tmp_path):
    """A reader must see which games were held back without knowing the day it ran."""
    import joblib,json as _json
    from scripts.report_real_points_readiness import run
    ds=tmp_path/'real_points.joblib'
    # Live-season dates with a boundary past them: a run() that drops the argument
    # falls back to the default and refuses, so this pins pass-through too.
    joblib.dump({'dates':np.array(['2026-10-20']*5),'games':np.array([str(i) for i in range(5)])},ds)
    run(ds,holdout_from='2026-11-01')
    assert _json.loads((tmp_path/'readiness.json').read_text())['holdout_from']=='2026-11-01'


def test_the_boundary_comes_from_a_given_day_not_from_the_clock():
    """A now()-derived cutoff would make one dataset answer differently tomorrow."""
    from scripts.report_real_points_readiness import holdout_boundary,HOLDOUT_DAYS
    assert HOLDOUT_DAYS==14
    # Deliberately not today: with today's date a clock read returns the same
    # answer, and the test would pass while proving nothing about derivation.
    assert holdout_boundary('2025-03-10')=='2025-02-24'
    assert holdout_boundary('2024-01-05')=='2023-12-22'
    assert holdout_boundary('2025-03-10',days=0)=='2025-03-10'


def test_live_season_games_are_evaluation_only_and_never_fit():
    """Moving the guard does not teach the model the live season: the masks are unchanged."""
    d={'dates':np.array(['2026-10-20']*30),'games':np.array([str(i) for i in range(30)])}
    got=coverage(d,holdout_from='2026-11-01')
    assert got['evaluation']['player_games']==30
    assert got['fit']['player_games']==0 and got['calibration']['player_games']==0
    assert got['selection']['player_games']==0


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
