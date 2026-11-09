import json
import pytest
from src.points_collection_preflight import validate_plan,attempted_games,validate_response


def fixture():
    p={'credit_ceiling':10,'candidates':[{'id':'123','nba_game_id':'0022500001','date':'2025-10-22','split':'holdout'}]}
    protocol={'purpose':'fresh_holdout','created_at':'2026-10-26T12:00:00Z','quote_horizon':'historical_closing','required_cohorts':['short_history'],'model_selection_rule':'fixed before evaluation'}
    return p,protocol

def test_new_quote_is_not_new_holdout():
    p,proto=fixture()
    with pytest.raises(ValueError,match='exposed'):validate_plan(p,proto,{'0022500001'},set())
    assert validate_plan(p,proto,set(),set())

def test_duplicate_attempt_budget_protected_and_order():
    p,proto=fixture()
    with pytest.raises(ValueError,match='already attempted'):validate_plan(p,proto,set(),{'123'})
    p['credit_ceiling']=1
    with pytest.raises(ValueError,match='budget'):validate_plan(p,proto,set(),set())
    p['credit_ceiling']=10;p['candidates'][0]['date']='2026-10-22'
    with pytest.raises(ValueError,match='Protected'):validate_plan(p,proto,set(),set())
    p['candidates'].append({'id':'456','nba_game_id':'0022400001','date':'2025-10-21','split':'calibration_selection'});p['credit_ceiling']=20
    p['candidates'][0]['date']='2025-10-20'
    with pytest.raises(ValueError,match='Chronological'):validate_plan(p,proto,set(),set())

def test_uncertain_attempts_across_batches_are_not_reset(tmp_path):
    p=tmp_path/'real_points_paid_replacements_02';p.mkdir();(p/'state.json').write_text(json.dumps({'attempts':{'123':{'status':'uncertain_failure'}}}))
    assert '123' in attempted_games(tmp_path)

def test_wrong_provider_game_rejected():
    with pytest.raises(ValueError,match='planned'):validate_response({'data':{'props':[{'game_id':'wrong','game_date':'2025-10-22'}]}},{'id':'123','date':'2025-10-22'})

def test_unapproved_plan_never_reaches_network(tmp_path,monkeypatch):
    from scripts import collect_paid_points_batch as c
    p,_=fixture();monkeypatch.setattr(c,'OUT',tmp_path);(tmp_path/'plan.json').write_text(json.dumps(p))
    monkeypatch.setattr(c.requests,'post',lambda *a,**k:pytest.fail('Network called before preflight'))
    with pytest.raises(ValueError,match='preflight'):c.run()
