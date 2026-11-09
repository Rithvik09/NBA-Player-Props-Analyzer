"""Real Flask/helper/feature builder/models/Postgres; only data providers stubbed."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.serving_provenance import build_manifest,verify_manifest,evidence,verify_evidence


def test_manifest_and_evidence_reject_drift(tmp_path):
    (tmp_path/'model_version.txt').write_text('fixture')
    m=build_manifest(tmp_path);verify_manifest(m,tmp_path)
    r=evidence(m,{'a':np.nan},1.5,'points',{'over_probability':.5});verify_evidence(r)
    r['line']=2.5
    with pytest.raises(ValueError):verify_evidence(r)
    (tmp_path/'model_version.txt').write_text('changed')
    with pytest.raises(ValueError):verify_manifest(m,tmp_path)


@pytest.fixture
def application(monkeypatch):
    import os
    if 'codex_serving_test_' not in os.environ.get('DATABASE_URL',''):
        pytest.skip('Requires an explicitly isolated codex_serving_test_* database')
    monkeypatch.setenv('APP_BACKGROUND_TASKS','0');monkeypatch.setenv('NIGHTLY_RETRAIN','0')
    monkeypatch.delenv('ODDS_API_KEY',raising=False)
    # Reject all external HTTP, even if a provider mock was missed.
    import requests
    monkeypatch.setattr(requests.sessions.Session,'request',lambda *a,**k:(_ for _ in ()).throw(AssertionError('External HTTP forbidden')))
    from src import app
    import tests.test_feature_defaults as f
    import src.basketball_betting_helper as bbh
    import src.live_injuries,src.prediction_schedule
    eff=f._serve_efficiency(monkeypatch);stats=eff['_stats']
    helper=app.betting_helper
    monkeypatch.setattr(helper,'get_player_stats',lambda _:stats)
    monkeypatch.setattr(helper.ml_predictor,'get_player_context',lambda *a:{'team_id':1610612738})
    monkeypatch.setattr(helper.ml_predictor,'get_team_context',lambda *a,**k:{'pace':100.,'rest_days':2})
    monkeypatch.setattr(helper,'_detect_game_type',lambda *a:None)
    monkeypatch.setattr(helper,'_get_referee_features',lambda **k:{'ref_foul_rate':0.,'ref_home_bias':.5,'ref_pace_tendency':100.})
    monkeypatch.setattr(helper._precomputed,'refresh',lambda *a,**k:defaultdict(dict))
    monkeypatch.setattr(src.live_injuries,'matchup_injury_context',lambda *a:{})
    monkeypatch.setattr(bbh,'TeamGameLog',lambda **k:f._CannedEndpoint(pd.DataFrame()))
    monkeypatch.setattr(src.prediction_schedule,'current_game_schedule',lambda *a:{'game_id':'0029900001','game_start':'2099-01-01T00:00:00Z','source':'offline_fixture'})
    return app.app.test_client(),helper


def test_http_model_database_replay_and_push(application):
    client,helper=application
    response=client.post('/analyze_prop',json={'player_id':2544,'prop_type':'points','line':20.,'opponent_team_id':1610612747,'is_home':False})
    result=response.get_json();assert response.status_code==200,result
    assert result['model_inference_succeeded'],result
    assert result['prediction_logged'] and result['log_id'] and result['model_manifest_id']
    with helper.get_db() as c:
        row=c.execute('SELECT e.evidence,m.manifest FROM prediction_evidence e JOIN serving_manifests m USING(manifest_id) WHERE prediction_id=%s',(result['log_id'],)).fetchone()
    verify_evidence(row[0]);assert row[0]['manifest_id']==result['model_manifest_id']
    from scripts.replay_served_prediction import replay
    replayed=replay(result['log_id']);assert replayed['exact_match']
    assert replayed['result']['over_probability']==result['over_probability']
    graded=client.post(f"/logs/{result['log_id']}/result",json={'actual_result':20.})
    assert graded.status_code==200 and graded.get_json()['outcome']=='PUSH'
    with helper.get_db() as c:
        row=c.execute('SELECT actual_outcome,correct,result_recorded_at FROM prediction_logs WHERE id=%s',(result['log_id'],)).fetchone()
        assert row[0]=='PUSH' and row[1] is None and row[2]
    assert client.get('/logs').status_code==200
    assert client.get('/accuracy').status_code==200
    status=client.get('/serving/provenance');assert status.status_code==200
    assert status.get_json()['manifest']['manifest_id']==result['model_manifest_id']
    import os
    if os.environ.get('SERVING_TEST_REPORT_DIR'):
        output=Path(os.environ['SERVING_TEST_REPORT_DIR']);output.mkdir(parents=True,exist_ok=True)
        (output/'http_roundtrip.json').write_text(json.dumps({'analysis':result,'replay':replayed,'grading':graded.get_json(),'database_outcome':list(row),'source_boundary':'Canned cached-provider fixtures; real feature builder, loaded models, Flask request and PostgreSQL persistence/replay/grading'},indent=2,default=str))


def test_failed_model_and_invalid_line_cannot_create_forecast(application,monkeypatch):
    client,helper=application
    with helper.get_db() as c:before=c.execute('SELECT COUNT(*) FROM prediction_logs').fetchone()[0]
    clf=helper.ml_predictor.prop_models['points']['calibrated_clf']
    monkeypatch.setattr(clf,'predict_proba',lambda *a:(_ for _ in ()).throw(RuntimeError('injected')))
    body={'player_id':2544,'prop_type':'points','line':20.,'opponent_team_id':1610612747,'is_home':False}
    result=client.post('/analyze_prop',json=body).get_json()
    assert result['prediction_mode']=='statistical_fallback' and result['recommendation']=='PASS' and not result['prediction_logged']
    with helper.get_db() as c:assert c.execute('SELECT COUNT(*) FROM prediction_logs').fetchone()[0]==before
    assert client.post('/analyze_prop',json={**body,'line':'nan'}).status_code==400


def test_evidence_failure_rolls_back_prediction(application,monkeypatch):
    client,helper=application
    import src.serving_provenance as provenance
    with helper.get_db() as c:
        before=c.execute('SELECT COUNT(*) FROM prediction_logs').fetchone()[0]
    monkeypatch.setattr(provenance,'verify_evidence',lambda *a:(_ for _ in ()).throw(ValueError('injected evidence failure')))
    result=client.post('/analyze_prop',json={'player_id':2544,'prop_type':'points','line':20.,'opponent_team_id':1610612747,'is_home':False}).get_json()
    assert result['model_inference_succeeded'] and not result['prediction_logged']
    with helper.get_db() as c:
        assert c.execute('SELECT COUNT(*) FROM prediction_logs').fetchone()[0]==before
