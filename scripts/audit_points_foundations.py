"""Offline foundational checks; does not launch the web app or change models."""
from pathlib import Path
from datetime import datetime,timezone
import json,io,contextlib
import joblib,numpy as np,pandas as pd
from unittest.mock import patch
from src.prospective_points import digest,save_new
from src.points_calibration_research_v2 import validate
from src.historical_inputs import load_season_logs
from src.models import EnhancedMLPredictor

ROOT=Path('logs/points_foundations_audit_03')


def run():
    ROOT.mkdir(exist_ok=False)
    before={str(p):digest(p) for p in Path('models').rglob('*') if p.is_file()}
    b=joblib.load('logs/points_evidence_representation_01/inputs.joblib');validate(b);d=b['data'];x=d['X']
    logs=pd.concat([load_season_logs(s,include_postseason=True) for s in ['2022-23','2023-24','2024-25','2025-26']],ignore_index=True)
    logs['GAME_ID']=logs.GAME_ID.astype(str).str.zfill(10)
    lookup=logs.set_index(['GAME_ID','PLAYER_ID']);idx=pd.MultiIndex.from_arrays([d['games'].astype(str),d['players'].astype(int)])
    aligned=lookup.reindex(idx)
    truth={'rows':len(x),'unique_player_games':len(set(zip(d['games'],d['players']))),'nba_source_identity_missing':int(aligned.PTS.isna().sum()),'points_mismatches':int((aligned.PTS.to_numpy()!=d['points']).sum()),'dates_mismatches':int((pd.to_datetime(aligned.GAME_DATE).to_numpy()!=pd.to_datetime(d['dates']).to_numpy()).sum()),'labels_exactly_points_greater_than_line':bool(np.array_equal(d['y'],d['points']>d['lines'])),'pushes':int((d['points']==d['lines']).sum()),'protected_season_rows':int((pd.to_datetime(d['dates'])>='2026-10-01').sum()),'note':'Existing importer provenance validation rerun; provider IDs are mapped to NBA IDs, not directly comparable across namespaces.'}
    # Explicit network prohibition, including unexpected calls during model loading.
    buffer=io.StringIO()
    with patch('socket.socket.connect',side_effect=AssertionError('Audit prohibits network')),contextlib.redirect_stdout(buffer):
        predictor=EnhancedMLPredictor()
        examples={}
        for name,feature in [('ordinary',{'recent_avg':20.,'season_avg':18.,'stddev':6.}),('missing_variance',{'recent_avg':20.,'season_avg':18.,'stddev':float('nan')}),('missing_mean',{'recent_avg':float('nan'),'season_avg':18.,'stddev':6.})]:
            result=predictor.predict(feature,20.5,'points')
            examples[name]={'returned_probability':result['over_probability'],'finite_predicted_points':bool(np.isfinite(result['predicted_value'])),'recommendation':result['recommendation'],'confidence':result['confidence']}
        # Fault injection: reproduce how a loaded classifier failure is exposed.
        with patch.object(predictor.prop_models['points']['calibrated_clf'],'predict_proba',side_effect=RuntimeError('audit injected failure')):
            fallback=predictor.predict({'recent_avg':20.,'season_avg':18.,'stddev':6.},20.5,'points')
        sample=[]
        # Research feature replay tests predict() only, NOT the full live feature builder.
        for i in np.linspace(0,len(x)-1,24,dtype=int):
            results=[predictor.predict(x.iloc[i].to_dict(),float(line),'points') for line in [10.5,20.5,30.5]]
            sample.append({'game':str(d['games'][i]),'player':int(d['players'][i]),'p':[r['over_probability'] for r in results],'nonfinite_points':sum(not np.isfinite(r['predicted_value']) for r in results)})
    meta=json.loads(Path('models/model_metadata.json').read_text())
    serving={'loaded_props':sorted(predictor.prop_models),'global_models_ready':predictor.models_trained,'effective_policy':getattr(predictor,'feature_policy','legacy'),'model_version':predictor.model_version,'metadata_training_cutoff':meta['props']['points'].get('training_outcomes_through'),'metadata_schema_features':len(meta['schema']['numeric_features']),'examples':examples,'injected_classifier_failure_response':fallback,'failure_response_has_mode_or_error':any(k in fallback for k in ['prediction_mode','inference_error','model_available','fallback_reason']),'replay':sample,'replay_monotonicity_violations':int(sum(sum(np.diff(r['p'])>1e-10) for r in sample)),'replay_limitation':'Cached research feature vectors passed to serving predict; no claim of full live builder parity or held-out predictive performance.'}
    frozen=json.loads(Path('logs/real_points_paid_01/frozen_before.json').read_text())
    for group in ['hashes','serving']:
        for path,h in frozen[group].items():assert digest(path)==h,path
    assert before=={str(p):digest(p) for p in Path('models').rglob('*') if p.is_file()}
    save_new(ROOT/'report.json',{'created_at':datetime.now(timezone.utc).isoformat(),'data_truth':truth,'serving':serving,'frozen_and_serving_hashes_unchanged':True,'network_calls':0,'model_hashes':before})
    (ROOT/'inference_stdout.txt').write_text(buffer.getvalue())
    print(json.dumps({'truth':truth,'serving':{k:v for k,v in serving.items() if k!='replay'}},indent=2))

if __name__=='__main__':run()
