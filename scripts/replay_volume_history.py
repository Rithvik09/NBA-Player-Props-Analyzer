"""Reproduce every history-volume fit and forecast offline, then audit cohort overlap."""
import json,socket
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scripts.research_volume_history import fit,predict,ARMS
from scripts.train_traceable_points import masks,protected,write as save_new
from src.serving_provenance import file_hash,safe
from scripts.train_selected_points import paired_losses


def write(path,value):
    if path.exists():
        assert json.loads(path.read_text())==safe(value), "Verification evidence changed"
    else:save_new(path,value)


def run():
    socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Network forbidden'))
    root=Path('logs/points_volume_history_02');manifest=json.loads((root/'manifest.json').read_text())
    for p,h in manifest['files'].items():assert file_hash(root/p)==h,p
    plan=json.loads((root/'plan.json').read_text());assert protected()==plan['protected_before']
    b=joblib.load(root/'inputs.joblib');d=b['data'];train,test=masks(b);x=d['X'];dates=pd.to_datetime(d['dates'])
    saved=json.loads((root/'predictions.json').read_text());models=joblib.load(root/'models.joblib');maxdiff=0.
    for arm in ARMS:
        for day in [None,*sorted(set(dates[test]))]:
            fm=train if day is None else np.asarray(dates<day)
            vm=test if day is None else np.asarray(dates==day)&test
            m=fit(x.loc[fm],d['lines'][fm],b['market'][fm],d['y'][fm],d['games'][fm],arm)
            key='fixed_'+arm if day is None else arm+'_'+str(day)
            p=predict(m,x.loc[vm],d['lines'][vm],b['market'][vm]);expected=predict(models[key],x.loc[vm],d['lines'][vm],b['market'][vm])
            assert np.array_equal(p,expected),key
            rowmask=np.ones(test.sum(),bool) if day is None else np.asarray(dates[test]==day)
            assert np.array_equal(p,np.array(saved[('fixed_' if day is None else 'walk_')+arm])[rowmask])
    # Compare both groups in the same games as a descriptive check, never a qualifying subset.
    games=d['games'][test];short=x.games_played.to_numpy()[test]<10;shared=sorted(set(games[short])&set(games[~short]));overlap=np.isin(games,shared)
    diagnostics={}
    for name,cm in [('short',short),('eligible',~short)]:
        m=overlap&cm
        diagnostics[name]=dict(rows=int(m.sum()),games=len(set(games[m])),shared_model_vs_market=paired_losses(d['y'][test][m],np.array(saved['fixed_shared'])[m],np.array(saved['market'])[m],games[m]))
    previous=json.loads(Path('logs/points_volume_history_01/predictions.json').read_text())
    delta=max(float(np.max(np.abs(np.array(saved[k])-previous[k]))) for k in saved)
    write(root/'verification.json',dict(all_100_models_refit_exact=True,all_prediction_blocks_exact=True,protected_unchanged=True,credits_spent=0,
        numerical_revision='Replaced BLAS dot products with elementwise reductions; first run retained. Second run treats all RuntimeWarnings as errors.',max_probability_change_from_first_run=delta,
        common_game_diagnostic=diagnostics,common_game_count=len(shared),common_game_qualifies=False,
        caution='Only9 games contain both cohorts; history length and calendar/season context cannot be cleanly separated. No subset selected for qualification.'))
    print(json.dumps(diagnostics,indent=2));print('Exact refit/replay passed; previous-run maximum difference',delta)

if __name__=='__main__':run()
