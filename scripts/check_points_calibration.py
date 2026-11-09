"""Independently verify saved one-dimensional Platt calibrators without BLAS."""
import json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scripts.train_expanded_points import load_expanded,cohort_weights,normalized_weights
from src.points_candidate import normalize_points_inputs
from src.verified_training import expand_points,classifier_frame
from scripts.points_pilot import metrics


def run():
    out=Path('logs/expanded_points_experiment_v4');m=json.loads((out/'manifest.json').read_text());d=load_expanded(Path('logs/expanded_points_v4'))
    d['X']=normalize_points_inputs(d['X']).astype(np.float32)
    _,y,idx,lines=expand_points(d,['recent_avg','season_avg','stddev']);dates=pd.to_datetime(d['dates'][idx]);core=d['original_cohort'][idx]
    results={}
    for name in ['compact_expanded','compact_original']:
        model=joblib.load(out/f'{name}.joblib');assert model.calibrator.method=='sigmoid'
        cohort=m['selections'][name]['classifier'][1]
        raw=model.classifier.predict_proba(classifier_frame(d['X'].iloc[idx],lines,model.columns).astype(np.float32))[:,1]
        mask=np.asarray((dates>='2025-03-01')&(dates<'2025-07-01'))&(core if cohort=='original' else True)
        weights=normalized_weights(cohort_weights(core,cohort),mask)
        # Match the persisted input transform, then use float64 elementwise sums.
        logits=np.log(np.clip(raw,1e-7,1-1e-7)/(1-np.clip(raw,1e-7,1-1e-7))).astype(float)
        x=logits[mask];target=y[mask]
        def objective(theta):
            z=theta[0]*x+theta[1];p=expit(z);err=weights*(p-target)
            loss=np.sum(weights*(np.logaddexp(0,z)-target*z))+.5*theta[0]**2
            grad=np.array([np.sum(err*x)+theta[0],np.sum(err)])
            return loss,grad
        original=np.array([model.calibrator.estimator.coef_[0,0],model.calibrator.estimator.intercept_[0]])
        solution=minimize(objective,original,jac=True,method='L-BFGS-B',options={'gtol':1e-9,'ftol':1e-14,'maxiter':500})
        expected=expit(solution.x[0]*logits+solution.x[1]);actual=model.calibrator.predict(raw)
        difference=float(np.max(np.abs(expected-actual)));test=np.asarray(dates>='2025-07-01')&core
        results[name]={'independent_solver_success':bool(solution.success),'original_coefficients':original.tolist(),
            'independent_coefficients':solution.x.tolist(),'maximum_probability_difference':difference,
            'original_mean_gradient':(objective(original)[1]/weights.sum()).tolist(),
            'independent_evaluation_metrics':metrics(y[test],expected[test])}
        assert np.isfinite(actual).all() and difference<.001,results[name]
    (out/'calibration_numeric_check.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


if __name__=='__main__':run()
