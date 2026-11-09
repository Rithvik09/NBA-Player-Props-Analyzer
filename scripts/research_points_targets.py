"""Offline predeclared target-by-shooting factorial comparison."""
from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBClassifier, XGBRegressor
from src.prospective_points import save_new, digest
from src.points_v5 import prepare
from src.points_calibration_research_v2 import validate, fit
from src.points_training_repair import keep_synthetic, unique_outcomes, assert_disjoint_games, outcome_weights, monotone_constraints
from src.verified_training import classifier_frame
from src.points_line_experiments import expand_with_offsets
from src.points_target_research import fixed_threshold_expansion, TargetCandidate, FIXED_THRESHOLDS
from scripts.points_pilot import metrics

ROOT = Path('logs/points_target_experiment_01')
DATA = Path('logs/points_evidence_representation_01')


def run():
    b = joblib.load(DATA/'inputs.joblib'); validate(b); d = b['data']; x = prepare(d['X'])
    dates = pd.to_datetime(d['dates']); base = np.asarray(dates < '2024-10-01'); cal = np.asarray((dates >= '2024-10-01') & (dates < '2025-07-01')); evaluation = np.asarray(dates >= '2025-07-01')
    assert_disjoint_games(d['games'][base], d['games'][cal], d['games'][evaluation])
    old = json.loads(Path('logs/points_evidence_feature_experiment_01/plan.json').read_text())
    core = old['groups']['core']
    efficiency = [c for c in x if c.startswith(('reliable_', 'support_'))]
    volume = [c for c in x if c.startswith('volume_')]
    groups = {'core': core, 'efficiency': core+efficiency, 'volume': core+volume, 'shooting': core+efficiency+volume}
    # Keep the exact previous column order for the reproducibility control.
    groups['shooting'] = old['groups']['core_reliable_shooting']
    parts = []; hashes = {}
    for season in ['2022-23','2023-24','2024-25','2025-26']:
        path = DATA/(season+'.joblib'); a = joblib.load(path); m = pd.to_datetime(a['dates']) < '2024-10-01'
        parts.append({'X': a['X'].loc[m], **{k: a[k][m] for k in ['y','games','players']}}); hashes[str(path)] = digest(path)
    sx = pd.concat([a['X'] for a in parts], ignore_index=True); sy = np.concatenate([a['y'] for a in parts]); sg = np.concatenate([a['games'] for a in parts]); sp = np.concatenate([a['players'] for a in parts])
    keep = keep_synthetic(sg, sp, d['games'][base], d['players'][base]); sx = prepare(sx.loc[keep].reset_index(drop=True)); sy = sy[keep]; sg = sg[keep]; sp = sp[keep]
    unique_outcomes(np.r_[sg, d['games'][base]], np.r_[sp, d['players'][base]])
    assert_disjoint_games(sg, d['games'][cal], d['games'][evaluation])
    h = x.games_played.to_numpy(); cohorts = {'short_history': h<10, 'eligible_history': h>=10}
    for mask in cohorts.values():
        for split in [cal, evaluation]:
            if (mask & split).sum()<100 or len(set(d['games'][mask & split]))<20: raise ValueError('Insufficient cohort support')
    ROOT.mkdir(exist_ok=False)
    save_new(ROOT/'plan.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'historical_independent': False, 'groups': groups, 'parameters':old['parameters'], 'targets':['centered','fixed','points'], 'fixed_thresholds':list(FIXED_THRESHOLDS), 'points_probability_scale':10, 'base_before':'2024-10-01','calibration_before':'2025-07-01', 'policy':'12 predeclared arms. Identical outcome population, total weight1 per outcome. No method selection on comparison. Point-target model excludes all line-derived features. Nonnegative slope calibration on prior real quotes only.', 'base_outcomes':len(sy)+int(base.sum()),'base_games':len(set(sg)|set(d['games'][base])), 'input_sha256':digest(DATA/'inputs.joblib'), 'training_sources':hashes, 'source_hashes':{p:digest(p) for p in [__file__, 'src/points_target_research.py','src/points_line_experiments.py','src/points_training_repair.py','src/points_calibration_research_v2.py']}})
    candidates = {}; means = {}
    for target in ['centered','fixed','points']:
        for group, columns in groups.items():
            name = target+'_'+group
            if target == 'points':
                xx = pd.concat([sx[columns], x.loc[base, columns]], ignore_index=True).astype(np.float32); yy = np.r_[sy, d['points'][base]]; weights = np.ones(len(yy))
                model = XGBRegressor(**old['parameters'], objective='reg:squarederror')
            else:
                if target == 'centered': xx, yy, idx, _ = expand_with_offsets({'X':sx,'y':sy}, columns, [-8,-4,0,4,8])
                else: xx, yy, idx, _ = fixed_threshold_expansion({'X':sx,'y':sy}, columns)
                xx = pd.concat([xx, classifier_frame(x, d['lines'], columns).loc[base]], ignore_index=True).astype(np.float32); yy = np.r_[yy,d['y'][base]]; weights = outcome_weights(idx,int(base.sum()))
                model = XGBClassifier(**old['parameters'], eval_metric='logloss', monotone_constraints=monotone_constraints(list(xx)))
            model.fit(xx, yy, sample_weight=weights)
            candidate = TargetCandidate(columns, model, target, {})
            raw = candidate.raw(d['X'], d['lines'])
            candidate.calibrations = {cohort: fit(raw[cal & m],d['y'][cal & m],'intercept_slope') for cohort,m in cohorts.items()}
            joblib.dump(candidate, ROOT/(name+'.joblib')); candidates[name] = candidate
            if target=='points': means[name] = model.predict(x[columns])
            print(name+' fitted and frozen before comparison', flush=True)
    raw = {n:c.raw(d['X'],d['lines']) for n,c in candidates.items()}
    probs = {n:c.predict_over(d['X'],d['lines']) for n,c in candidates.items()}; probs.update(market=b['market'],coin_flip=np.full(len(x),.5))
    center = .7*x.recent_avg.to_numpy()+.3*x.season_avg.to_numpy(); synthetic_lines = np.maximum(.5,np.floor(center)+.5); synth_y = (d['points']>synthetic_lines).astype(int)
    synthetic_raw = {n:c.raw(d['X'],synthetic_lines) for n,c in candidates.items()}
    report = {'historical_independent':False,'promoted':False,'cohorts':{},'monotonicity':{}}
    for n,c in candidates.items():
        previous = None; violations = 0
        for offset in np.arange(-5,5.01,.5):
            p = c.predict_over(d['X'].loc[evaluation],np.maximum(.5,d['lines'][evaluation]+offset))
            if previous is not None: violations += int((p>previous+1e-10).sum())
            previous = p
        if violations: raise ValueError('Monotonicity violation')
        report['monotonicity'][n] = {'comparisons':int(evaluation.sum())*20,'violations':violations}
    for cohort, mask in cohorts.items():
        m = mask & evaluation
        report['cohorts'][cohort] = {'rows':int(m.sum()),'games':len(set(d['games'][m])), 'metrics':{n:metrics(d['y'][m],p[m]) for n,p in probs.items()}, 'synthetic_center_metrics_same_rows':{n:metrics(synth_y[m],p[m]) for n,p in synthetic_raw.items()}, 'points_errors':{n:{'mae':float(np.mean(abs(p[m]-d['points'][m]))),'rmse':float(np.sqrt(np.mean((p[m]-d['points'][m])**2)))} for n,p in means.items()}, 'label_disagreement_rate':float(np.mean(synth_y[m]!=d['y'][m])), 'feature_relationships':{feature:{outcome:float(spearmanr(x.loc[m,feature],values[m]).statistic) for outcome,values in {'points':d['points'],'points_minus_center':d['points']-synthetic_lines,'points_minus_market_line':d['points']-d['lines'],'market_line_minus_center':d['lines']-synthetic_lines}.items()} for feature in efficiency+volume}}
    # Existing centered controls must reproduce the preceding experiment exactly.
    oldreport = json.loads(Path('logs/points_evidence_feature_experiment_01/report.json').read_text())
    for cohort in cohorts:
        for a,z in [('centered_core','core'),('centered_shooting','core_reliable_shooting')]:
            assert report['cohorts'][cohort]['metrics'][a] == oldreport['cohorts'][cohort]['metrics'][z], (cohort,a)
    save_new(ROOT/'report.json', report)
    pd.DataFrame({'game':d['games'][evaluation],'player':d['players'][evaluation],'over':d['y'][evaluation],'history':h[evaluation],**{n:p[evaluation] for n,p in probs.items()},**{'raw_'+n:p[evaluation] for n,p in raw.items()}}).to_csv(ROOT/'predictions.csv',index=False)
    print(json.dumps({k:{n:{z:v[z] for z in ['auc','log_loss']} for n,v in r['metrics'].items()} for k,r in report['cohorts'].items()},indent=2))

if __name__=='__main__': run()
