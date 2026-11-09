"""Train separate repaired candidates, without opening the prospective outcomes."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_squared_error
from src.points_v5 import prepare, PointsV5Candidate, EXTRA_COLUMNS, POLICY
from src.verified_training import expand_points, classifier_frame, ProbabilityCalibration
from src.prospective_points import digest, save_new
from scripts.investigate_points_features import PARAMS
from scripts.train_selected_points import fit_calibration, paired_losses
from scripts.points_pilot import metrics, model_hashes


def run(output, real_path=None):
    output.mkdir(parents=True, exist_ok=False)
    serving_before = model_hashes()
    frozen = Path('logs/expanded_points_experiment_v4')
    old = {name: joblib.load(frozen/file) for name, file in
           [('frozen161', 'development_selected.joblib'), ('frozen_compact', 'compact_expanded.joblib')]}
    groups = {'repaired': list(dict.fromkeys(old['frozen161'].columns+EXTRA_COLUMNS)),
              'compact_retrained': list(dict.fromkeys(old['frozen_compact'].columns+EXTRA_COLUMNS[:4]))}
    protocol = {'created_at': datetime.now(timezone.utc).isoformat(), 'policy': POLICY,
                'fit_before': '2025-01-01', 'calibration_fit': ['2025-03-01', '2025-05-01'],
                'calibration_selection': ['2025-05-01', '2025-07-01'],
                'evaluation': '2025-26 exposed retrospective research, separately by short/eligible history; not prospective',
                'synthetic_lines': 'Unchanged five half-point lines around prior mean. Never mislabeled real.',
                'real_lines': str(real_path) if real_path else None,
                'real_mix': 'When supplied, one real record per player/game; real and synthetic fitting blocks each total half the fitting weight. Calibration/selection/evaluation use real rows when every chronological split has at least100 rows and20 games; otherwise real training is blocked.',
                'groups': groups, 'hyperparameters': PARAMS,
                'classifier_settings': {'max_depth': 3, 'min_child_weight': 50, 'reg_lambda': 50, 'colsample_bytree': 1},
                'source_hashes': {p: digest(p) for p in ['src/points_v5.py', 'scripts/build_points_v5.py', __file__]},
                'frozen_reference_hashes': {p.name: digest(p) for p in frozen.glob('*.joblib')},
                'dataset_hashes': {p.name: digest(p) for p in Path('logs/points_v5_data').glob('*.joblib')}}
    save_new(output/'protocol.json', protocol)
    parts = [joblib.load(f'logs/points_v5_data/{s}.joblib') for s in ['2022-23', '2023-24', '2024-25', '2025-26']]
    if any(p['feature_policy'] != POLICY for p in parts):
        raise ValueError('Feature policy mismatch')
    data = {'X': pd.concat([p['X'] for p in parts], ignore_index=True)}
    for key in ('y', 'dates', 'games', 'players'):
        data[key] = np.concatenate([p[key] for p in parts])
    raw = data['X'].copy()
    data['X'] = prepare(data['X']).astype(np.float32)
    dates = pd.to_datetime(data['dates'])
    X, y, idx, lines = expand_points(data, groups['repaired'])
    X = X.astype(np.float32)
    cdates = dates[idx]
    games = data['games'][idx]
    weights = np.ones(len(y))
    real = joblib.load(real_path) if real_path else None
    if real is not None:
        if not real.get('provenance_validated') or real['feature_policy'] != POLICY:
            raise ValueError('Real data must pass the as-of importer')
        rd = pd.to_datetime(real['dates'])
        if (rd >= '2026-10-01').any():
            raise ValueError('Prospective season is forbidden in training/calibration')
        for mask in [rd < '2025-01-01', (rd >= '2025-03-01') & (rd < '2025-05-01'),
                     (rd >= '2025-05-01') & (rd < '2025-07-01'), rd >= '2025-07-01']:
            if mask.sum() < 100 or len(set(real['games'][mask])) < 20:
                raise ValueError('Insufficient real-line chronological split: require100 rows and20 games each')
        rx = classifier_frame(prepare(real['X']), real['lines'], groups['repaired']).astype(np.float32)
        nsynthetic = len(y)
        fit_synthetic = np.sum(cdates < '2025-01-01')
        weights = np.concatenate([np.ones(len(y)), np.full(len(rd), fit_synthetic/np.sum(rd < '2025-01-01'))])
        X = pd.concat([X, rx], ignore_index=True)
        y = np.concatenate([y, real['y']])
        cdates = cdates.append(pd.DatetimeIndex(rd))
        games = np.concatenate([games, real['games']])
        analysis_rows = np.arange(len(y)) >= nsynthetic
    else:
        analysis_rows = np.ones(len(y), dtype=bool)
    fit = np.asarray(dates < '2025-01-01')
    cfit = np.asarray(cdates < '2025-01-01')
    calfit = np.asarray((cdates >= '2025-03-01') & (cdates < '2025-05-01')) & analysis_rows
    caldev = np.asarray((cdates >= '2025-05-01') & (cdates < '2025-07-01')) & analysis_rows
    calall = calfit | caldev
    test = np.asarray(dates >= '2025-07-01')
    ctest = np.asarray(cdates >= '2025-07-01') & analysis_rows
    models = {}
    for name, cols in groups.items():
        classifier = XGBClassifier(**{**PARAMS, **protocol['classifier_settings']}, eval_metric='logloss')
        classifier.fit(X.loc[cfit, cols+['decision_line', 'recent_minus_line', 'season_minus_line', 'standardized_edge']], y[cfit], sample_weight=weights[cfit]/weights[cfit].mean())
        regressor = XGBRegressor(**PARAMS, objective='reg:squarederror').fit(data['X'].loc[fit, cols], data['y'][fit])
        cp = classifier.predict_proba(X.loc[:, cols+['decision_line', 'recent_minus_line', 'season_minus_line', 'standardized_edge']])[:, 1]
        scores = {method: metrics(y[caldev], fit_calibration(method, cp[calfit], y[calfit]).predict(cp[caldev])) for method in ('raw', 'sigmoid')}
        method = min(scores, key=lambda m: scores[m]['log_loss'])
        model = PointsV5Candidate(cols, regressor, classifier, fit_calibration(method, cp[calall], y[calall]),
                                  feature_profile=POLICY+'_'+name, regressor_columns=cols)
        models[name] = model
        joblib.dump(model, output/f'{name}.joblib')
        save_new(output/f'{name}_calibration.json', {'selected': method, 'scores': scores})
        print('trained', name, method, flush=True)
    # All model/calibration choices saved before scoring the exposed research set.
    eval_frame = real['X'].loc[pd.to_datetime(real['dates']) >= '2025-07-01'] if real else raw.iloc[idx[ctest]]
    eval_lines = real['lines'][pd.to_datetime(real['dates']) >= '2025-07-01'] if real else lines[ctest]
    probabilities = {name: model.predict_over(eval_frame, eval_lines) for name, model in {**old, **models}.items()}
    report = {'training_player_games': int(fit.sum()), 'training_short_history_rows': int((fit & (raw.games_played < 10)).sum()),
              'real_training_rows': int(np.sum(pd.to_datetime(real['dates']) < '2025-01-01')) if real else 0,
              'evaluation_line_type': 'real' if real else 'synthetic', 'populations': {}}
    short = eval_frame.games_played.to_numpy() < 10
    for name, mask in [('short_history', short), ('eligible_history', ~short), ('all', np.ones(len(short), dtype=bool))]:
        stats = {arm: metrics(y[ctest][mask], p[mask]) for arm, p in probabilities.items()}
        rmask = test & ((raw.games_played < 10) if name == 'short_history' else (raw.games_played >= 10) if name == 'eligible_history' else True)
        for arm, model in {**old, **models}.items():
            stats[arm]['points_rmse_synthetic_cohort'] = float(np.sqrt(mean_squared_error(data['y'][rmask], model.predict(raw.loc[rmask]))))
        report['populations'][name] = {'classifier_rows': int(mask.sum()), 'games': len(set(games[ctest][mask])), 'arms': stats,
            'paired_vs_frozen161': paired_losses(y[ctest][mask], probabilities['repaired'][mask], probabilities['frozen161'][mask], games[ctest][mask]),
            'paired_vs_frozen_compact': paired_losses(y[ctest][mask], probabilities['repaired'][mask], probabilities['frozen_compact'][mask], games[ctest][mask])}
    assert model_hashes() == serving_before
    report['serving_unchanged'] = True
    report['candidate_hashes'] = {p.name: digest(p) for p in output.glob('*.joblib')}
    save_new(output/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--real-data', type=Path)
    args = p.parse_args(); run(args.output, args.real_data)
