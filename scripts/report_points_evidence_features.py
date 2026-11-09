"""Offline paired game-cluster uncertainty for representation controls."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.prospective_points import save_new

ROOT = Path('logs/points_evidence_feature_experiment_01')
PAIRS = [('core_reliable_shooting', 'core_raw_shooting'),
         ('core_qualified_injury', 'core_legacy_injury'),
         ('full_represented', 'legacy_full'),
         ('core_reliable_shooting', 'core'),
         ('core_qualified_injury', 'core'), ('core_combined', 'core')]


def run():
    frame = pd.read_csv(ROOT / 'predictions.csv')
    result = {'limitation': 'Previously exposed retrospective outcomes; exploratory unadjusted 95% intervals, not independent validation.', 'bootstrap_draws': 500, 'cohorts': {}}
    for cohort, mask in [('short_history', frame.history < 10), ('eligible_history', frame.history >= 10)]:
        d = frame.loc[mask]
        y = d.over.to_numpy()
        games, ids = np.unique(d.game, return_inverse=True)
        names = sorted({n for pair in PAIRS for n in pair})
        probs = {n: np.clip(d[n].to_numpy(), 1e-12, 1 - 1e-12) for n in names}
        losses = {n: -(y * np.log(p) + (1-y) * np.log1p(-p)) for n, p in probs.items()}
        aucs = {n: roc_auc_score(y, d['raw_' + n]) for n in names}
        draws = {pair: {'auc': [], 'loss': []} for pair in PAIRS}
        rng = np.random.default_rng(42)
        for _ in range(500):
            weights = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games))[ids]
            if len(set(y[weights > 0])) < 2:
                continue
            sample_auc = {n: roc_auc_score(y, d['raw_' + n], sample_weight=weights) for n in names}
            for a, b in PAIRS:
                draws[a, b]['auc'].append(sample_auc[a] - sample_auc[b])
                draws[a, b]['loss'].append(np.average(losses[a] - losses[b], weights=weights))
        result['cohorts'][cohort] = {'rows': len(d), 'games': len(games), 'comparisons': [
            {'arm': a, 'reference': b, 'auc_delta': aucs[a] - aucs[b],
             'auc_interval_95': np.quantile(draws[a, b]['auc'], [.025, .975]).tolist(),
             'log_loss_delta': float(np.mean(losses[a] - losses[b])),
             'log_loss_interval_95': np.quantile(draws[a, b]['loss'], [.025, .975]).tolist()}
            for a, b in PAIRS]}
    save_new(ROOT / 'representation_uncertainty.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
