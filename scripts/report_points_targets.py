"""Paired game-cluster uncertainty for predeclared target comparisons."""
from pathlib import Path
import scripts.report_points_evidence_features as report

if __name__ == '__main__':
    report.ROOT = Path('logs/points_target_experiment_01')
    report.PAIRS = [(t+'_'+g,t+'_core') for t in ['centered','fixed','points'] for g in ['efficiency','volume','shooting']]
    report.PAIRS += [(t+'_'+g,'centered_'+g) for t in ['fixed','points'] for g in ['core','shooting']]
    report.run()
