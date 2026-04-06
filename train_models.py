"""
Offline training script for the NBA props ML models.

Usage:
    python train_models.py [--players N] [--seasons N]

Fetches historical game logs for N active players over the last N seasons,
builds rolling-window training samples, trains both the classification model
(will the player go OVER the line?) and the regression model (what value will
the player produce?), then saves the fitted models + scaler to models/.
"""

import argparse
import sys
import os

# Make sure src package is importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.data_collector import TrainingDataCollector
from src.models import EnhancedMLPredictor


def main():
    parser = argparse.ArgumentParser(description='Train NBA props ML models')
    parser.add_argument('--players', type=int, default=100,
                        help='Number of active players to collect data for (default: 100)')
    parser.add_argument('--seasons', type=int, default=3,
                        help='Number of past seasons to include (default: 3)')
    parser.add_argument('--min-samples', type=int, default=500,
                        help='Minimum training samples required before training (default: 500)')
    args = parser.parse_args()

    collector  = TrainingDataCollector()
    predictor  = EnhancedMLPredictor()

    print(f"Fetching active player list...")
    player_ids = collector.get_active_player_ids(n=args.players)
    print(f"Collected {len(player_ids)} player IDs")

    seasons = collector._get_seasons(num_seasons=args.seasons)
    print(f"Training on seasons: {seasons}")
    print(f"Collecting training data — this will take a while due to API rate limits...\n")

    training_data = collector.collect_bulk(player_ids, seasons=seasons)

    print(f"\nTotal training samples collected: {len(training_data)}")

    if len(training_data) < args.min_samples:
        print(f"ERROR: Only {len(training_data)} samples collected, "
              f"need at least {args.min_samples}. Aborting.")
        sys.exit(1)

    print("Training models...")
    metrics = predictor.train(training_data)
    print(f"Done! AUC: {metrics['auc']:.3f}  RMSE: {metrics['rmse']:.3f}")
    print("Models saved to models/")
    print("  models/classification_model.joblib")
    print("  models/regression_model.joblib")
    print("  models/scaler.joblib")


if __name__ == '__main__':
    main()
