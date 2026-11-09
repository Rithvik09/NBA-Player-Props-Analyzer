"""Evaluate frozen paired predictions separately at 60m, 15m or request time."""
import argparse
import json
from pathlib import Path
from src.sharp_evidence import validate_holdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--horizon', choices=['60', '15', 'request'], required=True)
    parser.add_argument('--selection-outcomes-through', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.predictions.read_text())
    result = validate_holdout(rows, horizon_minutes=args.horizon if args.horizon == 'request' else int(args.horizon),
                              selection_outcomes_through=args.selection_outcomes_through)
    result['activation_changed'] = False
    result['baseline_requirement'] = 'Frozen baseline already includes all existing market features'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result))
