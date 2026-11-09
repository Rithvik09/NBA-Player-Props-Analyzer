"""Replay canonical source histories at explicit decisions, without activation.

Inputs are JSON lists with NBA game/player IDs, not provider IDs. Quote rows
must already have a verified identity crosswalk. See SOURCE_BLOCKERS.md.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from src.source_evidence import market_features, game_total_features, performance_features


def replay(decisions, quotes=(), totals=(), predictions=()):
    counts = Counter()
    results = []
    for decision in decisions:
        base = {k: decision[k] for k in ('cutoff', 'game_id', 'player_id', 'prop_type')}
        features = market_features(quotes, **base, decision_line=decision.get('line'))
        features.update(game_total_features(totals, cutoff=base['cutoff'], game_id=base['game_id']))
        features.update(performance_features(predictions, cutoff=base['cutoff'],
                        player_id=base['player_id'], prop_type=base['prop_type'],
                        model_version=decision.get('model_version')))
        counts.update(features.keys())
        results.append({'decision': decision, 'features': features})
    return {'decision_count': len(decisions), 'observed_counts': dict(counts),
            'activation_changed': False, 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions', type=Path, required=True)
    for name in ('quotes', 'totals', 'predictions'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = {}
    hashes = {}
    for name in ('decisions', 'quotes', 'totals', 'predictions'):
        path = getattr(args, name)
        if path:
            data = path.read_bytes()
            inputs[name] = json.loads(data)
            if not isinstance(inputs[name], list):
                raise ValueError(name + ' must contain a JSON list')
            hashes[str(path)] = hashlib.sha256(data).hexdigest()
    result = replay(**inputs)
    result['input_sha256'] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print({k: v for k, v in result.items() if k != 'results'})
