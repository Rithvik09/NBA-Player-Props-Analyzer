"""Frozen, paired real-line validation. No synthetic lines or model updates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    """Exclusive create: an existing forecast or settlement is never revised."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def validate_quote(quote, now, horizon='request_time'):
    now = pd.Timestamp(now)
    observed, published, start = [pd.Timestamp(quote[k]) for k in
                                 ('observed_at', 'published_at', 'game_start')]
    if any(t.tzinfo is None for t in (now, observed, published, start)):
        raise ValueError('Timezone required')
    if not published <= observed <= now < start:
        raise ValueError('Quote/prediction must precede official tipoff; no future source')
    if (now-observed).total_seconds() > 600 or (now-published).total_seconds() > 900:
        raise ValueError('Quote too old for prospective validation')
    minutes = (start-now).total_seconds()/60
    if horizon == '60m' and not 50 <= minutes <= 70:
        raise ValueError('Outside 60-minute window')
    if horizon == '15m' and not 10 <= minutes <= 20:
        raise ValueError('Outside 15-minute window')
    if horizon not in ('request_time', '60m', '15m'):
        raise ValueError('Unknown horizon')
    if quote.get('prop_type') != 'points' or quote.get('scope') != 'prop' or quote.get('is_alternate'):
        raise ValueError('Expected main player-points line')
    if not np.isfinite(float(quote['line'])) or float(quote['line']) < 0:
        raise ValueError('Invalid line')
    for key in ('over_price', 'under_price'):
        if not np.isfinite(float(quote[key])) or abs(float(quote[key])) < 100:
            raise ValueError('Invalid American odds')
    return minutes


def select_quotes(quotes):
    """One lexicographically first book per player; no outcome/edge selection."""
    result = {}
    for quote in sorted(quotes, key=lambda q: (q['bookmaker'], q['line'])):
        result.setdefault((str(quote['game_id']), int(quote['player_id'])), quote)
    return list(result.values())


def settlement(forecast, summary, boxes):
    gid = forecast['quote']['game_id']
    if str(summary.get('gameId')) != gid or str(boxes.get('gameId')) != gid:
        raise ValueError('Wrong NBA outcome identity')
    if int(summary.get('gameStatus', 0)) != 3:
        return {'status': 'pending'}
    people = [p for side in ('homeTeam', 'awayTeam') for p in boxes[side]['players']
              if int(p['personId']) == forecast['quote']['player_id']]
    if len(people) != 1:
        return {'status': 'missing_player'}
    p = people[0]
    stats = p.get('statistics', {})
    minutes = str(stats.get('minutes', ''))
    # NBA v3 uses MM:SS; absent players/ambiguous source are not silently voided.
    if not minutes:
        return {'status': 'dnp'} if p.get('notPlayingReason') or p.get('status') == 'INACTIVE' else {'status': 'missing_minutes'}
    try:
        played = sum(float(v) for v in minutes.split(':')) > 0
    except ValueError:
        return {'status': 'missing_minutes'}
    if not played:
        return {'status': 'dnp'}
    points = stats.get('points')
    if points is None or not np.isfinite(float(points)):
        return {'status': 'missing_points'}
    points = float(points)
    line = forecast['quote']['line']
    return {'status': 'push' if points == line else 'settled', 'points': points,
            'over': None if points == line else int(points > line)}


def paired_report(forecasts, outcomes):
    """First forecast per player/game/horizon; game-cluster paired uncertainty."""
    from sklearn.metrics import roc_auc_score
    groups = {}
    counts = {}
    identities = set()
    for f in forecasts:
        identity = (f['quote']['game_id'], f['quote']['player_id'], f['horizon'])
        if identity in identities:
            raise ValueError('Duplicate player/game/horizon forecast')
        identities.add(identity)
        o = outcomes.get(f['id'], {'status': 'pending'})
        counts[o['status']] = counts.get(o['status'], 0)+1
        if o['status'] not in ('settled', 'push'):
            continue
        groups.setdefault((f['cohort'], f['horizon']), []).append((f, o))
    result = {}
    for (cohort, horizon), rows in groups.items():
        binary = [(f, o) for f, o in rows if o['status'] == 'settled']
        entry = {'player_games_with_points': len(rows), 'binary_player_games': len(binary),
                 'games': len({f['quote']['game_id'] for f, o in rows})}
        losses = {}
        for name in ('candidate', 'compact'):
            entry[name] = {'rmse': float(np.sqrt(np.mean([(f['predictions'][name]['points']-o['points'])**2 for f, o in rows])))}
            if binary:
                y = np.array([o['over'] for f, o in binary])
                p = np.clip([f['predictions'][name]['over_probability'] for f, o in binary], 1e-7, 1-1e-7)
                losses[name] = -(y*np.log(p)+(1-y)*np.log(1-p))
                entry[name].update(log_loss=float(losses[name].mean()), brier=float(np.mean((p-y)**2)),
                                   auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None)
        if binary:
            delta = losses['candidate']-losses['compact']
            games = np.array([f['quote']['game_id'] for f, o in binary])
            unique = np.unique(games)
            entry['candidate_minus_compact_log_loss'] = float(delta.mean())
            if len(unique) >= 2:
                rng = np.random.default_rng(42)
                clusters = [delta[games == g] for g in unique]
                samples = [np.concatenate([clusters[i] for i in rng.integers(len(unique), size=len(unique))]).mean() for _ in range(2000)]
                entry['game_cluster_95_ci'] = np.quantile(samples, [.025, .975]).tolist()
        entry['checkpoint_ready'] = cohort == 'eligible' and len(binary) >= 100
        entry['research_success'] = bool(entry['checkpoint_ready'] and entry['games'] >= 20
            and entry.get('candidate_minus_compact_log_loss', 0) <= -.002
            and entry.get('game_cluster_95_ci', [0, 0])[1] < 0
            and entry['candidate'].get('brier', 1) <= entry['compact'].get('brier', 0)
            and entry['candidate']['rmse'] <= entry['compact']['rmse'])
        result[cohort+'/'+horizon] = entry
    return {'forecasts': len(forecasts), 'status_counts': counts, 'comparisons': result,
            'note': 'No profit or closing-line-value claim; pushes excluded from binary metrics. Warmup excluded from primary eligible cohort.'}
