"""Experimental market-pressure components; never asserts bettor identity.

No serving activation or arbitrary composite weights. Game context and prop
evidence have distinct namespaces. Missing observations remain missing.
"""
from collections import defaultdict
from datetime import timedelta
import numpy as np
from .source_evidence import finite, instant, implied


def pressure_components(rows, *, cutoff, game_id, scope, player_id=None,
                        prop_type=None, lookback_minutes=60, fresh_minutes=30):
    if scope not in ('prop', 'game_total'):
        raise ValueError('Only over/under markets have an over-positive orientation')
    cutoff = instant(cutoff)
    grouped = defaultdict(list)
    for row in rows:
        try:
            if row['scope'] != scope or str(row['game_id']) != str(game_id):
                continue
            if scope == 'prop' and (int(row['player_id']) != int(player_id) or row['prop_type'] != prop_type):
                continue
            if row.get('is_live') or row.get('is_alternate') or row.get('team') or row.get('period') not in (None, '', 'full_game'):
                continue
            observed = instant(row['observed_at'])
            start = instant(row['game_start'])
            if not cutoff - timedelta(minutes=lookback_minutes) <= observed <= cutoff < start:
                continue
            # A missing publication time permits an observed split gap only;
            # it cannot establish quote freshness for movement confirmation.
            published = instant(row['published_at']) if row.get('published_at') else None
            if published is not None and published > observed:
                continue
            source, book = row['source'], row['bookmaker']
            if not isinstance(source, str) or not source.strip() or not isinstance(book, str) or not book.strip():
                continue
            grouped[(source, book)].append({**row, '_observed':observed, '_published':published})
        except (KeyError, TypeError, ValueError):
            continue
    gaps, moves, reverse, confirmations, prices = [], [], [], [], []
    books = set()
    # Multiple vendors can relay the same book. Keep one vendor's history per
    # book instead of counting the same underlying market several times.
    per_book = {}
    for (_, book), history in grouped.items():
        if book not in per_book or max(r['_observed'] for r in history) > max(r['_observed'] for r in per_book[book]):
            per_book[book] = history
    for book, history in per_book.items():
        history.sort(key=lambda row: row['_observed'])
        latest = history[-1]
        if (cutoff - latest['_observed']).total_seconds() > fresh_minutes * 60:
            continue
        books.add(book)
        try:
            tickets, money = finite(latest['tickets_fraction']), finite(latest['money_fraction'])
            if 0 <= tickets <= 1 and 0 <= money <= 1:
                gaps.append(money - tickets)
        except (KeyError, TypeError, ValueError):
            pass
        current_time = latest['_published']
        if current_time is None or (cutoff-current_time).total_seconds() > fresh_minutes*60:
            continue
        earlier = [r for r in history if r['_published'] is not None and r['_published'] < current_time]
        if not earlier:
            continue
        first = earlier[0]
        try:
            old_line, new_line = finite(first['line']), finite(latest['line'])
            if min(old_line, new_line) < 0:
                continue
            delta = new_line - old_line
            moves.append(delta)
            # Use the tickets observed BEFORE the move, not hindsight splits.
            tickets = finite(first['tickets_fraction'])
            if 0 <= tickets <= 1:
                reverse.append(float(np.sign(delta)) if delta*(tickets-.5) < 0 else 0.)
            money = finite(first['money_fraction'])
            if 0 <= tickets <= 1 and 0 <= money <= 1:
                confirmations.append((money-tickets) if delta*(money-tickets) > 0 else 0.)
        except (KeyError, TypeError, ValueError):
            pass
        try:
            if finite(first['line']) == finite(latest['line']):
                def probability(row):
                    over, under = implied(row['over_price']), implied(row['under_price'])
                    return over/(over+under)
                prices.append(probability(latest)-probability(first))
        except (KeyError, TypeError, ValueError):
            pass
    prefix = 'prop_' if scope == 'prop' else 'game_total_context_'
    values = {'money_ticket_gap':gaps, 'interval_line_change':moves,
              'reverse_movement_direction':reverse, 'money_move_confirmation':confirmations,
              'same_line_over_probability_change':prices}
    result = {prefix+key: float(np.median(value)) for key, value in values.items() if value}
    if moves:
        result[prefix+'movement_direction_agreement'] = float(np.mean(np.sign(moves)))
    if books:
        result[prefix+'observed_book_count'] = len(books)
    return result


def signed_score(baseline_over_probability, candidate_over_probability):
    """Percentage-point shift toward Over from an independently qualified model.

    This numeric conversion does not qualify a candidate or activate a feature.
    """
    baseline, candidate = map(finite, (baseline_over_probability, candidate_over_probability))
    if not 0 <= baseline <= 1 or not 0 <= candidate <= 1:
        raise ValueError('Probabilities must lie in [0,1]')
    return 100 * (candidate - baseline)


def validate_holdout(rows, *, horizon_minutes, selection_outcomes_through,
                     bootstrap_samples=2000, seed=73):
    """Paired game-day bootstrap on frozen, later, as-of predictions.

    Baseline must already contain existing market features. Caller is
    responsible for fitting both models exclusively before the selection cutoff.
    Refuses contaminated or duplicated rows instead of silently excluding them.
    CLV is deliberately not a substitute for predictive improvement.
    """
    if horizon_minutes not in (60, 15, 'request'):
        raise ValueError('Use 60, 15, or request as the evaluation horizon')
    frozen = instant(selection_outcomes_through)
    seen, clean = set(), []
    for row in rows:
        cutoff, start = instant(row['cutoff']), instant(row['game_start'])
        if horizon_minutes == 'request':
            if cutoff != instant(row['requested_at']):
                raise ValueError('Request-time cutoff must match the recorded request')
        elif abs((start-cutoff).total_seconds()-horizon_minutes*60) > 1:
            raise ValueError('Mixed decision horizons')
        if not frozen < cutoff < start < instant(row['result_recorded_at']):
            raise ValueError('Holdout overlaps selection or lacks later outcomes')
        if instant(row['input_latest_at']) > cutoff or instant(row['training_outcomes_through']) > frozen:
            raise ValueError('Future inputs or training outcomes')
        key = (str(row['game_id']), int(row['player_id']), row['prop_type'])
        if key in seen:
            raise ValueError('Duplicate decision')
        seen.add(key)
        y, baseline, candidate = map(finite, (row['over_outcome'], row['baseline_probability'], row['candidate_probability']))
        if y not in (0, 1) or not 0 <= baseline <= 1 or not 0 <= candidate <= 1:
            raise ValueError('Invalid outcome/probability; pushes must be excluded')
        from zoneinfo import ZoneInfo
        day = start.astimezone(ZoneInfo('America/New_York')).date().isoformat()
        p, q = np.clip([baseline, candidate], 1e-7, 1-1e-7)
        loss_delta = -y*np.log(q)-(1-y)*np.log(1-q)+y*np.log(p)+(1-y)*np.log(1-p)
        clean.append((day, key[0], y, loss_delta, (q-y)**2-(p-y)**2))
    days = sorted({r[0] for r in clean})
    games = {r[1] for r in clean}
    report = {'rows':len(clean), 'games':len(games), 'days':len(days),
              'horizon_minutes':horizon_minutes, 'qualified':False,
              'reason':'insufficient held-out evidence'}
    if len(days) < 30 or len(games) < 200 or min(sum(r[2] == y for r in clean) for y in (0,1)) < 50:
        return report
    clusters = [np.array([[r[3],r[4]] for r in clean if r[0] == day]) for day in days]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        sample = [clusters[i] for i in rng.integers(0, len(days), len(days))]
        estimates.append(np.concatenate(sample).mean(axis=0))
    bounds = np.quantile(estimates, [.025, .975], axis=0)
    mean = np.array([[r[3],r[4]] for r in clean]).mean(axis=0)
    report.update(log_loss_delta=float(mean[0]), brier_delta=float(mean[1]),
                  log_loss_delta_ci95=bounds[:,0].tolist(), brier_delta_ci95=bounds[:,1].tolist(),
                  qualified=bool(bounds[1,0] < 0 and bounds[1,1] <= 0),
                  reason='requires log-loss improvement and no Brier degradation at the 95% interval')
    return report
