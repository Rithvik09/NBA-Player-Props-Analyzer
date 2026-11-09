"""Strict, pure as-of adapters for missing market and prediction-history data.

These helpers do not enable a feature without observations. Unzoned legacy
timestamps and retrospective forecasts are rejected rather than assumed safe.
"""
import math
from datetime import datetime, timezone
import numpy as np


def instant(value):
    stamp=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if stamp.tzinfo is None:raise ValueError('Timezone-aware source timestamp required')
    return stamp.astimezone(timezone.utc)


def finite(value):
    value=float(value)
    if not math.isfinite(value):raise ValueError('Nonfinite source value')
    return value


def performance_features(rows, *, cutoff, player_id, prop_type, model_version, window=50):
    """One latest pregame prediction per game, exact model version, known grades.

    Scores describe genuinely earlier served predictions, not fitted values.
    Missing provenance cannot establish out-of-sample status. Pushes are excluded
    from binary accuracy/calibration, but remain valid absolute-error samples.
    """
    if not isinstance(model_version,str) or not model_version.strip():return {}
    cutoff=instant(cutoff);games={}
    for row in rows:
        try:
            if int(row['player_id'])!=int(player_id) or row['prop_type']!=prop_type or row['model_version']!=model_version:continue
            predicted=instant(row['predicted_at']);start=instant(row['game_start'])
            trained=instant(row['training_outcomes_through']);graded=instant(row['result_recorded_at'])
            if not trained<predicted<start<graded<cutoff:continue
            key=str(row['game_id'])
            if not key or key=='None':continue
            clean={**row,'_time':predicted,'actual':finite(row['actual']),
                'prediction':finite(row['prediction']),'line':finite(row['line']),
                'over_probability':finite(row['over_probability'])}
            if not 0<=clean['over_probability']<=1 or clean['actual']<0 or clean['line']<0:continue
            if key not in games or predicted>games[key]['_time']:games[key]=clean
        except (KeyError,ValueError,TypeError):continue
    prior=sorted(games.values(),key=lambda r:r['_time'])[-window:];out={}
    if len(prior)>=20:
        out['avg_prediction_error_player']=float(np.mean([abs(r['actual']-r['prediction']) for r in prior]))
    binary=[r for r in prior if r['actual']!=r['line']]
    if len(binary)>=20:
        out['model_accuracy_player']=float(np.mean([(r['over_probability']>=.5)==(r['actual']>r['line']) for r in binary]))
    if len(binary)>=50:
        error=0
        for index in range(5):
            bucket=[r for r in binary if min(4,int(r['over_probability']*5))==index]
            if bucket:error+=len(bucket)/len(binary)*abs(np.mean([r['over_probability'] for r in bucket])-np.mean([r['actual']>r['line'] for r in bucket]))
        out['calibration_score_player']=1-float(error)
    return out


def implied(price):
    price=finite(price)
    if abs(price)<100:raise ValueError('Expected American price')
    return 100/(price+100) if price>0 else -price/(-price+100)


def game_total_features(rows, *, cutoff, game_id, fresh_minutes=30):
    """Median fresh full-game bookmaker total at the decision cutoff."""
    cutoff = instant(cutoff)
    latest = {}
    for row in rows:
        try:
            if str(row['game_id']) != str(game_id) or row['market'] != 'totals':
                continue
            if row.get('is_alternate', False) or row.get('team') or row.get('period') not in (None, '', 'full_game'):
                continue
            observed, published, start = map(instant, (row['observed_at'], row['published_at'], row['game_start']))
            if not published <= observed <= cutoff < start:
                continue
            book = row['bookmaker']
            if not isinstance(book, str) or not book.strip() or not row.get('source'):
                continue
            line = finite(row['line'])
            if line <= 0:
                continue
            if book not in latest or (published, observed) > latest[book][:2]:
                latest[book] = (published, observed, line)
        except (KeyError, ValueError, TypeError):
            continue
    values = [line for published, _, line in latest.values()
              if (cutoff - published).total_seconds() <= fresh_minutes * 60]
    return {'implied_game_total': float(np.median(values))} if values else {}


def market_features(rows, *, cutoff, game_id, player_id, prop_type, decision_line=None,
                    fresh_minutes=30, stale_minutes=10):
    """Use observed pregame main-line quotes; keep implied price tied to its line.

    First-seen prices are not asserted to be the market's opening prices. A
    true opening quote requires the source's explicit ``is_opening`` flag.
    Collection time must be at least the source publication time; neither can
    be after the decision cutoff. No latest-summary or closing-line lookup.
    """
    cutoff=instant(cutoff);valid=[]
    for row in rows:
        try:
            if str(row['game_id'])!=str(game_id) or int(row['player_id'])!=int(player_id) or row['prop_type']!=prop_type or row.get('is_alternate',False):continue
            captured=instant(row['observed_at']);published=instant(row['published_at']);start=instant(row['game_start'])
            if not published<=captured<=cutoff<start:continue
            if not isinstance(row['bookmaker'],str) or not row['bookmaker'].strip() or not row.get('source'):continue
            valid.append({**row,'_observed':captured,'_published':published,'line':finite(row['line'])})
        except (KeyError,ValueError,TypeError):continue
    latest={}
    for row in sorted(valid,key=lambda r:(r['_published'],r['_observed'])):latest[row['bookmaker']]=row
    fresh=[r for r in latest.values() if (cutoff-r['_published']).total_seconds()<=fresh_minutes*60]
    if not fresh:return {}
    current=float(np.median([r['line'] for r in fresh]));out={'bookmaker_count':float(len(fresh))}
    if len(fresh)>=2:out['market_consensus_std']=float(np.std([r['line'] for r in fresh]))
    # Movement/velocity are same-book comparisons, then aggregated. Do not
    # manufacture movement merely by changing the set of contributing books.
    movements=[];speeds=[];openings=[]
    for row in fresh:
        first=[r for r in valid if r['bookmaker']==row['bookmaker'] and r.get('is_opening') is True and r['_published']<=row['_published']]
        if not first:continue
        opening=min(first,key=lambda r:r['_published']);openings.append(opening['line'])
        delta=row['line']-opening['line'];movements.append(delta)
        hours=(row['_published']-opening['_published']).total_seconds()/3600
        if hours>0:speeds.append(delta/hours)
    if openings:out['opening_line']=float(np.median(openings));out['line_movement']=float(np.median(movements))
    if speeds:out['line_velocity']=float(np.median(speeds))
    if decision_line is not None:
        prices=[]
        for row in fresh:
            if row['line']!=float(decision_line):continue
            try:
                over,under=implied(row['over_price']),implied(row['under_price'])
                prices.append(over/(over+under))
            except (KeyError,ValueError,TypeError):continue
        if prices:out['implied_over_prob']=float(np.mean(prices))
    # Stale means an observed unchanged old quote disagrees with fresh peer
    # main lines. Unknown/missing books are never assumed stale.
    stale=[r for r in latest.values() if (cutoff-r['_published']).total_seconds()>stale_minutes*60 and abs(r['line']-current)>=.5]
    # A provider polling timestamp does not reveal how long a quote was unchanged.
    if len(fresh)>=2 and all(r.get('timestamp_semantics')!='provider_last_seen_market' for r in latest.values()):
        out['stale_line_flag']=float(bool(stale))
    return out
