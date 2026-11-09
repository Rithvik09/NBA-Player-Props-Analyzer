from datetime import datetime, timedelta, timezone
import pytest
from src.sharp_evidence import pressure_components, signed_score, validate_holdout


def observation(time, line, **kw):
    return dict(scope='prop', game_id='g', player_id=1, prop_type='points',
                source='provider', bookmaker='book', game_start='2025-01-01T20:00:00Z',
                observed_at=f'2025-01-01T{time}:01Z', published_at=f'2025-01-01T{time}:00Z',
                line=line, over_price=-110, under_price=-110, **kw)


def pressure(rows, **kw):
    return pressure_components(rows, cutoff='2025-01-01T19:00:00Z', game_id='g',
                               scope='prop', player_id=1, prop_type='points', **kw)


def test_direction_uses_earlier_splits_and_excludes_future_wrong_scope():
    rows = [observation('18:10',20.5,tickets_fraction=.2,money_fraction=.6),
            observation('18:55',21.5,tickets_fraction=.8,money_fraction=.9)]
    result = pressure(rows)
    assert result['prop_reverse_movement_direction'] == 1
    assert result['prop_money_move_confirmation'] == pytest.approx(.4)
    assert result['prop_money_ticket_gap'] == pytest.approx(.1)
    assert 'prop_same_line_over_probability_change' not in result
    bad = [dict(rows[-1], observed_at='2025-01-01T19:30:00Z', line=99),
           dict(rows[-1], scope='game_total', line=230), dict(rows[-1], player_id=2)]
    assert pressure(rows+bad) == result
    assert signed_score(.6,.55) == pytest.approx(-5)


def test_missing_publication_cannot_confirm_movement_and_sources_do_not_duplicate_books():
    rows = [observation('18:10',20.5,tickets_fraction=.2,money_fraction=.6),
            observation('18:55',21.5,tickets_fraction=.2,money_fraction=.6)]
    copies = [dict(r,source='syndicated') for r in rows]
    assert pressure(rows+copies) == pressure(rows)
    result = pressure([dict(r,published_at=None) for r in rows])
    assert result['prop_money_ticket_gap'] == pytest.approx(.4)
    assert 'prop_interval_line_change' not in result
    game = [dict(r,scope='game_total',line=r['line']+200) for r in rows]
    context = pressure_components(game, cutoff='2025-01-01T19:00:00Z',game_id='g',scope='game_total')
    assert all(k.startswith('game_total_context_') for k in context)
    assert pressure(game) == {}


def heldout(n=240):
    result = []
    for i in range(n):
        start = datetime(2025,1,2,20,tzinfo=timezone.utc)+timedelta(days=i//4)
        cutoff = start-timedelta(hours=1)
        y = i % 2
        result.append(dict(game_id=str(i),player_id=1,prop_type='points',cutoff=cutoff.isoformat(),
                           requested_at=cutoff.isoformat(),game_start=start.isoformat(),
                           input_latest_at=cutoff.isoformat(),training_outcomes_through='2024-12-01T00:00:00Z',
                           result_recorded_at=(start+timedelta(hours=4)).isoformat(),over_outcome=y,
                           baseline_probability=.5,candidate_probability=.7 if y else .3))
    return result


def evaluate(rows, horizon=60):
    return validate_holdout(rows, horizon_minutes=horizon,
                            selection_outcomes_through='2025-01-01T00:00:00Z',bootstrap_samples=100)


def test_gate_rejects_small_samples_duplicates_future_inputs_and_wrong_horizon():
    rows = heldout()
    assert evaluate(rows)['qualified']
    assert not evaluate(rows[:10])['qualified']
    assert not evaluate([dict(r,candidate_probability=.5) for r in rows])['qualified']
    with pytest.raises(ValueError,match='Duplicate'): evaluate(rows+rows[:1])
    with pytest.raises(ValueError,match='Future'): evaluate([dict(rows[0],input_latest_at='2026-01-01T00:00:00Z')])
    with pytest.raises(ValueError,match='horizons'): evaluate(rows,15)
    assert evaluate(rows,'request')['qualified']
    with pytest.raises(ValueError,match='recorded request'):
        evaluate([dict(rows[0],requested_at='2025-01-01T00:00:00Z')],'request')
