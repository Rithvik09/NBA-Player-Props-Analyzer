import json
import pytest
from src.prospective_points import validate_quote, select_quotes, settlement, paired_report, save_new


def quote():
    return {'game_id': '0022600112', 'player_id': 1, 'bookmaker': 'a', 'line': 20.5,
            'over_price': -110, 'under_price': -110, 'scope': 'prop', 'prop_type': 'points',
            'observed_at': '2026-10-25T20:00:00Z', 'published_at': '2026-10-25T19:59:00Z',
            'game_start': '2026-10-25T21:00:00Z'}


def forecast(identity='one', cohort='eligible'):
    return {'id': identity, 'cohort': cohort, 'horizon': 'request_time', 'quote': quote(),
            'predictions': {'candidate': {'points': 23., 'over_probability': .7},
                            'compact': {'points': 21., 'over_probability': .5}}}


def test_timing_and_horizons_fail_closed():
    q = quote()
    assert validate_quote(q, '2026-10-25T20:00:01Z', '60m') > 59
    for moment in ['2026-10-25T19:59:59Z', '2026-10-25T20:11:00Z', '2026-10-25T21:00:00Z']:
        with pytest.raises(ValueError):
            validate_quote(q, moment)
    with pytest.raises(ValueError):
        validate_quote(q, '2026-10-25T20:00:01Z', '15m')
    with pytest.raises(ValueError):
        validate_quote({**q, 'published_at': '2026-10-25T20:01:00Z'}, '2026-10-25T20:02:00Z')


def test_one_book_selected_without_price_or_edge_cherry_pick():
    a = quote()
    b = {**a, 'bookmaker': 'b', 'line': 10.5}
    assert select_quotes([b, a, b]) == [a]


def test_forecasts_cannot_be_overwritten(tmp_path):
    path = tmp_path/'f.json'
    save_new(path, forecast())
    with pytest.raises(FileExistsError):
        save_new(path, {'changed': True})
    assert json.loads(path.read_text()) == forecast()


def test_final_identity_dnp_push_and_missing_outcomes():
    f = forecast()
    summary = {'gameId': f['quote']['game_id'], 'gameStatus': 3}
    person = {'personId': 1, 'statistics': {'minutes': '30:00', 'points': 23}}
    boxes = {'gameId': summary['gameId'], 'homeTeam': {'players': [person]}, 'awayTeam': {'players': []}}
    assert settlement(f, summary, boxes)['over'] == 1
    assert settlement(f, {**summary, 'gameStatus': 2}, boxes)['status'] == 'pending'
    with pytest.raises(ValueError):
        settlement(f, {**summary, 'gameId': 'wrong'}, boxes)
    f['quote']['line'] = 23
    assert settlement(f, summary, boxes)['status'] == 'push'
    person['statistics']['minutes'] = '0:00'
    assert settlement(f, summary, boxes)['status'] == 'dnp'
    person['statistics']['minutes'] = ''
    assert settlement(f, summary, boxes)['status'] == 'missing_minutes'
    boxes['homeTeam']['players'] = []
    assert settlement(f, summary, boxes)['status'] == 'missing_player'


def test_paired_metrics_separate_warmup_and_exclude_push_binary():
    a, b, c = forecast(), forecast('two', 'warmup'), forecast('three')
    b['quote']['player_id'] = 2
    c['quote']['player_id'] = 3
    report = paired_report([a, b, c], {'one': {'status': 'settled', 'points': 23., 'over': 1},
                                     'two': {'status': 'settled', 'points': 40., 'over': 1},
                                     'three': {'status': 'push', 'points': 21., 'over': None}})
    primary = report['comparisons']['eligible/request_time']
    assert primary['binary_player_games'] == 1
    assert primary['player_games_with_points'] == 2
    assert primary['candidate_minus_compact_log_loss'] < 0
    assert primary['candidate']['auc'] is None
    assert not primary['checkpoint_ready']
    assert report['comparisons']['warmup/request_time']['binary_player_games'] == 1


def test_report_rejects_duplicate_player_game():
    with pytest.raises(ValueError, match='Duplicate'):
        paired_report([forecast(), forecast('second-book')], {})
