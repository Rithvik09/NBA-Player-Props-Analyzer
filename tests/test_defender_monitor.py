import pandas as pd
import pytest
from scripts.monitor_defenders import grade_one, combined_stats


def fixture():
    f = {'game_id': '0022600113', 'player_id': 1, 'opponent_id': 20,
         'defender_id': 2, 'baseline_defender_id': 3,
         'predicted_at': '2026-10-25T18:00:00Z', 'game_start': '2026-10-25T22:00:00Z'}
    rows = pd.DataFrame([{'personIdOff': 1, 'personIdDef': 2, 'defTeamId': 20, 'partialPossessions': 30},
                         {'personIdOff': 1, 'personIdDef': 3, 'defTeamId': 20, 'partialPossessions': 10}])
    players = pd.DataFrame([{'PLAYER_ID': 1, 'MIN': 30}])
    return f, rows, players


def test_grade_identity_timing_and_target():
    f, rows, players = fixture()
    result = grade_one(f, rows, players)
    assert result == {'status': 'settled', 'target': 2, 'correct': True, 'baseline_correct': False}
    assert grade_one({**f, 'predicted_at': f['game_start']}, rows, players)['status'] == 'rejected_late'
    assert grade_one({**f, 'opponent_id': 30}, rows, players)['status'] == 'rejected_identity'


def test_missing_dnp_and_ties_do_not_become_losses():
    f, rows, players = fixture()
    assert grade_one(f, rows, players.iloc[:0])['status'] == 'missing_player'
    assert grade_one(f, rows, players.assign(MIN=0))['status'] == 'dnp'
    assert grade_one(f, rows.iloc[:0], players)['status'] == 'missing_matchups'
    assert grade_one(f, rows.assign(partialPossessions=10), players)['status'] == 'ambiguous_tie'


def test_union_dedup_and_fixed_gates():
    held = [{'game_id': 'g', 'player_id': i, 'correct': i < 49, 'baseline_correct': i < 10} for i in range(100)]
    overlap = [{**held[0], 'status': 'settled'}]
    result = combined_stats(held, overlap)
    assert result['deduplicated_union']['n'] == 100
    assert not result['assignment_gates_passed']
    held[49]['correct'] = True
    assert combined_stats(held, [])['assignment_gates_passed']
    assert not combined_stats(held, [])['seven_features_qualified']
    assert not combined_stats(held[:99], [])['assignment_gates_passed']
    with pytest.raises(ValueError, match='Duplicate'):
        combined_stats(held, overlap+overlap)


def test_unpublished_matchups_remain_missing_before_parser():
    from scripts.monitor_defenders import parse_matchup_response
    p={'boxScoreMatchups':{'gameId':'g','homeTeam':{'players':[]},'awayTeam':{'players':[]}}}
    assert parse_matchup_response(p,'g') is None
    with pytest.raises(ValueError,match='identity'):parse_matchup_response(p,'wrong')


def test_missing_history_blocks_all_state_advancement(tmp_path):
    from scripts.monitor_defenders import advance_history
    class NoUpdates:
        def observe_matchups(self, *args):
            pytest.fail('Must not partially advance history')
        def observe_boxes(self, *args):
            pytest.fail('Must not partially advance history')
    later = pd.DataFrame([{'GAME_ID': 'g', 'GAME_DATE': '2026-10-25'}])
    assert advance_history(NoUpdates(), later, tmp_path) == ['g']
    assert advance_history(NoUpdates(), later.iloc[:0], tmp_path) == []
