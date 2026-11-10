from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.points_v5 import prepare, injury_evidence, enrich_injuries


def test_short_windows_missing_not_fabricated_and_mature_values_unchanged():
    frame = pd.DataFrame({'games_played': [2, 10], 'trend_slope_5games': [12., 2.],
                          'minutes_last_3_games': [50., 90.], 'effective_fg_pct': [.9, .6],
                          'team_key_players_out': [np.nan, 0.], 'opp_key_players_out': [1., 0.]})
    out = prepare(frame)
    assert out.loc[0, ['trend_slope_5games', 'minutes_last_3_games', 'effective_fg_pct']].isna().all()
    assert out.loc[1, 'trend_slope_5games'] == 2.
    assert out.loc[1, 'minutes_last_3_games'] == 90.
    assert out.history_appearances.tolist() == [2, 10]
    assert out.team_injury_unknown.tolist() == [1, 0]
    assert frame.loc[0, 'trend_slope_5games'] == 12.  # no mutation of frozen inputs


def context(report):
    return SimpleNamespace(injuries=lambda team, date: report,
        inputs=SimpleNamespace(season='2026-27'),
        cutoff=lambda date: pd.Timestamp('2026-10-25T20:00:00Z'),
        resolve_name=lambda name: 1 if name == 'Known Player' else None,
        logs=pd.DataFrame([{'PLAYER_ID': 1, 'TEAM_ID': 10, 'GAME_DATE': '2026-10-24', 'MIN': 25},
                           {'PLAYER_ID': 2, 'TEAM_ID': 10, 'GAME_DATE': '2026-10-26', 'MIN': 40}]), prior_players={})


def roster():
    return {'season': '2026-27', 'fetched_at': pd.Timestamp('2026-10-25T19:00:00Z').timestamp(),
            'players': [{'nba_id': 2, 'team_id': 10, 'full_name': 'New Rookie'}]}


def test_roster_resolves_identity_but_unknown_role_is_not_zero():
    ctx = context([{'player': 'Known Player', 'status': 'Out'}, {'player': 'New Rookie', 'status': 'Out'}])
    evidence, roles = injury_evidence(ctx, 10, '2026-10-25', roster())
    assert evidence == {'injury_report_available': 1., 'out_count': 2., 'out_unresolved_count': 0.,
                        'out_unknown_role_count': 1., 'known_key_players_out': 1.}
    assert roles is None
    fixed = enrich_injuries({}, ctx, 10, 10, '2026-10-25', roster())
    assert np.isnan(fixed['team_key_players_out'])
    assert fixed['team_out_count'] == 2


def test_missing_report_distinct_from_explicit_empty_report():
    missing, roles = injury_evidence(context(None), 10, '2026-10-25')
    empty, empty_roles = injury_evidence(context([]), 10, '2026-10-25')
    assert missing['injury_report_available'] == 0 and np.isnan(missing['out_count']) and roles is None
    assert empty['injury_report_available'] == 1 and empty['out_count'] == 0 and empty_roles == []


@pytest.mark.parametrize('changes', [{'season': '2025-26'}, {'fetched_at': pd.Timestamp('2026-10-25T21:00:00Z').timestamp()},
                                   {'fetched_at': pd.Timestamp('2026-10-25T01:00:00Z').timestamp()}])
def test_wrong_season_future_or_stale_roster_rejected(changes):
    with pytest.raises(ValueError, match='Roster'):
        injury_evidence(context([{'player': 'New Rookie', 'status': 'Out'}]), 10, '2026-10-25', {**roster(), **changes})


def test_ambiguous_roster_name_stays_unresolved():
    r = roster(); r['players'].append({'nba_id': 3, 'team_id': 10, 'full_name': 'New Rookie'})
    evidence, roles = injury_evidence(context([{'player': 'New Rookie', 'status': 'Out'}]), 10, '2026-10-25', r)
    assert evidence['out_unresolved_count'] == 1 and roles is None


def test_real_training_rejects_prospective_games_and_post_tipoff_quotes():
    from scripts.build_real_points_training import validate_record
    summary = {'gameId': '0022400001', 'gameTimeUTC': '2024-10-25T23:00:00Z'}
    row = {'game_date': '2024-10-25', 'market': 'player_points', 'captured_at': '2024-10-25T22:30:00Z',
           'line': 20.5, 'over_odds': -110, 'under_odds': -110, 'bookmaker': 'a', 'source': 'archive'}
    assert validate_record(row, summary, set()) == pd.Timestamp(row['captured_at'])
    with pytest.raises(ValueError, match='Protected'):
        validate_record(row, summary, {'0022400001'})
    with pytest.raises(ValueError, match='Protected'):
        validate_record({**row, 'game_date': '2026-10-25'}, summary, set())
    with pytest.raises(ValueError, match='tipoff'):
        validate_record({**row, 'captured_at': '2024-10-25T23:00:00Z'}, summary, set())


def test_historical_cost_estimate_blocks_broad_request_before_charging(tmp_path, monkeypatch):
    from scripts import fetch_free_historical_props as feed
    monkeypatch.setenv('DUNKAPI_API_KEY', 'test-only-key')
    monkeypatch.setattr(feed, 'OUT', tmp_path)
    calls = []
    def post(url, **kwargs):
        calls.append(kwargs['json'])
        return SimpleNamespace(raise_for_status=lambda: None,
            json=lambda: {'data': {'estimate_only': True, 'credits': 1000}})
    monkeypatch.setattr(feed.requests, 'post', post)
    with pytest.raises(ValueError, match='Estimated cost'):
        feed.fetch(player_id='1')
    assert len(calls) == 1 and calls[0]['estimate_only']
    assert not list(tmp_path.glob('budget_*.json'))


def test_existing_frozen_validation_sources_still_match():
    from scripts.validate_prospective_points import protocol
    # Compared against the constant, not a literal, so a policy bump cannot
    # leave this assertion quietly pinning a retired version.
    from src.audited_features import POLICY_VERSION
    assert protocol()['feature_policy'] == POLICY_VERSION
