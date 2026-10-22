"""Current data identities, source failures and game-specific crews."""
import json
import time
from unittest.mock import patch

import pytest
from bs4 import BeautifulSoup

from src import current_rosters as rosters
from src import live_injuries as injuries
from src import live_officials as officials
from src.injury_tracker import InjuryTracker


def test_espn_json_columns_and_season():
    data = {'season': {'displayName': rosters.season_now()}, 'injuries': [
        {'displayName': 'Phoenix Suns', 'injuries': [
            {'athlete': {'displayName': 'Player'}, 'status': 'Out', 'date': '2026-10-22',
             'details': {'type': 'Ankle', 'returnDate': '2026-11-01'}}]}]}
    with patch.object(rosters, 'get_json', return_value=data):
        result = injuries._fetch_espn_json()['Phoenix Suns'][0]
    assert result['status'] == 'Out'
    assert result['severity'] == 1
    assert result['injury'] == 'Ankle'
    assert result['return'] == '2026-11-01'
    data['season']['displayName'] = '2020-21'
    with patch.object(rosters, 'get_json', return_value=data), pytest.raises(ValueError):
        injuries._fetch_espn_json()


def test_espn_five_column_html():
    soup = BeautifulSoup('<tr><td>Player</td><td>PG</td><td>Oct 25</td><td>Out</td><td>Ankle</td></tr>', 'html.parser')
    row = injuries._parse_espn_rows(soup.select('tr'))[0]
    assert (row['status'], row['injury'], row['return']) == ('Out', 'Ankle', 'Oct 25')


def test_unavailable_does_not_mean_healthy():
    with patch.object(injuries, '_fetch_espn', return_value={}), patch.object(injuries, '_fetch_rotowire', return_value={}):
        with pytest.raises(injuries.InjuryDataUnavailable):
            injuries.fetch_live_injuries.__wrapped__()
    with patch.object(injuries, 'summarise_team', side_effect=injuries.InjuryDataUnavailable):
        with pytest.raises(injuries.InjuryDataUnavailable):
            InjuryTracker().get_team_injuries(1610612756)


def test_abbreviation_resolves_full_team():
    data = {'Phoenix Suns': [{'player': 'Example', 'severity': 1, 'status': 'Out'}]}
    with patch.object(injuries, 'fetch_live_injuries', return_value=data):
        assert injuries.summarise_team('PHX')['out_count'] == 1


def test_old_roster_not_used(tmp_path, monkeypatch):
    path = tmp_path / 'roster.json'
    monkeypatch.setattr(rosters, 'SNAPSHOT', path)
    path.write_text(json.dumps({'season': '2020-21', 'fetched_at': time.time(), 'players': []}))
    assert rosters.load_snapshot() is None
    path.write_text(json.dumps({'season': rosters.season_now(), 'fetched_at': time.time() - 22000, 'players': []}))
    assert rosters.load_snapshot() is None


def test_current_team_overrides_old_player_info():
    from src.basketball_betting_helper import BasketballBettingHelper
    helper = BasketballBettingHelper.__new__(BasketballBettingHelper)
    with patch.object(rosters, 'current_player', return_value={'team_id': 1610612738}), patch('src.api_cache.fetch_player_info') as old:
        assert helper._get_player_team_id(1) == 1610612738
        old.assert_not_called()


def test_different_matchups_do_not_reuse_crew():
    from src.basketball_betting_helper import BasketballBettingHelper as Helper
    def crew(team, opponent):
        return ['Ref A'] if team == 1 else ['Ref B']
    data = {'refs': {'ref a': {'foul_rate': 30, 'home_win_pct': .4, 'pace': 99},
                     'ref b': {'foul_rate': 50, 'home_win_pct': .6, 'pace': 105}}}
    with patch.object(officials, 'matchup_officials', side_effect=crew):
        assert Helper._get_referee_features(None, data, 1, 2)['ref_foul_rate'] == 30
        assert Helper._get_referee_features(None, data, 3, 4)['ref_foul_rate'] == 50


def test_provider_game_ids_are_not_nba_ids():
    event = {'id': '401234567', 'date': '2026-10-22T23:00Z', 'competitions': [
        {'status': {'type': {'completed': True}}, 'competitors': [
            {'team': {'abbreviation': 'BOS'}, 'homeAway': 'home', 'winner': True},
            {'team': {'abbreviation': 'NY'}, 'homeAway': 'away'}]}]}
    summary = {'gameInfo': {'officials': [{'fullName': 'Ref A'}]}}
    game = officials.parse_game(event, summary, {'BOS': 1, 'NY': 2})
    assert game['espn_game_id'] == '401234567'
    assert 'game_id' not in game
    assert game['officials'] == ['Ref A']
    assert game['total_fouls'] is None


def test_modern_summary_uses_explicit_home_team():
    from src.nba_officials import summary_officials
    data = {'gameStatus': 3, 'gameEt': '2026-10-22T19:00Z',
            'homeTeam': {'teamId': 2, 'score': 90}, 'awayTeam': {'teamId': 1, 'score': 100},
            'officials': [{'name': 'Actual Ref', 'personId': 42}]}
    refs, home_won, day, scores = summary_officials(data)
    assert refs == [{'name': 'Actual Ref', 'id': 42}]
    assert home_won is False
    assert day == '2026-10-22'
    assert scores == [(2, 90), (1, 100)]
    data['gameStatus'] = 1
    assert summary_officials(data)[1] is None


def test_historical_training_excludes_current_injuries(pg_dsn):
    from scripts.train_models import _load_precomputed_player_data
    data = _load_precomputed_player_data(pg_dsn)
    assert data['injury_status'] == {}


def test_la_clippers_alias_keeps_injuries():
    data = {'LA Clippers': [{'player': 'Player', 'severity': 1, 'status': 'Out'}]}
    with patch.object(injuries, 'fetch_live_injuries', return_value=data):
        assert injuries.summarise_team('Los Angeles Clippers')['out_count'] == 1
        assert injuries.summarise_team('LAC')['out_count'] == 1
