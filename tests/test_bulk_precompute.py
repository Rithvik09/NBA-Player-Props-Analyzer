"""Bulk collectors preserve shooting denominators and historical membership."""
import pandas as pd
import pytest
from src import precompute_jobs as jobs


class Endpoint:
    def __init__(self, frame):
        self.frame = frame

    def get_data_frames(self):
        return [self.frame]


def test_bulk_zones_merge_traded_rows_without_double_counting_corners(monkeypatch):
    from nba_api.stats.endpoints import leaguedashplayershotlocations as endpoint
    data = []
    for makes, attempts in [(2., 4.), (1., 6.)]:
        row = {('', 'PLAYER_ID'): 999}
        for zone in ('Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Above the Break 3'):
            row[(zone, 'FGM')] = 0.
            row[(zone, 'FGA')] = 0.
        row[('Corner 3', 'FGM')], row[('Corner 3', 'FGA')] = makes, attempts
        # These subtotal components must not also be accumulated.
        row[('Left Corner 3', 'FGA')] = attempts
        data.append(row)
    frame = pd.DataFrame(data)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    monkeypatch.setattr(endpoint, 'LeagueDashPlayerShotLocations', lambda **kw: Endpoint(frame))
    rows = jobs.compute_shot_zone_breakdown('2022-23')
    assert rows[0]['player_id'] == 999
    assert rows[0]['corner3_fga_pct'] == 1.
    assert rows[0]['corner3_fg_pct'] == pytest.approx(.3)
    assert sum(rows[0][k] for k in rows[0] if k.endswith('_fga_pct')) == 1.


def test_bulk_quarters_include_players_from_the_requested_season(monkeypatch):
    calls = []
    def endpoint(**kw):
        calls.append(kw)
        return Endpoint(pd.DataFrame([{'PLAYER_ID': 999, 'PTS': float(kw['period']), 'MIN': 5.}]))
    monkeypatch.setattr(jobs.leaguedashplayerstats, 'LeagueDashPlayerStats', endpoint)
    monkeypatch.setattr(jobs.time, 'sleep', lambda _: None)
    rows = jobs.compute_quarter_splits('2022-23')
    assert len(calls) == 4
    assert all(c['season'] == '2022-23' for c in calls)
    assert rows == [{'player_id': 999, 'q1_avg': 1., 'q2_avg': 2., 'q3_avg': 3., 'q4_avg': 4., 'q4_min_pg': 5.}]


def test_yoy_does_not_use_future_seasons_or_count_a_missed_season(monkeypatch):
    calls = []
    def averages(season):
        calls.append(season)
        pts = {'2021-22': 10., '2022-23': 99., '2023-24': 20.}[season]
        if season == '2022-23':
            return {222: {'PTS': pts, 'FGA': 9., 'FTA': 2.}}
        return {999: {'PTS': pts, 'FGA': 10., 'FTA': 2.}}
    monkeypatch.setattr(jobs, '_season_player_averages', averages)
    monkeypatch.setattr(jobs.commonallplayers, 'CommonAllPlayers', lambda **kw: Endpoint(pd.DataFrame([
        {'PERSON_ID': 999, 'FROM_YEAR': '2021'}])))
    rows = jobs.compute_player_yoy('2023-24')
    assert rows[0]['yoy_pts_change'] == 10.
    assert rows[0]['seasons_in_league'] == 2
    assert set(calls) <= {'2021-22', '2022-23', '2023-24'}


def test_bulk_vs_opponent_uses_historical_players(monkeypatch):
    from nba_api.stats.endpoints import leaguegamelog
    frame = pd.DataFrame([
        {'PLAYER_ID': 999, 'MATCHUP': 'BOS vs. LAL', 'FGA': 10., 'FTA': 2., 'PTS': 20., 'FG_PCT': .5, 'MIN': 30.},
        {'PLAYER_ID': 999, 'MATCHUP': 'BOS @ LAL', 'FGA': 20., 'FTA': 4., 'PTS': 30., 'FG_PCT': .4, 'MIN': 32.},
    ])
    monkeypatch.setattr(leaguegamelog, 'LeagueGameLog', lambda **kw: Endpoint(frame))
    rows = jobs.compute_player_vs_opponent('2022-23')
    assert len(rows) == 1
    assert rows[0]['gp'] == 2
    assert rows[0]['avg_stat_pts'] == 25.
    assert rows[0]['ts_pct'] == pytest.approx(25. / (2 * (15. + .44 * 3.)))


def test_rest_retry_can_target_only_missing_teams(monkeypatch):
    from nba_api.stats.endpoints import teamdashboardbygeneralsplits
    monkeypatch.setattr(jobs.teams, 'get_teams', lambda: [{'id': 11}, {'id': 22}])
    calls = []
    def endpoint(**kw):
        calls.append(kw)
        assert kw['team_id'] == 22
        data = [{'GROUP_SET': 'DAYS REST', 'GROUP_VALUE': '0 Days Rest', 'PTS': 110., 'DEF_RATING': 112., 'PACE': 99.},
                {'GROUP_SET': 'DAYS REST', 'GROUP_VALUE': '2 Days Rest', 'PTS': 120., 'DEF_RATING': 108., 'PACE': 101.}]
        return Endpoint(pd.DataFrame(data))
    monkeypatch.setattr(teamdashboardbygeneralsplits, 'TeamDashboardByGeneralSplits', endpoint)
    monkeypatch.setattr(jobs.time, 'sleep', lambda _: None)
    rows = jobs.compute_team_rest_splits('2022-23', team_ids=[22])
    assert rows == [{'team_id': 22, 'b2b_def_rating': 112., 'b2b_pace': 99.,
                     'b2b_pts_allowed': 110., 'rested_def_rating': 108., 'rested_pace': 101.}]
    assert len(calls) == 2 and all(c['timeout'] == 10 for c in calls)


def test_membership_falls_back_to_bio_without_inventing_inactive_players(monkeypatch):
    jobs._season_player_members.cache_clear()
    def unavailable(season):
        raise RuntimeError('base endpoint unavailable')
    monkeypatch.setattr(jobs, '_season_player_averages', unavailable)
    monkeypatch.setattr(jobs.leaguedashplayerbiostats, 'LeagueDashPlayerBioStats', lambda **kw: Endpoint(pd.DataFrame([
        {'PLAYER_ID': 11, 'GP': 30}, {'PLAYER_ID': 22, 'GP': 0}])))
    assert jobs._season_player_members('2014-15') == frozenset({11})
    jobs._season_player_members.cache_clear()


def test_on_off_uses_status_labels_and_reuses_successful_team_cache(monkeypatch, tmp_path):
    from nba_api.stats.endpoints import teamplayeronoffsummary
    monkeypatch.setattr(jobs, '__file__', str(tmp_path / 'src' / 'jobs.py'))
    monkeypatch.setattr(jobs.teams, 'get_teams', lambda: [{'id': 11}])
    calls = []
    class Response:
        def get_data_frames(self):
            # Match the real endpoint order: OFF comes before ON.
            return [pd.DataFrame(),
                    pd.DataFrame([{'VS_PLAYER_ID': 999, 'COURT_STATUS': 'Off', 'NET_RATING': -3.}]),
                    pd.DataFrame([{'VS_PLAYER_ID': 999, 'COURT_STATUS': 'On', 'NET_RATING': 8.}])]
    def endpoint(**kw):
        calls.append(kw)
        return Response()
    monkeypatch.setattr(teamplayeronoffsummary, 'TeamPlayerOnOffSummary', endpoint)
    expected = [{'player_id': 999, 'on_court_net_rating': 8.,
                 'off_court_net_rating': -3., 'on_off_differential': 11.}]
    assert jobs.compute_on_off_ratings('2022-23') == expected
    assert jobs.compute_on_off_ratings('2022-23') == expected
    assert len(calls) == 1


def test_historical_averages_survive_restart_cache_clear(monkeypatch, tmp_path):
    jobs._season_player_averages.cache_clear()
    monkeypatch.setattr(jobs, '__file__', str(tmp_path / 'src' / 'jobs.py'))
    calls = []
    def endpoint(**kw):
        calls.append(kw)
        return Endpoint(pd.DataFrame([{'PLAYER_ID': 999, 'PTS': 12., 'FGA': 10.}]))
    monkeypatch.setattr(jobs.leaguedashplayerstats, 'LeagueDashPlayerStats', endpoint)
    monkeypatch.setattr(jobs.time, 'sleep', lambda _: None)
    first = jobs._season_player_averages('2022-23')
    jobs._season_player_averages.cache_clear()
    assert jobs._season_player_averages('2022-23') == first
    assert len(calls) == 1
    jobs._season_player_averages.cache_clear()


def test_synergy_ratio_fetch_uses_equivalent_mode_if_first_fails(monkeypatch):
    modes = []
    def endpoint(**kw):
        modes.append(kw['per_mode_simple'])
        if len(modes) == 1:
            raise RuntimeError('cached endpoint failure')
        return Endpoint(pd.DataFrame([{'PLAYER_ID': 999, 'PPP': 1.1, 'POSS_PCT': .2}]))
    monkeypatch.setattr(jobs.synergyplaytypes, 'SynergyPlayTypes', endpoint)
    frame = jobs._synergy_frame(season='2023-24', play_type_nullable='Cut')
    assert modes == ['Totals', 'PerGame']
    assert frame.iloc[0]['PPP'] == 1.1


def test_empty_shot_zones_do_not_poison_all_player_shares(monkeypatch):
    from nba_api.stats.endpoints import leaguedashplayershotlocations as endpoint
    data = []
    for pid, attempts in [(11, 2.), (22, 0.)]:
        row = {('', 'PLAYER_ID'): pid}
        for zone in ('Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range', 'Corner 3', 'Above the Break 3'):
            row[(zone, 'FGM')] = float('nan')
            row[(zone, 'FGA')] = float('nan')
        row[('Restricted Area', 'FGM')] = attempts
        row[('Restricted Area', 'FGA')] = attempts
        data.append(row)
    frame = pd.DataFrame(data)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    monkeypatch.setattr(endpoint, 'LeagueDashPlayerShotLocations', lambda **kw: Endpoint(frame))
    rows = jobs.compute_shot_zone_breakdown('2022-23')
    assert len(rows) == 1 and rows[0]['player_id'] == 11
    assert rows[0]['rim_fga_pct'] == rows[0]['rim_fg_pct'] == 1.
    assert rows[0]['paint_fga_pct'] == 0.
