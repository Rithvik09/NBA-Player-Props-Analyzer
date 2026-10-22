"""Compare complete model inputs with equal history, date and source presence."""
import numpy as np
import pytest

from src.ml_features import NUMERIC_FEATURE_KEYS, build_feature_vector
import tests.test_feature_defaults as fixture


def differing_features(train, serve):
    a = build_feature_vector(train).X.iloc[0]
    b = build_feature_vector(serve).X.iloc[0]
    assert list(a.index) == list(b.index) == NUMERIC_FEATURE_KEYS
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return {k: (float(a[k]), float(b[k])) for k in a.index
            if not np.isclose(a[k], b[k], rtol=1e-9, atol=1e-9)}


def varied_game_log():
    frame = fixture._canned_game_log()
    for i in range(len(frame)):
        for col in ('PTS AST REB STL BLK TOV FGA FTA OREB DREB FGM FG3M FG3A MIN PLUS_MINUS FG_PCT FG3_PCT').split():
            frame.loc[i, col] *= .7 + (i % 4) * .2
        frame.loc[i, 'WL'] = 'W' if i % 3 else 'L'
        frame.loc[i, 'MATCHUP'] = 'BOS @ LAL' if i % 2 == 0 else 'BOS vs. LAL'
    return frame


@pytest.mark.parametrize('vary', [False, True], ids=['flat', 'varying_box_scores'])
def test_all_416_inputs_match_with_equal_history(monkeypatch, vary):
    if vary:
        frame = varied_game_log()
        monkeypatch.setattr(fixture, '_canned_game_log', lambda: frame.copy())
    context = {'pace': 110.} if vary else {}
    train = fixture._train_feature_dict(monkeypatch, context)
    serve = fixture._serve_feature_dict_with_game_log(monkeypatch, context)
    assert not differing_features(train, serve)


def test_populated_sources_retain_official_efg_and_tracking_units(monkeypatch):
    frame = varied_game_log()
    monkeypatch.setattr(fixture, '_canned_game_log', lambda: frame.copy())
    pre = {
        'advanced': {(2544, '2024-25'): {'efg_pct': .61}},
        'shot_prof': {(2544, '2024-25'): {'open_shot_freq': .7, 'tight_shot_freq': .2, 'catch_shoot_freq': .4, 'pullup_freq': .5}},
        'tracking': {(2544, '2024-25'): {
            'time_of_poss_pg': 3.2, 'touches_pg': 64., 'avg_drib_per_touch': 2.3,
            'elbow_touches_pg': 4.}},
        'q_splits': {(2544, '2024-25'): {'q1_avg': 6.2, 'q4_avg': 4.1}},
        'clutch': {(2544, '2024-25'): {'clutch_min_pg': 2.7}},
    }
    sources = {
        'advanced': {'efg_pct_official': .61},
        'shot_profile': {'open_shot_frequency': .7, 'tight_shot_frequency': .2, 'catch_shoot_frequency': .4, 'pullup_frequency': .5},
        'tracking': {'tracking_time_of_poss_pg': 3.2, 'tracking_touches_pg': 64.,
                     'tracking_avg_drib_per_touch': 2.3, 'tracking_elbow_touches_pg': 4.},
        'quarters': {'q1_avg': 6.2, 'q4_avg': 4.1},
        'clutch': {'clutch_min_per_game': 2.7},
    }
    train = fixture._train_feature_dict(monkeypatch, {}, pre)
    serve = fixture._serve_feature_dict_with_game_log(monkeypatch, {}, sources)
    assert serve['effective_fg_pct'] == pytest.approx(.61)
    assert serve['time_of_possession_per_game'] == pytest.approx(3.2)
    keys = ('effective_fg_pct', 'time_of_possession_per_game', 'touches_per_possession',
            'avg_dribbles_per_touch', 'elbow_touches_per_game', 'first_quarter_avg',
            'fourth_quarter_avg', 'clutch_minutes_per_game', 'contested_shot_pct',
            'wide_open_shot_pct', 'catch_and_shoot_pct', 'pull_up_shot_pct')
    for key in keys:
        assert serve[key] == pytest.approx(train[key]), key


@pytest.mark.parametrize('matchup, expected', [('BOS vs. LAL', True), ('BOS vs LAL', True), ('BOS @ LAL', False)])
def test_training_recognizes_nba_home_marker(matchup, expected):
    from scripts.train_models import _parse_matchup
    assert _parse_matchup(matchup) == ('BOS', 'LAL', expected)


@pytest.mark.parametrize('prop_type', ['assists', 'rebounds', 'three_pointers'])
def test_varying_log_parity_across_props(monkeypatch, prop_type):
    frame = varied_game_log()
    monkeypatch.setattr(fixture, '_canned_game_log', lambda: frame.copy())
    train = fixture._train_feature_dict(monkeypatch, {}, prop_type=prop_type)
    serve = fixture._serve_feature_dict_with_game_log(monkeypatch, {}, prop_type=prop_type)
    assert not differing_features(train, serve)


def test_historical_lookup_never_substitutes_a_future_season():
    from scripts.train_models import _player_season_row
    table = {2544: {'season': '2025-26', 'efg_pct': .9},
             (2544, '2022-23'): {'season': '2022-23', 'efg_pct': .4}}
    assert _player_season_row(table, 2544, '2022-23')['efg_pct'] == .4
    assert _player_season_row(table, 2544, '2023-24') == {}


def test_analyze_passes_real_log_features_and_preserves_model_average_definitions(monkeypatch):
    """Cover the actual caller, including mutations after prepare_features."""
    from collections import defaultdict
    import pandas as pd
    import src.basketball_betting_helper as bbh
    from src.models import EnhancedMLPredictor
    import src.live_injuries

    frame = varied_game_log()
    monkeypatch.setattr(fixture, '_canned_game_log', lambda: frame.copy())
    eff = fixture._serve_efficiency(monkeypatch)
    stats = eff['_stats']
    helper = bbh.BasketballBettingHelper.__new__(bbh.BasketballBettingHelper)
    predictor = EnhancedMLPredictor.__new__(EnhancedMLPredictor)
    predictor._model_version = 'fixture'
    predictor.get_player_context = lambda *a: {'team_id': 1610612738}
    predictor.get_team_context = lambda *a, **kw: {'pace': 100., 'rest_days': 2}
    captured = []
    def predict(features, line, prop_type=None):
        captured.append(dict(features))
        return {'over_probability': .5, 'predicted_value': features['season_avg'],
                'recommendation': 'PASS', 'confidence': 'LOW', 'edge': 0.}
    predictor.predict = predict
    helper.ml_predictor = predictor
    helper.current_season = '2024-25'
    helper.get_player_stats = lambda _: stats
    helper._detect_game_type = lambda *a: None
    helper.log_prediction = lambda **kw: None
    helper._get_referee_features = lambda **kw: {'ref_foul_rate': 0., 'ref_home_bias': .5, 'ref_pace_tendency': 100.}
    class Store:
        def refresh(self, *a, **kw):
            return defaultdict(dict)
    helper._precomputed = Store()
    monkeypatch.setattr(src.live_injuries, 'matchup_injury_context', lambda *a: {})
    monkeypatch.setattr(bbh, 'TeamGameLog', lambda **kw: fixture._CannedEndpoint(pd.DataFrame()))
    import src.feature_context
    original = src.feature_context.apply_serving_context
    def at_fixture_date(features, inputs, team_context=None, opponent_context=None):
        original(features, {**inputs, '_prediction_date': '2025-01-21'}, team_context, opponent_context)
    monkeypatch.setattr(src.feature_context, 'apply_serving_context', at_fixture_date)
    result = helper.analyze_prop_bet(2544, 'points', 19.5, 1610612747, is_home=False)
    assert captured, result
    assert captured[0]['season_avg'] == pytest.approx(stats['points']['avg'])
    assert captured[0]['recent_avg'] == pytest.approx(stats['points']['last5_avg'])
    assert captured[0]['location_avg'] == pytest.approx(stats['points']['away_avg'])
    assert captured[0]['touches_per_game'] == pytest.approx(eff['_log_summary']['means']['FGA'] + eff['_log_summary']['means']['AST'])
    train = fixture._train_feature_dict(monkeypatch, {'pace': 100., 'rest_days': 2})
    assert differing_features(train, captured[0]) == {}



def test_audit_detects_removing_the_serving_correction(monkeypatch):
    import src.feature_context
    monkeypatch.setattr(src.feature_context, 'apply_serving_context', lambda *a, **kw: None)
    train = fixture._train_feature_dict(monkeypatch, {})
    serve = fixture._serve_feature_dict_with_game_log(monkeypatch, {})
    diffs = differing_features(train, serve)
    assert 'touches_per_game' in diffs
    assert 'shot_clock_management' in diffs


@pytest.mark.parametrize('vary', [False, True], ids=['flat_20_game_history', 'varying_20_game_history'])
def test_long_history_windows_match_training(monkeypatch, vary):
    frame = long_game_log(vary)
    monkeypatch.setattr(fixture, '_canned_game_log', lambda: frame.copy())
    train = fixture._train_feature_dict(monkeypatch, {}, history_count=20)
    serve = fixture._serve_feature_dict_with_game_log(monkeypatch, {}, history_count=20)
    assert differing_features(train, serve) == {}
    assert serve['consecutive_under_games'] <= 10


def long_game_log(vary=False):
    import pandas as pd
    frame = pd.concat([fixture._canned_game_log(), fixture._canned_game_log()], ignore_index=True).iloc[:24].copy()
    frame['GAME_DATE'] = pd.date_range('2025-01-01', periods=24, freq='2D').strftime('%b %d, %Y')
    frame['Game_ID'] = [f'0022400{i:04d}' for i in range(24)]
    if vary:
        for i in range(len(frame)):
            frame.loc[i, 'PTS'] *= .7 + (i % 4) * .2
            frame.loc[i, 'MIN'] *= .8 + (i % 3) * .2
            frame.loc[i, 'WL'] = 'W' if i % 3 else 'L'
    return frame
