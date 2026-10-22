"""Train and serve must agree on what "no data" means for a feature.

``ml_features.build_feature_vector`` ends with ``safe.get(k, 0.0)``, so a
missing feature never raises — it silently becomes zero. Every builder
therefore used to carry its own fallback literal, and the literals had
drifted: ``team_pace`` fell back to 100.0 while training and to 0 while
serving, and a pace of 0 is not a low value, it is 90 possessions outside
anything the model was fitted on.

These tests run the real training builder and the real inference builder
side by side with every optional data source empty, and assert both land on
``FEATURE_DEFAULTS``. The nba_api constructors are patched with canned
frames — nothing here touches the network or the database.
"""
from __future__ import annotations

import ast
import hashlib
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_features import (  # noqa: E402
    FEATURE_DEFAULTS,
    NUMERIC_FEATURE_KEYS,
    build_feature_vector,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every module that populates a raw feature dict. All five read their
# fallbacks from FEATURE_DEFAULTS; the ast scan below is what keeps it that
# way when someone adds a feature in a hurry.
BUILDER_FILES = (
    "scripts/train_models.py",             # trains the shipped boosters
    "scripts/update_incremental_models.py",  # trains the incremental models
    "src/data_collector.py",               # trains via EnhancedMLPredictor.train
    "src/models.py",                       # serves the shipped boosters
    "src/basketball_betting_helper.py",    # serves the incremental models
)

# Features the trainer computes from the game log rather than looking up, so
# the shared default is its no-data floor and does not fire on a row that has
# a game log at all. The inference builder still has to match the default —
# ``test_serve_emits_the_shared_default_for_every_key`` covers that half.
TRAIN_SIDE_DERIVED = {
    "rest_days": "days between consecutive GAME_DATEs",
    "true_shooting_pct": "PTS / 2(FGA + 0.44 FTA); default covers a zero denominator",
    "effective_fg_pct": "(FGM + 0.5 FG3M) / FGA",
    "rebound_percentage": "(OREB + DREB) / 100",
    "corner_3_pct": "recent FG3_PCT * 1.05",
    "garbage_time_minutes_pct": "blowout game share * 0.15",
    "arena_capacity": "real capacity from arena_data.ARENA_DATA",
}

# The trainer needs 11+ prior games before it emits a row, so the canned log
# is 14 flat games — identical values keep every derived feature a clean
# number and leave the defaults as the only source of variation.
_GAME_LOG_COLUMNS = {
    "MIN": 30.0, "PTS": 20.0, "REB": 5.0, "AST": 4.0, "STL": 1.0, "BLK": 0.5,
    "TOV": 2.0, "FG3M": 2.0, "FGA": 15.0, "FGM": 7.0, "FG3A": 5.0, "FTA": 4.0,
    "FTM": 3.0, "FG_PCT": 0.47, "FG3_PCT": 0.36, "FT_PCT": 0.8, "OREB": 1.0,
    "DREB": 4.0, "PF": 2.0, "PLUS_MINUS": 3.0,
}
_N_GAMES = 14

# The precompute tables build_training_examples loads up front. Empty is the
# case under test: every per-player and per-team lookup misses, which is
# exactly when a default fires.
_PRECOMPUTE_TABLES = (
    "advanced clutch hustle shot_prof play_types on_off shot_zones q_splits "
    "tracking scoring yoy team_stats team_foul rest_splits standings "
    "opp_shot_zones opp_synergy home_away lineup_stats injury_status "
    "vs_opponent dvp_rolling referee game_officials game_broadcasts "
    "game_team_stats"
).split()


class _CannedEndpoint:
    """Stands in for an nba_api endpoint object."""

    def __init__(self, df):
        self._df = df

    def get_data_frames(self):
        return [self._df]


def _canned_game_log():
    dates = pd.date_range("2025-01-01", periods=_N_GAMES, freq="2D")
    return pd.DataFrame([
        {
            "Game_ID": f"002240000{i:02d}",
            "GAME_DATE": d.strftime("%b %d, %Y"),
            "MATCHUP": "BOS vs. LAL",
            "WL": "W",
            **_GAME_LOG_COLUMNS,
        }
        for i, d in enumerate(dates)
    ])


def _train_feature_dict(monkeypatch, team_context):
    """Run the real training builder and return the raw dict it hands to
    ``build_feature_vector`` for the first row it produces."""
    import scripts.train_models as train_models

    def _player_game_log(**kwargs):
        # Playoff / play-in pulls have nothing to add for a regular-season log.
        if kwargs.get("season_type_all_star") in ("Playoffs", "PlayIn"):
            return _CannedEndpoint(pd.DataFrame())
        return _CannedEndpoint(_canned_game_log())

    monkeypatch.setattr(
        train_models.leaguedashplayerstats, "LeagueDashPlayerStats",
        lambda **kw: _CannedEndpoint(pd.DataFrame([{"PLAYER_ID": 2544, "MIN": 1000.0}])),
    )
    monkeypatch.setattr(train_models.playergamelog, "PlayerGameLog", _player_game_log)
    monkeypatch.setattr(
        train_models, "CommonPlayerInfo", lambda **kw: _CannedEndpoint(pd.DataFrame()))
    monkeypatch.setattr(
        train_models, "_load_precomputed_player_data",
        lambda db_path: {table: {} for table in _PRECOMPUTE_TABLES})

    captured: list[dict] = []

    def _capture(raw_features):
        captured.append(dict(raw_features))
        return build_feature_vector(raw_features)

    monkeypatch.setattr(train_models, "build_feature_vector", _capture)

    class _Precomputed:
        def refresh(self, force=False):
            return {}

    class _Predictor:
        def get_team_context(self, team_id):
            return dict(team_context)

    train_models.build_training_examples(
        "2024-25", 1, _Predictor(), _Precomputed(), db_path="unused")

    assert captured, "the training builder produced no rows"
    return captured[0]


def _serve_feature_dict(team_context):
    """Run the real inference builder with the same empty sources."""
    from src.models import EnhancedMLPredictor

    # __new__ rather than __init__: prepare_features touches no instance
    # state, and __init__ loads joblibs off disk.
    predictor = EnhancedMLPredictor.__new__(EnhancedMLPredictor)
    player_stats = {
        "values": [_GAME_LOG_COLUMNS["PTS"]] * (_N_GAMES - 1),
        "last5_avg": _GAME_LOG_COLUMNS["PTS"],
        "avg": _GAME_LOG_COLUMNS["PTS"],
    }
    return predictor.prepare_features(
        player_stats, {}, dict(team_context), dict(team_context))


# A context dict that is truthy but carries none of the keys the builder
# wants is the shape that produced the original bug: src/models.py read
# ``team_context.get('pace', 0)`` inside ``if team_context:``, which skipped
# the ``setdefault('team_pace', 100.0)`` below it.
CONTEXT_SHAPES = {
    "absent": {},
    "present_but_empty_of_the_keys_we_want": {"unrelated_key": 1},
}


@pytest.fixture(params=sorted(CONTEXT_SHAPES), ids=sorted(CONTEXT_SHAPES))
def context_shape(request):
    return CONTEXT_SHAPES[request.param]


# ───────────────────────────────────────────────── the invariant under test
def test_train_and_serve_agree_on_every_shared_default(monkeypatch, context_shape):
    """With every source empty, both builders land on FEATURE_DEFAULTS."""
    train = _train_feature_dict(monkeypatch, context_shape)
    serve = _serve_feature_dict(context_shape)

    mismatched = {}
    for key, default in FEATURE_DEFAULTS.items():
        if key in TRAIN_SIDE_DERIVED:
            continue
        got_train, got_serve = train.get(key), serve.get(key)
        if got_train is None or got_serve is None:
            mismatched[key] = (got_train, got_serve, default)
        elif float(got_train) != default or float(got_serve) != default:
            mismatched[key] = (float(got_train), float(got_serve), default)

    assert not mismatched, (
        "train/serve default divergence:\n" + "\n".join(
            f"  {k}: train={t!r} serve={s!r} expected={d!r}"
            for k, (t, s, d) in sorted(mismatched.items())
        )
    )


def test_serve_emits_the_shared_default_for_every_key(context_shape):
    """No exemptions on the inference side: it has no game log to derive from,
    so every key in the table must come out of the table."""
    serve = _serve_feature_dict(context_shape)
    wrong = {
        key: serve.get(key)
        for key, default in FEATURE_DEFAULTS.items()
        if key not in serve or float(serve[key]) != default
    }
    assert not wrong, f"inference builder did not use the shared default: {wrong}"


def test_train_side_derived_features_still_reach_the_model(monkeypatch):
    """The exempt features are exempt because the trainer computes them, not
    because they are missing. Guards against the exemption list turning into
    a place to hide a dropped feature."""
    train = _train_feature_dict(monkeypatch, {})
    absent = [k for k in TRAIN_SIDE_DERIVED if k not in train]
    assert not absent, f"exempt but never emitted by the trainer: {absent}"


def test_zero_floor_is_unreachable_for_shared_defaults(monkeypatch, context_shape):
    """build_feature_vector's ``safe.get(k, 0.0)`` is the last-resort floor.
    For a feature with a shared default it must never be what lands in X.

    The trainer's derived features are skipped on the train side only: a
    computed 0.0 and a floored 0.0 are indistinguishable from here, and the
    first test already pins the ones that are genuine lookups.
    """
    sides = (
        (_train_feature_dict(monkeypatch, context_shape), "train", TRAIN_SIDE_DERIVED),
        (_serve_feature_dict(context_shape), "serve", ()),
    )
    for raw, side, exempt in sides:
        row = build_feature_vector(raw).X.iloc[0]
        floored = [
            key for key, default in FEATURE_DEFAULTS.items()
            if default != 0.0 and key not in exempt and row[key] == 0.0
        ]
        assert not floored, f"{side} fell through to the 0.0 floor for {floored}"


# ───────────────────────────────────────────────────────── structural guards
def _hardcoded_fallbacks(path):
    """``x.get('feature', 1.23)`` / ``x.setdefault('feature', 1.23)`` where
    'feature' has a shared default — the pattern that drifted."""
    tree = ast.parse(open(os.path.join(REPO_ROOT, path)).read())
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and len(node.args) == 2):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "setdefault")):
            continue
        key, fallback = node.args
        if not (isinstance(key, ast.Constant) and key.value in FEATURE_DEFAULTS):
            continue
        if (isinstance(fallback, ast.Constant)
                and isinstance(fallback.value, (int, float))
                and not isinstance(fallback.value, bool)):
            found.append(f"{path}:{node.lineno} {key.value} -> {fallback.value!r}")
    return found


@pytest.mark.parametrize("path", BUILDER_FILES)
def test_no_builder_carries_its_own_fallback_literal(path):
    offenders = _hardcoded_fallbacks(path)
    assert not offenders, (
        "read the value from ml_features.FEATURE_DEFAULTS instead:\n  "
        + "\n  ".join(offenders)
    )


def test_every_shared_default_is_a_real_feature():
    unknown = sorted(set(FEATURE_DEFAULTS) - set(NUMERIC_FEATURE_KEYS))
    assert not unknown, f"FEATURE_DEFAULTS names features the schema has no slot for: {unknown}"


def test_numeric_feature_keys_unchanged():
    """The shipped boosters are positional — FEATURE_DEFAULTS must not have
    added, removed or reordered anything."""
    assert len(NUMERIC_FEATURE_KEYS) == 416
    assert len(set(NUMERIC_FEATURE_KEYS)) == 416, "duplicate feature name"
    digest = hashlib.sha256("\n".join(NUMERIC_FEATURE_KEYS).encode()).hexdigest()
    assert digest == (
        "8248083f3537d6c865537835bfaff2ebfa69bce8710f8678d773d7b7956a90b8"
    ), "NUMERIC_FEATURE_KEYS changed content or order"
