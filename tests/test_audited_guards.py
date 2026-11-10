"""Fail-closed guards in the audited contract, which nothing else asserts.

`tests/test_full_feature_audit.py` covers the *happy* path thoroughly: that all
416 specs exist, that every snapshot field reads the completed prior season,
that poisoned future rows cannot move a value, and that the training builder
and the serving adapter emit byte-identical frames. What it never does is drive
the contract off the road.

That matters here more than it usually would. Every divergence this project has
found so far was silent: a feature that defaulted to 0.0 instead of refusing to
be computed, a write that dropped `season` and starved ten tables, a poller
whose key was never in any environment. The guards below are the places where
`build_audited_features` and the serving adapter decide to raise instead of
returning a plausible number. A guard that stops raising does not fail a test
suite -- it just quietly starts training on preseason games or on NaN. So each
one gets an assertion.

Fixture note: the module-level invariants (416 specs, one disposition per
originally-inactive feature) already raise at *import* time, so importing
`src.audited_features` at all is itself a check. These tests deliberately use
the same `fixture()` the existing audit tests use, so a drift in the shared
fixture shows up in both files rather than only here.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.audited_features import POLICY_VERSION, build_audited_features, ratio
from tests.test_full_feature_audit import BOS, LAL, fixture

DATE = "2025-01-21"


# --------------------------------------------------------------------------
# ratio(): the shared divide-by-zero policy
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a,b,expected",
    [
        (10.0, 4.0, 2.5),
        (10.0, 0.0, 0.0),      # no ZeroDivisionError
        (10.0, -1.0, 0.0),     # negative denominators are nonsense, not signed
        (0.0, 0.0, 0.0),
    ],
)
def test_ratio_returns_zero_for_unusable_denominators(a, b, expected):
    """A zero denominator must give 0.0, never NaN, inf or an exception.

    Every shooting split, rate and per-possession figure in the contract runs
    through `ratio`. If it ever returned NaN for a zero denominator, the final
    finiteness guard would reject whole rows for players who simply took no
    free throws -- and if it raised, a single such player would abort a
    training run.
    """
    result = ratio(a, b)
    assert result == pytest.approx(expected)
    assert np.isfinite(result)
    assert isinstance(result, float)


# --------------------------------------------------------------------------
# build_audited_features(): the four fail-closed guards
# --------------------------------------------------------------------------

def test_no_prior_boxscores_raises_instead_of_inventing_a_player():
    """With nothing before the prediction date there is nothing to predict from.

    The dangerous alternative is not a crash, it is a full 416-value row of
    defaults that looks exactly like a real one.
    """
    logs, _raw, inputs = fixture()
    # Every game on or after the prediction date -> empty prior window.
    future = logs.copy()
    future["GAME_DATE"] = pd.Timestamp(DATE) + pd.Timedelta(days=1)

    with pytest.raises(ValueError, match="require prior player boxscores"):
        build_audited_features(future, "PTS", 2544, BOS, LAL, DATE, False, inputs)


def test_missing_target_column_raises_rather_than_scoring_zeros():
    """An absent target column must not be read as a player who scored nothing."""
    logs, _raw, inputs = fixture()

    with pytest.raises(ValueError, match="Missing target column"):
        build_audited_features(logs, "NOT_A_REAL_COLUMN", 2544, BOS, LAL, DATE, False, inputs)


@pytest.mark.parametrize(
    "prefix,label",
    [("001", "preseason"), ("003", "all-star")],
)
def test_non_regular_game_types_are_rejected(prefix, label):
    """Only regular season (002), playoffs (004) and play-in (005) are allowed.

    `HistoricalInputs` filters game ids when it builds team boxscores, but
    `build_audited_features` is handed the player history directly -- by
    training, by the serving adapter and by nine scripts. So the guard has to
    live here too, and an exhibition game must not become a training row.
    """
    logs, _raw, inputs = fixture()
    poisoned = logs.copy()
    poisoned.loc[0, "GAME_ID"] = prefix + "2400001"

    with pytest.raises(ValueError, match="Unexpected NBA game type"):
        build_audited_features(poisoned, "PTS", 2544, BOS, LAL, DATE, False, inputs)


@pytest.mark.parametrize("prefix", ["002", "004", "005"])
def test_regular_playoff_and_playin_game_types_are_accepted(prefix):
    """The mirror of the rejection test: the allowed prefixes must stay allowed.

    Without this, tightening the guard to regular season only would silently
    drop every playoff and play-in row from training and no test would notice.
    """
    logs, _raw, inputs = fixture()
    relabelled = logs.copy()
    relabelled["GAME_ID"] = [prefix + f"240000{i:02d}" for i in range(len(relabelled))]

    features, _gaps = build_audited_features(
        relabelled, "PTS", 2544, BOS, LAL, DATE, False, inputs
    )
    assert features["games_played"] > 0


def test_a_nonfinite_feature_raises_instead_of_reaching_the_model():
    """A corrupt dated team cache must abort the row, not emit inf.

    XGBoost accepts NaN, so an inf or NaN that escapes here does not crash
    training -- it becomes a split. `inputs.team_context` is the realistic
    source of such a value, since it averages whatever the dated cache holds.
    """
    logs, _raw, inputs = fixture()
    inputs.team_context = lambda team_id, date: {
        "pace": float("inf"),
        "offensive_rating": 110.0,
        "defensive_rating": 108.0,
        "days_rest": 2,
    }

    with pytest.raises(ValueError, match="Nonfinite audited feature"):
        build_audited_features(logs, "PTS", 2544, BOS, LAL, DATE, False, inputs)


def test_the_happy_path_returns_every_key_finite_and_sorted_gaps():
    """The guards must not be firing on ordinary input -- the control case."""
    logs, _raw, inputs = fixture()
    features, gaps = build_audited_features(logs, "PTS", 2544, BOS, LAL, DATE, False, inputs)

    assert all(np.isfinite(float(v)) for v in features.values())
    assert gaps == sorted(gaps), "callers treat the gap list as a stable ordering"
    assert len(set(gaps)) == len(gaps), "a duplicated gap would double-count a missing source"
    assert set(gaps) <= set(features), "every reported gap must name a real feature"


# --------------------------------------------------------------------------
# audited_runtime.prepare(): the serving-side guards
# --------------------------------------------------------------------------

def _serving_stats(inputs, **overrides):
    stats = {
        "_audited_inputs": inputs,
        "_boxscores": overrides.pop("_boxscores", None),
        "_prediction_date": DATE,
        "_prediction_game_id": "0022400001",
        "_prop_type": "points",
        "_player_id": 2544,
        "_team_id": BOS,
        "_opponent_team_id": LAL,
        "is_home": True,
    }
    stats.update(overrides)
    return stats


def test_prepare_rejects_inputs_from_the_wrong_season():
    """A 2024-25 cache must not be used to serve a 2025-26 prediction.

    This is the exact shape of the train/serve mismatch the audited contract
    exists to remove, so serving has to refuse rather than reach for whatever
    the cache happens to hold.
    """
    from src.audited_runtime import prepare

    logs, _raw, inputs = fixture()
    stats = _serving_stats(inputs, _boxscores=logs.to_dict("records"),
                           _prediction_date="2026-01-21")

    with pytest.raises(ValueError, match="season does not match"):
        prepare(stats, {})


def test_prepare_requires_raw_prior_boxscores():
    from src.audited_runtime import prepare

    _logs, _raw, inputs = fixture()

    with pytest.raises(ValueError, match="requires raw prior player boxscores"):
        prepare(_serving_stats(inputs, _boxscores=[]), {})


def test_prepare_requires_a_known_team_and_opponent():
    """Serving without a team id must fail closed, not default to a team."""
    from src.audited_runtime import prepare

    logs, _raw, inputs = fixture()
    stats = _serving_stats(inputs, _boxscores=logs.to_dict("records"), _team_id=None)

    with pytest.raises(ValueError, match="requires known team and opponent"):
        prepare(stats, {})  # empty player_context -> no team_id fallback either


def test_prepare_does_not_mutate_the_inputs_it_is_given():
    """`_inputs` is lru_cached, so mutating it would leak between predictions.

    `prepare` defends against that by copying the inputs and the recovery
    context before stamping the game id and clearing the per-prediction
    caches. If that copy is ever dropped, one player's injury cutoff cache and
    game id would silently serve the next player's prediction.
    """
    from src.audited_runtime import prepare

    logs, _raw, inputs = fixture()
    inputs.prediction_game_id = "ORIGINAL"
    original_recovery = inputs.recovery
    inputs.recovery._injury_cache = {"sentinel": "must survive"}
    inputs.recovery._cutoff_cache = {"sentinel": "must survive"}

    features = prepare(_serving_stats(inputs, _boxscores=logs.to_dict("records")), {})

    assert features["_feature_policy"] == POLICY_VERSION
    assert inputs.prediction_game_id == "ORIGINAL", "game id was stamped on the shared object"
    assert inputs.recovery is original_recovery, "recovery context was replaced in place"
    assert inputs.recovery._injury_cache == {"sentinel": "must survive"}
    assert inputs.recovery._cutoff_cache == {"sentinel": "must survive"}


def test_prepare_reports_gaps_and_policy_for_enforce_to_act_on():
    """`enforce` turns `_missing_features` into NaN, so `prepare` must supply it.

    These two keys are the whole interface between the two halves: drop them
    and `enforce` raises, or worse, a missing source stays 0.0.
    """
    from src.audited_runtime import prepare, enforce

    logs, _raw, inputs = fixture()
    features = prepare(_serving_stats(inputs, _boxscores=logs.to_dict("records")), {})

    assert features["_feature_policy"] == POLICY_VERSION
    assert isinstance(features["_missing_features"], list)

    cleaned = enforce(features, POLICY_VERSION)
    for key in features["_missing_features"]:
        assert np.isnan(cleaned[key]), f"{key} was reported missing but served a number"


# --------------------------------------------------------------------------
# attach_live_injuries(): a bounded swallow
# --------------------------------------------------------------------------

def test_live_injury_failure_leaves_no_partial_context(monkeypatch, caplog):
    """The blanket `except Exception` must fail closed and say so.

    Swallowing is defensible here -- a flaky ESPN endpoint should not take down
    a prediction -- but only if the result is *no* live context rather than a
    half-filled one, and only if it is audible. A silent partial snapshot would
    present stale injuries as current.
    """
    from src import audited_runtime

    def boom(_bucket):
        raise RuntimeError("endpoint changed shape")

    monkeypatch.setattr(audited_runtime, "_live_injury_snapshot", boom)

    _logs, _raw, inputs = fixture()
    with caplog.at_level(logging.WARNING, logger="src.audited_runtime"):
        audited_runtime.attach_live_injuries(inputs, pd.Timestamp(DATE))

    assert not hasattr(inputs, "live_injury_context"), "partial context survived a failure"
    assert any("live injury source unavailable" in r.message for r in caplog.records), \
        "the swallow must be audible in the log"
