"""/accuracy and /bias under PostgreSQL.

Neither report had any coverage, which is how three bugs shipped together
in the SQLite port:

* ``get_accuracy_stats`` ran ``ROUND(AVG(edge) * 100, 1)``. PostgreSQL only
  defines the two-argument ``ROUND`` for ``numeric``, so on ``double
  precision`` it raises.
* The same function selected a bare ``player_name`` alongside ``GROUP BY
  player_id``, which PostgreSQL rejects.

  Both were swallowed by a blanket ``except`` that returns ``{}``, so the
  endpoint answered ``200 {}`` and the tab rendered empty.
* Every ``ROUND(100.0 * ...)`` and ``AVG(CASE ... 1.0 ... END)`` column is
  PostgreSQL ``numeric``, which psycopg loads as ``Decimal`` and ``jsonify``
  writes out as a JSON *string*. ``static/js/main.js`` calls ``.toFixed`` on
  those values, so the /bias tab threw.

The JSON assertions below read the serialised body rather than the dicts,
because the string only appears once ``jsonify`` has run.
"""
from __future__ import annotations

import pytest

from src.basketball_betting_helper import BasketballBettingHelper


# Columns the front end feeds to ``.toFixed`` — a string in any of them is
# the H4 defect, whatever its value.
_NUMERIC_KEYS = frozenset({
    "accuracy_pct", "avg_edge_pct", "recent_accuracy", "total_graded",
    "total_correct", "total_predictions", "total", "correct", "n",
    "high_conf_count", "high_conf_correct", "med_conf_count",
    "med_conf_correct", "avg_error", "mae", "avg_prob", "actual_over_rate",
})

_INSERT = """
    INSERT INTO prediction_logs
        (timestamp, player_id, player_name, prop_type, line, is_home,
         predicted_value, over_probability, edge, confidence,
         actual_result, correct, recommendation)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'OVER')
"""


def _graded_rows():
    """16 graded rows: 12 'points' for Luka, 4 'rebounds' for LeBron.

    The counts clear every ``HAVING`` floor in ``get_bias_report`` for
    'points' (>= 10) and for home and away alike (>= 5 each: 8 and 8),
    while 'rebounds' stays below the by-prop floor.
    """
    rows = []
    for i in range(12):
        correct = 1 if i < 6 else 0
        rows.append((f"2026-01-{i + 1:02d}T20:00:00", 77, "Luka Doncic",
                     "points", 25.5, i % 2, 26.5, 0.60, 0.05,
                     "HIGH" if i < 8 else "MEDIUM",
                     30.0 if correct else 20.0, correct))
    for i in range(4):
        correct = 1 if i < 3 else 0
        rows.append((f"2026-02-{i + 1:02d}T20:00:00", 2544, "LeBron James",
                     "rebounds", 7.5, i % 2, 8.5, 0.55, 0.03, "MEDIUM",
                     9.0 if correct else 5.0, correct))
    return rows


@pytest.fixture
def helper(db_conn, db_url):
    """Helper on the test schema, with the graded rows above already in it."""
    h = BasketballBettingHelper(db_name=db_url)
    with db_conn.cursor() as cur:
        cur.executemany(_INSERT, _graded_rows())
    return h


@pytest.fixture
def client(helper):
    """Flask client sharing the seeded schema (``db_conn`` patches connect)."""
    from src import app as _app
    return _app.app.test_client()


def _walk(payload):
    """Yield every ``(key, value)`` pair in a nested dict/list payload."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _walk(item)


def _assert_numbers_are_numbers(payload):
    for key, value in _walk(payload):
        if key in _NUMERIC_KEYS and value is not None:
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"{key} serialised as {type(value).__name__} ({value!r}); "
                f"main.js calls .toFixed on it and would throw"
            )


# --------------------------------------------------------------- H3: /accuracy

def test_accuracy_stats_returns_data(helper):
    """The blanket except must not be reached — ``{}`` means the SQL raised."""
    stats = helper.get_accuracy_stats()

    assert stats != {}, "get_accuracy_stats swallowed an error and returned {}"
    assert stats["overall"]["total_graded"] == 16
    assert stats["overall"]["total_correct"] == 9
    assert stats["overall"]["accuracy_pct"] == pytest.approx(56.3)
    assert stats["total_predictions"] == 16

    by_prop = {r["prop_type"]: r for r in stats["by_prop"]}
    assert by_prop["points"]["accuracy_pct"] == pytest.approx(50.0)
    assert by_prop["rebounds"]["accuracy_pct"] == pytest.approx(75.0)
    # ROUND(AVG(edge) * 100, 1) — the two-argument ROUND on double precision
    assert by_prop["points"]["avg_edge_pct"] == pytest.approx(5.0)
    assert by_prop["rebounds"]["avg_edge_pct"] == pytest.approx(3.0)


def test_accuracy_stats_one_row_per_player(helper):
    """GROUP BY player_id with player_name selected — a GroupingError today."""
    by_player = helper.get_accuracy_stats()["by_player"]

    assert [r["player_id"] for r in by_player] == [77, 2544]  # ORDER BY total DESC
    assert by_player[0]["player_name"] == "Luka Doncic"
    assert by_player[0]["total"] == 12
    assert by_player[0]["correct"] == 6
    assert by_player[0]["accuracy_pct"] == pytest.approx(50.0)


def test_accuracy_stats_collapses_renamed_player(helper, db_conn):
    """One player_id that picked up a second spelling stays one row.

    Grouping by ``(player_id, player_name)`` would split the player in two
    and push a real name off the top-10; the aggregate must not.
    """
    with db_conn.cursor() as cur:
        cur.executemany(_INSERT, [
            ("2026-03-01T20:00:00", 77, "Luka Dončić", "points", 25.5, 1,
             26.5, 0.60, 0.05, "HIGH", 30.0, 1),
        ])

    by_player = helper.get_accuracy_stats()["by_player"]
    luka = [r for r in by_player if r["player_id"] == 77]
    assert len(luka) == 1, f"player 77 split across {len(luka)} rows"
    assert luka[0]["total"] == 13
    assert luka[0]["player_name"] in ("Luka Doncic", "Luka Dončić")


def test_accuracy_endpoint_emits_json_numbers(client):
    r = client.get("/accuracy")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload != {}
    _assert_numbers_are_numbers(payload)


# -------------------------------------------------------------------- H4: /bias

def test_bias_report_returns_data(helper):
    report = helper.get_bias_report()

    assert report != {}
    by_prop = {r["prop_type"]: r for r in report["by_prop"]}
    assert set(by_prop) == {"points"}  # 'rebounds' is under the HAVING floor
    assert by_prop["points"]["n"] == 12
    assert by_prop["points"]["avg_error"] == pytest.approx(1.5)
    assert by_prop["points"]["mae"] == pytest.approx(5.0)
    assert by_prop["points"]["accuracy_pct"] == pytest.approx(50.0)
    assert by_prop["points"]["actual_over_rate"] == pytest.approx(0.5)

    assert {r["location"] for r in report["by_location"]} == {"home", "away"}
    assert {r["confidence"] for r in report["by_confidence"]} == {"HIGH", "MEDIUM"}


def test_bias_endpoint_emits_json_numbers(client):
    r = client.get("/bias")
    assert r.status_code == 200
    payload = r.get_json()
    assert payload != {}
    _assert_numbers_are_numbers(payload)


@pytest.mark.parametrize("report", ["get_accuracy_stats", "get_bias_report"])
def test_reports_contain_no_decimals(helper, report):
    """Decimal is what ``jsonify`` turns into a string — none must escape.

    Asserted on the dicts as well as the wire format so a regression points
    at the query rather than at Flask.
    """
    from decimal import Decimal

    for key, value in _walk(getattr(helper, report)()):
        assert not isinstance(value, Decimal), f"{key} is a Decimal: {value!r}"



# ------------------------------------------------- the blanket except itself

@pytest.mark.parametrize("method, empty", [
    ("get_prediction_logs", []),
    ("get_accuracy_stats", {}),
    ("get_bias_report", {}),
])
def test_report_errors_are_logged_not_swallowed(method, empty, monkeypatch, caplog):
    """An induced SQL error must reach the structured log.

    Both PostgreSQL bugs above shipped because these handlers answered
    ``print()`` + an empty container: the endpoint returned 200 with nothing
    in it and nothing landed in the logs. The empty return is deliberate —
    callers depend on the shape — so visibility is the whole fix.
    """
    import logging

    import psycopg

    helper = BasketballBettingHelper.__new__(BasketballBettingHelper)
    boom = psycopg.ProgrammingError('column "edge" does not exist')

    def _explode():
        raise boom

    monkeypatch.setattr(helper, "get_db", _explode)

    with caplog.at_level(logging.ERROR, logger="src.basketball_betting_helper"):
        assert getattr(helper, method)() == empty

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, f"{method} swallowed the error with nothing in the log"
    assert any(r.exc_info for r in errors), (
        f"{method} logged without a traceback — use log.exception so the "
        f"failing SQL is identifiable"
    )
