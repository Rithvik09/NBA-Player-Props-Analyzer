"""The training run's per-player precompute loader must actually load.

``_load_precomputed_player_data`` opens a ``dict_row`` connection and
calls ``db.table_columns`` on it inside a ``try`` whose handler returns
``{}``. While those helpers indexed rows positionally that raised
``KeyError: 0``, so all 11 per-player tables — advanced, clutch, hustle,
shot_profile, play_types, on_off, shot_zones, quarter_splits, tracking,
scoring_breakdown, yoy — loaded empty on every training run regardless of
what was in the database, starving 114 of the 416 features.

The absence of this test is what let that through: the helpers were
tested, the loader on top of them was not.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_player_tables_load_on_a_dict_row_connection(db_conn, db_url):
    from scripts.train_models import _load_precomputed_player_data

    db_conn.execute(
        "CREATE TABLE player_advanced_stats ("
        "  player_id BIGINT, season VARCHAR(7), usage_pct DOUBLE PRECISION)"
    )
    db_conn.execute(
        "INSERT INTO player_advanced_stats VALUES (203999, '2025-26', 31.5)"
    )

    data = _load_precomputed_player_data(db_url)

    # Both index forms the loader promises in its docstring.
    assert data["advanced"][203999]["usage_pct"] == 31.5
    assert data["advanced"][(203999, "2025-26")]["usage_pct"] == 31.5


def test_latest_fallback_is_ordered_by_season_not_insertion(db_conn, db_url):
    from scripts.train_models import _load_precomputed_player_data
    db_conn.execute('CREATE TABLE player_advanced_stats (player_id BIGINT, season TEXT, efg_pct DOUBLE PRECISION)')
    db_conn.execute("INSERT INTO player_advanced_stats VALUES (2544, '2025-26', .6), (2544, '2022-23', .4)")
    data = _load_precomputed_player_data(db_url)
    assert data['advanced'][2544]['efg_pct'] == .6
    assert data['advanced'][(2544, '2022-23')]['efg_pct'] == .4
