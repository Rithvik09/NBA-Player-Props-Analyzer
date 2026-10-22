"""Every season-keyed insert must explicitly persist the requested season."""
import inspect
import re

import pytest
from src import precompute_jobs as jobs

UPSERTS = [fn for name, fn in vars(jobs).items()
           if name.startswith('upsert_') and callable(fn)
           and 'ON CONFLICT (player_id, season)' in inspect.getsource(fn)]


class CaptureConnection:
    def __init__(self):
        self.calls = []

    def cursor(self):
        return self

    def execute(self, sql, params):
        self.calls.append((sql, params))

    def commit(self):
        pass


@pytest.mark.parametrize('upsert', UPSERTS, ids=lambda fn: fn.__name__)
def test_explicit_season_is_inserted(upsert):
    conn = CaptureConnection()
    upsert(conn, [{'player_id': 2544}], updated_at=123, season='2022-23')
    sql, params = conn.calls[0]
    columns = re.search(r'INSERT INTO \w+\s*\((.*?)\)\s*VALUES', sql, re.S).group(1).split(',')
    columns = [c.strip() for c in columns]
    assert len(columns) == len(params) == sql.count('%s')
    assert params[columns.index('season')] == '2022-23'
