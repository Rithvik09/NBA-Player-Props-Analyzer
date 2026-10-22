"""A partial source is checkpointed as partial and retried without losing rows."""
from scripts import rebuild_precomputed as runner


def test_partial_rest_source_retries_only_missing_teams(monkeypatch):
    from nba_api.stats.static import teams
    ids = [t['id'] for t in teams.get_teams()]
    calls, written = [], []
    def collect(season, team_ids=None):
        calls.append(team_ids)
        return [{'team_id': tid} for tid in (ids[:20] if team_ids is None else team_ids)]
    monkeypatch.setattr(runner.jobs, 'compute_team_rest_splits', collect)
    monkeypatch.setattr(runner, 'write_rows', lambda writer, rows, season: written.append(len(rows)))
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    spec = next(s for s in runner.PLAN if s[0] == 'team_rest_splits')
    result = runner.run_stage(spec, '2022-23', {}, 3)
    assert result['status'] == 'complete'
    assert result['rows'] == 30 and result['attempts'] == 2
    assert calls == [None, ids[20:]]
    assert written == [20, 30]


def test_empty_required_source_does_not_report_success(monkeypatch):
    monkeypatch.setattr(runner.jobs, 'compute_quarter_splits', lambda season: [])
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    spec = next(s for s in runner.PLAN if s[0] == 'player_quarter_splits')
    result = runner.run_stage(spec, '2022-23', {}, 2)
    assert result['status'] == 'incomplete'
    assert result['rows'] == 0
    assert result['attempts'] == 2


def test_completed_stage_does_not_refetch(monkeypatch):
    def fail(**kw):
        raise AssertionError('completed stage must not fetch')
    monkeypatch.setattr(runner.jobs, 'compute_quarter_splits', fail)
    spec = next(s for s in runner.PLAN if s[0] == 'player_quarter_splits')
    previous = {'status': 'complete', 'rows': 539}
    assert runner.run_stage(spec, '2022-23', {spec[0]: previous}, 3) == previous


def test_changed_collector_revision_is_refetched(monkeypatch):
    calls = []
    def collect(season):
        calls.append(season)
        return [{'player_id': 999}]
    monkeypatch.setattr(runner.jobs, 'compute_synergy_play_types', collect)
    monkeypatch.setattr(runner, 'write_rows', lambda *a: None)
    spec = next(s for s in runner.PLAN if s[0] == 'player_play_types')
    result = runner.run_stage(spec, '2022-23', {spec[0]: {'status': 'complete', 'rows': 400}}, 1)
    assert calls == ['2022-23']
    assert result['revision'] == runner.REVISIONS['player_play_types']


def test_partial_stage_never_reads_a_potentially_later_db_snapshot(monkeypatch):
    def no_db(*a, **kw):
        raise AssertionError('unversioned DB snapshot is not historical resume data')
    monkeypatch.setattr(runner.db, 'connect', no_db)
    monkeypatch.setattr(runner.jobs, 'compute_team_rest_splits', lambda **kw: [])
    monkeypatch.setattr(runner.time, 'sleep', lambda _: None)
    spec = next(s for s in runner.PLAN if s[0] == 'team_rest_splits')
    result = runner.run_stage(spec, '2022-23', {spec[0]: {'status': 'incomplete', 'rows': 20}}, 1)
    assert result['status'] == 'incomplete'
