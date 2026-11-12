"""Each study verifies its own frozen dependencies; none borrows another study's gate."""
import json
import re
import subprocess
from pathlib import Path
import joblib
import numpy as np
import pytest
from src.prospective_points import digest
from src.study_provenance import (verify_recorded_hashes, close_study, verify_closed,
                                  recorded_or_live_hashes, verify_released_blobs)

ROOT = Path(__file__).resolve().parents[1]


def git(cwd, *args):
    return subprocess.run(('git',)+args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def init_repo(tmp_path, monkeypatch):
    """A real repository at tmp_path: close_study checks released content against its commits."""
    monkeypatch.setattr('src.study_provenance.ROOT', tmp_path)
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.email', 'studies@example.invalid')
    git(tmp_path, 'config', 'user.name', 'Study provenance tests')
    git(tmp_path, 'config', 'commit.gpgsign', 'false')
    return tmp_path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    return init_repo(tmp_path, monkeypatch)


def commit(tmp_path, message='the content the studies read'):
    # Forced add and --no-verify keep a developer's global excludes and hooks out of the fixture.
    git(tmp_path, 'add', '-A', '-f')
    git(tmp_path, 'commit', '-q', '--no-verify', '-m', message)
    return git(tmp_path, 'rev-parse', 'HEAD')


def pinned(tmp_path, names=('a.bin', 'b.bin', 'c.bin')):
    for i, name in enumerate(names):
        (tmp_path/name).write_bytes(bytes([i])*8)
    return {str(tmp_path/name): digest(tmp_path/name) for name in names}


def test_matching_digests_pass(tmp_path):
    verify_recorded_hashes(pinned(tmp_path), 'unit')


def test_every_drifted_path_is_named(tmp_path):
    recorded = pinned(tmp_path)
    first, untouched, last = recorded
    Path(first).write_bytes(b'changed')
    Path(last).write_bytes(b'changed too')
    with pytest.raises(ValueError, match='unit') as info:
        verify_recorded_hashes(recorded, 'unit')
    assert first in str(info.value) and last in str(info.value)
    assert untouched not in str(info.value)


def test_missing_file_is_drift(tmp_path):
    recorded = pinned(tmp_path)
    missing = list(recorded)[1]
    Path(missing).unlink()
    with pytest.raises(ValueError, match=re.escape(missing)):
        verify_recorded_hashes(recorded, 'unit')


def test_close_study_writes_once_and_is_idempotent(tmp_path, repo):
    recorded = pinned(tmp_path)
    first_path, _, last_path = recorded
    commit(tmp_path)
    study = tmp_path/'study'
    first = close_study(study, 'terminal', recorded, [last_path, first_path])
    path = study/'closed.json'
    assert set(first) == {'closed_at', 'reason', 'git_commit', 'source_hashes_at_close', 'released_paths'}
    assert first['reason'] == 'terminal' and first['source_hashes_at_close'] == recorded
    assert first['released_paths'] == sorted([first_path, last_path])
    assert first['git_commit'] is None or re.fullmatch(r'[0-9a-f]{40}', first['git_commit'])
    assert json.loads(path.read_text()) == first
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns
    assert close_study(study, 'a different reason', {}, []) == first
    assert path.read_bytes() == before and path.stat().st_mtime_ns == mtime


def test_verify_closed_pins_the_record_not_the_working_tree(tmp_path):
    recorded = pinned(tmp_path)
    study = tmp_path/'study'
    close_study(study, 'terminal', recorded, [])
    verify_closed(study, recorded)
    with pytest.raises(ValueError):
        verify_closed(study, {**recorded, list(recorded)[0]: '0'*64})
    with pytest.raises(ValueError):
        verify_closed(study, dict(list(recorded.items())[1:]))


def test_release_is_per_path_not_wholesale(tmp_path, repo):
    recorded = pinned(tmp_path)
    released, live, _ = recorded
    commit(tmp_path)
    study = tmp_path/'study'
    close_study(study, 'terminal', recorded, [released])
    Path(released).write_bytes(b'released sources move on after closure')
    verify_closed(study, recorded)
    Path(live).write_bytes(b'an unreleased dependency drifted')
    with pytest.raises(ValueError, match=re.escape(live)) as info:
        verify_closed(study, recorded)
    assert released not in str(info.value)


def test_released_paths_must_be_pinned(tmp_path):
    recorded = pinned(tmp_path)
    study = tmp_path/'study'
    with pytest.raises(ValueError, match='never pinned'):
        close_study(study, 'terminal', recorded, [str(tmp_path/'unpinned.bin')])
    assert not (study/'closed.json').exists()
    record = close_study(study, 'terminal', recorded, [])
    record['released_paths'] = [str(tmp_path/'unpinned.bin')]
    (study/'closed.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='never pinned'):
        verify_closed(study, recorded)


def test_released_path_answers_from_the_record_not_the_tree(tmp_path, repo):
    recorded = pinned(tmp_path)
    released, live, _ = recorded
    commit(tmp_path)
    study = tmp_path/'study'
    close_study(study, 'terminal', recorded, [released])
    Path(released).write_bytes(b'the released script was edited')
    Path(live).write_bytes(b'an unreleased dependency drifted')
    answered = recorded_or_live_hashes(study, list(recorded))
    # The released path keeps its closure digest, so a study that pins its own
    # script can still recompute; every other path reports what is on disk now.
    assert answered[released] == recorded[released]
    assert answered[live] == digest(live) != recorded[live]


def test_recorded_or_live_hashes_is_all_live_without_a_closure_record(tmp_path):
    recorded = pinned(tmp_path)
    changed = list(recorded)[0]
    Path(changed).write_bytes(b'drifted')
    answered = recorded_or_live_hashes(tmp_path/'never_closed', list(recorded))
    assert answered == {p: digest(p) for p in recorded}
    assert answered[changed] != recorded[changed]


def test_verify_closed_requires_a_closure_record(tmp_path):
    with pytest.raises(ValueError, match='closed.json'):
        verify_closed(tmp_path/'never_closed', {})


def test_closure_commit_must_hold_the_released_content(tmp_path, repo):
    (tmp_path/'unrelated.txt').write_text('before the study had a script\n')
    before = commit(tmp_path, 'nothing of the study yet')
    source = tmp_path/'scripts'/'train.py'
    source.parent.mkdir()
    source.write_text('ARM = 1\n')
    at_close = commit(tmp_path, 'the content the study read')
    recorded = {str(source): digest(source)}
    source.write_text('ARM = 2\n')
    moved_on = commit(tmp_path, 'the released script moved on')
    study = tmp_path/'study'
    # The newest commit holds other content, and the oldest does not hold the file
    # at all: a closure record may name neither.
    for bad in (moved_on, before):
        with pytest.raises(ValueError, match=re.escape(str(source))) as info:
            close_study(study, 'terminal', recorded, [str(source)], commit=bad)
        assert bad in str(info.value)
        assert not (study/'closed.json').exists()
    record = close_study(study, 'terminal', recorded, [str(source)], commit=at_close)
    assert record['git_commit'] == at_close
    verify_closed(study, recorded)


def test_closure_refuses_an_unreleased_source_that_already_drifted(tmp_path, repo):
    recorded = pinned(tmp_path)
    released, drifted, _ = recorded
    commit(tmp_path)
    Path(drifted).write_bytes(b'edited before anyone closed the study')
    study = tmp_path/'study'
    # Closing now would write a record that fails the first time it is verified.
    with pytest.raises(ValueError, match=re.escape(drifted)) as info:
        close_study(study, 'terminal', recorded, [released])
    assert released not in str(info.value)
    assert not (study/'closed.json').exists()


def test_commit_defaults_to_head(tmp_path, repo):
    recorded = pinned(tmp_path)
    released = list(recorded)[0]
    head = commit(tmp_path)
    assert close_study(tmp_path/'study', 'terminal', recorded, [released])['git_commit'] == head


def test_releasing_nothing_needs_no_resolvable_commit(tmp_path, repo):
    recorded = pinned(tmp_path)
    # Nothing committed yet, so HEAD does not resolve. With nothing released there
    # is no content for a commit to evidence; the digests still identify the tree.
    assert close_study(tmp_path/'unresolved', 'terminal', recorded, [])['git_commit'] is None
    head = commit(tmp_path)
    assert close_study(tmp_path/'resolved', 'terminal', recorded, [])['git_commit'] == head


def test_releasing_a_path_requires_a_resolvable_commit(tmp_path, repo):
    recorded = pinned(tmp_path)
    study = tmp_path/'study'
    with pytest.raises(ValueError, match='Cannot resolve'):
        close_study(study, 'terminal', recorded, [list(recorded)[0]])
    assert not (study/'closed.json').exists()


def staged_only(tmp_path):
    """A source whose pinned revision was staged and never committed: a blob holds it."""
    source = tmp_path/'scripts'/'train.py'
    source.parent.mkdir()
    source.write_text('ARM = 1\n')
    recorded = {str(source): digest(source)}
    blob = git(tmp_path, 'hash-object', '-w', str(source))
    source.write_text('ARM = 2\n')
    return source, recorded, blob


def test_closure_can_evidence_released_content_by_blob(tmp_path, repo):
    source, recorded, blob = staged_only(tmp_path)
    head = commit(tmp_path, 'only the later content was ever committed')
    study = tmp_path/'study'
    record = close_study(study, 'terminal', recorded, [str(source)], blobs={str(source): blob})
    assert record['released_blobs'] == {str(source): blob}
    # No commit holds the released content; HEAD only dates the closure.
    assert record['git_commit'] == head
    verify_closed(study, recorded)


def test_verify_closed_rejects_blob_evidence_that_no_longer_holds(tmp_path, repo):
    source, recorded, blob = staged_only(tmp_path)
    commit(tmp_path, 'only the later content was ever committed')
    study = tmp_path/'study'
    close_study(study, 'terminal', recorded, [str(source)], blobs={str(source): blob})
    verify_closed(study, recorded)
    # Repoint the record at a blob holding different content, as a prune and
    # re-tag would: the released path has no commit behind it, so a caller that
    # trusts released_paths alone would still report the study as verified.
    other = git(tmp_path, 'hash-object', '-w', str(source))
    closed = json.loads((study/'closed.json').read_text())
    closed['released_blobs'][str(source)] = other
    (study/'closed.json').write_text(json.dumps(closed))
    with pytest.raises(ValueError, match='Blob does not hold'):
        verify_closed(study, recorded)


def test_closure_refuses_a_blob_that_does_not_hold_the_recorded_content(tmp_path, repo):
    source, recorded, _ = staged_only(tmp_path)
    other = git(tmp_path, 'hash-object', '-w', str(source))
    commit(tmp_path)
    study = tmp_path/'study'
    for blob, expected in ((other, other), ('0'*40, 'unreadable')):
        with pytest.raises(ValueError, match=re.escape(expected)) as info:
            close_study(study, 'terminal', recorded, [str(source)], blobs={str(source): blob})
        assert str(source) in str(info.value)
        assert not (study/'closed.json').exists()


def test_a_blob_cannot_evidence_a_path_the_closure_does_not_release(tmp_path, repo):
    recorded = pinned(tmp_path)
    released, unreleased, _ = recorded
    commit(tmp_path)
    blob = git(tmp_path, 'hash-object', '-w', unreleased)
    study = tmp_path/'study'
    with pytest.raises(ValueError, match=re.escape(unreleased)):
        close_study(study, 'terminal', recorded, [released], blobs={unreleased: blob})
    assert not (study/'closed.json').exists()


# Each of these once called validate_prospective_points.protocol(), which gates
# the unrelated v2 study: it failed them when v2 drifted and verified nothing
# they actually read. Comments are stripped so a line explaining history passes.
@pytest.mark.parametrize('script', ['scripts/report_real_points_readiness.py',
                                    'scripts/validate_points_v5.py', 'scripts/train_points_v6.py'])
def test_unrelated_scripts_do_not_borrow_the_v2_gate(script):
    code = '\n'.join(line.split('#', 1)[0] for line in (ROOT/script).read_text().splitlines())
    assert 'validate_prospective_points' not in code


def readiness_tree(tmp_path, monkeypatch):
    """The v5 layout: protocol pins src/points_v5.py and the sidecar script by absolute path."""
    monkeypatch.chdir(tmp_path)
    init_repo(tmp_path, monkeypatch)
    (tmp_path/'logs/prospective_points_v1').mkdir(parents=True)
    (tmp_path/'logs/prospective_points_v1/protocol.json').write_text('{"study": "v1"}')
    (tmp_path/'logs/points_v5_synthetic').mkdir()
    (tmp_path/'logs/points_v5_synthetic/repaired.joblib').write_bytes(b'candidate')
    (tmp_path/'src').mkdir()
    (tmp_path/'scripts').mkdir()
    (tmp_path/'src/points_v5.py').write_text('POLICY = 1\n')
    script = tmp_path/'scripts/validate_points_v5.py'
    script.write_text('def run(): pass\n')
    sources = {'src/points_v5.py': digest('src/points_v5.py'), str(script): digest(script)}
    protocol = {'candidate_path': 'logs/points_v5_synthetic/repaired.joblib',
                'candidate_sha256': digest('logs/points_v5_synthetic/repaired.joblib'),
                'source_hashes': sources, 'v1_protocol_sha256': digest('logs/prospective_points_v1/protocol.json')}
    study = tmp_path/'logs/prospective_points_v5'
    study.mkdir()
    (study/'protocol.json').write_text(json.dumps(protocol))
    commit(tmp_path)  # The closure names the commit holding the script it releases.
    close_study(study, 'terminal', sources, [str(script)])
    dataset = tmp_path/'real_points.joblib'
    joblib.dump({'dates': np.array(['2024-12-01', '2025-04-01', '2025-06-01', '2025-11-01']),
                 'games': np.array(['1', '2', '3', '4'])}, dataset)
    return study, dataset


def test_readiness_live_verifies_artifacts_and_tolerates_the_released_script(tmp_path, monkeypatch):
    from scripts.report_real_points_readiness import run
    study, dataset = readiness_tree(tmp_path, monkeypatch)
    (tmp_path/'scripts/validate_points_v5.py').write_text('def run(): pass  # edited after closure\n')
    run(dataset)
    assert json.loads((tmp_path/'readiness.json').read_text())['frozen_studies_verified'] is True
    (tmp_path/'logs/points_v5_synthetic/repaired.joblib').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Frozen v5 drift'):
        run(dataset)


def test_readiness_still_live_verifies_the_unreleased_feature_source(tmp_path, monkeypatch):
    from scripts.report_real_points_readiness import run
    study, dataset = readiness_tree(tmp_path, monkeypatch)
    (tmp_path/'src/points_v5.py').write_text('POLICY = 2\n')
    with pytest.raises(ValueError, match='src/points_v5.py'):
        run(dataset)
    assert not (tmp_path/'readiness.json').exists()


def test_readiness_refuses_a_revised_closure_record(tmp_path, monkeypatch):
    from scripts.report_real_points_readiness import run
    study, dataset = readiness_tree(tmp_path, monkeypatch)
    closed = json.loads((study/'closed.json').read_text())
    closed['source_hashes_at_close']['src/points_v5.py'] = '0'*64
    (study/'closed.json').write_text(json.dumps(closed))
    with pytest.raises(ValueError):
        run(dataset)


def frozen_copy(tmp_path):
    frozen = tmp_path/'frozen'
    frozen.mkdir()
    for name in ('development_selected.joblib', 'compact_expanded.joblib'):
        (frozen/name).write_bytes(name.encode())
    record = tmp_path/'protocol.json'
    record.write_text(json.dumps({'frozen_reference_hashes': {p.name: digest(p) for p in frozen.glob('*.joblib')}}))
    return frozen, record


def test_train_points_v6_pins_the_v4_arms_it_scores(tmp_path):
    from scripts.train_points_v6 import verify_frozen_reference
    frozen, record = frozen_copy(tmp_path)
    verify_frozen_reference(frozen, record)
    (frozen/'compact_expanded.joblib').write_bytes(b'retrained in place')
    with pytest.raises(ValueError, match='compact_expanded.joblib'):
        verify_frozen_reference(frozen, record)


def test_train_points_v6_first_run_has_nothing_to_pin_yet(tmp_path):
    from scripts.train_points_v6 import verify_frozen_reference
    frozen, record = frozen_copy(tmp_path)
    verify_frozen_reference(frozen, tmp_path/'absent.json')
    record.write_text(json.dumps({'frozen_reference_hashes': {}}))
    with pytest.raises(ValueError):
        verify_frozen_reference(frozen, record)


PIN_KEYS = ('source_hashes', 'source_sha256')


def study_records():
    """Every frozen study record under logs/, with the source digests it pinned."""
    for name in ('protocol.json', 'manifest.json'):
        for path in sorted((ROOT/'logs').rglob(name)):
            body = json.loads(path.read_text())
            if isinstance(body, dict):
                for key in PIN_KEYS:
                    if isinstance(body.get(key), dict):
                        yield path, key, body[key]


def test_no_study_record_pins_source_content_the_tree_no_longer_holds():
    if not (ROOT/'logs').exists():
        pytest.skip('logs/ is gitignored (.gitignore:20) and absent on a fresh clone')
    offenders = []
    for path, key, pins in study_records():
        closed = path.parent/'closed.json'
        record = json.loads(closed.read_text()) if closed.exists() else {}
        released = set(record.get('released_paths', ()))
        blobs = record.get('released_blobs', {})
        if blobs:
            # Released content that no commit holds: the closure names the blob that does.
            try:
                verify_released_blobs(pins, blobs)
            except ValueError as gone:
                offenders.append(f'{path.parent.name}/{path.name} [{key}] {gone}')
        for pin, sha in sorted(pins.items()):
            if pin in released:
                continue  # Closed and explicitly released: its content may move on.
            live = Path(pin) if Path(pin).is_absolute() else ROOT/pin
            if not live.exists():
                offenders.append(f'{path.parent.name}/{path.name} [{key}] {pin}: gone from the tree')
            elif digest(live) != sha:
                offenders.append(f'{path.parent.name}/{path.name} [{key}] {pin}: content differs')
    assert not offenders, (
        'Study records pin source content the tree no longer holds:\n  '+'\n  '.join(offenders)+
        '\nEither restore the pinned content, or -- if the study is terminal -- close it with '
        'src.study_provenance.close_study(directory, reason, recorded, released, commit=<the '
        'newest commit holding those digests>), which releases exactly those paths from live '
        'verification. A study that can still be rerun must not be closed.')
