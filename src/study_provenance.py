"""Per-study provenance: a study pins, and re-verifies, only what it reads."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from src.prospective_points import digest, save_new
from src.serving_provenance import ROOT


def verify_recorded_hashes(recorded, label):
    """Re-digest paths a frozen record pinned; raise naming every drifted path."""
    drifted = []
    for path, sha in recorded.items():
        try:
            if digest(path) != sha:
                drifted.append(path)
        except FileNotFoundError:
            drifted.append(path+' (missing)')
    if drifted:
        raise ValueError(f'{label} drift: '+', '.join(drifted))


def _repo_relative(path):
    """Git addresses a blob by repo-relative path; a pinned key may be absolute."""
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError(f'{path} lies outside {ROOT}; no commit can be checked for it') from None


def _git(*args):
    return subprocess.check_output(('git',)+args, cwd=ROOT, stderr=subprocess.DEVNULL)


def verify_released_at_commit(recorded, released, commit):
    """Released content may move on, so only this commit still evidences it: it must hold it."""
    wrong = []
    for path in sorted(released):
        relative = _repo_relative(path)
        try:
            blob = _git('show', f'{commit}:{relative}')
        except (OSError, subprocess.CalledProcessError):
            wrong.append(path+' (absent there)')
            continue
        if hashlib.sha256(blob).hexdigest() != recorded[path]:
            wrong.append(path)
    if wrong:
        raise ValueError(f'Commit {commit} does not hold the recorded content of: '+', '.join(wrong))


def verify_released_blobs(recorded, blobs):
    """Content staged but never committed survives only as a blob; that blob must still hold it."""
    wrong = []
    for path, blob_sha in sorted(blobs.items()):
        try:
            content = _git('cat-file', 'blob', blob_sha)
        except (OSError, subprocess.CalledProcessError):
            wrong.append(f'{path} (blob {blob_sha} unreadable)')
            continue
        if hashlib.sha256(content).hexdigest() != recorded[path]:
            wrong.append(f'{path} (blob {blob_sha})')
    if wrong:
        raise ValueError('Blob does not hold the recorded content of: '+', '.join(wrong))


def close_study(directory, reason, recorded, released, commit=None, blobs=None):
    """Record once that a study is closed, its sources at closure, and which may move on.

    The commit named must hold exactly what `recorded` claims for every released
    path -- released content was last correct at the closing commit, which is not
    always HEAD -- and every unreleased path must still live-match, or the record
    would fail the first time anything verified it.

    `blobs` maps a released path to the blob that holds its recorded content when
    no commit does, because the content was staged and never committed. Such a
    path is evidenced by its blob instead of by the commit, and the record carries
    `released_blobs` so a reader can find it.
    """
    path = Path(directory)/'closed.json'
    blobs = dict(blobs or {})
    stale = set(released)-set(recorded)
    if stale:
        raise ValueError('Cannot release paths the protocol never pinned: '+', '.join(sorted(stale)))
    unreleased_blobs = set(blobs)-set(released)
    if unreleased_blobs:
        raise ValueError('Cannot evidence by blob a path the closure does not release: '
                         +', '.join(sorted(unreleased_blobs)))
    verify_released_blobs(recorded, blobs)
    at_commit = [p for p in released if p not in blobs]
    try:
        resolved = _git('rev-parse', f'{commit or "HEAD"}^{{commit}}').decode().strip()
    except (OSError, subprocess.CalledProcessError):
        if at_commit or commit is not None:
            raise ValueError(f'Cannot resolve {commit or "HEAD"} under {ROOT}; '
                             'a closure cannot release content no commit can evidence') from None
        resolved = None  # Nothing left for a commit to hold; the digests still identify the tree.
    if resolved is not None:
        verify_released_at_commit(recorded, at_commit, resolved)
    verify_recorded_hashes({k: v for k, v in recorded.items() if k not in set(released)},
                           f'{directory} unreleased source at closure')
    record = {'closed_at': datetime.now(timezone.utc).isoformat(), 'reason': reason,
              'git_commit': resolved, 'source_hashes_at_close': dict(recorded),
              'released_paths': sorted(released)}
    if blobs:  # Only the records that need it carry the key; a closure is never rewritten.
        record['released_blobs'] = {p: blobs[p] for p in sorted(blobs)}
    try:
        save_new(path, record)
    except FileExistsError:
        return json.loads(path.read_text())  # Closed once; never re-closed.
    return record


def recorded_or_live_hashes(directory, paths):
    """Live digests, except for paths a closure record released, which keep theirs.

    A study that pins its own script cannot re-digest itself after an edit without
    declaring itself changed. Released paths therefore answer from the closure
    record, so the comparison still catches drift in every path still live.
    """
    released = {}
    path = Path(directory)/'closed.json'
    if path.exists():
        closed = json.loads(path.read_text())
        released = {p: closed['source_hashes_at_close'][p] for p in closed['released_paths']}
    return {p: released[p] if p in released else digest(p) for p in paths}


def verify_closed(directory, recorded):
    """Only a closed study's released sources may move on; the rest stay live-verified."""
    path = Path(directory)/'closed.json'
    if not path.exists():
        raise ValueError(f'No closed.json under {directory}; a live study must re-digest its sources')
    closed = json.loads(path.read_text())
    if closed['source_hashes_at_close'] != dict(recorded):
        raise ValueError(f'{path} disagrees with the protocol it closed')
    released = set(closed['released_paths'])
    stale = released-set(recorded)
    if stale:
        raise ValueError(f'{path} releases paths the protocol never pinned: '+', '.join(sorted(stale)))
    # A blob-evidenced path has no commit behind it, so the blob is the whole of
    # its provenance: if it is pruned or repointed the content is gone, and a
    # caller that skipped this check would still report the study as verified.
    verify_released_blobs(recorded, closed.get('released_blobs') or {})
    verify_recorded_hashes({k: v for k, v in recorded.items() if k not in released},
                           f'{directory} unreleased source')
