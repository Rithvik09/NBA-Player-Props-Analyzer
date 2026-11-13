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


def _reachable():
    """Every object reachable from a ref, which is exactly what survives `git gc`.

    Reachability is the question, not how it is reached: a ref pointing straight at
    a blob and a blob inside a commit's tree both qualify, and `--points-at` would
    see only the first.
    """
    return set(_git('rev-list', '--objects', '--all', '--no-object-names').decode().split())


def verify_released_blobs(recorded, blobs):
    """Content staged but never committed survives only as a blob; that blob must still hold it.

    Holding it is not enough. `git cat-file` reads a dangling blob perfectly well
    right up until the gc that prunes it, so a readable-only check would report a
    study as verified while its sole evidence was one command from deletion. The
    blob must also be reachable from some ref.
    """
    if not blobs:
        return  # Most records evidence nothing by blob; do not pay for the object walk.
    wrong, unprotected = [], []
    reachable = _reachable()
    for path, blob_sha in sorted(blobs.items()):
        try:
            content = _git('cat-file', 'blob', blob_sha)
        except (OSError, subprocess.CalledProcessError):
            wrong.append(f'{path} (blob {blob_sha} unreadable)')
            continue
        if hashlib.sha256(content).hexdigest() != recorded[path]:
            wrong.append(f'{path} (blob {blob_sha})')
        elif blob_sha not in reachable:
            unprotected.append(f'{path} (blob {blob_sha})')
    if wrong:
        raise ValueError('Blob does not hold the recorded content of: '+', '.join(wrong))
    if unprotected:
        raise ValueError('Dangling blob evidence, which no ref keeps from the next git gc: '
                         +', '.join(unprotected)+'. Tag it, e.g. git tag provenance/<study>-<path> <blob>')


def content_exists(sha256):
    """Whether any object in this repository holds that content, reachable or not.

    A path may be declared lost only when nothing holds its recorded content, so
    this is the check that keeps `lost` from laundering an ordinary stale pin into
    a closed one. Unreachable is not gone: a dangling blob can be tagged back, so
    it counts as holding the content. The walk reads every object, which is why it
    runs at closure time only and never on a verification path.
    """
    listing = _git('cat-file', '--batch-all-objects',
                   '--batch-check=%(objectname) %(objecttype) %(objectsize)').decode()
    blobs = [line.split()[0] for line in listing.splitlines()
             if line.strip() and line.split()[1] == 'blob']
    if not blobs:
        return False
    data = subprocess.run(('git', 'cat-file', '--batch'), cwd=ROOT,
                          input=('\n'.join(blobs)+'\n').encode(),
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    at = 0
    for _ in blobs:
        end = data.index(b'\n', at)
        size = int(data[at:end].split()[2])
        if hashlib.sha256(data[end+1:end+1+size]).hexdigest() == sha256:
            return True
        at = end+1+size+1
    return False


def _resolve_commit(commit):
    """The full sha a revision names, or None when the repository has no such commit."""
    try:
        return _git('rev-parse', f'{commit}^{{commit}}').decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _already_closed(path, reason, recorded, released, commit, blobs, lost=()):
    """Read the record a previous close wrote, once the caller's claim is shown to agree.

    Nothing about the world is re-checked. Unreleased sources were live-verified at
    closure, and verify_closed is what re-checks them from then on; re-verifying here
    made a second close fail for drift the record already accounts for, so closing was
    idempotent only while nothing had moved.

    The caller's own claim must still agree, because a second call describing a
    different closure is a mistake rather than a no-op, and quietly handing back the
    stored record would hide it.
    """
    closed = json.loads(path.read_text())
    differs = [name for name, claim, stored in (
        ('reason', reason, closed['reason']),
        ('source_hashes_at_close', dict(recorded), closed['source_hashes_at_close']),
        ('released_paths', sorted(released), closed['released_paths']),
        ('released_blobs', blobs, closed.get('released_blobs') or {}),
        ('lost_paths', sorted(lost), closed.get('lost_paths') or []))
        if claim != stored]
    # commit=None asks for whatever HEAD is rather than naming one, so it cannot
    # disagree; only an explicitly named commit is a claim about which commit.
    if commit is not None and _resolve_commit(commit) != closed['git_commit']:
        differs.append('git_commit')
    if differs:
        raise ValueError(f'{path} is already closed and names a different closure; '
                         'a study is closed once. Disagrees on: '+', '.join(differs))
    return closed


def close_study(directory, reason, recorded, released, commit=None, blobs=None, lost=None):
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
    lost = set(lost or ())
    stale = set(released)-set(recorded)
    if stale:
        raise ValueError('Cannot release paths the protocol never pinned: '+', '.join(sorted(stale)))
    unreleased_blobs = set(blobs)-set(released)
    if unreleased_blobs:
        raise ValueError('Cannot evidence by blob a path the closure does not release: '
                         +', '.join(sorted(unreleased_blobs)))
    unreleased_lost = lost-set(released)
    if unreleased_lost:
        raise ValueError('Cannot declare lost a path the closure does not release: '
                         +', '.join(sorted(unreleased_lost)))
    if path.exists():
        return _already_closed(path, reason, recorded, released, commit, blobs, lost)
    recoverable = sorted(p for p in lost if content_exists(recorded[p]))
    if recoverable:
        raise ValueError('Declared lost, but the recorded content can still be recovered from '
                         'this repository: '+', '.join(recoverable)+'. Evidence it by commit or '
                         'blob instead, or tag the blob that holds it.')
    verify_released_blobs(recorded, blobs)
    at_commit = [p for p in released if p not in blobs and p not in lost]
    resolved = _resolve_commit(commit or 'HEAD')
    if resolved is None and (at_commit or commit is not None):
        raise ValueError(f'Cannot resolve {commit or "HEAD"} under {ROOT}; '
                         'a closure cannot release content no commit can evidence')
    # resolved stays None only when nothing is released; the digests still identify the tree.
    if resolved is not None:
        verify_released_at_commit(recorded, at_commit, resolved)
    verify_recorded_hashes({k: v for k, v in recorded.items() if k not in set(released)},
                           f'{directory} unreleased source at closure')
    record = {'closed_at': datetime.now(timezone.utc).isoformat(), 'reason': reason,
              'git_commit': resolved, 'source_hashes_at_close': dict(recorded),
              'released_paths': sorted(released)}
    if blobs:  # Only the records that need it carry the key; a closure is never rewritten.
        record['released_blobs'] = {p: blobs[p] for p in sorted(blobs)}
    if lost:  # Stated outright: this study's inputs can no longer be reproduced.
        record['lost_paths'] = sorted(lost)
    try:
        save_new(path, record)
    except FileExistsError:  # Lost a race to another closer; that record is the one.
        return _already_closed(path, reason, recorded, released, commit, blobs, lost)
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
