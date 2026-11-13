"""A cache refresh must not quietly replace good data, or silently void a study's pin."""
import json
import numpy as np
import pandas as pd
import pytest
from scripts import refresh_season_logs as mod

COLUMNS = ['SEASON_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
           'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS']


def frame(rows, columns=COLUMNS):
    return pd.DataFrame(rows, columns=columns)


def row(pid, gid, date='2026-10-24', season=22026):
    return [season, pid, f'P{pid}', 1610612748, 'MIA', 'Miami Heat', gid, date, 'MIA @ CHA', 'L', 27, 13]


def cache(tmp_path, season, rows):
    d = tmp_path/'logs'/'points_pilot_cache'
    d.mkdir(parents=True, exist_ok=True)
    p = d/f'players_{season}.json'
    p.write_text(frame(rows).to_json(orient='records'))
    return p


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    (tmp_path/'logs').mkdir(exist_ok=True)
    return tmp_path


def fetcher(rows, columns=COLUMNS):
    return lambda season: frame(rows, columns)


def test_a_refresh_writes_the_cache_in_the_schema_the_readers_expect(repo):
    reference = cache(repo, '2025-26', [row(1, '0022500001', '2025-10-21', 22025)])
    out = mod.refresh('2026-27', fetch=fetcher([row(1, '0022600104'), row(2, '0022600104')]))
    written = json.loads(out.read_text())
    assert isinstance(written, list) and len(written) == 2
    assert list(written[0]) == COLUMNS, 'column order must match, since readers index positionally downstream'
    assert written[0]['GAME_DATE'] == '2026-10-24', 'plain date, not an iso timestamp'
    assert written[0]['SEASON_ID'] == '22026', 'string, as every committed-era cache file has it'
    assert json.loads(reference.read_text())[0]['PLAYER_ID'] == 1, 'other seasons untouched'


def test_a_timestamp_game_date_is_written_as_a_plain_date(repo):
    """The provider has returned both; the caches hold only plain dates."""
    cache(repo, '2025-26', [row(1, '0022500001', '2025-10-21', 22025)])
    stamped = frame([row(1, '0022600104', pd.Timestamp('2026-10-24T00:00:00'))])
    out = mod.refresh('2026-27', fetch=lambda s: stamped)
    assert json.loads(out.read_text())[0]['GAME_DATE'] == '2026-10-24'


def test_the_same_data_writes_the_same_bytes(repo):
    cache(repo, '2025-26', [row(1, '0022500001', '2025-10-21', 22025)])
    f = fetcher([row(1, '0022600104')])
    first = mod.refresh('2026-27', fetch=f).read_bytes()
    assert mod.refresh('2026-27', fetch=f).read_bytes() == first, 'digest pins need byte-stable output'


def test_a_failed_move_leaves_the_original_cache_and_no_fragment(repo, monkeypatch):
    """Readers expect a whole season; a half-written file must never take its place."""
    import pathlib
    existing = cache(repo, '2026-27', [row(1, '0022600104')])
    before = existing.read_bytes()
    monkeypatch.setattr(pathlib.Path, 'replace', lambda self, other: (_ for _ in ()).throw(OSError('disk')))
    with pytest.raises(OSError):
        mod.refresh('2026-27', fetch=fetcher([row(1, '0022600104'), row(2, '0022600104')]))
    assert existing.read_bytes() == before


def test_a_successful_refresh_leaves_no_staging_file(repo):
    cache(repo, '2025-26', [row(1, '0022500001', '2025-10-21', 22025)])
    out = mod.refresh('2026-27', fetch=fetcher([row(1, '0022600104')]))
    assert not list(out.parent.glob('*.partial')), 'staging file must not survive'


def test_an_empty_response_never_replaces_a_cache(repo):
    existing = cache(repo, '2026-27', [row(1, '0022600104'), row(2, '0022600104')])
    before = existing.read_bytes()
    with pytest.raises(ValueError, match='no rows'):
        mod.refresh('2026-27', fetch=fetcher([]))
    assert existing.read_bytes() == before


def test_duplicate_player_games_are_refused_not_deduplicated(repo):
    """load_season_logs raises on duplicates, so writing them would poison every reader."""
    cache(repo, '2025-26', [row(1, '0022500001', '2025-10-21', 22025)])
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        mod.refresh('2026-27', fetch=fetcher([row(1, '0022600104'), row(1, '0022600104')]))


def test_a_changed_schema_is_refused(repo):
    cache(repo, '2025-26', [row(1, '0022500001', '2025-10-21', 22025)])
    short = COLUMNS[:-1]
    with pytest.raises(ValueError, match='[Ss]chema'):
        mod.refresh('2026-27', fetch=lambda s: frame([row(1, '0022600104')[:-1]], short))


def test_a_shrinking_refresh_is_refused_unless_allowed(repo):
    """A truncated response looks exactly like a real one; only the row count betrays it."""
    existing = cache(repo, '2026-27', [row(i, f'00226001{i:02d}') for i in range(1, 6)])
    before = existing.read_bytes()
    with pytest.raises(ValueError, match='fewer'):
        mod.refresh('2026-27', fetch=fetcher([row(1, '0022600101')]))
    assert existing.read_bytes() == before
    out = mod.refresh('2026-27', fetch=fetcher([row(1, '0022600101')]), allow_shrink=True)
    assert len(json.loads(out.read_text())) == 1


def test_a_study_pin_on_the_cache_blocks_the_refresh_until_named(repo):
    existing = cache(repo, '2026-27', [row(1, '0022600104')])
    from src.prospective_points import digest
    study = repo/'logs'/'expanded_points_v6'
    study.mkdir(parents=True)
    (study/'manifest.json').write_text(json.dumps(
        {'input_hashes': {str(existing): digest(existing)}}))
    bigger = [row(1, '0022600104'), row(2, '0022600104')]
    with pytest.raises(ValueError, match='expanded_points_v6'):
        mod.refresh('2026-27', fetch=fetcher(bigger))
    assert len(json.loads(existing.read_text())) == 1, 'left untouched while a valid pin stands'
    out = mod.refresh('2026-27', fetch=fetcher(bigger), break_pins=True)
    assert len(json.loads(out.read_text())) == 2


def test_an_already_stale_pin_does_not_block_the_refresh(repo):
    """Only a pin that still holds is evidence; a drifted one is already void."""
    existing = cache(repo, '2026-27', [row(1, '0022600104')])
    study = repo/'logs'/'expanded_points_v4'
    study.mkdir(parents=True)
    (study/'manifest.json').write_text(json.dumps({'input_hashes': {str(existing): '0'*64}}))
    out = mod.refresh('2026-27', fetch=fetcher([row(1, '0022600104'), row(2, '0022600104')]))
    assert len(json.loads(out.read_text())) == 2
