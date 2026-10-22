"""Offline feature audit: equal histories, dates, venue, and source presence.

Runs the real training builder and serving get_player_stats/prepare_features.
The JSON records a disposition for each of the original 100 observations.
This tests input parity; it does not measure predictive accuracy or establish
point-in-time correctness of full-season precompute snapshots.
"""
from __future__ import annotations

import json
import ast
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pytest import MonkeyPatch
import tests.test_feature_defaults as fixture
from tests.test_feature_parity_audit import differing_features, varied_game_log, long_game_log
from src.ml_features import NUMERIC_FEATURE_KEYS

EQUIVALENT = set('team_key_players_out opp_key_players_out opp_injury_impact vs_team_avg matchup_games'.split())
FIXTURE_CONTEXT = set('games_played is_home_game_num arena_altitude arena_capacity rivalry_game'.split())
SEASON_CONTEXT = set('lg_pts_fb lg_opp_pts_fb lg_pts_off_tov lg_opp_pts_off_tov lg_fga lg_fg_pct lg_fg3a lg_tov lg_stl days_into_season season_phase_numeric games_remaining_approx season_phase playoff_implications playoff_seeding_impact'.split())
UPSTREAM_PRESENT = set('ceiling_game_frequency recent_variance_spike games_since_return'.split())


def disposition(key):
    if key in EQUIVALENT:
        return ('equivalent_model_input', 'Missing raw key and explicit zero both vectorize to zero. Explicit source-preserving zeros now make the raw audit agree too.')
    if key in FIXTURE_CONTEXT:
        return ('unequal_fixture_context', 'Compare the same ten prior games, next-game venue and team/opponent IDs. The old fixture supplied thirteen values and no venue IDs.')
    if key in SEASON_CONTEXT:
        return ('season_context_and_definition', 'Use the same prediction date and shared season-era priors. Serving now matches the trainer calendar boundaries and discrete phase definitions.')
    if key in UPSTREAM_PRESENT:
        return ('fixture_bypassed_upstream', 'The old fixture omitted inputs calculated in analyze_prop_bet. The audited path now derives them once from the fetched box-score log, including threshold conventions.')
    return ('serving_formula_or_source', 'Serving previously used a constant, another metric, or default-filled aliases. The audited serving path now uses the trainer definition and preserves actual source-row presence.')


def run_scenario(name, varying=False, prop='points', history_count=10):
    with MonkeyPatch.context() as patch:
        if varying or history_count == 20:
            frame = long_game_log(varying) if history_count == 20 else varied_game_log()
            patch.setattr(fixture, '_canned_game_log', lambda: frame.copy())
        context = {'pace': 110.} if varying else {}
        train = fixture._train_feature_dict(patch, context, prop_type=prop, history_count=history_count)
        serve = fixture._serve_feature_dict_with_game_log(patch, context, prop_type=prop, history_count=history_count)
        diffs = differing_features(train, serve)
        return {'name': name, 'prop': prop, 'compared': len(NUMERIC_FEATURE_KEYS),
                'differences': [{'feature': k, 'train': a, 'serve': b} for k, (a, b) in diffs.items()]}



def feature_definitions(source, function=None):
    tree = ast.parse(source)
    if function:
        tree = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    out = {}
    for node in ast.walk(tree):
        entries = []
        if isinstance(node, ast.Dict):
            entries = zip(node.keys, node.values)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == 'setdefault' and len(node.args) >= 2):
            entries = [(node.args[0], node.args[1])]
        for key, value in entries:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                out.setdefault(key.value, []).append({
                    'line': key.lineno, 'expression': ast.get_source_segment(source, value)})
    return out

def main():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / 'tests/fixtures/feature_parity_baseline.json').read_text())
    scenarios = [run_scenario('equal_flat_history')]
    scenarios += [run_scenario('equal_varying_history', True, prop)
                  for prop in ('points', 'assists', 'rebounds', 'three_pointers')]
    training = feature_definitions((root / 'scripts/train_models.py').read_text())
    previous_source = subprocess.run(['git', 'show', 'HEAD:src/models.py'], cwd=root, capture_output=True, text=True, check=True).stdout
    previous = feature_definitions(previous_source, 'prepare_features')
    current = feature_definitions((root / 'src/feature_context.py').read_text())
    scenarios += [run_scenario('equal_flat_20_games', history_count=20),
                  run_scenario('equal_varying_20_games', True, history_count=20)]
    observations = []
    for row in baseline['differences']:
        category, evidence = disposition(row['feature'])
        observations.append({**row, 'classification': category,
                             'evidence': evidence,
                             'training_definition': training.get(row['feature'], []),
                             'previous_serving_definition': previous.get(row['feature'], []),
                             'current_context_definition': current.get(row['feature'], []),
                             'status': 'covered_by_equal_input_regression'})
    counts = {}
    for row in observations:
        key = row['classification']
        counts[key] = counts.get(key, 0) + 1
    report = {
        'feature_count': len(NUMERIC_FEATURE_KEYS), 'original_observations': len(observations),
        'classification_counts': counts, 'scenarios': scenarios,
        'remaining_differences': sum(len(s['differences']) for s in scenarios),
        'investigated_observations': observations,
        'additional_confirmed_findings': [
            'Historical training ignored the loader tuple keys and could read a later player season. It now selects the requested season.',
            'The NBA home marker vs. was not recognized by the training matchup parser.',
            'The analysis caller overwrote recent/season averages with location averages after preparing features. Location now remains a separate feature.',
            'Upstream default-filled shot-frequency aliases bypassed the shared defaults; real source-row presence now controls the derived aliases.',
            'The on/off collector assigned the first NBA frame to on-court even though the provider returns off-court first. It now selects explicit court status and recollects corrected rows.',
            'Historical collectors selected current active players, and YoY selected the latest career seasons. Bulk season-specific sources now include historical players and exclude future seasons.',
            'Null shot-zone counts propagated into NaN shares. The collector now normalizes empty zone counts before calculating shares.',
        ],
        'crawl_coverage_report': 'logs/rebuild_coverage.json',
        'collected_data_validation': 'logs/crawl_data_validation.json',
        'limits': [
            'Parity compares feature inputs, not AUC or betting returns.',
            'Populated advanced/tracking/clutch/quarter rows are checked by tests/test_feature_parity_audit.py.',
            'Full-season precompute tables remain season snapshots, not as-of-game historical records. Same-season look-ahead is not resolved by this audit.',
            'Team and player-vs-opponent tables are single snapshots; the historical crawl overwrites those with each later season.',
            'Live injury, market, and matchup context can legitimately differ from historical neutral inputs.',
            'Historical opponent back-to-back context still uses a rest-snapshot heuristic; equal-empty-source fixtures do not validate its schedule accuracy.',
            'Serving keeps the latest 20 games, whereas historical training uses all prior games in a season. Equal-history tests do not validate that operational window choice.',
            'The unused incremental serving builder is outside this audit scope.',
            'Shipped models and the 416-feature order are unchanged.',
        ],
    }
    target = root / 'logs/feature_parity_audit.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('feature_count', 'original_observations', 'classification_counts', 'remaining_differences')}, indent=2))
    if report['remaining_differences']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
