"""Paired v5 sidecar: reuse fresh v1 quotes, never backdate a repaired forecast."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.prospective_points import digest, save_new, validate_quote, paired_report
from src.points_v5 import enrich_injuries, POLICY
from src.historical_inputs import HistoricalInputs
from src.study_provenance import recorded_or_live_hashes

OUT = Path('logs/prospective_points_v5')
V1 = Path('logs/prospective_points_v1')
MODELS = Path('logs/points_v5_synthetic')


def now():
    return datetime.now(timezone.utc).isoformat()


def run():
    if not (MODELS/'report.json').exists():
        print('V5 candidate not ready; original validation unaffected.')
        return
    # v1 and own-source integrity is the stored != config comparison below, which
    # pins v1's protocol and src/points_v5.py live. The former verify_v1() here was
    # redundant with it and gated on logs/prospective_points_v2, a study this
    # sidecar never reads. This file's own digest answers from closed.json: pinning
    # it live made any edit here declare the study changed and refuse to recompute.
    OUT.mkdir(exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = {'policy': POLICY, 'candidate_path': str(MODELS/'repaired.joblib'),
                  'candidate_sha256': digest(MODELS/'repaired.joblib'),
                  'source_hashes': recorded_or_live_hashes(OUT, ['src/points_v5.py', __file__]),
                  'v1_protocol_sha256': digest(V1/'protocol.json'),
                  'analysis': 'Same-game/player/line/horizon paired comparison against both frozen v1 models. Early-season warmup and eligible cohorts separate. First100 eligible binary forecasts checkpoint; at least20 games for research-success criteria. No model updates.',
                  'lines_used_for_training': 'synthetic; real-line importer is separate and has no effect on this frozen candidate'}
        path = OUT/'protocol.json'
        if path.exists():
            stored = json.loads(path.read_text())
            if {k:v for k,v in stored.items() if k!='frozen_at'} != config:
                raise ValueError('V5 study changed; create a new version rather than overwriting it')
        else:
            save_new(path, {**config, 'frozen_at': now()})
        model = joblib.load(MODELS/'repaired.joblib')
        contexts, rejected = {}, []
        for source in sorted((V1/'forecasts').glob('*.json')):
            f = json.loads(source.read_text()); dest = OUT/'forecasts'/source.name
            if dest.exists():
                continue
            if pd.Timestamp(f['quote']['game_start']) <= pd.Timestamp.now(tz='UTC'):
                continue  # exposed outcomes are never relabeled prospective
            try:
                validate_quote(f['quote'], now(), f['horizon'])
                if f['protocol_sha256'] != config['v1_protocol_sha256']:
                    raise ValueError('Reference forecast protocol mismatch')
                run_path = Path(f['source_run'])
                if digest(run_path/'inputs.joblib') != f['inputs_sha256']:
                    raise ValueError('Archived source changed')
                if str(run_path) not in contexts:
                    raw = joblib.load(run_path/'inputs.joblib')
                    roster = json.loads((run_path/'roster.json').read_text())
                    inputs = HistoricalInputs(roster['season'], raw['raw_precomputed'], raw['player_logs'], audited=True)
                    injury = run_path/'injuries.json'
                    if injury.exists():
                        inputs.live_injury_context = json.loads(injury.read_text())
                    contexts[str(run_path)] = inputs, roster
                inputs, roster = contexts[str(run_path)]
                inputs.prediction_cutoff = pd.Timestamp(f['feature_cutoff'])
                inputs.prediction_game_id = f['quote']['game_id']
                inputs.recovery._injury_cache = {}; inputs.recovery._cutoff_cache = {}
                board = json.loads((run_path/'scoreboard.json').read_text())['scoreboard']['games']
                game = next(g for g in board if str(g['gameId']) == f['quote']['game_id'])
                player = next(p for p in roster['players'] if p.get('nba_id') == f['quote']['player_id'])
                team = int(player['team_id']); teams = [int(game[s]['teamId']) for s in ('homeTeam', 'awayTeam')]
                if team not in teams:
                    raise ValueError('Roster/game identity mismatch')
                opp = next(t for t in teams if t != team)
                date = pd.Timestamp(f['quote']['game_start']).tz_convert('America/New_York').tz_localize(None).normalize()
                features = enrich_injuries(f['features'], inputs.recovery, team, opp, date, roster)
                frame = pd.DataFrame([features], dtype=float)
                prediction = {'points': float(model.predict(frame)[0]), 'over_probability': float(model.predict_over(frame, [f['quote']['line']])[0])}
                generated = now(); validate_quote(f['quote'], generated, f['horizon'])
                if not np.isfinite(prediction['points']) or not 0 <= prediction['over_probability'] <= 1:
                    raise ValueError('Invalid candidate prediction')
                save_new(dest, {**f, 'generated_at': generated, 'reference_generated_at': f['generated_at'],
                    'reference_forecast_sha256': digest(source), 'v5_protocol_sha256': digest(path),
                    'predictions': {'candidate': prediction, 'compact': f['predictions']['compact'],
                                    'frozen161': f['predictions']['candidate']},
                    'features': {k: float(v) if v is not None and np.isfinite(float(v)) else None for k,v in features.items()}})
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append({'id': f['id'], 'reason': str(exc)})
        forecasts = [json.loads(p.read_text()) for p in (OUT/'forecasts').glob('*.json')]
        outcomes = {p.stem: json.loads(p.read_text()) for p in (V1/'settlements').glob('*.json')}
        report = {'updated_at': now(), 'vs_frozen_compact': paired_report(forecasts, outcomes),
                  'vs_frozen161': paired_report([{**f, 'predictions': {**f['predictions'], 'compact': f['predictions']['frozen161']}} for f in forecasts], outcomes),
                  'rejected': rejected, 'old_exposed_game_excluded': '0022600112', 'candidate_sha256': config['candidate_sha256']}
        tmp = OUT/'report.tmp'; tmp.write_text(json.dumps(report, indent=2)); tmp.replace(OUT/'report.json')
        print('V5 prospective paired forecasts:', len(forecasts))


if __name__ == '__main__':
    run()
