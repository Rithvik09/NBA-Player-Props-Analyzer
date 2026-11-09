"""Record actual supplied training examples; never infer missing legacy history."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from src.serving_provenance import ROOT, file_hash, safe


def record_training_inputs(examples, config):
    digest=hashlib.sha256()
    dates=[]
    identities=set()
    for e in examples:
        row={'prop':e.prop_type,'player_id':e.player_id,'game_id':e.game_id,
             'game_date':e.game_date.isoformat(),'outcome':e.y,'minutes':e.minutes,
             'feature_policy':e.feature_policy,'features':e.X.to_dict(orient='split')}
        digest.update(json.dumps(safe(row),sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n')
        dates.append(e.game_date.isoformat())
        identities.add((str(e.game_id),int(e.player_id)))
    sources={str(p.relative_to(ROOT)):file_hash(p) for folder in ('src','scripts') for p in sorted((ROOT/folder).glob('*.py'))}
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        commit=None  # Source hashes still identify an exported source tree.
    return {'schema_version':1,'recorded_at':datetime.now(timezone.utc).isoformat(),
        'git_commit':commit,'source_hashes':sources,'config':safe(config),
        'packages':{d.metadata['Name']:d.version for d in distributions() if d.metadata['Name']},
        'python':__import__('platform').python_version(),
        'examples_sha256':digest.hexdigest(),'examples_count':len(examples),
        'unique_player_games':len(identities),'min_outcome_date':min(dates,default=None),
        'max_outcome_date':max(dates,default=None),
        'scope':'All supplied examples, including calibration/evaluation; digest preserves order, targets, identities and feature values. Does not certify raw-source point-in-time availability.'}
