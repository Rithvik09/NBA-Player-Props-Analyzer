"""Guard future calibration research against base-training outcome overlap."""
import numpy as np
from src.points_calibration_research_v2 import validate


def validate_disjoint_calibration(bundle, base_training_games):
    fitting,selection,support=validate(bundle)
    games=np.asarray(bundle['data']['games']).astype(str)
    overlap=set(games[fitting|selection]) & set(map(str,base_training_games))
    if overlap:
        raise ValueError(f'Calibration/selection games overlap base training: {len(overlap)} games; use forward out-of-fold predictions or a base model fitted before calibration')
    return fitting,selection,support
