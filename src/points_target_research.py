"""Offline target controls; independent of frozen prospective pipelines."""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr
from src.points_v5 import prepare
from src.verified_training import classifier_frame
from src.points_calibration_research_v2 import probability

FIXED_THRESHOLDS = (.5, 5.5, 10.5, 15.5, 20.5, 25.5, 30.5, 40.5)


def fixed_threshold_expansion(data, columns):
    """Thresholds cannot depend on player features or realized points."""
    indices = np.repeat(np.arange(len(data['X'])), len(FIXED_THRESHOLDS))
    lines = np.tile(FIXED_THRESHOLDS, len(data['X']))
    x = classifier_frame(data['X'].iloc[indices].reset_index(drop=True), lines, columns)
    return x, (np.asarray(data['y'])[indices] > lines).astype(int), indices, lines


@dataclass
class TargetCandidate:
    columns: list
    model: object
    kind: str
    calibrations: dict

    def raw(self, frame, lines):
        x = prepare(frame)
        if self.kind == 'points':
            # Fixed scale, not optimized on comparison outcomes. Calibration learns
            # a nonnegative slope on the earlier calibration period only.
            mean = self.model.predict(x[self.columns])
            return np.clip(ndtr((mean - np.asarray(lines)) / 10.), 1e-6, 1-1e-6)
        return self.model.predict_proba(classifier_frame(x, lines, self.columns))[:, 1]

    def predict_over(self, frame, lines):
        raw = self.raw(frame, lines)
        short = frame.games_played.to_numpy() < 10
        out = np.empty(len(raw))
        for cohort, mask in [('short_history', short), ('eligible_history', ~short)]:
            out[mask] = probability(raw[mask], self.calibrations[cohort])
        return out
