"""Shared fit/inference repairs for the expanded points research candidate."""
from dataclasses import dataclass
import numpy as np
from src.verified_training import SelectedPointsCandidate

MATCHUP_MEANS=('vs_team_avg','vs_opp_avg_pts','vs_opp_avg_min','vs_opp_fg_pct','vs_opp_ts_pct')
NORMALIZATION='points_missing_matchup_v1'


def normalize_points_inputs(frame):
    """A mean over zero meetings is unavailable, not a measured zero.

    Counts stay zero; the explicitly defined own-average H2H fallback is retained.
    Genuine observed zero scoring/shooting percentages remain zero.
    """
    result=frame.copy()
    required=[c for c in MATCHUP_MEANS if c in result]
    if required:
        if 'matchup_games' not in result:raise ValueError('Matchup count required to distinguish missing means')
        unknown=result.matchup_games.isna() | (result.matchup_games<=0)
        result.loc[unknown,required]=np.nan
    return result


@dataclass
class ExpandedPointsCandidate(SelectedPointsCandidate):
    input_normalization:str=NORMALIZATION

    def _normalize(self,frame):
        if self.input_normalization!=NORMALIZATION:raise ValueError('Unsupported points input normalization')
        return normalize_points_inputs(frame)

    def predict(self,frame):
        return super().predict(self._normalize(frame))

    def predict_over(self,frame,lines):
        return super().predict_over(self._normalize(frame),lines)
