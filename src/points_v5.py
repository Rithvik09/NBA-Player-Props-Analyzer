"""Separate candidate contract: honest short windows and partial injury evidence."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.points_candidate import ExpandedPointsCandidate, normalize_points_inputs
from src.historical_injuries import normalize_name

POLICY = 'points_short_history_injuries_v5'
INJURY_COLUMNS = [f'{side}_{name}' for side in ('team', 'opp') for name in
    ('injury_report_available', 'out_count', 'out_unresolved_count', 'out_unknown_role_count', 'known_key_players_out')]
EXTRA_COLUMNS = ['history_appearances', 'short_history', 'team_injury_unknown', 'opp_injury_unknown']+INJURY_COLUMNS
MINIMUM = {**{f'trend_slope_{n}games': n for n in (5, 10)},
           **{f'minutes_last_{n}_games': n for n in (3, 5, 7)},
           **{f'last_{n}_games_trend': n for n in (3, 5, 10)},
           **{f'rolling_{n}day_avg': n for n in (7, 14, 30)},
           'true_shooting_pct': 5, 'effective_fg_pct': 5,
           'fg_pct_recent': 5, 'fg3_pct_recent': 5, 'ft_pct_recent': 5,
           'minutes_volatility': 5, 'stddev': 5}
# Legacy history is capped at20; a 30-appearance feature never has a complete
# window in this contract, so keep that slot unavailable rather than relabel it.


def prepare(frame):
    out = normalize_points_inputs(frame).copy()
    n = pd.to_numeric(out['games_played'], errors='raise')
    if n.isna().any() or (n < 1).any():
        raise ValueError('At least one strictly prior appearance is required')
    out['history_appearances'] = n
    out['short_history'] = (n < 10).astype(float)
    for key, minimum in MINIMUM.items():
        if key in out:
            out.loc[n < minimum, key] = np.nan
    for side in ('team', 'opp'):
        out[side+'_injury_unknown'] = out[side+'_key_players_out'].isna().astype(float)
    for col in INJURY_COLUMNS:
        if col not in out:
            out[col] = np.nan
    return out


def injury_evidence(ctx, team, date, roster=None):
    report = ctx.injuries(int(team), date)
    if report is None:
        return {'injury_report_available': 0., 'out_count': np.nan, 'out_unresolved_count': np.nan,
                'out_unknown_role_count': np.nan, 'known_key_players_out': np.nan}, None
    roster_ids = {}
    if roster is not None:
        cutoff = ctx.cutoff(date)
        observed = pd.Timestamp(roster['fetched_at'], unit='s', tz='UTC')
        if roster['season'] != ctx.inputs.season or not 0 <= (cutoff-observed).total_seconds() <= 21600:
            raise ValueError('Roster must be current-season, fresh and observed before cutoff')
        for row in roster['players']:
            if row.get('nba_id') and int(row['team_id']) == int(team):
                roster_ids.setdefault(normalize_name(row['full_name'], keep_suffix=True), set()).add(int(row['nba_id']))
    outs = [r for r in report if r['status'] == 'Out']
    unresolved, unknown, roles = 0, 0, []
    for row in outs:
        ids = roster_ids.get(normalize_name(row['player'], keep_suffix=True), set())
        pid = next(iter(ids)) if len(ids) == 1 else None
        if len(ids) > 1:
            unresolved += 1
            continue
        if pid is None:
            pid = ctx.resolve_name(row['player'])
        if pid is None:
            unresolved += 1
            continue
        past = ctx.logs[(ctx.logs.PLAYER_ID == pid) & (ctx.logs.TEAM_ID == int(team)) &
                        (pd.to_datetime(ctx.logs.GAME_DATE) < pd.Timestamp(date))].sort_values('GAME_DATE').tail(10)
        role = float(past.MIN.mean()) if len(past) else ctx.prior_players.get(pid, {}).get('MIN')
        if role is None or not np.isfinite(role):
            unknown += 1
        else:
            roles.append(float(role))
    evidence = {'injury_report_available': 1., 'out_count': float(len(outs)),
                'out_unresolved_count': float(unresolved), 'out_unknown_role_count': float(unknown),
                'known_key_players_out': float(sum(m >= 20 for m in roles))}
    return evidence, roles if not unresolved and not unknown else None


def enrich_injuries(features, ctx, team, opponent, date, roster=None):
    result = dict(features)
    for side, tid in [('team', team), ('opp', opponent)]:
        evidence, roles = injury_evidence(ctx, tid, date, roster)
        result.update({side+'_'+k: v for k, v in evidence.items()})
        result[side+'_key_players_out'] = float(sum(m >= 20 for m in roles)) if roles is not None else np.nan
        if side == 'opp':
            result['opp_injury_impact'] = min(1., sum(roles)/240) if roles is not None else np.nan
    return result


@dataclass
class PointsV5Candidate(ExpandedPointsCandidate):
    repair_policy: str = POLICY

    def _normalize(self, frame):
        if self.repair_policy != POLICY:
            raise ValueError('Unknown repair contract')
        return prepare(frame)
