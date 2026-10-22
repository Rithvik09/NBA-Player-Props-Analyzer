"""Serving features with the definitions used by the historical trainer.

Summaries are made from the already fetched box scores. Official advanced
percentages retain their own keys; they must not replace differently defined
box-score ratios just because the names sound similar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml_features import FEATURE_DEFAULTS


def summarize_game_log(frame):
    if frame.empty:
        return {}
    hist = frame.copy()
    hist['GAME_DATE'] = pd.to_datetime(hist['GAME_DATE'])
    hist = hist.sort_values('GAME_DATE')
    cols = ('MIN PTS REB AST STL BLK TOV FGA FGM FG3A FG3M FTA OREB DREB PF '
            'PLUS_MINUS FG_PCT FG3_PCT FT_PCT').split()
    means = {c: float(pd.to_numeric(hist[c], errors='coerce').fillna(0).mean())
             for c in cols if c in hist}
    recent = {c: float(pd.to_numeric(hist[c], errors='coerce').fillna(0).tail(5).mean())
              for c in cols if c in hist}
    pm = pd.to_numeric(hist.get('PLUS_MINUS', pd.Series(0., index=hist.index)))
    wl = hist.get('WL', pd.Series('', index=hist.index)).astype(str)
    pts = pd.to_numeric(hist.get('PTS', pd.Series(0., index=hist.index)))
    mins = pd.to_numeric(hist.get('MIN', pd.Series(0., index=hist.index)))
    return {'means': means, 'recent': recent, 'count': len(hist),
            'minutes_std': float(mins.std()) if len(mins) > 1 else 0.0,
            'minutes_last5_total': float(mins.tail(5).sum()),
            'minutes': mins.tolist(),
            'blowout': float((pm.abs() > 15).mean()),
            'close': float((pm.abs() <= 5).mean()),
            'points_std': float(pts.std()) if len(pts) > 1 else 0.0,
            'points_ceiling': float((pts >= means.get('PTS', 0) * 1.5).mean()),
            'wins_last10': float((wl.tail(10) == 'W').sum()),
            'wl': wl.tolist(),
            'points_wins': float(pts[wl == 'W'].mean()) if (wl == 'W').any() else means.get('PTS', 0),
            'points_losses': float(pts[wl == 'L'].mean()) if (wl == 'L').any() else means.get('PTS', 0),
            'positive_points_games': int((pts > 0).sum()),
            'dates': hist['GAME_DATE'].dt.strftime('%Y-%m-%d').tolist(),
            'matchups': hist.get('MATCHUP', pd.Series('', index=hist.index)).tolist(),
            'total_ast': float(pd.to_numeric(hist.get('AST', pd.Series(0., index=hist.index))).sum()),
            'total_tov': float(pd.to_numeric(hist.get('TOV', pd.Series(0., index=hist.index))).sum()),
            'total_reb': float(pd.to_numeric(hist.get('REB', pd.Series(0., index=hist.index))).sum()),
            'total_min': float(mins.sum()),
            'total_fga': float(pd.to_numeric(hist.get('FGA', pd.Series(0., index=hist.index))).sum()),
            'total_pts': float(pts.sum()),
            'recent_points_per_shot': float(pts.tail(5).sum()) / max(float(pd.to_numeric(hist.get('FGA', pd.Series(0., index=hist.index))).tail(5).sum()), 1.)}


def apply_log_features(features, summary, prediction_date=None):
    """Update features only when a real log exists, retaining no-data defaults."""
    if not summary:
        return
    m, r = summary['means'], summary['recent']
    mins, pts = m.get('MIN', 0), m.get('PTS', 0)
    fga, fta, ast = m.get('FGA', 0), m.get('FTA', 0), m.get('AST', 0)
    reb = m.get('OREB', 0) + m.get('DREB', 0)
    per36 = max(mins / 36, 0.1)
    consistency = max(0.0, 1 - summary['points_std'] / max(pts, 1))
    features.update({
        'fg3_pct_recent': r.get('FG3_PCT', 0), 'fg_pct_recent': r.get('FG_PCT', 0),
        'fga_per_game': fga, 'fg3a_per_game': m.get('FG3A', 0), 'fta_per_game': fta,
        'oreb_per_game': m.get('OREB', 0), 'dreb_per_game': m.get('DREB', 0),
        'plus_minus_avg': m.get('PLUS_MINUS', 0), 'fouls_per_game': m.get('PF', 0),
        'win_rate_last10': summary['wins_last10'] / min(10, summary['count']),
        'points_per_shot': summary['total_pts'] / max(summary['total_fga'], 1),
        'ast_to_tov_ratio': summary['total_ast'] / max(summary['total_tov'], 1),
        'reb_rate_per_36': summary['total_reb'] / max(summary['total_min'], 1) * 36,
        'minutes_volatility': summary['minutes_std'],
        'blowout_game_pct': summary['blowout'], 'close_game_pct': summary['close'],
        'ceiling_game_frequency': summary['points_ceiling'],
        'true_shooting_pct': pts / (2 * (fga + .44 * fta)) if fga + .44 * fta > 0 else FEATURE_DEFAULTS['true_shooting_pct'],
        'assist_percentage': ast / max(fga, 1), 'rebound_percentage': reb / 100,
        'on_court_plus_minus': m.get('PLUS_MINUS', 0), 'net_rating': m.get('PLUS_MINUS', 0),
        'corner_3_pct': r.get('FG3_PCT', 0) * 1.05,
        'above_break_3_pct': r.get('FG3_PCT', 0),
        'corner_three_pct': r.get('FG3_PCT', 0), 'above_break_three_pct': r.get('FG3_PCT', 0),
        'restricted_area_fg_pct': r.get('FG_PCT', 0),
        'garbage_time_minutes_pct': summary['blowout'] * .15,
        'typical_substitution_minute': min(48, mins + 3),
        'paint_touch_frequency': summary['positive_points_games'] * .1,
        'touches_per_game': fga + ast, 'front_court_touches_per_game': fga + ast,
        'paint_touches_per_game': m.get('REB', 0) * .5,
        'clutch_performance_score': consistency,
        'shot_selection_rating': r.get('FG_PCT', 0), 'bad_shot_frequency': 1 - r.get('FG_PCT', 0),
        'performance_when_leading': summary['points_wins'],
        'performance_when_trailing': summary['points_losses'],
        'pts_vs_top10_defenses': pts, 'pts_vs_bottom10_defenses': pts,
        'minutes_fatigue_score': min(1., summary['minutes_last5_total'] / max(mins * 5, 1)) if mins > 0 else .5,
        'rebound_positioning_score': (summary['total_reb'] / max(summary['total_min'], 1) * 36) / max(reb * 36 / max(mins, 1), .1),
        'pct_fga_2pt': 1 - m.get('FG3A', 0) / max(fga, 1),
        'pct_fga_3pt': m.get('FG3A', 0) / max(fga, 1),
        'paint_pts_per_game': features.get('pct_pts_paint', FEATURE_DEFAULTS['pct_pts_paint']) * pts,
        'avg_points_per_touch': pts / max(features.get('tracking_touches_pg', 50.), 1.),
        'team_l10_wins': summary['wins_last10'],
        'consistency_score': consistency,
        'scoring_efficiency_trend': summary['recent_points_per_shot'] - summary['total_pts'] / max(summary['total_fga'], 1.),
        'performance_by_score_differential': summary['points_wins'] - summary['points_losses'],
        'usage_trend': r.get('FGA', 0) - fga,
    })
    # An official EFG of zero follows the trainer's box-score fallback.
    features['effective_fg_pct'] = features.get('efg_pct_official') or (
        (m.get('FGM', fga * .47) + .5 * m.get('FG3M', m.get('FG3A', 0) * .36)) / max(fga, 1))
    # The caller's default-filled official value is not evidence of a row.
    # Pass presence separately; absent rows must use the box-score fallback.
    if not summary.get('official_efg_present', False):
        features['effective_fg_pct'] = (m.get('FGM', fga * .47) + .5 * m.get('FG3M', m.get('FG3A', 0) * .36)) / max(fga, 1)
    pace = features.get('team_pace', FEATURE_DEFAULTS['team_pace'])
    p100 = 100 / max(pace * mins / 48 if mins > 0 else 50, 1)
    for stat, key in [('PTS', 'pts'), ('AST', 'ast'), ('STL', 'stl'), ('BLK', 'blk'), ('TOV', 'tov')]:
        features[key + '_per_100'] = m.get(stat, 0) * p100
    features['reb_per_100'] = reb * p100
    for zone, alias in [('rim', 'rim_fga_per_game'), ('paint', 'paint_fga_per_game'), ('midrange', 'mid_range_fga_per_game')]:
        features[alias] = features.get(zone + '_fga_pct', FEATURE_DEFAULTS[zone + '_fga_pct']) * fga
    features['paint_attempts_per_game'] = features['paint_fga_per_game']
    features['restricted_area_attempts'] = features['rim_fga_per_game']
    features['usage_rate_with_star_out'] = fga * 1.1
    features['ft_rate'] = features['fta_rate_player'] = fta / max(fga, 1.)
    features['and_one_frequency'] = min(.3, fta / max(fga * 3, 1.))
    for n in (3, 5, 7):
        features[f'minutes_last_{n}_games'] = float(sum(summary['minutes'][-n:]))
    features['avg_minutes_last_3'] = float(np.mean(summary['minutes'][-3:]))
    features['paint_touch_to_points'] = features.get('pct_pts_paint', FEATURE_DEFAULTS['pct_pts_paint']) * pts / max(features.get('paint_fga_pct', FEATURE_DEFAULTS['paint_fga_pct']) * fga, .1)
    features['historical_game_script_avg'] = features['expected_game_script'] = summary['blowout'] - summary['close']
    features['blowout_probability'], features['close_game_probability'] = summary['blowout'], summary['close']
    wl = summary['wl']
    streak = 0
    for result in reversed(wl[-10:]):
        if result != wl[-1]:
            break
        streak += 1
    features['team_current_streak'] = float(streak if wl[-1] == 'W' else -streak)
    dates = pd.to_datetime(summary['dates'])
    games_since, missed = 0, 0.
    for gap in (dates[::-1][:-1] - dates[::-1][1:]).days:
        if gap > 5:
            missed = max(0., (gap - 2) / 2)
            break
        games_since += 1
    features['games_since_return'], features['missed_games_before_return'] = float(games_since), missed
    date = pd.Timestamp(prediction_date or pd.Timestamp.now().normalize())
    gap = max(0, (date - dates[-1]).days)
    rest = max(0, gap - 1)
    features.update(rest_days=float(rest), days_since_last_game=float(gap), is_back_to_back=float(rest == 0), b2b_flag=float(rest == 0))
    own, opp = rest, features.get('days_rest_opponent', 2.)
    features.update(rest_advantage=own-opp, rest_advantage_abs=abs(own-opp), both_teams_rested=float(own >= 2 and opp >= 2))


def apply_serving_context(features, inputs, team_context=None, opponent_context=None):
    """Apply audited definitions using real source rows, preserving presence."""
    summary = inputs.get('_log_summary')
    if not summary:
        return
    sources = inputs.get('_feature_sources', {})
    adv, tracking = sources.get('advanced', {}), sources.get('tracking', {})
    clutch, quarters = sources.get('clutch', {}), sources.get('quarters', {})
    lineup = sources.get('lineup', {})
    profile = sources.get('shot_profile', {})
    for key, source in [('contested_shot_pct', 'tight_shot_frequency'),
                        ('catch_and_shoot_pct', 'catch_shoot_frequency'),
                        ('pull_up_shot_pct', 'pullup_frequency')]:
        features[key] = profile.get(source, FEATURE_DEFAULTS[key])
    features['wide_open_shot_pct'] = (max(0., profile['open_shot_frequency'] - profile['tight_shot_frequency'])
        if 'open_shot_frequency' in profile and 'tight_shot_frequency' in profile
        else FEATURE_DEFAULTS['wide_open_shot_pct'])
    summary = dict(summary, official_efg_present='efg_pct_official' in adv)
    date = pd.Timestamp(inputs.get('_prediction_date') or pd.Timestamp.now().normalize())
    apply_log_features(features, summary, date)
    mins = summary['means'].get('MIN', 0)
    std = float(np.std(list(inputs.get('values') or [0.])))
    recent = features.get('recent_avg', 0.)
    values = list(reversed(inputs.get('values') or [0.]))
    for n in (3, 5, 10):
        features[f'last_{n}_games_trend'] = float(np.mean(values[-n:])) - features['season_avg']
    window = values[-10:]
    proxy = float(np.mean(window))
    over, under = 0, 0
    for value in reversed(window):
        if value > proxy:
            if under:
                break
            over += 1
        else:
            if over:
                break
            under += 1
    features['consecutive_over_games'], features['consecutive_under_games'] = float(over), float(under)
    features.update({
        'avg_prediction_error_player': std * .5,
        'volatility_ratio': float(np.std(values[-5:]) / max(std, .01)) if len(values) >= 5 else 1.,
        'momentum_score': (recent - float(np.mean(values[-10:-5] if len(values) >= 10 else values[:max(1, len(values)//2)]))) / max(abs(float(np.mean(values[-10:-5] if len(values) >= 10 else values[:max(1, len(values)//2)]))), .1),
        'games_above_season_avg_14day': float(sum(v > features['season_avg'] for v in values[-14:])) if len(values) >= 14 else 0.,
        'model_accuracy_player': max(.55, min(.85, .85 - .025 * std)),
        'calibration_score_player': max(.55, min(.90, .90 - .030 * std)),
        'recent_variance_spike': float(np.std(values[-3:]) / max(std, .1) - 1.) if len(values) >= 3 else 0.,
        'first_quarter_avg': quarters.get('q1_avg', recent * .25),
        'fourth_quarter_avg': quarters.get('q4_avg', recent * .25),
        'clutch_minutes_per_game': clutch.get('clutch_min_per_game', mins * .15),
        'avg_dribbles_per_touch': tracking.get('tracking_avg_drib_per_touch', 2.),
        'elbow_touches_per_game': tracking.get('tracking_elbow_touches_pg', 0.),
        'time_of_possession_per_game': tracking.get('tracking_time_of_poss_pg', mins * .25),
        'touches_per_possession': tracking.get('tracking_touches_pg', 50.) / max(features['team_pace'], 1.),
        'lineup_stability_score': lineup.get('lineup_continuity', 1.),
        'bench_strength': lineup.get('bench_strength', 0.),
        'lineups_played_count': lineup.get('lineups_played_count', 1.),
        'teammate_chemistry_score': .5 + .3 * lineup.get('lineup_continuity', .5) + .001 * lineup.get('top_lineup_net_rating', 0.),
        'shot_quality_vs_expected': ((adv.get('ts_pct_official', .55) or .55) - .55) * (adv.get('usg_pct_official', .20) or .20) * 5.,
        'shot_clock_management': 12.,
        'defender_switching_frequency': features.get('opp_lineup_changes_last5', 0.) / 5.,
    })
    # The trainer uses season-era priors; the old server used a single fixed
    # prior (or today's team snapshot) for every historical season.
    year = date.year if date.month >= 10 else date.year - 1
    league = LEAGUE_AVGS_BY_SEASON.get(year, LEAGUE_AVGS_BY_SEASON[2024])
    for key, val in league.items():
        features['lg_' + key] = val
    start, end = pd.Timestamp(f'{year}-10-22'), pd.Timestamp(f'{year+1}-04-14')
    days = max(0, (date - start).days)
    fraction = days / max(1, (end - start).days)
    phase = 0 if date.month in (10, 11) else (1 if date.month in (12, 1) else (2 if date.month in (2, 3) else 3))
    features.update(days_into_season=float(days),
                    games_remaining_approx=float(max(0, int(82 * (1-fraction)))),
                    season_phase_numeric=0. if fraction < .2 else (2. if fraction > .8 else 1.),
                    season_phase=float(phase))
    rank, gb = features.get('team_conf_rank', 8.), features.get('team_games_back', 5.)
    features['playoff_implications'] = float(phase >= 2 and rank <= 12 and gb <= 6)
    features['playoff_seeding_impact'] = min(1., max(0., phase/3)) * (1. if 6 <= rank <= 12 else max(.2, 1-abs(rank-9)/8))
    pvo = sources.get('vs_opponent', {})
    features['vs_team_avg'] = pvo.get('vs_opp_avg_pts', 0.)
    features['matchup_games'] = pvo.get('vs_opp_gp', 0.)
    h2h = features['vs_team_avg'] if features['matchup_games'] > 0 else features.get('season_avg', 0.)
    features['head_to_head_avg'] = features['vs_team_last_season_avg'] = h2h
    opp_abbrev = inputs.get('_opponent_abbrev')
    matches = [i for i, matchup in enumerate(summary['matchups']) if opp_abbrev and opp_abbrev in str(matchup)]
    features['vs_team_win_pct'] = float(np.mean([summary['wl'][i] == 'W' for i in matches])) if matches else .5
    team_id, opponent_id = inputs.get('_team_id'), inputs.get('_opponent_team_id')
    from src.arena_data import ARENA_DATA, calculate_travel_metrics
    from nba_api.stats.static import teams
    team_map = {t['id']: t for t in teams.get_teams()}
    team = team_map.get(int(team_id)) if team_id else None
    opponent = team_map.get(int(opponent_id)) if opponent_id else None
    if not opp_abbrev and opponent:
        opp_abbrev = opponent['abbreviation']
        matches = [i for i, matchup in enumerate(summary['matchups']) if opp_abbrev in str(matchup)]
        features['vs_team_win_pct'] = float(np.mean([summary['wl'][i] == 'W' for i in matches])) if matches else .5
    features['rivalry_game'] = float(bool(team and opponent and (team['abbreviation'], opponent['abbreviation']) in RIVALRY_PAIRS))
    arena = ARENA_DATA.get(int(team_id) if team_id else 0, {})
    features['arena_altitude'] = arena.get('altitude', 0.)
    features['arena_capacity'] = arena.get('capacity', FEATURE_DEFAULTS['arena_capacity'])
    is_home = inputs.get('is_home')
    known = is_home in (0, 1)
    features['home_court_advantage_rating'] = (3.5 if is_home else -3.5) if known else 0.
    features['is_home_game_num'] = float(is_home) if known else 1.
    features['travel_distance'] = 0.
    if known and team_id and opponent_id and summary['matchups']:
        prev = str(summary['matchups'][-1]).split()
        prev_opp = teams.find_team_by_abbreviation(prev[-1]) if len(prev) >= 3 else None
        prev_loc = team_id if 'vs.' in prev or 'vs' in prev else (prev_opp['id'] if prev_opp else None)
        cur_loc = team_id if is_home else opponent_id
        if prev_loc:
            tz, coast, distance = calculate_travel_metrics(prev_loc, cur_loc, is_home)
            features.update(time_zone_change=float(tz), coast_to_coast=float(coast), travel_distance=float(distance))
    if len(values) == len(summary['matchups']):
        home_vals = [v for v, matchup in zip(values, summary['matchups']) if 'vs' in str(matchup)]
        away_vals = [v for v, matchup in zip(values, summary['matchups']) if '@' in str(matchup)]
        home_mean = float(np.mean(home_vals)) if home_vals else features['season_avg']
        away_mean = float(np.mean(away_vals)) if away_vals else features['season_avg']
        features['vs_team_home_away_split'] = home_mean - away_mean
        features['player_vs_arena'] = (away_mean if is_home == 0 else home_mean) - features['season_avg']
        if opp_abbrev:
            versus = [v for v, matchup in zip(values, summary['matchups']) if opp_abbrev in str(matchup)]
            features['career_vs_defender'] = (float(np.mean(versus)) if versus else features['season_avg']) - features['season_avg']
            features['recent_vs_defender'] = (float(np.mean(versus[-5:])) if versus else features['season_avg']) - features['season_avg']
    pace = (team_context or {}).get('pace', 100.)
    opp_pace = (opponent_context or {}).get('pace', 100.)
    features['implied_game_total'] = (pace + opp_pace) * 1.1
    if 'player_pace' not in adv:
        features['player_pace'] = pace
    # Prefer supplied recent/rest source values; otherwise the training
    # fallback is the opponent snapshot pace rather than a fixed 100.
    for key in ('opp_pace_last5', 'opp_b2b_pace', 'opp_rested_pace'):
        if key not in inputs:
            features[key] = opp_pace
    score = features.get('primary_defender_score01', 0.)
    gap = (date - pd.Timestamp(summary['dates'][-1])).days
    features['primary_defender_active'] = score * (1. if gap <= 3 else (.7 if gap <= 7 else .3)) if score > 0 else 0.
    # Missing context dictionaries previously looked different in raw JSON,
    # although build_feature_vector zero-filled both sides identically.
    for key in ('team_key_players_out', 'opp_key_players_out', 'opp_injury_impact'):
        features.setdefault(key, 0.)


LEAGUE_AVGS_BY_SEASON = {
    2018: {"fga": 89.2, "fg_pct": 0.461, "fg3a": 32.0, "tov": 14.1,
           "stl": 7.6, "pts_fb": 13.5, "opp_pts_fb": 13.5,
           "pts_off_tov": 17.6, "opp_pts_off_tov": 17.6},
    2019: {"fga": 88.8, "fg_pct": 0.460, "fg3a": 34.1, "tov": 14.5,
           "stl": 7.7, "pts_fb": 13.0, "opp_pts_fb": 13.0,
           "pts_off_tov": 17.2, "opp_pts_off_tov": 17.2},
    2020: {"fga": 88.1, "fg_pct": 0.466, "fg3a": 34.6, "tov": 14.0,
           "stl": 7.8, "pts_fb": 13.6, "opp_pts_fb": 13.6,
           "pts_off_tov": 16.7, "opp_pts_off_tov": 16.7},
    2021: {"fga": 88.1, "fg_pct": 0.466, "fg3a": 34.6, "tov": 13.8,
           "stl": 7.6, "pts_fb": 13.7, "opp_pts_fb": 13.7,
           "pts_off_tov": 16.7, "opp_pts_off_tov": 16.7},
    2022: {"fga": 88.3, "fg_pct": 0.466, "fg3a": 35.2, "tov": 14.1,
           "stl": 7.6, "pts_fb": 13.9, "opp_pts_fb": 13.9,
           "pts_off_tov": 16.6, "opp_pts_off_tov": 16.6},
    2023: {"fga": 89.5, "fg_pct": 0.474, "fg3a": 35.1, "tov": 14.0,
           "stl": 7.5, "pts_fb": 14.5, "opp_pts_fb": 14.5,
           "pts_off_tov": 16.4, "opp_pts_off_tov": 16.4},
    2024: {"fga": 88.4, "fg_pct": 0.467, "fg3a": 37.5, "tov": 13.9,
           "stl": 7.6, "pts_fb": 14.3, "opp_pts_fb": 14.3,
           "pts_off_tov": 15.9, "opp_pts_off_tov": 15.9},
    2025: {"fga": 89.2, "fg_pct": 0.465, "fg3a": 38.2, "tov": 13.7,
           "stl": 7.5, "pts_fb": 14.4, "opp_pts_fb": 14.4,
           "pts_off_tov": 15.8, "opp_pts_off_tov": 15.8},
}

RIVALRY_PAIRS = frozenset({
    ("BOS","LAL"),("LAL","BOS"),
    ("BOS","PHI"),("PHI","BOS"),
    ("LAL","BOS"),("LAL","LAC"),("LAC","LAL"),
    ("GSW","CLE"),("CLE","GSW"),
    ("GSW","LAL"),("LAL","GSW"),
    ("MIA","BOS"),("BOS","MIA"),
    ("BOS","NYK"),("NYK","BOS"),
    ("NYK","BKN"),("BKN","NYK"),
    ("NYK","CHI"),("CHI","NYK"),
    ("CHI","DET"),("DET","CHI"),
    ("CHI","NYK"),("CHI","CLE"),("CLE","CHI"),
    ("UTA","HOU"),("HOU","UTA"),
    ("OKC","DEN"),("DEN","OKC"),
    ("DAL","SAS"),("SAS","DAL"),
    ("DAL","HOU"),("HOU","DAL"),
    ("SAS","HOU"),("HOU","SAS"),
    ("BOS","WAS"),("WAS","BOS"),  # legacy
    ("PHI","NYK"),("NYK","PHI"),
    ("PHI","BKN"),("BKN","PHI"),
    ("MIL","TOR"),("TOR","MIL"),
    ("MIL","CHI"),("CHI","MIL"),
})
