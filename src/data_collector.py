import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, TeamGameLog
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
import time
from datetime import datetime


def _dvp_position_keys(raw_pos: str):
    """
    Map a position string (e.g. 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'G-F' …)
    to the tuple (dvp_pos, pos_group) used as keys in PrecomputedStore.

    dvp_pos   – the single position used to look up dvp_by_position  (e.g. 'SG')
    pos_group – the broad group used for team_special_defenders       ('G', 'F', or 'C')
    """
    p = raw_pos.upper().strip()
    # Pure center
    if p in ('C',):
        return 'C', 'C'
    # Pure guard
    if p in ('PG', 'SG', 'G'):
        return 'SG', 'G'
    # Pure forward
    if p in ('SF', 'PF', 'F'):
        return 'SF', 'F'
    # Hybrid: bias toward the primary token
    if p.startswith('G') or p.endswith('G'):
        return 'SG', 'G'
    if 'C' in p:
        return 'C', 'C'
    if p.startswith('F') or p.endswith('F'):
        return 'SF', 'F'
    # Default
    return 'SF', 'F'


def _set_dvp_defaults(features: dict) -> None:
    """
    Zero-fill DVP features when precomputed data is unavailable.
    Zero deltas = league-average defence, consistent with the inference fallback.
    """
    features['dvp_gp'] = 0
    for _k in ('pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov'):
        features[f'dvp_{_k}_delta'] = 0.0
    features['primary_defender_score01'] = 0.0


class TrainingDataCollector:
    """
    Builds rolling-window training samples from NBA game logs.
    For game i, features come from games 0..i-1 so there's no leakage.
    The simulated line is the rolling 10-game avg, matching what the model sees at inference.

    Team/opponent context (pace, defensive rating, rest days, etc.) is fetched once per
    team per session from the current season and injected into all training samples.
    This matches the inference pipeline in prepare_features(), which also uses the
    current season's team context.  The full ~45-feature set is built here so the
    model trains on exactly what it will see at prediction time.
    """

    PROP_COL_MAP = {
        'points':         'PTS',
        'assists':        'AST',
        'rebounds':       'REB',
        'steals':         'STL',
        'blocks':         'BLK',
        'turnovers':      'TOV',
        'three_pointers': 'FG3M',
    }

    MIN_PRIOR_GAMES = 10   # need at least this many prior games to build a sample
    MIN_TOTAL_GAMES = 20   # skip players with very thin history

    # Neutral defaults used when team context cannot be fetched
    _DEFAULT_TEAM_CTX = {
        'pace':             100.0,
        'offensive_rating': 110.0,
        'defensive_rating': 110.0,
        'recent_form':        0.5,
        'rest_days':            2,
        'injury_impact':      0.1,
        'key_players_out':      0,
        'total_players_out':    0,
    }

    def __init__(self):
        self._team_context_cache = {}   # abbrev -> context dict (or None on failure)
        self._team_id_map = {}          # abbrev -> team id int

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_seasons(self, num_seasons=3):
        """Returns the N most recent season strings like ['2024-25', '2023-24', ...]"""
        current_year = datetime.now().year
        current_month = datetime.now().month
        latest_year = current_year - 1 if 1 <= current_month <= 7 else current_year
        return [f"{latest_year - i}-{str(latest_year - i + 1)[2:]}" for i in range(num_seasons)]

    def _fetch_game_log(self, player_id, season):
        """Fetches one season's game log, returns empty DataFrame on failure."""
        try:
            games = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season
            ).get_data_frames()[0]
            time.sleep(0.6)
            return games
        except Exception as e:
            print(f"  Could not fetch {season} for player {player_id}: {e}")
            time.sleep(0.6)
            return pd.DataFrame()

    def _get_team_id(self, abbrev):
        """Map team abbreviation (e.g. 'LAL') to its NBA API team id."""
        if not self._team_id_map:
            all_teams = nba_teams.get_teams()
            self._team_id_map = {t['abbreviation']: t['id'] for t in all_teams}
        return self._team_id_map.get(abbrev)

    def _fetch_team_context(self, abbrev):
        """
        Fetch current-season pace / defensive-rating / rest-days for a team.
        Results are cached by abbreviation so each team is fetched at most once.
        Returns a context dict, or None on failure.
        """
        if abbrev in self._team_context_cache:
            return self._team_context_cache[abbrev]

        try:
            team_id = self._get_team_id(abbrev)
            if not team_id:
                self._team_context_cache[abbrev] = None
                return None

            current_season = self._get_seasons(1)[0]
            games = TeamGameLog(
                team_id=team_id,
                season=current_season
            ).get_data_frames()[0]
            time.sleep(0.6)

            if games.empty:
                self._team_context_cache[abbrev] = None
                return None

            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            games = games.sort_values('GAME_DATE', ascending=False)
            recent = games.head(10)

            # --- pace (Oliver possession formula) ---
            fga  = float(recent['FGA'].mean())  if 'FGA'  in recent.columns else 85.0
            oreb = float(recent['OREB'].mean()) if 'OREB' in recent.columns else 10.0
            tov  = float(recent['TOV'].mean())  if 'TOV'  in recent.columns else 13.0
            fta  = float(recent['FTA'].mean())  if 'FTA'  in recent.columns else 20.0
            pace = float(max(fga - oreb + tov + 0.44 * fta, 90.0))

            # --- defensive rating estimate ---
            pts_per_game    = float(recent['PTS'].mean())
            avg_pm          = float(recent['PLUS_MINUS'].mean()) if 'PLUS_MINUS' in recent.columns else 0.0
            pts_allowed_est = pts_per_game - avg_pm
            def_rating      = float(max(90.0, min(130.0, pts_allowed_est / (pace / 100))))
            off_rating      = float(pts_per_game / (pace / 100)) if pace > 0 else 110.0

            # --- recent form (win pct over last 10) ---
            form = float((recent['WL'] == 'W').sum() / len(recent)) if 'WL' in recent.columns else 0.5

            # --- rest days proxy (average gap between recent games) ---
            if len(games) >= 2:
                dates = games['GAME_DATE'].values
                gaps  = [
                    (dates[i - 1] - dates[i]) / np.timedelta64(1, 'D')
                    for i in range(1, min(6, len(dates)))
                ]
                rest_days = max(1, int(np.mean(gaps)))
            else:
                rest_days = 2

            context = {
                'pace':             pace,
                'offensive_rating': off_rating,
                'defensive_rating': def_rating,
                'recent_form':      form,
                'rest_days':        rest_days,
                # Injury data cannot be fetched retroactively; use neutral values
                'injury_impact':     0.1,
                'key_players_out':   0,
                'total_players_out': 0,
            }
            self._team_context_cache[abbrev] = context
            return context

        except Exception as e:
            print(f"  Could not fetch team context for {abbrev}: {e}")
            self._team_context_cache[abbrev] = None
            return None

    @staticmethod
    def _parse_matchup(matchup_str):
        """
        Parse 'ABC vs. XYZ' -> (team='ABC', opp='XYZ', is_home=1)
              'ABC @ XYZ'   -> (team='ABC', opp='XYZ', is_home=0)
        Returns (None, None, 0.5) on unexpected format.
        """
        s = str(matchup_str).upper().strip()
        parts = s.split()
        if len(parts) >= 3:
            team = parts[0]
            opp  = parts[2]
            is_home = 1 if 'VS.' in s else 0
            return team, opp, is_home
        return None, None, 0.5

    @staticmethod
    def _team_features(ctx, prefix):
        """
        Build feature dict from a team context dict, applying the given key prefix
        ('team_' or 'opp_').  Mirrors the keys written by prepare_features().
        """
        d = ctx or TrainingDataCollector._DEFAULT_TEAM_CTX
        if prefix == 'team_':
            return {
                'team_pace':               d['pace'],
                'team_off_rating':         d['offensive_rating'],
                'team_def_rating':         d['defensive_rating'],
                'team_form':               d['recent_form'],
                'rest_days':               d['rest_days'],
                'team_injuries':           d['injury_impact'],
                'team_injury_impact':      d['injury_impact'],
                'team_key_players_out':    d['key_players_out'],
                'team_total_players_out':  d['total_players_out'],
            }
        else:  # opp_
            return {
                'opp_pace':              d['pace'],
                'opp_def_rating':        d['defensive_rating'],
                'opp_form':              d['recent_form'],
                'opp_injuries':          d['injury_impact'],
                'opp_injury_impact':     d['injury_impact'],
                'opp_key_players_out':   d['key_players_out'],
                'opp_total_players_out': d['total_players_out'],
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_player_samples(self, player_id, prop_type, seasons=None,
                               dvp_map=None, dvp_pos_avgs=None, defenders_map=None,
                               player_position=None):
        """
        Rolling-window training samples for one player/prop.
        Returns list of {features, result, line, prop_type} dicts.

        Features built here exactly mirror what prepare_features() + analyze_prop_bet()
        produce at inference time — including efficiency stats, location splits,
        trend slope, and b2b flag — so the model trains on the full ~45-feature set.
        """
        col = self.PROP_COL_MAP.get(prop_type)
        if not col:
            return []

        if seasons is None:
            seasons = self._get_seasons()

        all_games = []
        for season in seasons:
            df = self._fetch_game_log(player_id, season)
            if not df.empty:
                all_games.append(df)

        if not all_games:
            return []

        games_df = pd.concat(all_games, ignore_index=True)
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        games_df = games_df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)

        if len(games_df) < self.MIN_TOTAL_GAMES:
            return []

        # --- coerce all numeric columns we'll use ---
        for _c in [col, 'FG_PCT', 'FT_PCT', 'MIN', 'FGA', 'FTA', 'TOV', 'FG3_PCT', 'FG3A', 'OREB', 'DREB', 'PLUS_MINUS', 'PF', 'WL']:
            if _c in games_df.columns:
                if _c == 'WL':
                    continue  # handled separately
                games_df[_c] = pd.to_numeric(games_df[_c], errors='coerce').fillna(0)

        stat_values = games_df[col].values.astype(float)

        # Pre-build efficiency arrays (parallel to stat_values)
        fg_pct_vals  = games_df['FG_PCT'].values.astype(float) if 'FG_PCT' in games_df.columns else np.full(len(games_df), 0.45)
        ft_pct_vals  = games_df['FT_PCT'].values.astype(float) if 'FT_PCT' in games_df.columns else np.full(len(games_df), 0.75)
        min_vals     = games_df['MIN'].values.astype(float)    if 'MIN'    in games_df.columns else np.full(len(games_df), 24.0)
        usage_vals   = (
            games_df['FGA'].values.astype(float) +
            0.44 * games_df['FTA'].values.astype(float) +
            games_df['TOV'].values.astype(float)
        ) if all(c in games_df.columns for c in ('FGA', 'FTA', 'TOV')) else np.full(len(games_df), 18.0)

        # --- additional arrays for extended features ---
        fg3_pct_vals = games_df['FG3_PCT'].values.astype(float) if 'FG3_PCT' in games_df.columns else np.full(len(games_df), 0.33)
        fga_vals     = games_df['FGA'].values.astype(float)     if 'FGA'    in games_df.columns else np.full(len(games_df), 15.0)
        fg3a_vals    = games_df['FG3A'].values.astype(float)    if 'FG3A'   in games_df.columns else np.full(len(games_df), 5.0)
        fta_vals     = games_df['FTA'].values.astype(float)     if 'FTA'    in games_df.columns else np.full(len(games_df), 4.0)
        oreb_vals    = games_df['OREB'].values.astype(float)    if 'OREB'   in games_df.columns else np.full(len(games_df), 1.0)
        dreb_vals    = games_df['DREB'].values.astype(float)    if 'DREB'   in games_df.columns else np.full(len(games_df), 3.0)
        plus_minus_v = games_df['PLUS_MINUS'].values.astype(float) if 'PLUS_MINUS' in games_df.columns else np.zeros(len(games_df))
        pf_vals      = games_df['PF'].values.astype(float)      if 'PF'     in games_df.columns else np.full(len(games_df), 2.0)
        wl_vals      = np.array([1.0 if str(w).upper() == 'W' else 0.0 for w in games_df.get('WL', pd.Series([''] * len(games_df)))])
        pts_vals     = games_df['PTS'].values.astype(float) if 'PTS' in games_df.columns else stat_values

        # Rest-days array: actual gap between consecutive games
        game_dates = games_df['GAME_DATE'].values  # numpy datetime64
        rest_days_arr = np.full(len(games_df), 2.0)
        for k in range(1, len(games_df)):
            gap = (game_dates[k] - game_dates[k - 1]) / np.timedelta64(1, 'D')
            rest_days_arr[k] = max(0.0, float(gap) - 1.0)   # days of rest = gap - 1

        # --- parse MATCHUP column ---
        team_abbrevs  = []
        opp_abbrevs   = []
        is_home_flags = []
        for matchup in games_df.get('MATCHUP', pd.Series(dtype=str)):
            team, opp, is_home = self._parse_matchup(matchup)
            team_abbrevs.append(team)
            opp_abbrevs.append(opp)
            is_home_flags.append(is_home)

        is_home_arr = np.array(is_home_flags, dtype=float)

        # --- pre-fetch team context for every unique abbreviation ---
        unique_abbrevs = {a for a in team_abbrevs + opp_abbrevs if a}
        for abbrev in unique_abbrevs:
            self._fetch_team_context(abbrev)

        samples = []
        for i in range(self.MIN_PRIOR_GAMES, len(games_df)):
            prior  = stat_values[:i]
            last5  = prior[-5:]
            last10 = prior[-10:]

            recent_avg   = float(np.mean(last5))
            season_avg   = float(np.mean(prior))
            stddev       = float(np.std(last10))
            max_recent   = float(np.max(last5))
            min_recent   = float(np.min(last5))
            games_played = i

            line     = float(np.mean(last10))
            hit_rate = 0.5    # neutral — don't leak label into training features
            edge     = ((recent_avg - line) / line) if line > 0 else 0.0
            result   = float(stat_values[i])

            # --- home / away location features ---
            is_home    = is_home_flags[i]
            home_idxs  = [j for j in range(i) if is_home_arr[j] == 1]
            away_idxs  = [j for j in range(i) if is_home_arr[j] == 0]
            home_vals  = stat_values[home_idxs]
            away_vals  = stat_values[away_idxs]

            home_avg_f   = float(np.mean(home_vals)) if len(home_vals) > 0 else recent_avg
            away_avg_f   = float(np.mean(away_vals)) if len(away_vals) > 0 else recent_avg
            home_games_n = len(home_idxs)
            away_games_n = len(away_idxs)

            if is_home == 1 and len(home_vals) > 0:
                loc_src = home_vals[-5:] if len(home_vals) >= 5 else home_vals
                location_avg = float(np.mean(loc_src))
            elif is_home == 0 and len(away_vals) > 0:
                loc_src = away_vals[-5:] if len(away_vals) >= 5 else away_vals
                location_avg = float(np.mean(loc_src))
            else:
                location_avg = recent_avg

            # --- efficiency features (from game log, no extra API calls) ---
            prior_fg  = fg_pct_vals[:i]
            prior_ft  = ft_pct_vals[:i]
            prior_min = min_vals[:i]
            prior_usg = usage_vals[:i]

            avg_minutes    = float(np.mean(prior_min))
            recent_minutes = float(np.mean(prior_min[-5:]))
            fg_pct         = float(np.mean(prior_fg))
            recent_fg_pct  = float(np.mean(prior_fg[-5:]))
            ft_pct         = float(np.mean(prior_ft))
            usage_rate     = float(np.mean(prior_usg))

            # --- trend slope: polyfit on the last 5 stat values ---
            if len(last5) >= 3:
                _z = np.polyfit(range(len(last5)), last5, 1)
                trend_slope = float(_z[0])
            else:
                trend_slope = 0.0

            # --- rest days & b2b (exact from game dates) ---
            rest_days_val = float(rest_days_arr[i])
            b2b_flag      = int(rest_days_val <= 1)

            features = {
                'recent_avg':    recent_avg,
                'season_avg':    season_avg,
                'stddev':        stddev,
                'max_recent':    max_recent,
                'min_recent':    min_recent,
                'games_played':  games_played,
                'hit_rate':      hit_rate,
                'edge':          edge,
                'is_home':       float(is_home),
                'location_avg':  location_avg,
                # location splits
                'home_avg':      home_avg_f,
                'away_avg':      away_avg_f,
                'home_games':    home_games_n,
                'away_games':    away_games_n,
                # efficiency
                'avg_minutes':    avg_minutes,
                'recent_minutes': recent_minutes,
                'fg_pct':         fg_pct,
                'recent_fg_pct':  recent_fg_pct,
                'ft_pct':         ft_pct,
                'usage_rate':     usage_rate,
                # momentum / schedule
                'trend_slope': trend_slope,
                'b2b_flag':    b2b_flag,
            }

            # ---- extended game-log features ----
            prior_fg3p  = fg3_pct_vals[:i]
            prior_fga   = fga_vals[:i]
            prior_fg3a  = fg3a_vals[:i]
            prior_fta   = fta_vals[:i]
            prior_oreb  = oreb_vals[:i]
            prior_dreb  = dreb_vals[:i]
            prior_pm    = plus_minus_v[:i]
            prior_pf    = pf_vals[:i]
            prior_wl    = wl_vals[:i]
            prior_pts   = pts_vals[:i]

            # per-game averages
            fg3_pct_recent  = float(np.mean(prior_fg3p[-5:])) if len(prior_fg3p) >= 5 else float(np.mean(prior_fg3p))
            fga_per_game    = float(np.mean(prior_fga))
            fg3a_per_game   = float(np.mean(prior_fg3a))
            fta_per_game    = float(np.mean(prior_fta))
            oreb_per_game   = float(np.mean(prior_oreb))
            dreb_per_game   = float(np.mean(prior_dreb))
            plus_minus_avg  = float(np.mean(prior_pm))
            fouls_per_game  = float(np.mean(prior_pf))
            win_rate_last10 = float(np.mean(prior_wl[-10:])) if len(prior_wl) >= 10 else float(np.mean(prior_wl)) if len(prior_wl) > 0 else 0.5

            # efficiency derived metrics
            _mean_fga  = float(np.mean(prior_fga[-5:])) if len(prior_fga) >= 5 else float(np.mean(prior_fga))
            _mean_pts5 = float(np.mean(prior[-5:]))
            _seas_pts  = float(np.mean(prior))
            _seas_fga  = float(np.mean(prior_fga))
            points_per_shot  = float(_mean_pts5 / max(_mean_fga, 1.0))
            ast_vals_p       = games_df['AST'].values.astype(float)[:i] if 'AST' in games_df.columns else np.zeros(i)
            tov_vals_p       = games_df['TOV'].values.astype(float)[:i] if 'TOV' in games_df.columns else np.ones(i)
            ast_to_tov_ratio = float(np.mean(ast_vals_p[-5:])) / max(float(np.mean(tov_vals_p[-5:])), 0.5)
            reb_rate_per_36  = float(np.mean((prior_oreb + prior_dreb))) / max(float(np.mean(min_vals[:i])), 1.0) * 36.0

            # trend/variance metrics
            _seas_pts_per_shot       = float(_seas_pts / max(_seas_fga, 1.0))
            scoring_efficiency_trend = float(points_per_shot - _seas_pts_per_shot)
            _seas_usg  = float(np.mean(usage_vals[:i]))
            _rec_usg   = float(np.mean(usage_vals[max(0, i - 5):i]))
            usage_trend       = float(_rec_usg - _seas_usg)
            minutes_volatility = float(np.std(min_vals[:i])) if i >= 3 else 0.0

            # game script features (need PLUS_MINUS to infer margin)
            _margins = np.abs(prior_pm)
            blowout_game_pct = float(np.mean(_margins > 15)) if len(_margins) > 0 else 0.0
            close_game_pct   = float(np.mean(_margins < 5))  if len(_margins) > 0 else 0.0

            # consistency and ceiling
            _cv = float(np.std(prior)) / max(float(np.mean(prior)), 0.1)
            consistency_score      = max(0.0, 1.0 - _cv)
            ceiling_game_frequency = float(np.mean(prior > _seas_pts * 1.5)) if _seas_pts > 0 else 0.0
            _rec3_std  = float(np.std(prior[-3:])) if len(prior) >= 3 else 0.0
            _seas_std  = float(np.std(prior))      if len(prior) >= 3 else 1.0
            recent_variance_spike = float(_rec3_std / max(_seas_std, 0.1) - 1.0)

            # trend slopes (polyfit)
            def _ext_slope(arr):
                if len(arr) < 2:
                    return 0.0
                try:
                    return float(np.polyfit(range(len(arr)), arr, 1)[0])
                except Exception:
                    return 0.0

            _last3_ext  = prior[-3:]
            _last10_ext = prior[-10:]
            last_3_games_trend  = _ext_slope(_last3_ext)
            last_5_games_trend  = _ext_slope(last5)   # last5 already computed earlier in loop
            last_10_games_trend = _ext_slope(_last10_ext)

            # vs season average
            _seas_avg_val = float(np.mean(prior))
            games_above_season_avg_last5 = int(np.sum(np.array(last5) > _seas_avg_val))

            # schedule density
            _today_dt       = game_dates[i]
            days_since_last_game = float((game_dates[i] - game_dates[i - 1]) / np.timedelta64(1, 'D')) if i > 0 else 2.0
            _seven_days_ago = _today_dt - np.timedelta64(7, 'D')
            games_in_last_7_days = int(np.sum(game_dates[:i] >= _seven_days_ago))

            features.update({
                'fg3_pct_recent':               fg3_pct_recent,
                'fga_per_game':                 fga_per_game,
                'fg3a_per_game':                fg3a_per_game,
                'fta_per_game':                 fta_per_game,
                'oreb_per_game':                oreb_per_game,
                'dreb_per_game':                dreb_per_game,
                'plus_minus_avg':               plus_minus_avg,
                'fouls_per_game':               fouls_per_game,
                'win_rate_last10':              win_rate_last10,
                'points_per_shot':              points_per_shot,
                'ast_to_tov_ratio':             ast_to_tov_ratio,
                'reb_rate_per_36':              reb_rate_per_36,
                'scoring_efficiency_trend':     scoring_efficiency_trend,
                'usage_trend':                  usage_trend,
                'minutes_volatility':           minutes_volatility,
                'blowout_game_pct':             blowout_game_pct,
                'close_game_pct':               close_game_pct,
                'consistency_score':            consistency_score,
                'ceiling_game_frequency':       ceiling_game_frequency,
                'recent_variance_spike':        recent_variance_spike,
                'last_3_games_trend':           last_3_games_trend,
                'last_5_games_trend':           last_5_games_trend,
                'last_10_games_trend':          last_10_games_trend,
                'games_above_season_avg_last5': float(games_above_season_avg_last5),
                'days_since_last_game':         days_since_last_game,
                'games_in_last_7_days':         float(games_in_last_7_days),
            })

            # --- team context (current-season proxy) ---
            team_ctx = self._team_context_cache.get(team_abbrevs[i]) if team_abbrevs[i] else None
            features.update(self._team_features(team_ctx, 'team_'))
            # override rest_days with actual game-log value (more accurate than season avg)
            features['rest_days'] = rest_days_val

            # --- opponent context (current-season proxy) ---
            opp_ctx = self._team_context_cache.get(opp_abbrevs[i]) if opp_abbrevs[i] else None
            features.update(self._team_features(opp_ctx, 'opp_'))

            # --- player matchup history vs this specific opponent ---
            opp = opp_abbrevs[i]
            prior_vs_opp = stat_values[[j for j in range(i) if opp_abbrevs[j] == opp]]
            if len(prior_vs_opp) > 0:
                vs_team_avg          = float(np.mean(prior_vs_opp))
                matchup_games        = len(prior_vs_opp)
                matchup_success_rate = float(np.mean([1.0 if v > line else 0.0 for v in prior_vs_opp]))
            else:
                vs_team_avg          = recent_avg
                matchup_games        = 0
                matchup_success_rate = 0.5

            features.update({
                'vs_team_avg':          vs_team_avg,
                'matchup_games':        matchup_games,
                'matchup_success_rate': matchup_success_rate,
                'pos_pts_allowed':  0.0,
                'pos_def_rating':   110.0,
                'effective_fg_pct': 0.47,
                'injury_risk': 0.0,
            })

            # ---- DVP (Defence vs Position) deltas + primary defender ----
            # Uses current-season precomputed data passed in from retrain().
            # Zero defaults = league-average defence (consistent with inference fallback).
            if dvp_map is not None and player_position and opp_abbrevs[i]:
                _dvp_pos, _pos_group = _dvp_position_keys(player_position)
                _opp_id = self._get_team_id(opp_abbrevs[i])
                if _opp_id:
                    _dvp     = dvp_map.get((int(_opp_id), _dvp_pos), {})
                    _dvp_avg = (dvp_pos_avgs or {}).get(_dvp_pos, {})
                    features['dvp_gp'] = int(_dvp.get('gp', 0))
                    for _k in ('pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov'):
                        features[f'dvp_{_k}_delta'] = (
                            float(_dvp.get(_k, 0.0)) - float(_dvp_avg.get(_k, 0.0))
                        )
                    _defs = (defenders_map or {}).get((int(_opp_id), _pos_group), [])
                    features['primary_defender_score01'] = float(
                        (_defs[0] if _defs else {}).get('score01', 0.0) or 0.0
                    )
                else:
                    _set_dvp_defaults(features)
            else:
                _set_dvp_defaults(features)

            # ---- Calendar position features (derived from game date) ----
            try:
                from datetime import date as _date
                _game_dt = game_dates[i]  # numpy datetime64
                _game_py_date = pd.Timestamp(_game_dt).date()
                _year = _game_py_date.year if _game_py_date.month >= 10 else _game_py_date.year - 1
                _season_start = _date(_year, 10, 18)
                _season_end   = _date(_year + 1, 4, 15)
                _total_days   = max((_season_end - _season_start).days, 1)
                _elapsed      = max((_game_py_date - _season_start).days, 0)
                _phase        = min(1.0, _elapsed / _total_days)
                features['days_into_season']       = float(_elapsed)
                features['season_phase_numeric']   = float(_phase)
                features['games_remaining_approx'] = float(max(0.0, 82.0 * (1.0 - _phase)))
            except Exception:
                features.setdefault('days_into_season', 90.0)
                features.setdefault('season_phase_numeric', 0.5)
                features.setdefault('games_remaining_approx', 40.0)

            # ---- Return-from-injury trajectory (from game log gaps) ----
            try:
                _games_since_return = 0
                _missed_before = 0.0
                _found_gap = False
                for _ri in range(i - 1, max(0, i - 20), -1):
                    _gap_days = float((game_dates[_ri + 1] - game_dates[_ri]) / np.timedelta64(1, 'D'))
                    if _gap_days > 5 and not _found_gap:
                        _missed_before = max(0.0, (_gap_days - 2) / 2.0)
                        _found_gap = True
                        break
                    elif not _found_gap:
                        _games_since_return += 1
                features['games_since_return']         = float(_games_since_return)
                features['missed_games_before_return'] = float(_missed_before)
            except Exception:
                features.setdefault('games_since_return', 0.0)
                features.setdefault('missed_games_before_return', 0.0)

            # ---- Features that need precomputed data (zero-fill if unavailable) ----
            for _k in ('ref_foul_rate', 'ref_home_bias', 'ref_pace_tendency',
                       'dvp_pts_delta_last5', 'dvp_pts_delta_last10',
                       'dvp_reb_delta_last5', 'dvp_ast_delta_last5', 'dvp_fg3m_delta_last5',
                       'opp_foul_rate_per48', 'opp_foul_rate_last5',
                       'primary_defender_active', 'opp_lineup_changes_last5'):
                features.setdefault(_k, 0.0)
            features.setdefault('ref_home_bias', 0.5)
            features.setdefault('opp_foul_rate_per48', 20.0)
            features.setdefault('opp_foul_rate_last5', 20.0)
            features.setdefault('primary_defender_active', 1.0)
            features.setdefault('implied_game_total', 220.0)

            # ---- Team-style + opponent baseline defaults (match models.py prepare_features defaults)
            # These are current-season precomputed values unavailable for historical training,
            # so every training sample gets the same league-average defaults. GB will learn near-zero
            # weight for them; at inference the real values serve as a contextual adjustment.
            _team_opp_defaults = {
                'team_pts_fb': 12.0, 'opp_pts_fb_allowed': 12.0,
                'team_pts_off_tov': 16.0, 'opp_pts_off_tov_allowed': 16.0,
                'opp_pts_paint': 44.0, 'opp_fga': 86.0, 'opp_fg_pct': 0.47,
                'opp_fg3a': 35.0, 'opp_fg3_pct': 0.36, 'opp_tov': 14.0,
                'opp_stl': 7.0, 'opp_blk': 5.0, 'opp_off_rating': 110.0,
                'opp_def_rating_last5': 110.0, 'opp_blocks_per_game_last5': 5.0,
                'opp_steals_per_game_last5': 7.0,
                'lg_pts_fb': 12.0, 'lg_opp_pts_fb': 12.0,
                'lg_pts_off_tov': 16.0, 'lg_opp_pts_off_tov': 16.0,
                'lg_fga': 86.0, 'lg_fg_pct': 0.47, 'lg_fg3a': 35.0,
                'lg_tov': 14.0, 'lg_stl': 7.0,
            }
            for _k, _dv in _team_opp_defaults.items():
                features.setdefault(_k, _dv)

            # Zero-fill new player-specific precomputed features (not available historically)
            _new_defaults = {
                'usg_pct_official': 0.18, 'ts_pct_official': 0.55, 'efg_pct_official': 0.50,
                'ast_pct_official': 0.15, 'oreb_pct_official': 0.05, 'dreb_pct_official': 0.15,
                'reb_pct_official': 0.10, 'pie': 0.10, 'player_off_rating': 110.0,
                'player_def_rating': 110.0, 'player_pace': 100.0, 'net_rating_player': 0.0,
                'player_age': 26.0, 'player_height_inches': 78.0, 'player_weight': 220.0,
                'years_experience': 5.0, 'clutch_pts_per_game': 0.0, 'clutch_fg_pct': 0.45,
                'clutch_fg3_pct': 0.33, 'clutch_fta_per_game': 0.0, 'clutch_plus_minus': 0.0,
                'clutch_min_per_game': 0.0, 'clutch_games': 0, 'contested_shots_per_game': 3.0,
                'deflections_per_game': 1.0, 'charges_drawn_per_game': 0.1,
                'screen_assists_per_game': 0.5, 'open_shot_fg_pct': 0.50,
                'open_shot_frequency': 0.30, 'tight_shot_fg_pct': 0.38, 'tight_shot_frequency': 0.15,
                'catch_shoot_fg_pct': 0.40, 'catch_shoot_frequency': 0.25,
                'pullup_fg_pct': 0.40, 'pullup_frequency': 0.20,
                'iso_poss_pct': 0.0, 'iso_ppp': 0.9, 'pnr_bh_poss_pct': 0.0, 'pnr_bh_ppp': 0.9,
                'pnr_roll_poss_pct': 0.0, 'pnr_roll_ppp': 0.9, 'spotup_poss_pct': 0.0,
                'spotup_ppp': 1.0, 'transition_poss_pct': 0.0, 'transition_ppp': 1.1,
                'postup_poss_pct': 0.0, 'cut_poss_pct': 0.0, 'on_court_net_rating': 0.0,
                'off_court_net_rating': 0.0, 'on_off_differential': 0.0,
                'rim_fga_pct': 0.25, 'rim_fg_pct': 0.62, 'paint_fga_pct': 0.30,
                'paint_fg_pct': 0.55, 'midrange_fga_pct': 0.20, 'midrange_fg_pct': 0.42,
                'corner3_fga_pct': 0.10, 'corner3_fg_pct': 0.38, 'above_break3_fga_pct': 0.25,
                'above_break3_fg_pct': 0.35, 'q1_avg': 0.0, 'q2_avg': 0.0, 'q3_avg': 0.0,
                'q4_avg': 0.0, 'q4_min_per_game': 0.0, 'opp_rim_fg_pct_allowed': 0.62,
                'opp_paint_fg_pct_allowed': 0.55, 'opp_midrange_fg_pct_allowed': 0.42,
                'opp_corner3_fg_pct_allowed': 0.38, 'opp_above_break3_fg_pct_allowed': 0.35,
                'rim_shot_quality_matchup': 0.0, 'midrange_shot_quality_matchup': 0.0,
                'corner3_shot_quality_matchup': 0.0, 'above_break3_shot_quality_matchup': 0.0,
                'opp_pnr_ppp_allowed': 0.9, 'opp_iso_ppp_allowed': 0.9,
                'opp_spotup_ppp_allowed': 1.0, 'opp_transition_ppp_allowed': 1.1,
                'opp_postup_ppp_allowed': 0.9, 'pnr_matchup_advantage': 0.0,
                'iso_matchup_advantage': 0.0, 'spotup_matchup_advantage': 0.0,
                'transition_matchup_advantage': 0.0,
                # Group A: Derived Efficiency Features
                'touch_efficiency': 0.5,
                'pace_adjusted_variance': 2.0,
                'clutch_efficiency_delta': 0.0,
                'rim_volume_quality': 0.25,
                'play_specialization_score': 0.2,
                'best_play_type_ppp': 0.9,
                'ts_vs_zone_expected': 0.0,
                'usage_stability': 0.5,
                'recent_form_confidence': 0.75,
                'def_toughness_composite': 0.55,
                'load_efficiency_ratio': 1.0,
                'tracking_dist_per_touch': 0.05,
                # Group B: Historical vs Opponent
                'historical_avg_vs_opp': 0.0,
                'historical_fg_pct_vs_opp': 0.45,
                'historical_ts_pct_vs_opp': 0.55,
                'historical_games_vs_opp': 0,
                'historical_min_vs_opp': 30.0,
                # Group C: Team Rest Splits
                'opp_b2b_def_rating': 112.0,
                'opp_b2b_pace': 100.0,
                'opp_b2b_pts_allowed': 115.0,
                'opp_rested_def_rating': 110.0,
                'opp_rested_pace': 100.0,
                'opp_rest_def_rating_delta': 2.0,
                # Group D: YoY Stats
                'yoy_pts_change': 0.0,
                'yoy_ts_change': 0.0,
                'yoy_usage_change': 0.0,
                'seasons_in_league': 5,
            }
            for _k, _dv in _new_defaults.items():
                features.setdefault(_k, _dv)

            # ---- Zero-fill for new feature groups (Groups 1-6) ----
            _group_defaults = {
                # Group 1: Player Tracking Stats
                'tracking_avg_speed': 4.5, 'tracking_avg_speed_off': 4.8,
                'tracking_avg_speed_def': 4.2, 'tracking_dist_miles': 2.5,
                'tracking_dist_miles_off': 1.3, 'tracking_dist_miles_def': 1.2,
                'tracking_touches_pg': 50.0, 'tracking_time_of_poss_pg': 2.5,
                'tracking_avg_drib_per_touch': 1.5, 'tracking_passes_made_pg': 30.0,
                'tracking_potential_ast_pg': 5.0, 'tracking_secondary_ast_pg': 1.0,
                # Group 2: Team Standings
                'team_win_pct': 0.5, 'team_conf_rank': 8.0, 'team_games_back': 5.0,
                'team_current_streak': 0.0, 'team_l10_wins': 5.0, 'team_home_win_pct': 0.5,
                'opp_win_pct_standings': 0.5, 'opp_conf_rank': 8.0, 'opp_games_back': 5.0,
                'opp_current_streak_standings': 0.0, 'opp_l10_wins': 5.0,
                'opp_road_win_pct': 0.5, 'win_pct_diff': 0.0, 'is_playoff_race_game': 0.0,
                # Group 3: Scoring Breakdown
                'pct_pts_3pt': 0.25, 'pct_pts_paint': 0.30, 'pct_pts_ft': 0.15,
                'pct_pts_midrange': 0.20, 'pct_uast_fgm': 0.40, 'pct_ast_fgm': 0.60,
                # Group 4: Win/Loss Splits (computed inline)
                'stat_in_wins': 0.0, 'stat_in_losses': 0.0,
                'win_loss_performance_split': 0.0, 'over_rate_in_wins': 0.5,
                # Group 5: Opponent Rest
                'opp_days_rest': 2.0, 'opp_b2b': 0.0, 'rest_advantage': 0.0,
                # Group 6: Derived Features
                'ast_pct_to_usg_ratio': 0.83, 'defensive_burden': 58.0,
                'shot_profile_fit': 0.09, 'pace_adjusted_projection': 0.0,
                'form_momentum': 0.0,
                # Context features (player_context / team_context / opponent_context)
                # Conditional in prepare_features() — must exist in training samples
                # so the scaler/model learns their column shape.
                'vs_team_avg': 0.0, 'matchup_games': 0, 'matchup_success_rate': 0.0,
                'pos_pts_allowed': 0.0, 'pos_def_rating': 0.0,
                'effective_fg_pct': 0.47, 'injury_risk': 0.0,
                'team_off_rating': 110.0, 'team_def_rating': 110.0,
                'team_form': 0.5, 'team_injuries': 0.0,
                'opp_def_rating': 110.0, 'opp_form': 0.5, 'opp_injuries': 0.0,
                'team_injury_impact': 0.0, 'team_key_players_out': 0,
                'team_total_players_out': 0,
                'opp_injury_impact': 0.0, 'opp_key_players_out': 0,
                'opp_total_players_out': 0,
            }
            for _k, _dv in _group_defaults.items():
                features.setdefault(_k, _dv)

            # Compute Group 4 Win/Loss splits from wl_vals
            try:
                _seas_avg_wl = float(np.mean(prior)) if len(prior) > 0 else 0.0
                _win_idxs  = [j for j in range(i) if wl_vals[j] == 1.0]
                _loss_idxs = [j for j in range(i) if wl_vals[j] == 0.0]
                _win_stat_vals  = stat_values[_win_idxs]
                _loss_stat_vals = stat_values[_loss_idxs]
                _stat_in_wins   = float(np.mean(_win_stat_vals))  if len(_win_stat_vals)  > 0 else _seas_avg_wl
                _stat_in_losses = float(np.mean(_loss_stat_vals)) if len(_loss_stat_vals) > 0 else _seas_avg_wl
                features['stat_in_wins']   = _stat_in_wins
                features['stat_in_losses'] = _stat_in_losses
                features['win_loss_performance_split'] = _stat_in_wins - _stat_in_losses
                # over_rate_in_wins: fraction of wins where stat > line (use rolling line)
                _line_val = float(np.mean(prior[-10:])) if len(prior) >= 10 else _seas_avg_wl
                features['over_rate_in_wins'] = float(np.mean([1.0 if v > _line_val else 0.0 for v in _win_stat_vals])) if len(_win_stat_vals) > 0 else 0.5
            except Exception:
                pass  # keep the zero-fill defaults

            # Compute Group 6 Derived Features from already-available data
            try:
                _ast_pct_v2   = float(features.get('ast_pct_official', 0.15))
                _usg_pct_v2   = float(features.get('usg_pct_official', 0.18))
                features['ast_pct_to_usg_ratio'] = _ast_pct_v2 / max(_usg_pct_v2, 0.01)

                _opp_def_v2   = float(features.get('opp_def_rating_last5', 110.0))
                _opp_fgp_v2   = float(features.get('opp_fg_pct', 0.47))
                features['defensive_burden'] = _opp_def_v2 * (1.0 - _opp_fgp_v2)

                _pct3_v2      = float(features.get('pct_pts_3pt', 0.25))
                _ab3_v2       = float(features.get('opp_above_break3_fg_pct_allowed', 0.35))
                features['shot_profile_fit'] = _pct3_v2 * _ab3_v2

                _igt_v2       = float(features.get('implied_game_total', 220.0))
                _rec_avg_v2   = float(np.mean(prior[-5:])) if len(prior) >= 5 else float(np.mean(prior))
                features['pace_adjusted_projection'] = _rec_avg_v2 * (_igt_v2 / 220.0)

                _l3t_v2       = float(features.get('last_3_games_trend', 0.0))
                _l5t_v2       = float(features.get('last_5_games_trend', 0.0))
                features['form_momentum'] = (_l3t_v2 * 3.0 + _l5t_v2 * 2.0) / 5.0
            except Exception:
                pass  # keep defaults

            samples.append({
                'features':  features,
                'result':    result,
                'line':      line,
                'prop_type': prop_type,
            })

        return samples

    def collect_bulk(self, player_ids, prop_types=None, seasons=None,
                     dvp_map=None, dvp_pos_avgs=None, defenders_map=None):
        """Collects training samples across multiple players and prop types."""
        if prop_types is None:
            prop_types = list(self.PROP_COL_MAP.keys())

        all_samples = []
        total = len(player_ids) * len(prop_types)
        done  = 0

        for player_id in player_ids:
            # Resolve player position once per player (used for DVP lookup)
            player_pos = None
            if dvp_map is not None:
                try:
                    from nba_api.stats.endpoints import commonplayerinfo
                    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
                    time.sleep(0.4)
                    if not info.empty:
                        player_pos = str(info.iloc[0].get('POSITION', '') or '')
                except Exception:
                    player_pos = None

            for prop_type in prop_types:
                done += 1
                try:
                    samples = self.collect_player_samples(
                        player_id, prop_type, seasons,
                        dvp_map=dvp_map,
                        dvp_pos_avgs=dvp_pos_avgs,
                        defenders_map=defenders_map,
                        player_position=player_pos,
                    )
                    all_samples.extend(samples)
                    print(f"[{done}/{total}] player {player_id} / {prop_type}: "
                          f"{len(samples)} samples  (total so far: {len(all_samples)})")
                except Exception as e:
                    print(f"[{done}/{total}] Error for player {player_id}/{prop_type}: {e}")
                    continue

        return all_samples

    def get_active_player_ids(self, n=100):
        """IDs of the first n active players."""
        all_players = nba_players.get_players()
        active = [p for p in all_players if p['is_active']]
        return [p['id'] for p in active[:n]]
