import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, TeamGameLog
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
import time
from datetime import datetime


class TrainingDataCollector:
    """
    Builds rolling-window training samples from NBA game logs.
    For game i, features come from games 0..i-1 so there's no leakage.
    The simulated line is the rolling 10-game avg, matching what the model sees at inference.

    Team/opponent context (pace, defensive rating, rest days, etc.) is fetched once per
    team per session from the current season and injected into all training samples.
    This matches the inference pipeline in prepare_features(), which also uses the
    current season's team context.  The full 32-feature set is built here so the
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

    def collect_player_samples(self, player_id, prop_type, seasons=None):
        """
        Rolling-window training samples for one player/prop.
        Returns list of {features, result, line, prop_type} dicts.
        Each sample's feature dict has the same ~32 keys that prepare_features()
        builds at inference time.
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

        stat_values = pd.to_numeric(games_df[col], errors='coerce').fillna(0).values.astype(float)

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
            self._fetch_team_context(abbrev)   # populates cache

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

            # --- home / away features ---
            is_home = is_home_flags[i]
            prior_home = [stat_values[j] for j in range(i) if is_home_arr[j] == 1]
            prior_away = [stat_values[j] for j in range(i) if is_home_arr[j] == 0]

            if is_home == 1 and prior_home:
                loc_src = prior_home[-5:] if len(prior_home) >= 5 else prior_home
                location_avg = float(np.mean(loc_src))
            elif is_home == 0 and prior_away:
                loc_src = prior_away[-5:] if len(prior_away) >= 5 else prior_away
                location_avg = float(np.mean(loc_src))
            else:
                location_avg = recent_avg

            features = {
                'recent_avg':   recent_avg,
                'season_avg':   season_avg,
                'stddev':       stddev,
                'max_recent':   max_recent,
                'min_recent':   min_recent,
                'games_played': games_played,
                'hit_rate':     hit_rate,
                'edge':         edge,
                'is_home':      float(is_home),
                'location_avg': location_avg,
            }

            # --- team context features ---
            team_ctx = self._team_context_cache.get(team_abbrevs[i]) if team_abbrevs[i] else None
            features.update(self._team_features(team_ctx, 'team_'))

            # --- opponent context features ---
            opp_ctx = self._team_context_cache.get(opp_abbrevs[i]) if opp_abbrevs[i] else None
            features.update(self._team_features(opp_ctx, 'opp_'))

            # --- player matchup history vs this specific opponent ---
            opp = opp_abbrevs[i]
            prior_vs_opp = [stat_values[j] for j in range(i) if opp_abbrevs[j] == opp]
            if prior_vs_opp:
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
                # Position-level defensive stats can't be computed per historical game
                # without a per-game LeagueDashPtDefend call; use neutral defaults.
                'pos_pts_allowed': 0.0,
                'pos_def_rating':  110.0,
                # Injury risk is a real-time signal — not retroactively available.
                'injury_risk': 0.0,
            })

            samples.append({
                'features':  features,
                'result':    result,
                'line':      line,
                'prop_type': prop_type,
            })

        return samples

    def collect_bulk(self, player_ids, prop_types=None, seasons=None):
        """Collects training samples across multiple players and prop types."""
        if prop_types is None:
            prop_types = list(self.PROP_COL_MAP.keys())

        all_samples = []
        total = len(player_ids) * len(prop_types)
        done  = 0

        for player_id in player_ids:
            for prop_type in prop_types:
                done += 1
                try:
                    samples = self.collect_player_samples(player_id, prop_type, seasons)
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
