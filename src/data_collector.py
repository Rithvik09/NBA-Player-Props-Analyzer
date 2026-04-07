import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players
import time
from datetime import datetime


class TrainingDataCollector:
    """
    Builds rolling-window training samples from NBA game logs.
    For game i, features come from games 0..i-1 so there's no leakage.
    The simulated line is the rolling 10-game avg, matching what the model sees at inference.
    """

    PROP_COL_MAP = {
        'points':        'PTS',
        'assists':       'AST',
        'rebounds':      'REB',
        'steals':        'STL',
        'blocks':        'BLK',
        'turnovers':     'TOV',
        'three_pointers': 'FG3M',
    }

    MIN_PRIOR_GAMES = 10   # need at least this many prior games to build a sample
    MIN_TOTAL_GAMES = 20   # skip players with very thin history

    def __init__(self):
        pass

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_player_samples(self, player_id, prop_type, seasons=None):
        """Rolling-window training samples for one player/prop. Returns list of {features, result, line} dicts."""
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
        samples = []

        for i in range(self.MIN_PRIOR_GAMES, len(games_df)):
            prior  = stat_values[:i]
            last5  = prior[-5:]
            last10 = prior[-10:]

            recent_avg = float(np.mean(last5))
            season_avg = float(np.mean(prior))
            stddev     = float(np.std(last10))
            max_recent = float(np.max(last5))
            min_recent = float(np.min(last5))
            games_played = i

            line     = float(np.mean(last10))  # rolling 10-game avg as the simulated line
            hit_rate = 0.5  # neutral — don't leak label info into training features
            edge     = ((recent_avg - line) / line) if line > 0 else 0.0

            result = float(stat_values[i])

            samples.append({
                'features': {
                    'recent_avg':   recent_avg,
                    'season_avg':   season_avg,
                    'stddev':       stddev,
                    'max_recent':   max_recent,
                    'min_recent':   min_recent,
                    'games_played': games_played,
                    'hit_rate':     hit_rate,
                    'edge':         edge,
                    'is_home':      0.5,   # unknown for historical data
                    'location_avg': recent_avg,  # best proxy we have without play-by-play
                },
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
