import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players
import time
from datetime import datetime


class TrainingDataCollector:
    """
    Builds supervised training samples from historical NBA game logs.

    For each game i in a player's history the features are computed from
    all games 0..i-1 (i.e. information that was available *before* game i),
    the line is the rolling-10-game average (simulated market line), and the
    result is the actual stat value recorded in game i.  This mirrors exactly
    what prepare_features/predict see at inference time.
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

    MIN_PRIOR_GAMES = 10   # need at least this many games before we can build a sample
    MIN_TOTAL_GAMES = 20   # skip players with fewer total games

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_seasons(self, num_seasons=3):
        """Return the N most recent NBA season strings, e.g. ['2024-25','2023-24',...]"""
        current_year = datetime.now().year
        current_month = datetime.now().month
        latest_year = current_year - 1 if 1 <= current_month <= 7 else current_year
        return [f"{latest_year - i}-{str(latest_year - i + 1)[2:]}" for i in range(num_seasons)]

    def _fetch_game_log(self, player_id, season):
        """Fetch a single season's game log; returns empty DataFrame on failure."""
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
        """
        Generate rolling-window training samples for one player / prop type.

        Returns a list of dicts:
            {'features': {...}, 'result': float, 'line': float}
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
        # Sort oldest → newest so index i means "game number i in career window"
        games_df = games_df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)

        if len(games_df) < self.MIN_TOTAL_GAMES:
            return []

        stat_values = games_df[col].fillna(0).values.astype(float)
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

            # Simulated line: rolling 10-game average (mirrors sportsbook logic)
            line     = float(np.mean(last10))
            hit_rate = 0.5  # neutral placeholder — avoids leaking label info into training
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
                    # Neutral defaults so feature set matches log-based training samples
                    'is_home':      0.5,   # unknown home/away for historical data
                    'location_avg': recent_avg,  # best proxy without play-by-play
                },
                'result': result,
                'line':   line,
            })

        return samples

    def collect_bulk(self, player_ids, prop_types=None, seasons=None):
        """
        Collect training samples for a list of player IDs and prop types.
        Returns the combined list of sample dicts.
        """
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
        """Return IDs of the first n active NBA players."""
        all_players = nba_players.get_players()
        active = [p for p in all_players if p['is_active']]
        return [p['id'] for p in active[:n]]
