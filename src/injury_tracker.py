import time
import json
from pathlib import Path

class InjuryTracker:
    def __init__(self):
        self.injury_cache = {}
        self.cache_timestamps = {}
        self.cache_timeout = 3600  # 1 hour

    def get_team_injuries(self, team_id):
        """Use the shared live report; source failure must not mean healthy.

        Impact uses cached prior-season per-game statistics and severity.
        Missing player statistics are explicitly marked unavailable.
        """
        from nba_api.stats.static import teams
        from .live_injuries import summarise_team
        from .current_rosters import get_rosters, name_key
        team = teams.find_team_name_by_id(int(team_id))
        if team is None:
            raise ValueError('Unknown NBA team')
        report = summarise_team(team['full_name'])
        roster = get_rosters()
        by_name = {name_key(p['full_name']): p for p in (roster or {}).get('players', [])
                   if p['team_id'] == int(team_id)}
        # Historical per-game averages estimate player importance; never
        # divide these by games played a second time.
        from .current_rosters import season_now
        previous_year = int(season_now()[:4]) - 1
        baseline_season = f'{previous_year}-{str(previous_year + 1)[2:]}'
        baseline_path = Path(__file__).resolve().parents[1] / 'logs/precompute_cache' / f'player_averages_{baseline_season}.json'
        try:
            stats_by_id = {int(r['PLAYER_ID']): r for r in json.loads(baseline_path.read_text())}
        except (OSError, ValueError, KeyError):
            stats_by_id = {}
        injuries = []
        for entry in report['players']:
            player = by_name.get(name_key(entry['player']), {})
            baseline = stats_by_id.get(player.get('nba_id'))
            impact = self._calculate_player_impact({**baseline, 'GP': 1}) if baseline else None
            injuries.append({**entry, 'player_name': entry['player'],
                             'player_id': player.get('nba_id'),
                             'expected_return': entry['return'],
                             'impact_score': impact,
                             'impact_available': impact is not None})
        return {'active_injuries': injuries,
                'total_impact': sum((r['impact_score'] or 0) * r['severity'] for r in injuries),
                'impact_available': all(r['impact_available'] for r in injuries),
                'impact_baseline_season': baseline_season,
                'key_players_out': sum(r['severity'] >= .85 and (r['impact_score'] or 0) > .15 for r in injuries),
                'total_players_out': report['out_count'],
                'questionable_players': report['questionable_count'],
                'status': 'available', 'fetched_at': time.time()}

    def _calculate_player_impact(self, stats):
        """0–1 score based on mins/pts/reb/ast per game."""
        try:
            mpg = float(stats['MIN']) / float(stats['GP']) if float(stats['GP']) > 0 else 0
            ppg = float(stats['PTS']) / float(stats['GP']) if float(stats['GP']) > 0 else 0
            rpg = float(stats['REB']) / float(stats['GP']) if float(stats['GP']) > 0 else 0
            apg = float(stats['AST']) / float(stats['GP']) if float(stats['GP']) > 0 else 0

            impact_score = (
                0.4 * (mpg / 48.0) +
                0.3 * (ppg / 30.0) +
                0.15 * (rpg / 10.0) +
                0.15 * (apg / 10.0)
            )

            return min(max(impact_score, 0), 1)

        except Exception as e:
            print(f"couldn't calculate impact: {e}")
            return 0.1