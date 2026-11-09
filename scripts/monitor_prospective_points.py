"""Grade each wakeup; conserve quote credits until the next game has eligible players."""
import subprocess
import sys
from pathlib import Path
import pandas as pd
from src.current_rosters import load_snapshot, refresh_rosters
from src.historical_inputs import HistoricalInputs, load_season_logs
from src.nba_officials import nba_json
from scripts.build_expanded_points import eligible


def main():
    command = [sys.executable, '-m', 'scripts.validate_prospective_points']
    subprocess.run(command, check=True)
    sidecar = [sys.executable, '-m', 'scripts.validate_points_v5']
    # A sidecar failure must never stop the original study's collection.
    subprocess.run(sidecar, check=False)
    roster = load_snapshot() or refresh_rosters()
    logs = load_season_logs(roster['season'], include_postseason=True)
    day = pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()
    games = nba_json('scoreboardv3', GameDate=str(day.date()), LeagueID='00')['scoreboard']['games']
    future = sorted([g for g in games if pd.Timestamp(g['gameTimeUTC']) > pd.Timestamp.now(tz='UTC')], key=lambda g: g['gameTimeUTC'])
    if not future:
        print('No remaining games today; grading only.')
        return
    game = future[0]
    teams = {int(game['homeTeam']['teamId']), int(game['awayTeam']['teamId'])}
    players = {int(p['nba_id']) for p in roster['players'] if p.get('nba_id') and int(p['team_id']) in teams}
    inputs = HistoricalInputs(roster['season'], {}, logs)
    original = set(map(int, inputs.catalog(50).PLAYER_ID))
    histories = logs[pd.to_datetime(logs.GAME_DATE) < day].sort_values('GAME_DATE').groupby('PLAYER_ID')
    qualified = [int(pid) for pid, history in histories if int(pid) in players and eligible(history, int(pid) in original)]
    if not qualified and not Path('logs/prospective_points_v5/protocol.json').exists():
        print('Next game has no training-eligible players yet; reserving quote credits. Initial warmup forecasts remain pending/graded separately.')
        return
    subprocess.run(command+['--collect'], check=True)
    subprocess.run(sidecar, check=False)


if __name__ == '__main__':
    main()
