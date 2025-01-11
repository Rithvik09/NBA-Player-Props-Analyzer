from nba_api.stats.endpoints import CommonTeamRoster, PlayerCareerStats
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

class InjuryTracker:
    def __init__(self):
        self.injury_cache = {}
        self.cache_timeout = 3600  # 1 hour cache timeout
        self.last_cache_update = 0
    
    def get_team_injuries(self, team_id):
        """Get current injuries for a team"""
        try:
            current_time = time.time()
            
            # Check cache
            if (team_id in self.injury_cache and 
                current_time - self.last_cache_update < self.cache_timeout):
                return self.injury_cache[team_id]
            
            # Get team roster
            roster = CommonTeamRoster(team_id=team_id).get_data_frames()[0]
            time.sleep(0.6)  # Rate limiting
            
            # Fetch injury data from ESPN
            injuries = self._fetch_espn_injuries()
            
            # Match injuries with roster
            team_injuries = []
            for _, player in roster.iterrows():
                player_name = f"{player['PLAYER']}"
                if player_name in injuries:
                    injury_info = injuries[player_name]
                    
                    # Get player's stats to determine impact
                    career_stats = PlayerCareerStats(
                        player_id=player['PLAYER_ID']
                    ).get_data_frames()[0]
                    time.sleep(0.6)  # Rate limiting
                    
                    if len(career_stats) > 0:
                        recent_stats = career_stats.iloc[-1]
                        impact_score = self._calculate_player_impact(recent_stats)
                    else:
                        impact_score = 0.1  # Default for players without stats
                    
                    team_injuries.append({
                        'player_name': player_name,
                        'player_id': player['PLAYER_ID'],
                        'status': injury_info['status'],
                        'injury': injury_info['injury'],
                        'expected_return': injury_info['return'],
                        'impact_score': impact_score
                    })
            
            injury_analysis = {
                'active_injuries': team_injuries,
                'total_impact': sum(inj['impact_score'] for inj in team_injuries),
                'key_players_out': len([inj for inj in team_injuries if inj['impact_score'] > 0.15]),
                'total_players_out': len(team_injuries)
            }
            
            # Update cache
            self.injury_cache[team_id] = injury_analysis
            self.last_cache_update = current_time
            
            return injury_analysis
            
        except Exception as e:
            print(f"Error getting team injuries: {e}")
            return {
                'active_injuries': [],
                'total_impact': 0,
                'key_players_out': 0,
                'total_players_out': 0
            }

    def _fetch_espn_injuries(self):
        """Fetch current NBA injuries from ESPN"""
        try:
            url = "https://www.espn.com/nba/injuries"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            injuries = {}
            
            # Parse ESPN injury page
            for row in soup.find_all('tr', class_='Table__TR'):
                cells = row.find_all('td')
                if len(cells) >= 4:
                    player_name = cells[0].get_text().strip()
                    status = cells[1].get_text().strip()
                    injury = cells[2].get_text().strip()
                    exp_return = cells[3].get_text().strip()
                    
                    injuries[player_name] = {
                        'status': status,
                        'injury': injury,
                        'return': exp_return
                    }
            
            return injuries
            
        except Exception as e:
            print(f"Error fetching ESPN injuries: {e}")
            return {}

    def _calculate_player_impact(self, stats):
        """Calculate player's impact score based on recent stats"""
        try:
            # Basic impact calculation based on key statistics
            mpg = float(stats['MIN']) / float(stats['GP']) if float(stats['GP']) > 0 else 0
            ppg = float(stats['PTS']) / float(stats['GP']) if float(stats['GP']) > 0 else 0
            rpg = float(stats['REB']) / float(stats['GP']) if float(stats['GP']) > 0 else 0
            apg = float(stats['AST']) / float(stats['GP']) if float(stats['GP']) > 0 else 0
            
            # Weight different aspects of player's contribution
            impact_score = (
                0.4 * (mpg / 48.0) +  # Minutes played
                0.3 * (ppg / 30.0) +  # Scoring
                0.15 * (rpg / 10.0) + # Rebounding
                0.15 * (apg / 10.0)   # Playmaking
            )
            
            return min(max(impact_score, 0), 1)  # Normalize between 0 and 1
            
        except Exception as e:
            print(f"Error calculating player impact: {e}")
            return 0.1  # Default impact score