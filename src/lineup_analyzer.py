import requests
import json
import time
from datetime import datetime, timedelta
from nba_api.stats.endpoints import CommonTeamRoster, PlayerVsPlayer
from nba_api.stats.static import teams, players
import sqlite3

class LineupAnalyzer:
    def __init__(self, db_name='basketball_data.db'):
        self.db_name = db_name
        self.create_tables()
        # Cache for lineup data to avoid repeated API calls
        self.lineup_cache = {}
        self.injury_cache = {}
        
    def create_tables(self):
        """Create tables for lineup tracking"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lineup_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER,
                game_date DATE,
                starting_lineup TEXT,
                key_injuries TEXT,
                minutes_distribution TEXT,
                pace_impact REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_usage_rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                team_id INTEGER,
                usage_rate REAL,
                avg_minutes REAL,
                role TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_team_roster_analysis(self, team_id):
        """Get comprehensive team roster analysis"""
        try:
            # Get current roster
            roster = CommonTeamRoster(team_id=team_id).get_data_frames()[0]
            time.sleep(0.6)
            
            roster_analysis = {
                'total_players': len(roster),
                'positions': {},
                'key_players': [],
                'role_players': [],
                'rookies': []
            }
            
            # Analyze positions and roles
            for _, player in roster.iterrows():
                position = player['POSITION']
                player_info = {
                    'id': player['PLAYER_ID'],
                    'name': player['PLAYER'],
                    'position': position,
                    'number': player['NUM'],
                    'experience': player['EXP']
                }
                
                # Count positions
                if position not in roster_analysis['positions']:
                    roster_analysis['positions'][position] = 0
                roster_analysis['positions'][position] += 1
                
                # Categorize players (simplified logic - could be enhanced with more data)
                if player['EXP'] == 'R':
                    roster_analysis['rookies'].append(player_info)
                elif 'STAR' in player['PLAYER'].upper() or len(player['PLAYER'].split()) == 2:
                    # Simplified star detection - would need more sophisticated logic
                    roster_analysis['key_players'].append(player_info)
                else:
                    roster_analysis['role_players'].append(player_info)
            
            return roster_analysis
            
        except Exception as e:
            print(f"Error getting roster analysis: {e}")
            return self._get_default_roster_analysis()
    
    def analyze_lineup_impact(self, team_id, player_id, opponent_team_id):
        """Analyze how current lineup affects player performance"""
        try:
            roster_analysis = self.get_team_roster_analysis(team_id)
            
            # Simulate injury impact scenarios
            injury_scenarios = self._simulate_injury_scenarios(team_id, player_id)
            
            # Calculate usage rate changes
            usage_impact = self._calculate_usage_impact(team_id, player_id, injury_scenarios)
            
            # Pace analysis
            pace_impact = self._analyze_pace_impact(team_id, roster_analysis)
            
            # Matchup advantages/disadvantages
            matchup_analysis = self._analyze_positional_matchups(team_id, opponent_team_id)
            
            return {
                'roster_analysis': roster_analysis,
                'injury_scenarios': injury_scenarios,
                'usage_impact': usage_impact,
                'pace_impact': pace_impact,
                'matchup_analysis': matchup_analysis,
                'overall_impact': self._calculate_overall_impact(usage_impact, pace_impact, matchup_analysis)
            }
            
        except Exception as e:
            print(f"Error analyzing lineup impact: {e}")
            return self._get_default_lineup_impact()
    
    def _simulate_injury_scenarios(self, team_id, player_id):
        """Simulate various injury scenarios and their impact"""
        scenarios = {
            'healthy_lineup': {
                'probability': 0.7,
                'usage_modifier': 1.0,
                'pace_modifier': 1.0,
                'description': 'Full healthy roster available'
            },
            'minor_injuries': {
                'probability': 0.2,
                'usage_modifier': 1.1,
                'pace_modifier': 0.95,
                'description': '1-2 role players out'
            },
            'key_player_out': {
                'probability': 0.08,
                'usage_modifier': 1.25,
                'pace_modifier': 0.9,
                'description': 'Star player or key contributor injured'
            },
            'multiple_injuries': {
                'probability': 0.02,
                'usage_modifier': 1.4,
                'pace_modifier': 0.85,
                'description': 'Multiple key players injured'
            }
        }
        
        # Get actual injury data if available
        current_injuries = self._get_current_injuries(team_id)
        
        # Adjust probabilities based on actual injury data
        if current_injuries['key_players_out'] > 0:
            scenarios['key_player_out']['probability'] = 0.8
            scenarios['healthy_lineup']['probability'] = 0.1
        elif current_injuries['total_players_out'] > 2:
            scenarios['multiple_injuries']['probability'] = 0.6
            scenarios['healthy_lineup']['probability'] = 0.2
        
        return scenarios
    
    def _calculate_usage_impact(self, team_id, player_id, injury_scenarios):
        """Calculate how injuries affect player usage rate"""
        base_usage = self._get_player_usage_rate(player_id)
        
        usage_adjustments = {}
        for scenario, data in injury_scenarios.items():
            adjusted_usage = base_usage * data['usage_modifier']
            usage_adjustments[scenario] = {
                'base_usage': base_usage,
                'adjusted_usage': adjusted_usage,
                'usage_increase': (adjusted_usage - base_usage) / base_usage * 100,
                'probability': data['probability']
            }
        
        # Calculate expected usage rate
        expected_usage = sum(
            adj['adjusted_usage'] * adj['probability'] 
            for adj in usage_adjustments.values()
        )
        
        return {
            'base_usage_rate': base_usage,
            'expected_usage_rate': expected_usage,
            'scenario_breakdown': usage_adjustments,
            'usage_volatility': self._calculate_usage_volatility(usage_adjustments)
        }
    
    def _analyze_pace_impact(self, team_id, roster_analysis):
        """Analyze how roster composition affects team pace"""
        try:
            # Get team's baseline pace (simplified calculation)
            baseline_pace = 100.0  # Default NBA pace
            
            # Adjust based on roster composition
            pace_factors = {
                'young_players': len(roster_analysis.get('rookies', [])) * 0.02,
                'veteran_presence': len(roster_analysis.get('key_players', [])) * -0.01,
                'depth': max(0, (roster_analysis.get('total_players', 15) - 12) * 0.005)
            }
            
            adjusted_pace = baseline_pace * (1 + sum(pace_factors.values()))
            
            return {
                'baseline_pace': baseline_pace,
                'adjusted_pace': adjusted_pace,
                'pace_factors': pace_factors,
                'pace_impact_percentage': sum(pace_factors.values()) * 100
            }
            
        except Exception as e:
            print(f"Error analyzing pace impact: {e}")
            return {
                'baseline_pace': 100.0,
                'adjusted_pace': 100.0,
                'pace_factors': {},
                'pace_impact_percentage': 0.0
            }
    
    def _analyze_positional_matchups(self, team_id, opponent_team_id):
        """Analyze positional matchups between teams"""
        try:
            team_roster = self.get_team_roster_analysis(team_id)
            opponent_roster = self.get_team_roster_analysis(opponent_team_id)
            
            matchup_advantages = {}
            
            # Compare position strengths (simplified)
            for position in ['G', 'F', 'C']:
                team_count = team_roster['positions'].get(position, 0)
                opp_count = opponent_roster['positions'].get(position, 0)
                
                if team_count > opp_count:
                    advantage = 'TEAM'
                elif opp_count > team_count:
                    advantage = 'OPPONENT'
                else:
                    advantage = 'NEUTRAL'
                
                matchup_advantages[position] = {
                    'advantage': advantage,
                    'team_depth': team_count,
                    'opponent_depth': opp_count,
                    'impact_score': abs(team_count - opp_count) * 0.1
                }
            
            return {
                'positional_advantages': matchup_advantages,
                'overall_advantage': self._calculate_overall_matchup_advantage(matchup_advantages)
            }
            
        except Exception as e:
            print(f"Error analyzing matchups: {e}")
            return {'positional_advantages': {}, 'overall_advantage': 'NEUTRAL'}
    
    def _get_current_injuries(self, team_id):
        """Get current injury information for team"""
        # This would typically connect to an injury API or database
        # For now, return mock data
        return {
            'total_players_out': 1,
            'key_players_out': 0,
            'day_to_day': 2,
            'long_term': 0
        }
    
    def _get_player_usage_rate(self, player_id):
        """Get player's current usage rate"""
        # This would typically calculate from recent game logs
        # For now, return a reasonable default
        return 22.0  # Average NBA usage rate
    
    def _calculate_usage_volatility(self, usage_adjustments):
        """Calculate volatility in usage rate across scenarios"""
        usage_rates = [adj['adjusted_usage'] for adj in usage_adjustments.values()]
        if len(usage_rates) <= 1:
            return 0.0
        
        mean_usage = sum(usage_rates) / len(usage_rates)
        variance = sum((rate - mean_usage) ** 2 for rate in usage_rates) / len(usage_rates)
        
        return variance ** 0.5
    
    def _calculate_overall_matchup_advantage(self, matchup_advantages):
        """Calculate overall matchup advantage"""
        team_advantages = sum(1 for adv in matchup_advantages.values() if adv['advantage'] == 'TEAM')
        opp_advantages = sum(1 for adv in matchup_advantages.values() if adv['advantage'] == 'OPPONENT')
        
        if team_advantages > opp_advantages:
            return 'FAVORABLE'
        elif opp_advantages > team_advantages:
            return 'UNFAVORABLE'
        else:
            return 'NEUTRAL'
    
    def _calculate_overall_impact(self, usage_impact, pace_impact, matchup_analysis):
        """Calculate overall lineup impact on player performance"""
        impact_score = 0.0
        
        # Usage impact (40% weight)
        usage_increase = usage_impact.get('expected_usage_rate', 22.0) - usage_impact.get('base_usage_rate', 22.0)
        impact_score += (usage_increase / usage_impact.get('base_usage_rate', 22.0)) * 0.4
        
        # Pace impact (30% weight)
        pace_change = pace_impact.get('pace_impact_percentage', 0.0)
        impact_score += (pace_change / 100) * 0.3
        
        # Matchup impact (30% weight)
        matchup_advantage = matchup_analysis.get('overall_advantage', 'NEUTRAL')
        if matchup_advantage == 'FAVORABLE':
            impact_score += 0.05 * 0.3
        elif matchup_advantage == 'UNFAVORABLE':
            impact_score -= 0.05 * 0.3
        
        # Convert to performance multiplier
        performance_multiplier = 1.0 + impact_score
        
        return {
            'impact_score': impact_score,
            'performance_multiplier': performance_multiplier,
            'category': self._categorize_impact(impact_score),
            'explanation': self._explain_impact(impact_score, usage_impact, pace_impact, matchup_analysis)
        }
    
    def _categorize_impact(self, impact_score):
        """Categorize the overall impact"""
        if impact_score > 0.10:
            return 'VERY_POSITIVE'
        elif impact_score > 0.05:
            return 'POSITIVE'
        elif impact_score > -0.05:
            return 'NEUTRAL'
        elif impact_score > -0.10:
            return 'NEGATIVE'
        else:
            return 'VERY_NEGATIVE'
    
    def _explain_impact(self, impact_score, usage_impact, pace_impact, matchup_analysis):
        """Provide explanation for the lineup impact"""
        explanations = []
        
        usage_change = usage_impact.get('expected_usage_rate', 22.0) - usage_impact.get('base_usage_rate', 22.0)
        if usage_change > 2:
            explanations.append(f"Increased usage rate due to lineup changes (+{usage_change:.1f}%)")
        elif usage_change < -2:
            explanations.append(f"Decreased usage rate due to lineup changes ({usage_change:.1f}%)")
        
        pace_change = pace_impact.get('pace_impact_percentage', 0.0)
        if pace_change > 2:
            explanations.append(f"Faster pace expected (+{pace_change:.1f}%)")
        elif pace_change < -2:
            explanations.append(f"Slower pace expected ({pace_change:.1f}%)")
        
        matchup = matchup_analysis.get('overall_advantage', 'NEUTRAL')
        if matchup == 'FAVORABLE':
            explanations.append("Favorable positional matchups")
        elif matchup == 'UNFAVORABLE':
            explanations.append("Challenging positional matchups")
        
        if not explanations:
            explanations.append("Minimal lineup impact expected")
        
        return " | ".join(explanations)
    
    def _get_default_roster_analysis(self):
        """Return default roster analysis"""
        return {
            'total_players': 15,
            'positions': {'G': 6, 'F': 6, 'C': 3},
            'key_players': [],
            'role_players': [],
            'rookies': []
        }
    
    def _get_default_lineup_impact(self):
        """Return default lineup impact"""
        return {
            'roster_analysis': self._get_default_roster_analysis(),
            'injury_scenarios': {},
            'usage_impact': {'base_usage_rate': 22.0, 'expected_usage_rate': 22.0},
            'pace_impact': {'baseline_pace': 100.0, 'adjusted_pace': 100.0},
            'matchup_analysis': {'overall_advantage': 'NEUTRAL'},
            'overall_impact': {
                'impact_score': 0.0,
                'performance_multiplier': 1.0,
                'category': 'NEUTRAL',
                'explanation': 'No significant lineup impact detected'
            }
        }