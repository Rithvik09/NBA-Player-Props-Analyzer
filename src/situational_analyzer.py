"""
Advanced Situational Analysis for NBA Player Props
Tracks revenge games, milestones, contract situations, and other unique betting angles
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelog, CommonPlayerInfo, CommonTeamRoster, LeagueGameFinder
from nba_api.stats.static import players, teams
import time
import json

class SituationalAnalyzer:
    def __init__(self):
        self.milestone_tracking = {
            'points': [10, 20, 30, 40, 50],
            'assists': [5, 10, 15, 20],
            'rebounds': [10, 15, 20],
            'steals': [2, 3, 4, 5],
            'blocks': [2, 3, 4, 5]
        }
        
    def analyze_revenge_game(self, player_id, opponent_team_id):
        """Check if this is a revenge game scenario"""
        try:
            # Get player's team history
            player_info = CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            time.sleep(0.6)
            
            # Check if player previously played for opponent team
            # This is a simplified check - in reality you'd need historical team data
            revenge_factor = self._check_former_team(player_id, opponent_team_id)
            
            return {
                'is_revenge_game': revenge_factor['is_former_team'],
                'revenge_intensity': revenge_factor['intensity'],
                'games_since_trade': revenge_factor['games_since'],
                'historical_performance': revenge_factor['performance_boost']
            }
            
        except Exception as e:
            print(f"Error analyzing revenge game: {e}")
            return {
                'is_revenge_game': False,
                'revenge_intensity': 'None',
                'games_since_trade': None,
                'historical_performance': 1.0
            }
    
    def _check_former_team(self, player_id, opponent_team_id):
        """Check if player previously played for the opponent team"""
        try:
            # Get player's game logs from past seasons to check team history
            current_year = datetime.now().year
            seasons_to_check = [
                f"{current_year-2}-{str(current_year-1)[2:]}",
                f"{current_year-3}-{str(current_year-2)[2:]}"
            ]
            
            for season in seasons_to_check:
                try:
                    games = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season
                    ).get_data_frames()[0]
                    time.sleep(0.6)
                    
                    # Check if any games show player was on the opponent team
                    for _, game in games.iterrows():
                        matchup = game['MATCHUP']
                        # Parse team from matchup string
                        if '@' in matchup:
                            team_abbrev = matchup.split(' @ ')[0].split(' ')[-1]
                        else:
                            team_abbrev = matchup.split(' vs. ')[0].split(' ')[-1]
                            
                        # Check if this team matches opponent
                        team_info = teams.find_team_by_abbreviation(team_abbrev)
                        if team_info and team_info['id'] == opponent_team_id:
                            return {
                                'is_former_team': True,
                                'intensity': 'High' if season == seasons_to_check[0] else 'Medium',
                                'games_since': self._estimate_games_since_trade(),
                                'performance_boost': 1.15  # 15% boost for revenge games
                            }
                except:
                    continue
                    
            return {
                'is_former_team': False,
                'intensity': 'None',
                'games_since': None,
                'performance_boost': 1.0
            }
            
        except Exception as e:
            return {
                'is_former_team': False,
                'intensity': 'None',
                'games_since': None,
                'performance_boost': 1.0
            }
    
    def _estimate_games_since_trade(self):
        """Estimate games since potential trade"""
        return np.random.randint(10, 50)  # Simplified estimation
    
    def analyze_milestone_chase(self, player_id, player_stats, prop_type, line):
        """Analyze if player is chasing a career/season milestone"""
        try:
            milestones = {
                'season_high': self._check_season_high_chase(player_stats, prop_type, line),
                'career_milestone': self._check_career_milestone(player_id, prop_type, line),
                'round_number': self._check_round_number_chase(player_stats, prop_type),
                'consecutive_games': self._check_consecutive_streak(player_stats, prop_type, line)
            }
            
            # Calculate overall milestone motivation
            motivation_score = 0
            active_milestones = []
            
            if milestones['season_high']['is_close']:
                motivation_score += 0.3
                active_milestones.append('Season High')
                
            if milestones['career_milestone']['is_approaching']:
                motivation_score += 0.4
                active_milestones.append('Career Milestone')
                
            if milestones['round_number']['is_chasing']:
                motivation_score += 0.2
                active_milestones.append('Round Number')
                
            if milestones['consecutive_games']['has_streak']:
                motivation_score += 0.25
                active_milestones.append('Streak')
            
            return {
                'milestone_factor': min(motivation_score, 0.5),  # Cap at 50% boost
                'active_milestones': active_milestones,
                'details': milestones,
                'motivation_level': self._get_motivation_level(motivation_score)
            }
            
        except Exception as e:
            print(f"Error analyzing milestones: {e}")
            return {
                'milestone_factor': 0,
                'active_milestones': [],
                'details': {},
                'motivation_level': 'None'
            }
    
    def _check_season_high_chase(self, player_stats, prop_type, line):
        """Check if player is close to season high"""
        if prop_type not in player_stats:
            return {'is_close': False, 'current_high': 0, 'games_needed': 0}
            
        values = player_stats[prop_type].get('values', [])
        if not values:
            return {'is_close': False, 'current_high': 0, 'games_needed': 0}
            
        season_high = max(values)
        is_close = line >= season_high * 0.9  # Within 90% of season high
        
        return {
            'is_close': is_close,
            'current_high': season_high,
            'gap_to_high': max(0, season_high - line)
        }
    
    def _check_career_milestone(self, player_id, prop_type, line):
        """Check for major career milestones (simplified)"""
        career_milestones = {
            'points': [1000, 5000, 10000, 15000, 20000, 25000, 30000],
            'assists': [1000, 2500, 5000, 7500, 10000],
            'rebounds': [1000, 2500, 5000, 7500, 10000, 12500, 15000],
            'steals': [500, 1000, 1500, 2000, 2500],
            'blocks': [500, 1000, 1500, 2000, 2500]
        }
        
        # This is simplified - you'd need career totals from API
        estimated_career_total = np.random.randint(5000, 20000)  # Placeholder
        
        if prop_type in career_milestones:
            milestones = career_milestones[prop_type]
            next_milestone = next((m for m in milestones if m > estimated_career_total), None)
            
            if next_milestone:
                games_to_milestone = (next_milestone - estimated_career_total) / max(line, 1)
                is_approaching = games_to_milestone <= 10  # Within 10 games
                
                return {
                    'is_approaching': is_approaching,
                    'next_milestone': next_milestone,
                    'games_needed': int(games_to_milestone) if games_to_milestone > 0 else 0,
                    'career_total': estimated_career_total
                }
        
        return {
            'is_approaching': False,
            'next_milestone': None,
            'games_needed': 0,
            'career_total': estimated_career_total
        }
    
    def _check_round_number_chase(self, player_stats, prop_type):
        """Check if chasing round numbers (10, 20, 30 etc)"""
        if prop_type not in player_stats:
            return {'is_chasing': False, 'target': 0}
            
        avg = player_stats[prop_type].get('avg', 0)
        round_numbers = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        
        for num in round_numbers:
            if avg >= num * 0.85 and avg < num:  # Within 85% of round number
                return {
                    'is_chasing': True,
                    'target': num,
                    'current_avg': avg,
                    'gap': num - avg
                }
        
        return {'is_chasing': False, 'target': 0}
    
    def _check_consecutive_streak(self, player_stats, prop_type, line):
        """Check for active consecutive games streak"""
        if prop_type not in player_stats:
            return {'has_streak': False, 'streak_length': 0}
            
        values = player_stats[prop_type].get('values', [])[:10]  # Last 10 games
        
        if not values:
            return {'has_streak': False, 'streak_length': 0}
        
        # Check for consecutive games hitting the line
        streak = 0
        for value in values:
            if value >= line:
                streak += 1
            else:
                break
                
        has_streak = streak >= 3  # 3+ game streak
        
        return {
            'has_streak': has_streak,
            'streak_length': streak,
            'line': line
        }
    
    def _get_motivation_level(self, score):
        """Convert motivation score to readable level"""
        if score >= 0.4:
            return 'Very High'
        elif score >= 0.3:
            return 'High'
        elif score >= 0.2:
            return 'Medium'
        elif score >= 0.1:
            return 'Low'
        else:
            return 'None'
    
    def analyze_rest_advantage(self, team_id, opponent_team_id):
        """Analyze rest days advantage between teams"""
        try:
            # This would require game schedule data
            # Simplified implementation
            team_rest = np.random.randint(0, 4)
            opponent_rest = np.random.randint(0, 4)
            
            rest_differential = team_rest - opponent_rest
            
            advantage_level = 'None'
            if rest_differential >= 2:
                advantage_level = 'Significant'
            elif rest_differential == 1:
                advantage_level = 'Slight'
            elif rest_differential <= -2:
                advantage_level = 'Disadvantage'
            
            return {
                'team_rest_days': team_rest,
                'opponent_rest_days': opponent_rest,
                'rest_differential': rest_differential,
                'advantage_level': advantage_level,
                'fatigue_factor': max(0.8, 1 - abs(rest_differential) * 0.05)
            }
            
        except Exception as e:
            return {
                'team_rest_days': 1,
                'opponent_rest_days': 1,
                'rest_differential': 0,
                'advantage_level': 'None',
                'fatigue_factor': 1.0
            }
    
    def get_comprehensive_situation_analysis(self, player_id, opponent_team_id, player_stats, prop_type, line):
        """Get complete situational analysis"""
        try:
            analysis = {
                'revenge_game': self.analyze_revenge_game(player_id, opponent_team_id),
                'milestones': self.analyze_milestone_chase(player_id, player_stats, prop_type, line),
                'rest_advantage': self.analyze_rest_advantage(player_id, opponent_team_id),
                'overall_situation_boost': 0
            }
            
            # Calculate combined situational boost
            boost = 0
            
            if analysis['revenge_game']['is_revenge_game']:
                boost += 0.1 * (1 if analysis['revenge_game']['revenge_intensity'] == 'High' else 0.5)
                
            boost += analysis['milestones']['milestone_factor']
            
            if analysis['rest_advantage']['advantage_level'] == 'Significant':
                boost += 0.05
            elif analysis['rest_advantage']['advantage_level'] == 'Slight':
                boost += 0.03
                
            analysis['overall_situation_boost'] = min(boost, 0.25)  # Cap at 25%
            analysis['situation_grade'] = self._grade_situation(boost)
            
            return analysis
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            return {
                'revenge_game': {'is_revenge_game': False},
                'milestones': {'milestone_factor': 0, 'active_milestones': []},
                'rest_advantage': {'advantage_level': 'None'},
                'overall_situation_boost': 0,
                'situation_grade': 'D'
            }
    
    def _grade_situation(self, boost):
        """Grade the overall situation"""
        if boost >= 0.2:
            return 'A+'
        elif boost >= 0.15:
            return 'A'
        elif boost >= 0.1:
            return 'B'
        elif boost >= 0.05:
            return 'C'
        else:
            return 'D'