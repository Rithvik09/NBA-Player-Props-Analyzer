"""
Game-level predictions (moneyline, spread, total) using player projections.

This module aggregates player-level projections to predict game outcomes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class GamePredictor:
    """Predicts game outcomes by aggregating player projections."""
    
    def __init__(self, basketball_helper):
        """
        Initialize game predictor.
        
        Args:
            basketball_helper: BasketballBettingHelper instance for player projections
        """
        self.helper = basketball_helper
        
        # Home court advantage (points)
        self.HOME_ADVANTAGE = 3.5
        
        # Confidence thresholds
        self.HIGH_CONFIDENCE_SPREAD = 8.0  # 8+ point difference
        self.MEDIUM_CONFIDENCE_SPREAD = 4.0  # 4-8 point difference
    
    def predict_game(
        self,
        home_team_id: int,
        away_team_id: int,
        home_roster: List[int],
        away_roster: List[int],
        game_location: str = 'home'
    ) -> Dict:
        """
        Predict game outcome (moneyline, spread, total).
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            home_roster: List of player IDs for home team (top 8-10 rotation)
            away_roster: List of player IDs for away team (top 8-10 rotation)
            game_location: 'home' or 'away' (for neutral site games)
        
        Returns:
            Dict with predictions:
                - moneyline: {'winner': team_id, 'confidence': str, 'win_probability': float}
                - spread: {'favorite': team_id, 'line': float, 'confidence': str}
                - total: {'over_under': float, 'confidence': str}
                - team_projections: {'home': {}, 'away': {}}
        """
        # Project both teams
        home_projection = self._project_team(
            team_id=home_team_id,
            roster=home_roster,
            opponent_team_id=away_team_id,
            is_home=True if game_location == 'home' else False
        )
        
        away_projection = self._project_team(
            team_id=away_team_id,
            roster=away_roster,
            opponent_team_id=home_team_id,
            is_home=False
        )
        
        # Calculate predictions
        moneyline = self._predict_moneyline(home_projection, away_projection)
        spread = self._predict_spread(home_projection, away_projection, game_location)
        total = self._predict_total(home_projection, away_projection)
        
        return {
            'moneyline': moneyline,
            'spread': spread,
            'total': total,
            'team_projections': {
                'home': home_projection,
                'away': away_projection
            }
        }
    
    def _project_team(
        self,
        team_id: int,
        roster: List[int],
        opponent_team_id: int,
        is_home: bool
    ) -> Dict:
        """
        Project team stats by aggregating player projections.
        
        Args:
            team_id: Team ID
            roster: List of player IDs in rotation
            opponent_team_id: Opponent team ID
            is_home: Whether team is playing at home
        
        Returns:
            Dict with team projections
        """
        # Aggregate player projections
        team_points = 0.0
        team_assists = 0.0
        team_rebounds = 0.0
        team_three_pointers = 0.0
        team_turnovers = 0.0
        team_minutes = 0.0
        
        player_projections = []
        
        for player_id in roster:
            try:
                # Get player stats
                stats = self.helper.get_player_stats(player_id)
                
                # Get projected stats (use season averages with adjustments)
                pts_data = stats.get('points', {})
                ast_data = stats.get('assists', {})
                reb_data = stats.get('rebounds', {})
                fg3_data = stats.get('three_pointers', {})
                tov_data = stats.get('turnovers', {})
                
                # Use last 5 games average (more recent)
                player_pts = pts_data.get('last5_avg', pts_data.get('season_avg', 0)) or 0
                player_ast = ast_data.get('last5_avg', ast_data.get('season_avg', 0)) or 0
                player_reb = reb_data.get('last5_avg', reb_data.get('season_avg', 0)) or 0
                player_fg3 = fg3_data.get('last5_avg', fg3_data.get('season_avg', 0)) or 0
                player_tov = tov_data.get('last5_avg', tov_data.get('season_avg', 0)) or 0
                player_min = stats.get('minutes', {}).get('last5_avg', 25.0) or 25.0
                
                # Apply home/away adjustment (small)
                if is_home:
                    player_pts *= 1.02  # 2% boost at home
                
                team_points += player_pts
                team_assists += player_ast
                team_rebounds += player_reb
                team_three_pointers += player_fg3
                team_turnovers += player_tov
                team_minutes += player_min
                
                player_projections.append({
                    'player_id': player_id,
                    'points': player_pts,
                    'assists': player_ast,
                    'rebounds': player_reb,
                    'minutes': player_min
                })
                
            except Exception as e:
                print(f"Error projecting player {player_id}: {e}")
                continue
        
        # Get team context
        team_context = self.helper.ml_predictor.get_team_context(team_id) if hasattr(self.helper, 'ml_predictor') else {}
        team_style = team_context.get('style', {}) if team_context else {}
        
        # Pace adjustment (use defaults if not available)
        pace = float(team_style.get('pace') or team_style.get('PACE') or 100.0)
        pace_factor = pace / 100.0
        
        # Adjust for pace
        team_points *= pace_factor
        
        # Get team ratings (use defaults if not available)
        off_rating = float(team_style.get('off_rating') or team_style.get('OFF_RATING') or 110.0)
        def_rating = float(team_style.get('def_rating') or team_style.get('DEF_RATING') or 110.0)
        
        return {
            'team_id': team_id,
            'projected_points': round(team_points, 1),
            'projected_assists': round(team_assists, 1),
            'projected_rebounds': round(team_rebounds, 1),
            'projected_three_pointers': round(team_three_pointers, 1),
            'projected_turnovers': round(team_turnovers, 1),
            'total_minutes': round(team_minutes, 1),
            'pace': pace,
            'offensive_rating': off_rating,
            'defensive_rating': def_rating,
            'player_projections': player_projections
        }
    
    def _predict_moneyline(
        self,
        home_proj: Dict,
        away_proj: Dict
    ) -> Dict:
        """Predict moneyline winner."""
        home_pts = home_proj['projected_points']
        away_pts = away_proj['projected_points']
        
        # Add home court advantage
        home_pts += self.HOME_ADVANTAGE
        
        # Calculate margin
        margin = home_pts - away_pts
        
        # Determine winner
        if margin > 0:
            winner = home_proj['team_id']
            winner_name = 'home'
        else:
            winner = away_proj['team_id']
            winner_name = 'away'
            margin = abs(margin)
        
        # Calculate win probability (logistic function)
        # Each point of margin ≈ 2.8% increase in win probability
        # Formula: P(win) = 1 / (1 + exp(-margin / 3.5))
        win_prob = 1.0 / (1.0 + np.exp(-margin / 3.5))
        
        # Determine confidence
        if margin >= self.HIGH_CONFIDENCE_SPREAD:
            confidence = 'HIGH'
        elif margin >= self.MEDIUM_CONFIDENCE_SPREAD:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'winner': winner,
            'winner_team': winner_name,
            'win_probability': round(win_prob, 3),
            'expected_margin': round(margin, 1),
            'confidence': confidence
        }
    
    def _predict_spread(
        self,
        home_proj: Dict,
        away_proj: Dict,
        game_location: str
    ) -> Dict:
        """Predict point spread."""
        home_pts = home_proj['projected_points']
        away_pts = away_proj['projected_points']
        
        # Add home court advantage
        if game_location == 'home':
            home_pts += self.HOME_ADVANTAGE
        
        # Calculate spread (from home team perspective)
        # Positive spread = home team favored
        # Negative spread = away team favored
        spread = home_pts - away_pts
        
        # Determine favorite
        if spread > 0:
            favorite = home_proj['team_id']
            favorite_name = 'home'
        else:
            favorite = away_proj['team_id']
            favorite_name = 'away'
        
        # Confidence based on spread size
        abs_spread = abs(spread)
        if abs_spread >= self.HIGH_CONFIDENCE_SPREAD:
            confidence = 'HIGH'
        elif abs_spread >= self.MEDIUM_CONFIDENCE_SPREAD:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'spread': round(spread, 1),
            'favorite': favorite,
            'favorite_team': favorite_name,
            'line': round(abs(spread), 1),  # Absolute value
            'confidence': confidence,
            'home_implied_score': round(home_pts, 1),
            'away_implied_score': round(away_pts, 1)
        }
    
    def _predict_total(
        self,
        home_proj: Dict,
        away_proj: Dict
    ) -> Dict:
        """Predict total over/under."""
        home_pts = home_proj['projected_points']
        away_pts = away_proj['projected_points']
        
        # Combined pace factor
        avg_pace = (home_proj['pace'] + away_proj['pace']) / 2.0
        pace_factor = avg_pace / 100.0
        
        # Calculate total
        total = (home_pts + away_pts) * pace_factor
        
        # Defensive adjustments
        # Better defenses = lower scoring
        home_def = home_proj['defensive_rating']
        away_def = away_proj['defensive_rating']
        avg_def = (home_def + away_def) / 2.0
        
        # Adjust total based on defense
        # League average is ~110, each point above/below adjusts total by ~0.5 points
        def_adjustment = (110.0 - avg_def) * 0.5
        total += def_adjustment
        
        # Confidence based on pace
        if avg_pace > 105:  # Fast game
            confidence = 'HIGH'
            pace_note = 'Fast pace game'
        elif avg_pace < 95:  # Slow game
            confidence = 'MEDIUM'
            pace_note = 'Slow pace game'
        else:
            confidence = 'MEDIUM'
            pace_note = 'Average pace'
        
        return {
            'total': round(total, 1),
            'over_under': round(total, 1),
            'confidence': confidence,
            'pace_note': pace_note,
            'avg_pace': round(avg_pace, 1),
            'home_implied_score': round(home_pts, 1),
            'away_implied_score': round(away_pts, 1)
        }
    
    def get_roster_players(self, team_id: int, top_n: int = 10) -> List[int]:
        """
        Get top N players from a team's roster by minutes played.
        
        Args:
            team_id: Team ID
            top_n: Number of top players to return
        
        Returns:
            List of player IDs
        """
        try:
            # This would need to be implemented to fetch roster
            # For now, return empty list - caller should provide roster
            return []
        except Exception as e:
            print(f"Error getting roster for team {team_id}: {e}")
            return []

