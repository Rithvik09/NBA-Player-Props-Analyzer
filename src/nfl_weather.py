"""
NFL Weather Integration System
Real-time weather data integration for NFL games
Weather significantly impacts NFL prop betting outcomes
"""
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sqlite3

class NFLWeatherSystem:
    def __init__(self, db_name='nfl_data.db'):
        self.db_name = db_name
        self.api_key = None  # Would be set from environment variables
        
        # NFL stadiums with weather data requirements
        self.nfl_stadiums = {
            'ARI': {'name': 'State Farm Stadium', 'type': 'dome', 'city': 'Glendale', 'state': 'AZ'},
            'ATL': {'name': 'Mercedes-Benz Stadium', 'type': 'dome', 'city': 'Atlanta', 'state': 'GA'},
            'BAL': {'name': 'M&T Bank Stadium', 'type': 'outdoor', 'city': 'Baltimore', 'state': 'MD'},
            'BUF': {'name': 'Highmark Stadium', 'type': 'outdoor', 'city': 'Orchard Park', 'state': 'NY'},
            'CAR': {'name': 'Bank of America Stadium', 'type': 'outdoor', 'city': 'Charlotte', 'state': 'NC'},
            'CHI': {'name': 'Soldier Field', 'type': 'outdoor', 'city': 'Chicago', 'state': 'IL'},
            'CIN': {'name': 'Paycor Stadium', 'type': 'outdoor', 'city': 'Cincinnati', 'state': 'OH'},
            'CLE': {'name': 'Cleveland Browns Stadium', 'type': 'outdoor', 'city': 'Cleveland', 'state': 'OH'},
            'DAL': {'name': 'AT&T Stadium', 'type': 'dome', 'city': 'Arlington', 'state': 'TX'},
            'DEN': {'name': 'Empower Field at Mile High', 'type': 'outdoor', 'city': 'Denver', 'state': 'CO'},
            'DET': {'name': 'Ford Field', 'type': 'dome', 'city': 'Detroit', 'state': 'MI'},
            'GB': {'name': 'Lambeau Field', 'type': 'outdoor', 'city': 'Green Bay', 'state': 'WI'},
            'HOU': {'name': 'NRG Stadium', 'type': 'dome', 'city': 'Houston', 'state': 'TX'},
            'IND': {'name': 'Lucas Oil Stadium', 'type': 'dome', 'city': 'Indianapolis', 'state': 'IN'},
            'JAX': {'name': 'TIAA Bank Field', 'type': 'outdoor', 'city': 'Jacksonville', 'state': 'FL'},
            'KC': {'name': 'Arrowhead Stadium', 'type': 'outdoor', 'city': 'Kansas City', 'state': 'MO'},
            'LV': {'name': 'Allegiant Stadium', 'type': 'dome', 'city': 'Las Vegas', 'state': 'NV'},
            'LAC': {'name': 'SoFi Stadium', 'type': 'dome', 'city': 'Los Angeles', 'state': 'CA'},
            'LAR': {'name': 'SoFi Stadium', 'type': 'dome', 'city': 'Los Angeles', 'state': 'CA'},
            'MIA': {'name': 'Hard Rock Stadium', 'type': 'outdoor', 'city': 'Miami Gardens', 'state': 'FL'},
            'MIN': {'name': 'U.S. Bank Stadium', 'type': 'dome', 'city': 'Minneapolis', 'state': 'MN'},
            'NE': {'name': 'Gillette Stadium', 'type': 'outdoor', 'city': 'Foxborough', 'state': 'MA'},
            'NO': {'name': 'Caesars Superdome', 'type': 'dome', 'city': 'New Orleans', 'state': 'LA'},
            'NYG': {'name': 'MetLife Stadium', 'type': 'outdoor', 'city': 'East Rutherford', 'state': 'NJ'},
            'NYJ': {'name': 'MetLife Stadium', 'type': 'outdoor', 'city': 'East Rutherford', 'state': 'NJ'},
            'PHI': {'name': 'Lincoln Financial Field', 'type': 'outdoor', 'city': 'Philadelphia', 'state': 'PA'},
            'PIT': {'name': 'Acrisure Stadium', 'type': 'outdoor', 'city': 'Pittsburgh', 'state': 'PA'},
            'SF': {'name': 'Levi\'s Stadium', 'type': 'outdoor', 'city': 'Santa Clara', 'state': 'CA'},
            'SEA': {'name': 'Lumen Field', 'type': 'outdoor', 'city': 'Seattle', 'state': 'WA'},
            'TB': {'name': 'Raymond James Stadium', 'type': 'outdoor', 'city': 'Tampa', 'state': 'FL'},
            'TEN': {'name': 'Nissan Stadium', 'type': 'outdoor', 'city': 'Nashville', 'state': 'TN'},
            'WAS': {'name': 'FedExField', 'type': 'outdoor', 'city': 'Landover', 'state': 'MD'}
        }
    
    def get_weather_impact_analysis(self, home_team: str, game_date: str, prop_type: str) -> Dict[str, Any]:
        """
        Analyze weather impact on specific prop types
        Returns impact assessment and recommendations
        """
        try:
            stadium_info = self.nfl_stadiums.get(home_team, {})
            
            # Dome games have no weather impact
            if stadium_info.get('type') == 'dome':
                return {
                    'impact_level': 'NONE',
                    'stadium_type': 'dome',
                    'recommendation': 'NO_WEATHER_ADJUSTMENT',
                    'details': 'Game played in climate-controlled dome'
                }
            
            # Get weather data (simulated for now)
            weather_data = self._get_game_weather(stadium_info, game_date)
            
            # Analyze impact based on prop type
            impact_analysis = self._analyze_weather_impact(weather_data, prop_type)
            
            return {
                'impact_level': impact_analysis['level'],
                'stadium_type': 'outdoor',
                'weather_conditions': weather_data,
                'prop_adjustments': impact_analysis['adjustments'],
                'recommendation': impact_analysis['recommendation'],
                'details': impact_analysis['explanation']
            }
            
        except Exception as e:
            print(f"Error analyzing weather impact: {e}")
            return {
                'impact_level': 'UNKNOWN',
                'error': str(e)
            }
    
    def _get_game_weather(self, stadium_info: Dict, game_date: str) -> Dict[str, Any]:
        """
        Get weather conditions for game location and time
        In production, this would call OpenWeatherMap or similar API
        """
        # Simulated weather data based on location and season
        city = stadium_info.get('city', 'Unknown')
        state = stadium_info.get('state', 'Unknown')
        
        # Sample weather conditions based on geographic location and date
        sample_conditions = self._generate_sample_weather(city, state, game_date)
        
        return sample_conditions
    
    def _generate_sample_weather(self, city: str, state: str, game_date: str) -> Dict[str, Any]:
        """Generate realistic weather conditions based on location and date"""
        
        # Parse date to determine season impact
        try:
            date_obj = datetime.strptime(game_date, '%Y-%m-%d')
            month = date_obj.month
        except:
            month = 12  # Default to December (cold weather)
        
        # Regional weather patterns
        if state in ['FL', 'AZ', 'CA', 'TX', 'NV']:  # Warm weather states
            temperature = 75 if month in [9, 10] else 68 if month in [11, 12, 1] else 82
            wind_speed = 8
            precipitation = 'None' if state in ['AZ', 'CA', 'NV'] else 'Light Rain' if month in [6, 7, 8] else 'None'
        elif state in ['WI', 'MN', 'NY', 'MA', 'PA', 'OH', 'IL', 'MI']:  # Cold weather states
            temperature = 35 if month in [12, 1, 2] else 45 if month in [11, 3] else 55
            wind_speed = 15 if month in [11, 12, 1, 2] else 10
            precipitation = 'Snow' if month in [12, 1, 2] and temperature < 35 else 'None'
        else:  # Moderate climate states
            temperature = 50 if month in [11, 12, 1, 2] else 65
            wind_speed = 12
            precipitation = 'None'
        
        return {
            'temperature': temperature,
            'wind_speed': wind_speed,
            'precipitation': precipitation,
            'humidity': 65,
            'conditions': self._determine_conditions(temperature, wind_speed, precipitation),
            'forecast_confidence': 85
        }
    
    def _determine_conditions(self, temp: int, wind: int, precip: str) -> str:
        """Determine overall weather conditions description"""
        if precip == 'Snow':
            return 'Snowy'
        elif precip in ['Rain', 'Light Rain', 'Heavy Rain']:
            return 'Rainy'
        elif wind > 20:
            return 'Windy'
        elif temp < 32:
            return 'Freezing'
        elif temp < 45:
            return 'Cold'
        elif temp > 85:
            return 'Hot'
        else:
            return 'Clear'
    
    def _analyze_weather_impact(self, weather: Dict, prop_type: str) -> Dict[str, Any]:
        """Analyze how weather conditions impact specific prop types"""
        
        temp = weather.get('temperature', 70)
        wind = weather.get('wind_speed', 5)
        precip = weather.get('precipitation', 'None')
        conditions = weather.get('conditions', 'Clear')
        
        # Initialize impact assessment
        impact = {
            'level': 'LOW',
            'adjustments': {},
            'recommendation': 'NO_ADJUSTMENT',
            'explanation': 'Favorable weather conditions'
        }
        
        # Passing prop impacts
        if prop_type in ['passing_yards', 'passing_touchdowns', 'completions']:
            
            # Cold weather impact
            if temp < 40:
                impact['level'] = 'MODERATE'
                impact['adjustments']['cold_weather'] = -0.15  # 15% reduction
                impact['explanation'] = f'Cold weather ({temp}°F) typically reduces passing efficiency'
            
            # Wind impact (major factor for passing)
            if wind > 15:
                impact['level'] = 'HIGH' if wind > 25 else 'MODERATE'
                wind_penalty = min(0.25, wind * 0.01)  # Up to 25% reduction
                impact['adjustments']['wind_penalty'] = -wind_penalty
                impact['explanation'] = f'High winds ({wind} mph) significantly impact passing accuracy'
            
            # Precipitation impact
            if precip in ['Rain', 'Heavy Rain', 'Snow']:
                impact['level'] = 'HIGH'
                precip_penalty = 0.20 if precip == 'Heavy Rain' else 0.15 if precip == 'Snow' else 0.10
                impact['adjustments']['precipitation_penalty'] = -precip_penalty
                impact['explanation'] = f'{precip} conditions reduce passing performance'
            
        # Rushing prop impacts
        elif prop_type in ['rushing_yards', 'rushing_touchdowns', 'rushing_attempts']:
            
            # Cold/wet weather often favors rushing
            if temp < 45 or precip in ['Rain', 'Snow']:
                impact['level'] = 'MODERATE'
                impact['adjustments']['weather_boost'] = 0.10  # 10% increase
                impact['explanation'] = 'Cold/wet weather typically leads to more rushing attempts'
            
            # Snow specifically helps rushing games
            if precip == 'Snow':
                impact['adjustments']['snow_boost'] = 0.15
                impact['explanation'] = 'Snow conditions strongly favor ground game'
        
        # Receiving prop impacts (similar to passing but slightly less severe)
        elif prop_type in ['receiving_yards', 'receiving_touchdowns', 'receptions']:
            
            if temp < 40 or wind > 15 or precip in ['Rain', 'Snow']:
                impact['level'] = 'MODERATE'
                
                total_penalty = 0
                if temp < 40:
                    total_penalty += 0.12
                if wind > 15:
                    total_penalty += min(0.20, wind * 0.008)
                if precip in ['Rain', 'Heavy Rain', 'Snow']:
                    total_penalty += 0.15
                
                impact['adjustments']['weather_penalty'] = -total_penalty
                impact['explanation'] = f'Weather conditions ({conditions}) reduce passing game effectiveness'
        
        # Kicking props (if supported)
        elif prop_type in ['field_goals', 'extra_points']:
            if wind > 20:
                impact['level'] = 'HIGH'
                impact['adjustments']['wind_penalty'] = -0.25
                impact['explanation'] = f'High winds ({wind} mph) severely impact kicking accuracy'
        
        # Determine overall recommendation
        total_adjustment = sum(impact['adjustments'].values())
        
        if abs(total_adjustment) > 0.15:
            impact['recommendation'] = 'STRONG_ADJUSTMENT'
        elif abs(total_adjustment) > 0.08:
            impact['recommendation'] = 'MODERATE_ADJUSTMENT'
        elif abs(total_adjustment) > 0.03:
            impact['recommendation'] = 'SLIGHT_ADJUSTMENT'
        else:
            impact['recommendation'] = 'NO_ADJUSTMENT'
        
        return impact
    
    def get_historical_weather_trends(self, team: str, prop_type: str, weeks_back: int = 8) -> Dict[str, Any]:
        """Analyze historical weather impact trends for team/prop type"""
        
        try:
            # This would query historical weather data from database
            # For now, returning sample analysis
            
            outdoor_games = self._get_outdoor_games_count(team, weeks_back)
            
            trends = {
                'total_outdoor_games': outdoor_games,
                'weather_impacted_games': max(0, outdoor_games - 2),
                'average_temp': 52,
                'average_wind': 12,
                'games_with_precipitation': max(0, outdoor_games // 3),
                'prop_performance': {
                    'clear_weather_avg': 0,
                    'adverse_weather_avg': 0,
                    'weather_impact_factor': 0
                }
            }
            
            # Calculate performance differential
            if prop_type in ['passing_yards', 'passing_touchdowns']:
                trends['prop_performance'] = {
                    'clear_weather_avg': 275,
                    'adverse_weather_avg': 235,
                    'weather_impact_factor': -0.15
                }
            elif prop_type in ['rushing_yards', 'rushing_touchdowns']:
                trends['prop_performance'] = {
                    'clear_weather_avg': 110,
                    'adverse_weather_avg': 125,
                    'weather_impact_factor': 0.12
                }
            
            return trends
            
        except Exception as e:
            print(f"Error getting historical weather trends: {e}")
            return {}
    
    def _get_outdoor_games_count(self, team: str, weeks_back: int) -> int:
        """Count outdoor games for team in recent weeks"""
        
        stadium_type = self.nfl_stadiums.get(team, {}).get('type', 'outdoor')
        
        if stadium_type == 'dome':
            # Check away games only
            return min(weeks_back // 2, 4)  # Assume roughly half games are away
        else:
            # All home games are outdoor, plus some away games
            return min(weeks_back, 8)
    
    def store_weather_data(self, game_data: Dict[str, Any]) -> bool:
        """Store weather data in database for historical analysis"""
        
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO nfl_weather 
                (game_date, stadium, temperature, wind_speed, precipitation, humidity, conditions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                game_data.get('game_date'),
                game_data.get('stadium'),
                game_data.get('temperature'),
                game_data.get('wind_speed'),
                game_data.get('precipitation'),
                game_data.get('humidity'),
                game_data.get('conditions')
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error storing weather data: {e}")
            return False
    
    def get_weather_alerts(self, games_today: List[Dict]) -> List[Dict]:
        """Get weather alerts for games with significant weather impact"""
        
        alerts = []
        
        for game in games_today:
            home_team = game.get('home_team')
            game_date = game.get('date')
            
            # Skip dome games
            if self.nfl_stadiums.get(home_team, {}).get('type') == 'dome':
                continue
            
            weather_impact = self.get_weather_impact_analysis(home_team, game_date, 'passing_yards')
            
            if weather_impact.get('impact_level') in ['HIGH', 'MODERATE']:
                alerts.append({
                    'game': f"{game.get('away_team')} @ {home_team}",
                    'weather_conditions': weather_impact.get('weather_conditions', {}),
                    'impact_level': weather_impact.get('impact_level'),
                    'recommendation': weather_impact.get('recommendation'),
                    'affected_props': self._get_affected_prop_types(weather_impact)
                })
        
        return alerts
    
    def _get_affected_prop_types(self, weather_impact: Dict) -> List[str]:
        """Determine which prop types are affected by weather conditions"""
        
        affected_props = []
        impact_level = weather_impact.get('impact_level', 'LOW')
        
        if impact_level in ['HIGH', 'MODERATE']:
            weather_conditions = weather_impact.get('weather_conditions', {})
            temp = weather_conditions.get('temperature', 70)
            wind = weather_conditions.get('wind_speed', 5)
            precip = weather_conditions.get('precipitation', 'None')
            
            # Passing props affected by cold, wind, precipitation
            if temp < 45 or wind > 15 or precip in ['Rain', 'Snow']:
                affected_props.extend(['passing_yards', 'passing_touchdowns', 'receiving_yards'])
            
            # Rushing props may benefit from adverse passing weather
            if temp < 45 or precip in ['Rain', 'Snow']:
                affected_props.extend(['rushing_yards', 'rushing_touchdowns'])
            
            # Kicking affected by high winds
            if wind > 20:
                affected_props.extend(['field_goals', 'extra_points'])
        
        return affected_props
    
    def get_outdoor_stadium_teams(self):
        """Get list of NFL teams with outdoor stadiums"""
        outdoor_teams = []
        
        for team, stadium_info in self.nfl_stadiums.items():
            if stadium_info.get('type') == 'outdoor':
                outdoor_teams.append({
                    'team': team,
                    'stadium': stadium_info.get('name'),
                    'city': stadium_info.get('city'),
                    'state': stadium_info.get('state'),
                    'climate_zone': stadium_info.get('climate_zone', 'temperate')
                })
        
        return outdoor_teams
    
    def get_weekly_weather_outlook(self):
        """Get weather outlook for all outdoor stadiums for the upcoming week"""
        outlook = {
            'timestamp': datetime.now().isoformat(),
            'outdoor_stadiums': [],
            'weather_alerts': []
        }
        
        outdoor_teams = self.get_outdoor_stadium_teams()
        
        for team_info in outdoor_teams:
            team = team_info['team']
            # Simulate weather forecast for the week
            forecast = self._generate_weekly_forecast(team_info)
            
            outlook['outdoor_stadiums'].append({
                'team': team,
                'stadium': team_info['stadium'],
                'location': f"{team_info['city']}, {team_info['state']}",
                'forecast': forecast
            })
            
            # Check for weather alerts
            if any(day.get('precipitation_chance', 0) > 70 for day in forecast):
                outlook['weather_alerts'].append(f"{team}: High chance of precipitation this week")
            
            if any(day.get('wind_speed', 0) > 25 for day in forecast):
                outlook['weather_alerts'].append(f"{team}: High winds expected this week")
        
        return outlook
    
    def _generate_weekly_forecast(self, team_info):
        """Generate a 7-day weather forecast for a team's location"""
        forecast = []
        
        for day in range(7):
            date = datetime.now() + timedelta(days=day)
            
            # Simulate weather based on location and season
            weather_data = self._generate_sample_weather(
                team_info['city'], 
                team_info['state'], 
                date.strftime('%Y-%m-%d')
            )
            
            forecast.append({
                'date': date.strftime('%Y-%m-%d'),
                'day_of_week': date.strftime('%A'),
                'temperature': weather_data.get('temperature'),
                'condition': weather_data.get('condition'),
                'precipitation_chance': weather_data.get('precipitation_chance'),
                'wind_speed': weather_data.get('wind_speed'),
                'impact_level': self._assess_weather_impact(weather_data)
            })
        
        return forecast
    
    def _assess_weather_impact(self, weather_data):
        """Assess the impact level of weather conditions"""
        temp = weather_data.get('temperature', 70)
        wind = weather_data.get('wind_speed', 5)
        precip = weather_data.get('precipitation_chance', 0)
        
        impact_score = 0
        
        # Temperature impact
        if temp < 32 or temp > 95:
            impact_score += 3
        elif temp < 45 or temp > 85:
            impact_score += 2
        elif temp < 55 or temp > 75:
            impact_score += 1
        
        # Wind impact
        if wind > 25:
            impact_score += 3
        elif wind > 15:
            impact_score += 2
        elif wind > 10:
            impact_score += 1
        
        # Precipitation impact
        if precip > 80:
            impact_score += 3
        elif precip > 50:
            impact_score += 2
        elif precip > 30:
            impact_score += 1
        
        # Determine impact level
        if impact_score >= 6:
            return 'HIGH'
        elif impact_score >= 4:
            return 'MEDIUM'
        elif impact_score >= 2:
            return 'LOW'
        else:
            return 'NONE'
        return list(set(affected_props))  # Remove duplicates