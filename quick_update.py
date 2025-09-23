#!/usr/bin/env python3
"""
Quick Database Update - Essential data only for immediate use
"""

import sqlite3
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# NBA API imports
from nba_api.stats.static import players, teams

class QuickUpdater:
    def __init__(self):
        self.nba_db_path = 'basketball_data.db'
        self.nfl_db_path = 'nfl_data.db'
        
        # Current seasons
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # NBA season logic
        if current_month >= 10:  # October or later - new season
            self.nba_season = f"{current_year}-{str(current_year+1)[2:]}"
        else:  # Before October - previous season
            self.nba_season = f"{current_year-1}-{str(current_year)[2:]}"
            
        print(f"🚀 Quick update for NBA {self.nba_season} season")
        
    def quick_nba_update(self):
        """Quick NBA update with essential data"""
        print("🏀 Quick NBA update starting...")
        
        conn = sqlite3.connect(self.nba_db_path)
        cursor = conn.cursor()
        
        try:
            # 1. Update current players (fast)
            print("📥 Updating NBA players...")
            all_players = players.get_players()
            active_players = [p for p in all_players if p['is_active']]
            
            # Clear and update players table
            cursor.execute("DELETE FROM players")
            
            for player in active_players:
                cursor.execute('''
                    INSERT OR REPLACE INTO players (id, full_name, first_name, last_name, is_active)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    player['id'], 
                    player['full_name'],
                    player['full_name'].split()[0] if player['full_name'] else '',
                    player['full_name'].split()[-1] if player['full_name'] else '',
                    1
                ))
                    
            conn.commit()
            print(f"✅ Updated {len(active_players)} active NBA players")
            
            # 2. Update NBA teams
            print("📥 Updating NBA teams...")
            nba_teams = teams.get_teams()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS nba_teams (
                    id INTEGER PRIMARY KEY,
                    full_name TEXT,
                    abbreviation TEXT,
                    nickname TEXT,
                    city TEXT,
                    state TEXT,
                    year_founded INTEGER
                )
            ''')
            
            cursor.execute("DELETE FROM nba_teams")
            
            for team in nba_teams:
                cursor.execute('''
                    INSERT INTO nba_teams (id, full_name, abbreviation, nickname, city, state, year_founded)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    team['id'],
                    team['full_name'],
                    team['abbreviation'],
                    team['nickname'],
                    team['city'],
                    team['state'],
                    team['year_founded']
                ))
                
            conn.commit()
            print(f"✅ Updated {len(nba_teams)} NBA teams")
            
        except Exception as e:
            print(f"❌ Error in NBA update: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def quick_nfl_update(self):
        """Quick NFL update with essential data"""
        print("🏈 Quick NFL update starting...")
        
        conn = sqlite3.connect(self.nfl_db_path)
        cursor = conn.cursor()
        
        try:
            # 1. NFL Teams (2024 season)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS nfl_teams (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    abbreviation TEXT UNIQUE,
                    full_name TEXT,
                    city TEXT,
                    nickname TEXT,
                    conference TEXT,
                    division TEXT
                )
            ''')
            
            nfl_teams_2024 = [
                # AFC East
                ('BUF', 'Buffalo Bills', 'Buffalo', 'Bills', 'AFC', 'East'),
                ('MIA', 'Miami Dolphins', 'Miami', 'Dolphins', 'AFC', 'East'),
                ('NE', 'New England Patriots', 'New England', 'Patriots', 'AFC', 'East'),
                ('NYJ', 'New York Jets', 'New York', 'Jets', 'AFC', 'East'),
                
                # AFC North
                ('BAL', 'Baltimore Ravens', 'Baltimore', 'Ravens', 'AFC', 'North'),
                ('CIN', 'Cincinnati Bengals', 'Cincinnati', 'Bengals', 'AFC', 'North'),
                ('CLE', 'Cleveland Browns', 'Cleveland', 'Browns', 'AFC', 'North'),
                ('PIT', 'Pittsburgh Steelers', 'Pittsburgh', 'Steelers', 'AFC', 'North'),
                
                # AFC South
                ('HOU', 'Houston Texans', 'Houston', 'Texans', 'AFC', 'South'),
                ('IND', 'Indianapolis Colts', 'Indianapolis', 'Colts', 'AFC', 'South'),
                ('JAX', 'Jacksonville Jaguars', 'Jacksonville', 'Jaguars', 'AFC', 'South'),
                ('TEN', 'Tennessee Titans', 'Tennessee', 'Titans', 'AFC', 'South'),
                
                # AFC West
                ('DEN', 'Denver Broncos', 'Denver', 'Broncos', 'AFC', 'West'),
                ('KC', 'Kansas City Chiefs', 'Kansas City', 'Chiefs', 'AFC', 'West'),
                ('LV', 'Las Vegas Raiders', 'Las Vegas', 'Raiders', 'AFC', 'West'),
                ('LAC', 'Los Angeles Chargers', 'Los Angeles', 'Chargers', 'AFC', 'West'),
                
                # NFC East
                ('DAL', 'Dallas Cowboys', 'Dallas', 'Cowboys', 'NFC', 'East'),
                ('NYG', 'New York Giants', 'New York', 'Giants', 'NFC', 'East'),
                ('PHI', 'Philadelphia Eagles', 'Philadelphia', 'Eagles', 'NFC', 'East'),
                ('WAS', 'Washington Commanders', 'Washington', 'Commanders', 'NFC', 'East'),
                
                # NFC North
                ('CHI', 'Chicago Bears', 'Chicago', 'Bears', 'NFC', 'North'),
                ('DET', 'Detroit Lions', 'Detroit', 'Lions', 'NFC', 'North'),
                ('GB', 'Green Bay Packers', 'Green Bay', 'Packers', 'NFC', 'North'),
                ('MIN', 'Minnesota Vikings', 'Minnesota', 'Vikings', 'NFC', 'North'),
                
                # NFC South
                ('ATL', 'Atlanta Falcons', 'Atlanta', 'Falcons', 'NFC', 'South'),
                ('CAR', 'Carolina Panthers', 'Carolina', 'Panthers', 'NFC', 'South'),
                ('NO', 'New Orleans Saints', 'New Orleans', 'Saints', 'NFC', 'South'),
                ('TB', 'Tampa Bay Buccaneers', 'Tampa Bay', 'Buccaneers', 'NFC', 'South'),
                
                # NFC West
                ('ARI', 'Arizona Cardinals', 'Arizona', 'Cardinals', 'NFC', 'West'),
                ('LAR', 'Los Angeles Rams', 'Los Angeles', 'Rams', 'NFC', 'West'),
                ('SF', 'San Francisco 49ers', 'San Francisco', '49ers', 'NFC', 'West'),
                ('SEA', 'Seattle Seahawks', 'Seattle', 'Seahawks', 'NFC', 'West'),
            ]
            
            cursor.execute("DELETE FROM nfl_teams")
            
            for team in nfl_teams_2024:
                cursor.execute('''
                    INSERT INTO nfl_teams (abbreviation, full_name, city, nickname, conference, division)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', team)
                
            conn.commit()
            print(f"✅ Updated {len(nfl_teams_2024)} NFL teams for 2024 season")
            
            # 2. Key NFL Players (2024 active rosters)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS nfl_players (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    position TEXT,
                    team TEXT,
                    age INTEGER,
                    weight INTEGER,
                    height TEXT,
                    active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # Top 2024 NFL Players by position
            key_players_2024 = [
                # Star Quarterbacks
                ('Josh Allen', 'QB', 'BUF', 28, 245, '6-5'),
                ('Tua Tagovailoa', 'QB', 'MIA', 26, 217, '6-1'),
                ('Aaron Rodgers', 'QB', 'NYJ', 40, 225, '6-2'),
                ('Drake Maye', 'QB', 'NE', 22, 230, '6-4'),
                
                ('Lamar Jackson', 'QB', 'BAL', 27, 212, '6-2'),
                ('Joe Burrow', 'QB', 'CIN', 27, 221, '6-4'),
                ('Deshaun Watson', 'QB', 'CLE', 29, 215, '6-2'),
                ('Russell Wilson', 'QB', 'PIT', 35, 215, '5-11'),
                
                ('C.J. Stroud', 'QB', 'HOU', 22, 218, '6-3'),
                ('Anthony Richardson', 'QB', 'IND', 22, 244, '6-4'),
                ('Trevor Lawrence', 'QB', 'JAX', 25, 213, '6-6'),
                ('Will Levis', 'QB', 'TEN', 25, 229, '6-3'),
                
                ('Patrick Mahomes', 'QB', 'KC', 29, 225, '6-3'),
                ('Bo Nix', 'QB', 'DEN', 24, 217, '6-2'),
                ('Justin Herbert', 'QB', 'LAC', 26, 236, '6-6'),
                ('Gardner Minshew', 'QB', 'LV', 28, 225, '6-1'),
                
                ('Dak Prescott', 'QB', 'DAL', 31, 238, '6-2'),
                ('Daniel Jones', 'QB', 'NYG', 27, 230, '6-5'),
                ('Jalen Hurts', 'QB', 'PHI', 26, 223, '6-1'),
                ('Jayden Daniels', 'QB', 'WAS', 24, 210, '6-4'),
                
                ('Caleb Williams', 'QB', 'CHI', 22, 214, '6-1'),
                ('Jared Goff', 'QB', 'DET', 30, 217, '6-4'),
                ('Jordan Love', 'QB', 'GB', 26, 224, '6-4'),
                ('Sam Darnold', 'QB', 'MIN', 27, 225, '6-3'),
                
                ('Kirk Cousins', 'QB', 'ATL', 36, 202, '6-3'),
                ('Bryce Young', 'QB', 'CAR', 23, 204, '5-10'),
                ('Derek Carr', 'QB', 'NO', 33, 215, '6-3'),
                ('Baker Mayfield', 'QB', 'TB', 29, 215, '6-1'),
                
                ('Kyler Murray', 'QB', 'ARI', 27, 207, '5-10'),
                ('Matthew Stafford', 'QB', 'LAR', 36, 220, '6-3'),
                ('Brock Purdy', 'QB', 'SF', 24, 220, '6-1'),
                ('Geno Smith', 'QB', 'SEA', 34, 221, '6-3'),
                
                # Elite Running Backs
                ('Christian McCaffrey', 'RB', 'SF', 28, 205, '5-11'),
                ('Josh Jacobs', 'RB', 'GB', 26, 220, '5-10'),
                ('Saquon Barkley', 'RB', 'PHI', 27, 233, '6-0'),
                ('Derrick Henry', 'RB', 'BAL', 30, 247, '6-3'),
                ('Jonathan Taylor', 'RB', 'IND', 25, 226, '5-10'),
                ('Alvin Kamara', 'RB', 'NO', 29, 215, '5-10'),
                ('Nick Chubb', 'RB', 'CLE', 28, 227, '5-11'),
                ('Kenneth Walker III', 'RB', 'SEA', 24, 211, '5-9'),
                ('Breece Hall', 'RB', 'NYJ', 23, 217, '5-11'),
                ('Bijan Robinson', 'RB', 'ATL', 22, 215, '6-0'),
                
                # Top Wide Receivers
                ('Tyreek Hill', 'WR', 'MIA', 30, 185, '5-10'),
                ('Davante Adams', 'WR', 'LV', 31, 215, '6-1'),
                ('Stefon Diggs', 'WR', 'HOU', 30, 191, '6-0'),
                ('A.J. Brown', 'WR', 'PHI', 27, 226, '6-1'),
                ('CeeDee Lamb', 'WR', 'DAL', 25, 198, '6-2'),
                ('Ja\'Marr Chase', 'WR', 'CIN', 24, 201, '6-0'),
                ('Justin Jefferson', 'WR', 'MIN', 25, 202, '6-1'),
                ('Amon-Ra St. Brown', 'WR', 'DET', 25, 197, '5-11'),
                ('DK Metcalf', 'WR', 'SEA', 27, 229, '6-4'),
                ('Mike Evans', 'WR', 'TB', 31, 231, '6-5'),
                ('Cooper Kupp', 'WR', 'LAR', 31, 208, '6-2'),
                ('DeVonta Smith', 'WR', 'PHI', 26, 170, '6-0'),
                
                # Elite Tight Ends
                ('Travis Kelce', 'TE', 'KC', 35, 250, '6-5'),
                ('Mark Andrews', 'TE', 'BAL', 29, 256, '6-5'),
                ('T.J. Hockenson', 'TE', 'MIN', 27, 246, '6-5'),
                ('George Kittle', 'TE', 'SF', 31, 250, '6-4'),
                ('Evan Engram', 'TE', 'JAX', 30, 240, '6-3'),
                ('Kyle Pitts', 'TE', 'ATL', 24, 246, '6-6'),
                
                # Top Kickers
                ('Justin Tucker', 'K', 'BAL', 35, 183, '6-1'),
                ('Harrison Butker', 'K', 'KC', 29, 196, '6-4'),
                ('Tyler Bass', 'K', 'BUF', 27, 183, '5-10'),
                ('Chris Boswell', 'K', 'PIT', 33, 185, '6-2'),
                ('Jake Elliott', 'K', 'PHI', 29, 167, '5-9'),
                ('Brandon McManus', 'K', 'GB', 32, 201, '6-3'),
            ]
            
            cursor.execute("DELETE FROM nfl_players")
            
            for player in key_players_2024:
                cursor.execute('''
                    INSERT INTO nfl_players (name, position, team, age, weight, height, active)
                    VALUES (?, ?, ?, ?, ?, ?, TRUE)
                ''', player)
                
            conn.commit()
            print(f"✅ Updated {len(key_players_2024)} key NFL players for 2024 season")
            
            # 3. NFL weather data for key stadiums
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS nfl_weather (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT,
                    city TEXT,
                    latitude REAL,
                    longitude REAL,
                    conditions TEXT,
                    dome_stadium BOOLEAN DEFAULT FALSE
                )
            ''')
            
            weather_data = [
                ('BUF', 'Buffalo', 42.7738, -78.7870, 'Cold, Snow, Wind', False),
                ('MIA', 'Miami Gardens', 25.9580, -80.2389, 'Hot, Humid', False),
                ('NE', 'Foxborough', 42.0909, -71.2643, 'Cold, Rain, Wind', False),
                ('NYJ', 'East Rutherford', 40.8135, -74.0745, 'Cold, Wind', False),
                ('BAL', 'Baltimore', 39.2780, -76.6229, 'Moderate, Rain', False),
                ('CIN', 'Cincinnati', 39.0955, -84.5162, 'Cold, Rain', False),
                ('CLE', 'Cleveland', 41.5061, -81.6995, 'Cold, Snow, Wind', False),
                ('PIT', 'Pittsburgh', 40.4468, -80.0157, 'Cold, Snow, Rain', False),
                ('DEN', 'Denver', 39.7439, -105.0201, 'Altitude, Cold, Snow', False),
                ('KC', 'Kansas City', 39.0489, -94.4839, 'Cold, Wind, Rain', False),
                ('GB', 'Green Bay', 44.5013, -88.0622, 'Very Cold, Snow, Wind', False),
                ('CHI', 'Chicago', 41.8623, -87.6167, 'Cold, Snow, Wind', False),
                ('SEA', 'Seattle', 47.5952, -122.3316, 'Rain, Wind', False),
                ('SF', 'Santa Clara', 37.4030, -121.9699, 'Mild, Wind', False),
                # Dome stadiums
                ('ATL', 'Atlanta', 33.7577, -84.4008, 'Controlled Climate', True),
                ('NO', 'New Orleans', 29.9511, -90.0812, 'Controlled Climate', True),
                ('DET', 'Detroit', 42.3400, -83.0456, 'Controlled Climate', True),
                ('MIN', 'Minneapolis', 44.9739, -93.2581, 'Controlled Climate', True),
            ]
            
            cursor.execute("DELETE FROM nfl_weather")
            
            for weather in weather_data:
                cursor.execute('''
                    INSERT INTO nfl_weather (team, city, latitude, longitude, conditions, dome_stadium)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', weather)
                
            conn.commit()
            print(f"✅ Updated weather data for {len(weather_data)} NFL stadiums")
            
        except Exception as e:
            print(f"❌ Error in NFL update: {e}")
            conn.rollback()
        finally:
            conn.close()
            
    def verify_quick_update(self):
        """Verify quick update results"""
        print("\n🔍 Verifying quick database updates...")
        
        # NBA verification
        nba_conn = sqlite3.connect(self.nba_db_path)
        nba_cursor = nba_conn.cursor()
        
        print("🏀 NBA Database:")
        nba_tables = ['players', 'nba_teams', 'games', 'game_logs']
        for table in nba_tables:
            try:
                nba_cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = nba_cursor.fetchone()[0]
                print(f"  {table}: {count:,} records")
            except Exception as e:
                print(f"  {table}: Error - {e}")
        nba_conn.close()
        
        # NFL verification  
        nfl_conn = sqlite3.connect(self.nfl_db_path)
        nfl_cursor = nfl_conn.cursor()
        
        print("🏈 NFL Database:")
        nfl_tables = ['nfl_teams', 'nfl_players', 'nfl_weather']
        for table in nfl_tables:
            try:
                nfl_cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = nfl_cursor.fetchone()[0]
                print(f"  {table}: {count:,} records")
            except Exception as e:
                print(f"  {table}: Error - {e}")
        nfl_conn.close()
        
    def run_quick_update(self):
        """Run the quick database update"""
        print("⚡ Starting QUICK database update...")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*40)
        
        self.quick_nba_update()
        print()
        self.quick_nfl_update()
        print()
        self.verify_quick_update()
        
        print("\n✅ Quick database update completed!")

if __name__ == "__main__":
    updater = QuickUpdater()
    updater.run_quick_update()