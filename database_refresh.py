#!/usr/bin/env python3
"""
Database Refresh Script - Updates existing database with current 2024-25 data
Works with existing table structures
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

class DatabaseRefresher:
    def __init__(self):
        self.nba_db_path = 'basketball_data.db'
        self.nfl_db_path = 'nfl_data.db'
        
        # Current seasons
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # NBA season logic (Sept 23, 2025 = 2024-25 season)
        if current_month >= 9:  # September or later - current season
            self.nba_season = f"{current_year-1}-{str(current_year)[2:]}"  # 2024-25
        else:  # Before September - previous season  
            self.nba_season = f"{current_year-2}-{str(current_year-1)[2:]}"  # 2023-24
            
        # For NFL (Sept 23, 2025 = 2024 season)
        if current_month >= 2:  # February or later
            self.nfl_season = current_year
        else:
            self.nfl_season = current_year - 1
            
        print(f"🏀 Updating NBA {self.nba_season} season (Active)")
        print(f"🏈 Updating NFL {self.nfl_season} season (Active)")
        
    def refresh_nba_data(self):
        """Refresh NBA data with current rosters and stats"""
        print("🏀 Refreshing NBA database...")
        
        conn = sqlite3.connect(self.nba_db_path)
        cursor = conn.cursor()
        
        try:
            # 1. Update ALL active players for 2024-25 season
            print("📥 Getting current NBA players...")
            all_players = players.get_players()
            active_players = [p for p in all_players if p['is_active']]
            
            print(f"Found {len(active_players)} active NBA players")
            
            # Clear existing players and add current ones
            cursor.execute("DELETE FROM players")
            
            for player in active_players:
                # Insert with existing table structure
                cursor.execute('''
                    INSERT INTO players (id, full_name, first_name, last_name, is_active)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    player['id'], 
                    player['full_name'],
                    player['full_name'].split()[0] if player['full_name'] else '',
                    ' '.join(player['full_name'].split()[1:]) if len(player['full_name'].split()) > 1 else '',
                    1
                ))
                    
            conn.commit()
            print(f"✅ Updated {len(active_players)} current NBA players")
            
            # 2. Add NBA teams table if missing and populate
            print("📥 Updating NBA teams...")
            nba_teams = teams.get_teams()
            
            # Create NBA teams table with correct structure
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
            
            # 3. Clear old game logs to make room for fresh data
            cursor.execute("DELETE FROM game_logs")
            cursor.execute("DELETE FROM player_context")
            cursor.execute("DELETE FROM matchup_data")
            cursor.execute("DELETE FROM team_news")
            cursor.execute("DELETE FROM model_predictions")
            
            conn.commit()
            print("✅ Cleared old NBA analysis data for fresh start")
            
        except Exception as e:
            print(f"❌ Error refreshing NBA data: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def refresh_nfl_data(self):
        """Refresh NFL data with current 2024 season rosters"""
        print("🏈 Refreshing NFL database...")
        
        conn = sqlite3.connect(self.nfl_db_path)
        cursor = conn.cursor()
        
        try:
            # 1. Update NFL teams (use existing structure)
            print("📥 Updating NFL teams for 2024 season...")
            
            # Current 2024 NFL teams
            nfl_teams_2024 = [
                # AFC Teams
                ('BUF', 'Buffalo Bills', 'Buffalo', 'AFC East', 'AFC', 'Highmark Stadium', 'Outdoor'),
                ('MIA', 'Miami Dolphins', 'Miami', 'AFC East', 'AFC', 'Hard Rock Stadium', 'Open-Air'),
                ('NE', 'New England Patriots', 'Foxborough', 'AFC East', 'AFC', 'Gillette Stadium', 'Outdoor'),
                ('NYJ', 'New York Jets', 'East Rutherford', 'AFC East', 'AFC', 'MetLife Stadium', 'Outdoor'),
                
                ('BAL', 'Baltimore Ravens', 'Baltimore', 'AFC North', 'AFC', 'M&T Bank Stadium', 'Outdoor'),
                ('CIN', 'Cincinnati Bengals', 'Cincinnati', 'AFC North', 'AFC', 'Paycor Stadium', 'Outdoor'),
                ('CLE', 'Cleveland Browns', 'Cleveland', 'AFC North', 'AFC', 'FirstEnergy Stadium', 'Outdoor'),
                ('PIT', 'Pittsburgh Steelers', 'Pittsburgh', 'AFC North', 'AFC', 'Acrisure Stadium', 'Outdoor'),
                
                ('HOU', 'Houston Texans', 'Houston', 'AFC South', 'AFC', 'NRG Stadium', 'Retractable Roof'),
                ('IND', 'Indianapolis Colts', 'Indianapolis', 'AFC South', 'AFC', 'Lucas Oil Stadium', 'Retractable Roof'),
                ('JAX', 'Jacksonville Jaguars', 'Jacksonville', 'AFC South', 'AFC', 'TIAA Bank Field', 'Outdoor'),
                ('TEN', 'Tennessee Titans', 'Nashville', 'AFC South', 'AFC', 'Nissan Stadium', 'Outdoor'),
                
                ('DEN', 'Denver Broncos', 'Denver', 'AFC West', 'AFC', 'Empower Field at Mile High', 'Outdoor'),
                ('KC', 'Kansas City Chiefs', 'Kansas City', 'AFC West', 'AFC', 'Arrowhead Stadium', 'Outdoor'),
                ('LV', 'Las Vegas Raiders', 'Las Vegas', 'AFC West', 'AFC', 'Allegiant Stadium', 'Dome'),
                ('LAC', 'Los Angeles Chargers', 'Los Angeles', 'AFC West', 'AFC', 'SoFi Stadium', 'Dome'),
                
                # NFC Teams
                ('DAL', 'Dallas Cowboys', 'Arlington', 'NFC East', 'NFC', 'AT&T Stadium', 'Retractable Roof'),
                ('NYG', 'New York Giants', 'East Rutherford', 'NFC East', 'NFC', 'MetLife Stadium', 'Outdoor'),
                ('PHI', 'Philadelphia Eagles', 'Philadelphia', 'NFC East', 'NFC', 'Lincoln Financial Field', 'Outdoor'),
                ('WAS', 'Washington Commanders', 'Landover', 'NFC East', 'NFC', 'FedExField', 'Outdoor'),
                
                ('CHI', 'Chicago Bears', 'Chicago', 'NFC North', 'NFC', 'Soldier Field', 'Outdoor'),
                ('DET', 'Detroit Lions', 'Detroit', 'NFC North', 'NFC', 'Ford Field', 'Dome'),
                ('GB', 'Green Bay Packers', 'Green Bay', 'NFC North', 'NFC', 'Lambeau Field', 'Outdoor'),
                ('MIN', 'Minnesota Vikings', 'Minneapolis', 'NFC North', 'NFC', 'U.S. Bank Stadium', 'Dome'),
                
                ('ATL', 'Atlanta Falcons', 'Atlanta', 'NFC South', 'NFC', 'Mercedes-Benz Stadium', 'Retractable Roof'),
                ('CAR', 'Carolina Panthers', 'Charlotte', 'NFC South', 'NFC', 'Bank of America Stadium', 'Outdoor'),
                ('NO', 'New Orleans Saints', 'New Orleans', 'NFC South', 'NFC', 'Caesars Superdome', 'Dome'),
                ('TB', 'Tampa Bay Buccaneers', 'Tampa', 'NFC South', 'NFC', 'Raymond James Stadium', 'Outdoor'),
                
                ('ARI', 'Arizona Cardinals', 'Glendale', 'NFC West', 'NFC', 'State Farm Stadium', 'Retractable Roof'),
                ('LAR', 'Los Angeles Rams', 'Los Angeles', 'NFC West', 'NFC', 'SoFi Stadium', 'Dome'),
                ('SF', 'San Francisco 49ers', 'Santa Clara', 'NFC West', 'NFC', 'Levi\'s Stadium', 'Outdoor'),
                ('SEA', 'Seattle Seahawks', 'Seattle', 'NFC West', 'NFC', 'Lumen Field', 'Outdoor'),
            ]
            
            cursor.execute("DELETE FROM nfl_teams")
            
            for team in nfl_teams_2024:
                cursor.execute('''
                    INSERT INTO nfl_teams (id, name, city, division, conference, home_stadium, stadium_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', team)
                
            conn.commit()
            print(f"✅ Updated {len(nfl_teams_2024)} NFL teams for 2024 season")
            
            # 2. Update NFL players with current 2024 rosters (key players)
            print("📥 Updating key NFL players for 2024 season...")
            
            # Top NFL players by position for 2024 season
            nfl_players_2024 = [
                # Star Quarterbacks (current 2024 starters)
                ('Josh Allen', 'QB', 'BUF', 17, 1, '6-5', 245, 'Wyoming', 7),
                ('Tua Tagovailoa', 'QB', 'MIA', 1, 1, '6-1', 217, 'Alabama', 5),
                ('Aaron Rodgers', 'QB', 'NYJ', 8, 1, '6-2', 225, 'California', 20),
                ('Drake Maye', 'QB', 'NE', 10, 1, '6-4', 230, 'North Carolina', 1),
                
                ('Lamar Jackson', 'QB', 'BAL', 8, 1, '6-2', 212, 'Louisville', 7),
                ('Joe Burrow', 'QB', 'CIN', 9, 1, '6-4', 221, 'LSU', 5),
                ('Deshaun Watson', 'QB', 'CLE', 4, 1, '6-2', 215, 'Clemson', 8),
                ('Russell Wilson', 'QB', 'PIT', 3, 1, '5-11', 215, 'Wisconsin', 13),
                
                ('C.J. Stroud', 'QB', 'HOU', 7, 1, '6-3', 218, 'Ohio State', 2),
                ('Anthony Richardson', 'QB', 'IND', 5, 1, '6-4', 244, 'Florida', 2),
                ('Trevor Lawrence', 'QB', 'JAX', 16, 1, '6-6', 213, 'Clemson', 4),
                ('Will Levis', 'QB', 'TEN', 8, 1, '6-3', 229, 'Kentucky', 2),
                
                ('Patrick Mahomes', 'QB', 'KC', 15, 1, '6-3', 225, 'Texas Tech', 8),
                ('Bo Nix', 'QB', 'DEN', 10, 1, '6-2', 217, 'Oregon', 1),
                ('Justin Herbert', 'QB', 'LAC', 10, 1, '6-6', 236, 'Oregon', 5),
                ('Gardner Minshew', 'QB', 'LV', 15, 1, '6-1', 225, 'Washington State', 6),
                
                ('Dak Prescott', 'QB', 'DAL', 4, 1, '6-2', 238, 'Mississippi State', 9),
                ('Daniel Jones', 'QB', 'NYG', 8, 1, '6-5', 230, 'Duke', 6),
                ('Jalen Hurts', 'QB', 'PHI', 1, 1, '6-1', 223, 'Oklahoma', 4),
                ('Jayden Daniels', 'QB', 'WAS', 5, 1, '6-4', 210, 'LSU', 1),
                
                ('Caleb Williams', 'QB', 'CHI', 18, 1, '6-1', 214, 'USC', 1),
                ('Jared Goff', 'QB', 'DET', 16, 1, '6-4', 217, 'California', 9),
                ('Jordan Love', 'QB', 'GB', 10, 1, '6-4', 224, 'Utah State', 5),
                ('Sam Darnold', 'QB', 'MIN', 14, 1, '6-3', 225, 'USC', 7),
                
                ('Kirk Cousins', 'QB', 'ATL', 18, 1, '6-3', 202, 'Michigan State', 13),
                ('Bryce Young', 'QB', 'CAR', 9, 1, '5-10', 204, 'Alabama', 2),
                ('Derek Carr', 'QB', 'NO', 4, 1, '6-3', 215, 'Fresno State', 11),
                ('Baker Mayfield', 'QB', 'TB', 6, 1, '6-1', 215, 'Oklahoma', 7),
                
                ('Kyler Murray', 'QB', 'ARI', 1, 1, '5-10', 207, 'Oklahoma', 6),
                ('Matthew Stafford', 'QB', 'LAR', 9, 1, '6-3', 220, 'Georgia', 16),
                ('Brock Purdy', 'QB', 'SF', 13, 1, '6-1', 220, 'Iowa State', 3),
                ('Geno Smith', 'QB', 'SEA', 7, 1, '6-3', 221, 'West Virginia', 14),
                
                # Elite Running Backs
                ('Christian McCaffrey', 'RB', 'SF', 23, 1, '5-11', 205, 'Stanford', 8),
                ('Josh Jacobs', 'RB', 'GB', 8, 1, '5-10', 220, 'Alabama', 6),
                ('Saquon Barkley', 'RB', 'PHI', 26, 1, '6-0', 233, 'Penn State', 7),
                ('Derrick Henry', 'RB', 'BAL', 22, 1, '6-3', 247, 'Alabama', 9),
                ('Jonathan Taylor', 'RB', 'IND', 28, 1, '5-10', 226, 'Wisconsin', 4),
                ('Alvin Kamara', 'RB', 'NO', 41, 1, '5-10', 215, 'Tennessee', 8),
                ('Nick Chubb', 'RB', 'CLE', 24, 1, '5-11', 227, 'Georgia', 7),
                ('Kenneth Walker III', 'RB', 'SEA', 9, 1, '5-9', 211, 'Michigan State', 3),
                ('Breece Hall', 'RB', 'NYJ', 20, 1, '5-11', 217, 'Iowa State', 3),
                ('Bijan Robinson', 'RB', 'ATL', 7, 1, '6-0', 215, 'Texas', 2),
                
                # Top Wide Receivers
                ('Tyreek Hill', 'WR', 'MIA', 10, 1, '5-10', 185, 'West Alabama', 9),
                ('Davante Adams', 'WR', 'LV', 17, 1, '6-1', 215, 'Fresno State', 11),
                ('Stefon Diggs', 'WR', 'HOU', 1, 1, '6-0', 191, 'Maryland', 10),
                ('A.J. Brown', 'WR', 'PHI', 11, 1, '6-1', 226, 'Ole Miss', 6),
                ('CeeDee Lamb', 'WR', 'DAL', 88, 1, '6-2', 198, 'Oklahoma', 5),
                ('Ja\'Marr Chase', 'WR', 'CIN', 1, 1, '6-0', 201, 'LSU', 4),
                ('Justin Jefferson', 'WR', 'MIN', 18, 1, '6-1', 202, 'LSU', 5),
                ('Amon-Ra St. Brown', 'WR', 'DET', 14, 1, '5-11', 197, 'USC', 4),
                ('DK Metcalf', 'WR', 'SEA', 14, 1, '6-4', 229, 'Ole Miss', 6),
                ('Mike Evans', 'WR', 'TB', 13, 1, '6-5', 231, 'Texas A&M', 11),
                
                # Elite Tight Ends
                ('Travis Kelce', 'TE', 'KC', 87, 1, '6-5', 250, 'Cincinnati', 12),
                ('Mark Andrews', 'TE', 'BAL', 89, 1, '6-5', 256, 'Oklahoma', 7),
                ('T.J. Hockenson', 'TE', 'MIN', 87, 1, '6-5', 246, 'Iowa', 6),
                ('George Kittle', 'TE', 'SF', 85, 1, '6-4', 250, 'Iowa', 8),
                ('Evan Engram', 'TE', 'JAX', 17, 1, '6-3', 240, 'Ole Miss', 8),
                ('Kyle Pitts', 'TE', 'ATL', 8, 1, '6-6', 246, 'Florida', 4),
                
                # Top Kickers
                ('Justin Tucker', 'K', 'BAL', 9, 1, '6-1', 183, 'Texas', 13),
                ('Harrison Butker', 'K', 'KC', 7, 1, '6-4', 196, 'Georgia Tech', 8),
                ('Tyler Bass', 'K', 'BUF', 2, 1, '5-10', 183, 'Georgia Southern', 5),
                ('Chris Boswell', 'K', 'PIT', 9, 1, '6-2', 185, 'Rice', 9),
                ('Jake Elliott', 'K', 'PHI', 4, 1, '5-9', 167, 'Memphis', 8),
            ]
            
            cursor.execute("DELETE FROM nfl_players")
            
            for player in nfl_players_2024:
                cursor.execute('''
                    INSERT INTO nfl_players (name, position, team, jersey_number, is_active, height, weight, college, years_pro)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', player)
                
            conn.commit()
            print(f"✅ Updated {len(nfl_players_2024)} key NFL players for 2024 season")
            
            # 3. Clear old NFL data for fresh start
            cursor.execute("DELETE FROM nfl_game_logs")
            cursor.execute("DELETE FROM nfl_weather WHERE game_date < '2024-09-01'")  # Keep recent weather
            
            conn.commit()
            print("✅ Cleared old NFL analysis data for fresh start")
            
        except Exception as e:
            print(f"❌ Error refreshing NFL data: {e}")
            conn.rollback()
        finally:
            conn.close()
            
    def verify_refresh(self):
        """Verify the database refresh"""
        print("\n🔍 Verifying database refresh...")
        
        # NBA verification
        nba_conn = sqlite3.connect(self.nba_db_path)
        nba_cursor = nba_conn.cursor()
        
        print("🏀 NBA Database Status:")
        nba_tables = ['players', 'nba_teams', 'games', 'game_logs']
        for table in nba_tables:
            try:
                nba_cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = nba_cursor.fetchone()[0]
                print(f"  {table}: {count:,} records")
                
                if table == 'players' and count > 0:
                    nba_cursor.execute('SELECT full_name FROM players ORDER BY id LIMIT 5')
                    sample_players = nba_cursor.fetchall()
                    print(f"    Sample: {[p[0] for p in sample_players]}")
                    
            except Exception as e:
                print(f"  {table}: Error - {e}")
        nba_conn.close()
        
        # NFL verification
        nfl_conn = sqlite3.connect(self.nfl_db_path)
        nfl_cursor = nfl_conn.cursor()
        
        print("🏈 NFL Database Status:")
        nfl_tables = ['nfl_teams', 'nfl_players', 'nfl_game_logs']
        for table in nfl_tables:
            try:
                nfl_cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = nfl_cursor.fetchone()[0]
                print(f"  {table}: {count:,} records")
                
                if table == 'nfl_players' and count > 0:
                    nfl_cursor.execute('SELECT name, position, team FROM nfl_players WHERE position = "QB" LIMIT 5')
                    sample_qbs = nfl_cursor.fetchall()
                    print(f"    Sample QBs: {[(p[0], p[2]) for p in sample_qbs]}")
                    
            except Exception as e:
                print(f"  {table}: Error - {e}")
        nfl_conn.close()
        
    def run_refresh(self):
        """Run the complete database refresh"""
        print("🔄 Starting database refresh for current seasons...")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏀 Target NBA Season: {self.nba_season}")
        print(f"🏈 Target NFL Season: {self.nfl_season}")
        print("="*50)
        
        self.refresh_nba_data()
        print()
        self.refresh_nfl_data()
        print()
        self.verify_refresh()
        
        print(f"\n✅ Database refresh completed!")
        print(f"📊 Ready for {self.nba_season} NBA and {self.nfl_season} NFL season analysis")

if __name__ == "__main__":
    refresher = DatabaseRefresher()
    refresher.run_refresh()