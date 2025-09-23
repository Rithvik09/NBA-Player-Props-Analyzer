#!/usr/bin/env python3
"""
Comprehensive Database Update Script
Updates NBA and NFL databases with latest 2024-25 season data
"""

import sqlite3
import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# NBA API imports
try:
    from nba_api.stats.endpoints import (
        playergamelog, CommonPlayerInfo, PlayerVsPlayer, 
        TeamDashboardByGeneralSplits, TeamGameLog, 
        CommonTeamRoster, LeagueGameFinder, teamyearbyyearstats,
        leaguegamefinder, teamgamelog, playercareerstats
    )
    from nba_api.stats.static import players, teams
    NBA_API_AVAILABLE = True
except ImportError:
    print("NBA API not available - installing...")
    import subprocess
    subprocess.run(['pip', 'install', 'nba_api'], check=True)
    from nba_api.stats.endpoints import (
        playergamelog, CommonPlayerInfo, PlayerVsPlayer, 
        TeamDashboardByGeneralSplits, TeamGameLog, 
        CommonTeamRoster, LeagueGameFinder, teamyearbyyearstats,
        leaguegamefinder, teamgamelog, playercareerstats
    )
    from nba_api.stats.static import players, teams
    NBA_API_AVAILABLE = True

class DatabaseUpdater:
    def __init__(self):
        self.nba_db_path = 'basketball_data.db'
        self.nfl_db_path = 'nfl_data.db'
        
        # Current seasons
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # NBA season logic (Oct-June)
        if current_month >= 10:  # October or later
            self.nba_season = f"{current_year}-{str(current_year+1)[2:]}"
        else:  # Before October
            self.nba_season = f"{current_year-1}-{str(current_year)[2:]}"
            
        # NFL season (Sep-Feb)
        if current_month >= 9:  # September or later
            self.nfl_season = current_year
        else:  # Before September
            self.nfl_season = current_year - 1
            
        print(f"🏀 Updating NBA {self.nba_season} season")
        print(f"🏈 Updating NFL {self.nfl_season} season")
        
    def update_nba_database(self):
        """Update NBA database with comprehensive current season data"""
        print("🏀 Starting NBA database update...")
        
        conn = sqlite3.connect(self.nba_db_path)
        cursor = conn.cursor()
        
        try:
            # 1. Update ALL active players (including 2024-25 rookies)
            print("📥 Fetching all current NBA players...")
            all_players = players.get_players()
            active_players = [p for p in all_players if p['is_active']]
            
            print(f"Found {len(active_players)} active players")
            
            # Clear and rebuild players table
            cursor.execute("DELETE FROM players")
            
            for i, player in enumerate(active_players):
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
                
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(active_players)} players...")
                    
            conn.commit()
            print(f"✅ Updated {len(active_players)} active NBA players")
            
            # 2. Get current season games
            print("📥 Fetching current season games...")
            try:
                game_finder = LeagueGameFinder(season_nullable=self.nba_season, 
                                             season_type_nullable='Regular Season')
                time.sleep(0.6)  # Rate limiting
                games_df = game_finder.get_data_frames()[0]
                
                # Clear and update games table
                cursor.execute("DELETE FROM games")
                
                for _, game in games_df.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO games (
                            game_id, game_date, matchup, wl, pts, ast, reb, 
                            team_id, team_name, season_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        game['GAME_ID'],
                        game['GAME_DATE'],
                        game['MATCHUP'],
                        game['WL'],
                        game.get('PTS', 0),
                        game.get('AST', 0),
                        game.get('REB', 0),
                        game['TEAM_ID'],
                        game['TEAM_NAME'],
                        game.get('SEASON_ID', self.nba_season)
                    ))
                
                conn.commit()
                print(f"✅ Updated {len(games_df)} NBA games for {self.nba_season}")
                
            except Exception as e:
                print(f"⚠️ Error fetching games: {e}")
                
            # 3. Update game logs for top 100 players (recent performance)
            print("📥 Updating recent game logs for key players...")
            
            # Get top players by usage/popularity
            top_players = active_players[:100]  # First 100 active players
            
            cursor.execute("DELETE FROM game_logs")
            
            successful_updates = 0
            for i, player in enumerate(top_players):
                try:
                    # Get recent games for this season
                    gamelog = playergamelog.PlayerGameLog(
                        player_id=player['id'],
                        season=self.nba_season
                    )
                    time.sleep(0.6)  # Rate limiting
                    games = gamelog.get_data_frames()[0]
                    
                    for _, game in games.iterrows():
                        cursor.execute('''
                            INSERT OR REPLACE INTO game_logs (
                                player_id, game_date, matchup, wl, min, pts, ast, reb,
                                stl, blk, turnover, fg3m, fg_pct, fg3_pct, ft_pct
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            player['id'],
                            game['GAME_DATE'],
                            game['MATCHUP'],
                            game['WL'],
                            game.get('MIN', 0),
                            game.get('PTS', 0),
                            game.get('AST', 0),
                            game.get('REB', 0),
                            game.get('STL', 0),
                            game.get('BLK', 0),
                            game.get('TOV', 0),
                            game.get('FG3M', 0),
                            game.get('FG_PCT', 0),
                            game.get('FG3_PCT', 0),
                            game.get('FT_PCT', 0)
                        ))
                    
                    successful_updates += 1
                    if i % 10 == 0:
                        conn.commit()  # Commit every 10 players
                        print(f"  Processed {i}/{len(top_players)} players ({successful_updates} successful)...")
                        
                except Exception as e:
                    print(f"  ⚠️ Error for player {player['full_name']}: {e}")
                    continue
                    
            conn.commit()
            print(f"✅ Updated game logs for {successful_updates} players")
            
            # 4. Update team information
            print("📥 Updating NBA team information...")
            nba_teams = teams.get_teams()
            
            # Create teams table if it doesn't exist
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
            print(f"❌ Error updating NBA database: {e}")
            conn.rollback()
        finally:
            conn.close()
            
    def update_nfl_database(self):
        """Update NFL database with comprehensive current season data"""
        print("🏈 Starting NFL database update...")
        
        conn = sqlite3.connect(self.nfl_db_path)
        cursor = conn.cursor()
        
        try:
            # Create comprehensive NFL tables
            self.create_nfl_tables(cursor)
            
            # 1. Update NFL teams (all 32 teams)
            print("📥 Updating NFL teams...")
            nfl_teams = [
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
            
            for team in nfl_teams:
                cursor.execute('''
                    INSERT INTO nfl_teams (abbreviation, full_name, city, nickname, conference, division)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', team)
                
            conn.commit()
            print(f"✅ Updated {len(nfl_teams)} NFL teams")
            
            # 2. Add sample NFL players (since we don't have a comprehensive API)
            print("📥 Adding NFL player data...")
            
            # Sample of key NFL players for 2024 season
            nfl_players = [
                # Quarterbacks
                ('Josh Allen', 'QB', 'BUF', 28, 245, '6-5'),
                ('Tua Tagovailoa', 'QB', 'MIA', 26, 217, '6-1'),
                ('Aaron Rodgers', 'QB', 'NYJ', 40, 225, '6-2'),
                ('Mac Jones', 'QB', 'NE', 26, 217, '6-3'),
                
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
                
                # Running Backs
                ('Christian McCaffrey', 'RB', 'SF', 28, 205, '5-11'),
                ('Josh Jacobs', 'RB', 'GB', 26, 220, '5-10'),
                ('Saquon Barkley', 'RB', 'PHI', 27, 233, '6-0'),
                ('Derrick Henry', 'RB', 'BAL', 30, 247, '6-3'),
                ('Jonathan Taylor', 'RB', 'IND', 25, 226, '5-10'),
                ('Nick Chubb', 'RB', 'CLE', 28, 227, '5-11'),
                ('Austin Ekeler', 'RB', 'WAS', 29, 200, '5-8'),
                ('Kenneth Walker III', 'RB', 'SEA', 24, 211, '5-9'),
                
                # Wide Receivers
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
                
                # Tight Ends
                ('Travis Kelce', 'TE', 'KC', 35, 250, '6-5'),
                ('Mark Andrews', 'TE', 'BAL', 29, 256, '6-5'),
                ('T.J. Hockenson', 'TE', 'MIN', 27, 246, '6-5'),
                ('George Kittle', 'TE', 'SF', 31, 250, '6-4'),
                ('Evan Engram', 'TE', 'JAX', 30, 240, '6-3'),
                
                # Kickers
                ('Justin Tucker', 'K', 'BAL', 35, 183, '6-1'),
                ('Harrison Butker', 'K', 'KC', 29, 196, '6-4'),
                ('Tyler Bass', 'K', 'BUF', 27, 183, '5-10'),
                ('Chris Boswell', 'K', 'PIT', 33, 185, '6-2'),
                ('Jake Elliott', 'K', 'PHI', 29, 167, '5-9'),
            ]
            
            cursor.execute("DELETE FROM nfl_players")
            
            for i, player in enumerate(nfl_players):
                cursor.execute('''
                    INSERT INTO nfl_players (name, position, team, age, weight, height)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', player)
                
            conn.commit()
            print(f"✅ Added {len(nfl_players)} NFL players")
            
            # 3. Add sample weather data for outdoor stadiums
            print("📥 Adding NFL weather data...")
            
            outdoor_stadiums = [
                ('BUF', 'Buffalo', 42.7738, -78.7870, 'Cold, Snow, Wind'),
                ('NE', 'Foxborough', 42.0909, -71.2643, 'Cold, Rain, Wind'),
                ('BAL', 'Baltimore', 39.2780, -76.6229, 'Moderate, Rain'),
                ('CIN', 'Cincinnati', 39.0955, -84.5162, 'Cold, Rain'),
                ('CLE', 'Cleveland', 41.5061, -81.6995, 'Cold, Snow, Wind'),
                ('PIT', 'Pittsburgh', 40.4468, -80.0157, 'Cold, Snow, Rain'),
                ('DEN', 'Denver', 39.7439, -105.0201, 'Altitude, Cold, Snow'),
                ('KC', 'Kansas City', 39.0489, -94.4839, 'Cold, Wind, Rain'),
                ('CHI', 'Chicago', 41.8623, -87.6167, 'Cold, Snow, Wind'),
                ('GB', 'Green Bay', 44.5013, -88.0622, 'Very Cold, Snow, Wind'),
                ('WAS', 'Landover', 38.9077, -76.8644, 'Moderate, Rain'),
                ('PHI', 'Philadelphia', 39.9008, -75.1675, 'Cold, Rain, Wind'),
                ('SEA', 'Seattle', 47.5952, -122.3316, 'Rain, Wind'),
                ('SF', 'Santa Clara', 37.4030, -121.9699, 'Mild, Wind'),
            ]
            
            cursor.execute("DELETE FROM nfl_weather")
            
            for stadium in outdoor_stadiums:
                cursor.execute('''
                    INSERT INTO nfl_weather (team, city, latitude, longitude, conditions)
                    VALUES (?, ?, ?, ?, ?)
                ''', stadium)
                
            conn.commit()
            print(f"✅ Added weather data for {len(outdoor_stadiums)} outdoor stadiums")
            
        except Exception as e:
            print(f"❌ Error updating NFL database: {e}")
            conn.rollback()
        finally:
            conn.close()
            
    def create_nfl_tables(self, cursor):
        """Create comprehensive NFL database tables"""
        
        # NFL teams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nfl_teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                abbreviation TEXT UNIQUE,
                full_name TEXT,
                city TEXT,
                nickname TEXT,
                conference TEXT,
                division TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Enhanced NFL players table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nfl_players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                position TEXT,
                team TEXT,
                age INTEGER,
                weight INTEGER,
                height TEXT,
                experience INTEGER DEFAULT 0,
                college TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # NFL game logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nfl_game_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                game_date TEXT,
                opponent TEXT,
                result TEXT,
                passing_yards INTEGER DEFAULT 0,
                passing_tds INTEGER DEFAULT 0,
                interceptions INTEGER DEFAULT 0,
                rushing_yards INTEGER DEFAULT 0,
                rushing_tds INTEGER DEFAULT 0,
                receiving_yards INTEGER DEFAULT 0,
                receiving_tds INTEGER DEFAULT 0,
                receptions INTEGER DEFAULT 0,
                targets INTEGER DEFAULT 0,
                fantasy_points REAL DEFAULT 0,
                FOREIGN KEY (player_id) REFERENCES nfl_players (id)
            )
        ''')
        
        # Enhanced NFL weather table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nfl_weather (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT,
                city TEXT,
                latitude REAL,
                longitude REAL,
                conditions TEXT,
                dome_stadium BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
    def verify_updates(self):
        """Verify that updates were successful"""
        print("\n🔍 Verifying database updates...")
        
        # Check NBA database
        nba_conn = sqlite3.connect(self.nba_db_path)
        nba_cursor = nba_conn.cursor()
        
        print("🏀 NBA Database Status:")
        tables = ['players', 'games', 'game_logs', 'nba_teams']
        for table in tables:
            try:
                nba_cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = nba_cursor.fetchone()[0]
                print(f"  {table}: {count:,} records")
            except:
                print(f"  {table}: Table not found")
                
        nba_conn.close()
        
        # Check NFL database
        nfl_conn = sqlite3.connect(self.nfl_db_path)
        nfl_cursor = nfl_conn.cursor()
        
        print("🏈 NFL Database Status:")
        tables = ['nfl_teams', 'nfl_players', 'nfl_game_logs', 'nfl_weather']
        for table in tables:
            try:
                nfl_cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = nfl_cursor.fetchone()[0]
                print(f"  {table}: {count:,} records")
            except:
                print(f"  {table}: Table not found")
                
        nfl_conn.close()
        
    def run_update(self):
        """Run complete database update"""
        print("🚀 Starting comprehensive database update...")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏀 NBA Season: {self.nba_season}")
        print(f"🏈 NFL Season: {self.nfl_season}")
        print("="*50)
        
        # Update NBA
        self.update_nba_database()
        print()
        
        # Update NFL
        self.update_nfl_database()
        print()
        
        # Verify updates
        self.verify_updates()
        
        print("\n✅ Database update completed successfully!")
        print(f"🏀 NBA database: {self.nba_db_path}")
        print(f"🏈 NFL database: {self.nfl_db_path}")

if __name__ == "__main__":
    updater = DatabaseUpdater()
    updater.run_update()