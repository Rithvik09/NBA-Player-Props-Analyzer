#!/usr/bin/env python3
"""
Test Database Update and Functionality
"""

import sqlite3
import sys
import os
sys.path.append('src')

def test_database_status():
    """Test current database status and functionality"""
    
    print("🔍 Testing Database Status and Functionality")
    print("="*50)
    
    # Test NBA Database
    print("🏀 NBA Database Test:")
    try:
        nba_conn = sqlite3.connect('basketball_data.db')
        nba_cursor = nba_conn.cursor()
        
        # Check players
        nba_cursor.execute('SELECT COUNT(*) FROM players')
        player_count = nba_cursor.fetchone()[0]
        print(f"  Total players: {player_count:,}")
        
        # Test LeBron search
        nba_cursor.execute('SELECT id, full_name FROM players WHERE full_name LIKE "%LeBron%" OR full_name LIKE "%lebron%"')
        lebron_results = nba_cursor.fetchall()
        print(f"  LeBron James found: {lebron_results}")
        
        # Test other star players
        stars = ['Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo', 'Luka Doncic']
        for star in stars:
            nba_cursor.execute('SELECT id, full_name FROM players WHERE full_name LIKE ?', (f'%{star}%',))
            result = nba_cursor.fetchone()
            print(f"  {star}: {'✅ Found' if result else '❌ Not found'}")
        
        # Check games table
        nba_cursor.execute('SELECT COUNT(*) FROM games')
        games_count = nba_cursor.fetchone()[0]
        print(f"  Total games: {games_count:,}")
        
        # Check if NBA teams table exists
        nba_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nba_teams'")
        teams_table = nba_cursor.fetchone()
        if teams_table:
            nba_cursor.execute('SELECT COUNT(*) FROM nba_teams')
            teams_count = nba_cursor.fetchone()[0]
            print(f"  NBA teams: {teams_count}")
        else:
            print("  NBA teams table: ❌ Missing")
        
        nba_conn.close()
        
    except Exception as e:
        print(f"  ❌ NBA database error: {e}")
    
    print()
    
    # Test NFL Database
    print("🏈 NFL Database Test:")
    try:
        nfl_conn = sqlite3.connect('nfl_data.db')
        nfl_cursor = nfl_conn.cursor()
        
        # Check NFL teams
        nfl_cursor.execute('SELECT COUNT(*) FROM nfl_teams')
        nfl_teams_count = nfl_cursor.fetchone()[0]
        print(f"  NFL teams: {nfl_teams_count}")
        
        if nfl_teams_count > 0:
            nfl_cursor.execute('SELECT name FROM nfl_teams LIMIT 5')
            sample_teams = nfl_cursor.fetchall()
            print(f"  Sample teams: {[t[0] for t in sample_teams]}")
        
        # Check NFL players
        nfl_cursor.execute('SELECT COUNT(*) FROM nfl_players')
        nfl_players_count = nfl_cursor.fetchone()[0]
        print(f"  NFL players: {nfl_players_count}")
        
        if nfl_players_count > 0:
            nfl_cursor.execute('SELECT name, position, team FROM nfl_players WHERE position = "QB" LIMIT 5')
            sample_qbs = nfl_cursor.fetchall()
            print(f"  Sample QBs: {[(p[0], p[2]) for p in sample_qbs]}")
        
        # Check enhanced tables
        nfl_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%enhanced%'")
        enhanced_tables = nfl_cursor.fetchall()
        print(f"  Enhanced tables: {[t[0] for t in enhanced_tables]}")
        
        nfl_conn.close()
        
    except Exception as e:
        print(f"  ❌ NFL database error: {e}")

def test_player_search_function():
    """Test the player search functionality directly"""
    
    print("\n🔍 Testing Player Search Function:")
    print("="*40)
    
    try:
        from basketball_betting_helper import BasketballBettingHelper
        
        helper = BasketballBettingHelper()
        
        # Test searches
        test_queries = ['lebron', 'curry', 'durant', 'giannis', 'luka']
        
        for query in test_queries:
            print(f"  Searching '{query}'...")
            try:
                results = helper.get_player_suggestions(query)
                if results:
                    print(f"    ✅ Found {len(results)} results")
                    for result in results[:3]:  # Show top 3
                        print(f"      - {result.get('full_name', 'Unknown')} (ID: {result.get('id', 'N/A')})")
                else:
                    print(f"    ❌ No results found")
            except Exception as e:
                print(f"    ❌ Search error: {e}")
        
    except Exception as e:
        print(f"  ❌ Import error: {e}")

def create_simple_nba_teams_table():
    """Create NBA teams table if missing"""
    
    print("\n🔧 Creating missing NBA teams table...")
    
    try:
        from nba_api.stats.static import teams
        
        nba_conn = sqlite3.connect('basketball_data.db')
        nba_cursor = nba_conn.cursor()
        
        # Create table
        nba_cursor.execute('''
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
        
        # Get teams and insert
        nba_teams = teams.get_teams()
        
        nba_cursor.execute("DELETE FROM nba_teams")
        
        for team in nba_teams:
            nba_cursor.execute('''
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
        
        nba_conn.commit()
        nba_conn.close()
        
        print(f"  ✅ Created NBA teams table with {len(nba_teams)} teams")
        
    except Exception as e:
        print(f"  ❌ Error creating teams table: {e}")

def test_api_endpoints():
    """Test if API endpoints are working"""
    
    print("\n🌐 Testing API Endpoints:")
    print("="*30)
    
    import requests
    
    base_url = "https://5000-ix0lcpvzl84z9cg4mk5td-6532622b.e2b.dev"
    
    endpoints = [
        ('/', 'Home page'),
        ('/search_players?q=james', 'Player search'),
        ('/nba/analyze_moneyline', 'NBA moneyline (POST)'),
    ]
    
    for endpoint, description in endpoints:
        try:
            if 'POST' in description:
                # Test POST endpoint
                response = requests.post(f"{base_url}{endpoint}", 
                                       json={'home_team': 'LAL', 'away_team': 'GSW', 'home_odds': -110, 'away_odds': -110},
                                       timeout=5)
            else:
                # Test GET endpoint
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                
            if response.status_code == 200:
                print(f"  ✅ {description}: Working")
                if 'search' in endpoint and response.text:
                    print(f"    Response length: {len(response.text)} chars")
            else:
                print(f"  ❌ {description}: Status {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ {description}: Error - {e}")

if __name__ == "__main__":
    print("🧪 Database Update and Functionality Test")
    print(f"📅 {os.popen('date').read().strip()}")
    print()
    
    # Run tests
    test_database_status()
    create_simple_nba_teams_table()
    test_player_search_function()
    test_api_endpoints()
    
    print(f"\n✅ Database test completed!")