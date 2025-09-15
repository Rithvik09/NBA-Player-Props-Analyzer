#!/usr/bin/env python3
"""
Test NFL Player Search Enhancement
Tests the expanded NFL player database with 320+ players and enhanced search functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from nfl_betting_helper import NFLBettingHelper
import json

def test_nfl_player_search():
    """Test the enhanced NFL player search system"""
    print("🏈 Testing Enhanced NFL Player Search System")
    print("=" * 60)
    
    # Initialize NFL helper
    nfl_helper = NFLBettingHelper()
    
    # Test 1: Database size
    print("\n1. 📊 Testing Database Coverage")
    all_players = nfl_helper._get_comprehensive_nfl_players()
    print(f"   Total NFL players in database: {len(all_players)}")
    
    # Count by position
    positions = {}
    teams = {}
    for player in all_players:
        pos = player.get('position', 'UNK')
        team = player.get('team', 'UNK')
        positions[pos] = positions.get(pos, 0) + 1
        teams[team] = teams.get(team, 0) + 1
    
    print(f"   Positions covered: {len(positions)}")
    print(f"   Teams covered: {len(teams)}")
    print(f"   Position breakdown: {dict(sorted(positions.items(), key=lambda x: x[1], reverse=True))}")
    
    # Test 2: Search functionality
    print("\n2. 🔍 Testing Search Functionality")
    
    test_queries = [
        "Mahomes",
        "Josh Allen", 
        "Tyreek",
        "CeeDee",
        "Travis Kelce",
        "Lamar",
        "Brock Purdy",
        "Joe Burrow",
        "Justin Herbert",
        "Christian McCaffrey",
        "Derrick Henry",
        "Cooper Kupp",
        "Davante Adams"
    ]
    
    for query in test_queries:
        matches = nfl_helper.get_player_suggestions(query)
        print(f"   Query: '{query}' -> {len(matches)} matches")
        if matches:
            top_match = matches[0]
            print(f"      Top: {top_match['full_name']} ({top_match['position']}, {top_match['team']}) [Score: {top_match.get('match_score', 'N/A')}]")
    
    # Test 3: Team-based searches  
    print("\n3. 🏟️ Testing Team-Based Search")
    
    test_teams = ['KC', 'BUF', 'SF', 'BAL', 'DET']
    for team in test_teams:
        team_players = nfl_helper.get_players_by_team(team)
        print(f"   Team {team}: {len(team_players)} players")
        if team_players:
            print(f"      Sample: {', '.join([p['full_name'] for p in team_players[:3]])}")
    
    # Test 4: Position-based searches
    print("\n4. 🎯 Testing Position-Based Search")
    
    test_positions = ['QB', 'RB', 'WR', 'TE']
    for pos in test_positions:
        pos_players = nfl_helper.get_players_by_position(pos)
        print(f"   Position {pos}: {len(pos_players)} players")
    
    # Test 5: Advanced search with filters
    print("\n5. 🎛️ Testing Advanced Search with Filters")
    
    # Search for Mahomes with team filter
    mahomes_search = nfl_helper.search_players_advanced("Mahomes", {"team": "KC"})
    print(f"   'Mahomes' + team KC filter: {len(mahomes_search)} matches")
    
    # Search QBs with name filter
    qb_josh = nfl_helper.search_players_advanced("Josh", {"position": "QB"})
    print(f"   'Josh' + position QB filter: {len(qb_josh)} matches")
    
    # Test 6: Edge cases and partial matches
    print("\n6. 🔍 Testing Edge Cases")
    
    edge_cases = [
        "chubb",  # Should find Nick Chubb
        "mccaf",  # Should find McCaffrey
        "diggs",  # Should find Stefon Diggs (now with HOU)
        "a.j.",   # Should find A.J. Brown
        "cj",     # Should find C.J. Stroud
        "dk",     # Should find DK Metcalf
        "jj"      # Should find Justin Jefferson
    ]
    
    for query in edge_cases:
        matches = nfl_helper.get_player_suggestions(query)
        print(f"   Query: '{query}' -> {len(matches)} matches")
        if matches:
            print(f"      Top: {matches[0]['full_name']} ({matches[0]['position']}, {matches[0]['team']})")
    
    # Test 7: Team overview
    print("\n7. 🏈 Team Overview")
    all_teams = nfl_helper.get_all_teams()
    print(f"   Total teams: {len(all_teams)}")
    
    # Show teams with most/least players
    sorted_teams = sorted(all_teams, key=lambda x: x['player_count'], reverse=True)
    print(f"   Most players: {sorted_teams[0]['name']} ({sorted_teams[0]['player_count']} players)")
    print(f"   Least players: {sorted_teams[-1]['name']} ({sorted_teams[-1]['player_count']} players)")
    
    print("\n8. ✅ Testing Popular Players")
    
    # Test searches for very popular players that users will likely search for
    popular_players = [
        "Patrick Mahomes",
        "Josh Allen", 
        "Lamar Jackson",
        "Joe Burrow",
        "Tua Tagovailoa",
        "Aaron Rodgers",
        "Dak Prescott",
        "Russell Wilson",
        "Kyler Murray",
        "Justin Herbert",
        "Christian McCaffrey",
        "Saquon Barkley",
        "Derrick Henry",
        "Josh Jacobs",
        "Tyreek Hill",
        "Stefon Diggs",
        "Davante Adams",
        "Cooper Kupp",
        "Travis Kelce",
        "Mark Andrews"
    ]
    
    missing_players = []
    for player_name in popular_players:
        matches = nfl_helper.get_player_suggestions(player_name)
        found = any(match['full_name'].lower() == player_name.lower() for match in matches)
        if not found:
            missing_players.append(player_name)
        else:
            print(f"   ✅ Found: {player_name}")
    
    if missing_players:
        print(f"\n   ❌ Missing popular players: {missing_players}")
    else:
        print("   🎉 All popular players found!")
    
    print(f"\n🏈 NFL Player Search Enhancement Complete!")
    print(f"📊 Database now contains {len(all_players)} players across {len(teams)} teams")
    print(f"🔍 Enhanced search with match scoring and multiple search strategies")
    print(f"🎯 Position filtering, team filtering, and advanced search capabilities")
    return True

if __name__ == "__main__":
    test_nfl_player_search()