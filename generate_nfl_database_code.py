#!/usr/bin/env python3
"""
Generate Python code for the NFL database replacement
"""

import json

def generate_database_code():
    """Generate the Python code for the NFL database"""
    
    # Load the verified database
    with open('nfl_database_update.json', 'r') as f:
        players = json.load(f)
    
    # Start building the code
    code_lines = []
    code_lines.append("        return [")
    
    # Add each player
    for player in players:
        player_id = player['id']
        full_name = player['full_name']
        position = player['position']
        team = player['team']
        is_active = str(player['is_active']).capitalize()
        
        # Add any additional info for rookies
        extra_info = ""
        if 'draft_year' in player and player['draft_year'] == 2025:
            if 'draft_pick' in player:
                if player['draft_pick'] <= 32:
                    extra_info = f"  # 2025 Draft Pick #{player['draft_pick']}"
                elif player['draft_pick'] == 144:
                    extra_info = f"  # 2025 Round 5, Pick #{player['draft_pick']}"
                else:
                    extra_info = f"  # 2025 Draft Pick #{player['draft_pick']}"
            else:
                extra_info = "  # 2025 Rookie"
        
        line = f"            {{'id': '{player_id}', 'full_name': '{full_name}', 'position': '{position}', 'team': '{team}', 'is_active': {is_active}}},{extra_info}"
        code_lines.append(line)
    
    code_lines.append("        ]")
    
    # Write to file
    with open('nfl_database_replacement.txt', 'w') as f:
        f.write('\n'.join(code_lines))
    
    print(f"✅ Generated NFL database code with {len(players)} players")
    print("📄 Code saved to nfl_database_replacement.txt")

if __name__ == "__main__":
    generate_database_code()