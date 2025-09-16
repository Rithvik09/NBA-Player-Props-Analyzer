#!/usr/bin/env python3
"""
Replace the NFL database method in the helper file
"""

import re

def replace_nfl_database():
    """Replace the NFL database with the verified accurate one"""
    
    # Read the current file
    with open('src/nfl_betting_helper.py', 'r') as f:
        content = f.read()
    
    # Read the new database content
    with open('nfl_database_replacement.txt', 'r') as f:
        new_database = f.read()
    
    # Find the start and end of the current database
    start_pattern = r'(\s+)(def _get_comprehensive_nfl_players\(self\):.*?""".*?"""\s+)(return \[)'
    end_pattern = r'(\s+\])'
    
    # Find the method start
    method_match = re.search(start_pattern, content, re.DOTALL)
    if not method_match:
        print("❌ Could not find the method start")
        return False
    
    method_start = method_match.start()
    database_start = method_match.end() - len('return [')
    
    # Find the end of the database (the closing bracket)
    # Look for the next occurrence of ']' that ends the method
    remaining_content = content[database_start:]
    
    # Count brackets to find the matching closing bracket
    bracket_count = 0
    pos = 0
    found_end = False
    
    for i, char in enumerate(remaining_content):
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0:
                database_end = database_start + i + 1
                found_end = True
                break
    
    if not found_end:
        print("❌ Could not find the database end")
        return False
    
    # Replace the database section
    new_content = (
        content[:database_start] +
        new_database +
        content[database_end:]
    )
    
    # Write the updated content
    with open('src/nfl_betting_helper.py', 'w') as f:
        f.write(new_content)
    
    print("✅ Successfully replaced NFL database")
    print(f"   Replaced {database_end - database_start} characters with {len(new_database)} characters")
    
    return True

if __name__ == "__main__":
    success = replace_nfl_database()
    if success:
        print("🏈 NFL database replacement complete!")
    else:
        print("❌ Database replacement failed!")