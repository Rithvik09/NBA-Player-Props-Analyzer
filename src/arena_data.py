"""
Arena and environmental data for NBA teams
"""

# Arena data: altitude (feet), capacity, timezone
ARENA_DATA = {
    1610612737: {"name": "State Farm Arena", "altitude": 1050, "capacity": 18118, "timezone": "America/New_York", "city": "Atlanta"},
    1610612738: {"name": "TD Garden", "altitude": 10, "capacity": 19156, "timezone": "America/New_York", "city": "Boston"},
    1610612751: {"name": "Barclays Center", "altitude": 10, "capacity": 17732, "timezone": "America/New_York", "city": "Brooklyn"},
    1610612766: {"name": "Spectrum Center", "altitude": 750, "capacity": 19077, "timezone": "America/New_York", "city": "Charlotte"},
    1610612741: {"name": "United Center", "altitude": 600, "capacity": 20917, "timezone": "America/Chicago", "city": "Chicago"},
    1610612739: {"name": "Rocket Mortgage FieldHouse", "altitude": 650, "capacity": 19432, "timezone": "America/New_York", "city": "Cleveland"},
    1610612742: {"name": "American Airlines Center", "altitude": 430, "capacity": 19200, "timezone": "America/Chicago", "city": "Dallas"},
    1610612743: {"name": "Ball Arena", "altitude": 5280, "capacity": 19520, "timezone": "America/Denver", "city": "Denver"},  # High altitude!
    1610612765: {"name": "Little Caesars Arena", "altitude": 600, "capacity": 20332, "timezone": "America/New_York", "city": "Detroit"},
    1610612744: {"name": "Chase Center", "altitude": 10, "capacity": 18064, "timezone": "America/Los_Angeles", "city": "San Francisco"},
    1610612745: {"name": "Toyota Center", "altitude": 50, "capacity": 18055, "timezone": "America/Chicago", "city": "Houston"},
    1610612754: {"name": "Gainbridge Fieldhouse", "altitude": 715, "capacity": 17923, "timezone": "America/New_York", "city": "Indianapolis"},
    1610612746: {"name": "Crypto.com Arena", "altitude": 300, "capacity": 19068, "timezone": "America/Los_Angeles", "city": "Los Angeles"},
    1610612747: {"name": "Crypto.com Arena", "altitude": 300, "capacity": 19068, "timezone": "America/Los_Angeles", "city": "Los Angeles"},
    1610612763: {"name": "FedExForum", "altitude": 300, "capacity": 18119, "timezone": "America/Chicago", "city": "Memphis"},
    1610612748: {"name": "Kaseya Center", "altitude": 10, "capacity": 19600, "timezone": "America/New_York", "city": "Miami"},
    1610612749: {"name": "Fiserv Forum", "altitude": 600, "capacity": 17341, "timezone": "America/Chicago", "city": "Milwaukee"},
    1610612750: {"name": "Target Center", "altitude": 830, "capacity": 18798, "timezone": "America/Chicago", "city": "Minneapolis"},
    1610612740: {"name": "Smoothie King Center", "altitude": 10, "capacity": 16867, "timezone": "America/Chicago", "city": "New Orleans"},
    1610612752: {"name": "Madison Square Garden", "altitude": 30, "capacity": 19812, "timezone": "America/New_York", "city": "New York"},
    1610612760: {"name": "Paycom Center", "altitude": 1200, "capacity": 18203, "timezone": "America/Chicago", "city": "Oklahoma City"},
    1610612753: {"name": "Amway Center", "altitude": 100, "capacity": 18846, "timezone": "America/New_York", "city": "Orlando"},
    1610612755: {"name": "Wells Fargo Center", "altitude": 40, "capacity": 20478, "timezone": "America/New_York", "city": "Philadelphia"},
    1610612756: {"name": "Footprint Center", "altitude": 1100, "capacity": 18055, "timezone": "America/Phoenix", "city": "Phoenix"},
    1610612757: {"name": "Moda Center", "altitude": 50, "capacity": 19393, "timezone": "America/Los_Angeles", "city": "Portland"},
    1610612758: {"name": "Golden 1 Center", "altitude": 30, "capacity": 17608, "timezone": "America/Los_Angeles", "city": "Sacramento"},
    1610612759: {"name": "Frost Bank Center", "altitude": 650, "capacity": 18418, "timezone": "America/Chicago", "city": "San Antonio"},
    1610612761: {"name": "Scotiabank Arena", "altitude": 250, "capacity": 19800, "timezone": "America/Toronto", "city": "Toronto"},
    1610612762: {"name": "Delta Center", "altitude": 4200, "capacity": 18206, "timezone": "America/Denver", "city": "Salt Lake City"},  # High altitude
    1610612764: {"name": "Capital One Arena", "altitude": 50, "capacity": 20356, "timezone": "America/New_York", "city": "Washington"},
}

# Calculate timezone differences (hours from EST)
TIMEZONE_OFFSETS = {
    "America/New_York": 0,      # EST
    "America/Chicago": -1,       # CST
    "America/Denver": -2,        # MST
    "America/Phoenix": -2,       # MST (no DST)
    "America/Los_Angeles": -3,   # PST
    "America/Toronto": 0,        # EST
}

# Historical home court advantage (based on historical win rates at home vs away)
# These are approximate based on NBA historical data
HOME_COURT_ADVANTAGE = {
    1610612737: 0.58,  # Atlanta
    1610612738: 0.68,  # Boston (strong home court)
    1610612751: 0.55,  # Brooklyn
    1610612766: 0.54,  # Charlotte
    1610612741: 0.62,  # Chicago
    1610612739: 0.61,  # Cleveland
    1610612742: 0.60,  # Dallas
    1610612743: 0.70,  # Denver (altitude advantage!)
    1610612765: 0.56,  # Detroit
    1610612744: 0.64,  # Golden State
    1610612745: 0.59,  # Houston
    1610612754: 0.58,  # Indiana
    1610612746: 0.60,  # Clippers
    1610612747: 0.62,  # Lakers
    1610612763: 0.59,  # Memphis
    1610612748: 0.63,  # Miami
    1610612749: 0.61,  # Milwaukee
    1610612750: 0.57,  # Minnesota
    1610612740: 0.56,  # New Orleans
    1610612752: 0.65,  # Knicks (MSG advantage)
    1610612760: 0.60,  # OKC
    1610612753: 0.55,  # Orlando
    1610612755: 0.62,  # Philadelphia
    1610612756: 0.59,  # Phoenix
    1610612757: 0.61,  # Portland
    1610612758: 0.57,  # Sacramento
    1610612759: 0.63,  # San Antonio
    1610612761: 0.59,  # Toronto
    1610612762: 0.66,  # Utah (altitude + crowd)
    1610612764: 0.57,  # Washington
}

def get_arena_info(team_id):
    """Get arena information for a team"""
    return ARENA_DATA.get(team_id, {
        "name": "Unknown",
        "altitude": 0,
        "capacity": 18000,
        "timezone": "America/New_York",
        "city": "Unknown"
    })

def get_home_court_advantage(team_id):
    """Get historical home court advantage rating for a team"""
    return HOME_COURT_ADVANTAGE.get(team_id, 0.58)  # League average

def calculate_travel_metrics(last_game_team_id, current_game_team_id, is_home_game):
    """
    Calculate travel-related metrics between games
    
    Returns:
        - time_zone_change: hours of timezone difference
        - coast_to_coast: 1 if coast-to-coast travel, 0 otherwise
        - travel_distance: approximate miles (simplified calculation)
    """
    if last_game_team_id is None:
        return 0, 0, 0
    
    last_arena = get_arena_info(last_game_team_id)
    current_arena = get_arena_info(current_game_team_id)
    
    # Timezone change
    last_tz = last_arena.get("timezone", "America/New_York")
    current_tz = current_arena.get("timezone", "America/New_York")
    tz_change = abs(TIMEZONE_OFFSETS.get(current_tz, 0) - TIMEZONE_OFFSETS.get(last_tz, 0))
    
    # Coast to coast (3-hour timezone difference)
    coast_to_coast = 1 if tz_change >= 3 else 0
    
    # Simplified travel distance (rough estimate based on timezones and cities)
    # This is approximate - real implementation would use lat/long
    city_distances = {
        ("Atlanta", "Los Angeles"): 2200, ("Atlanta", "San Francisco"): 2500,
        ("Boston", "Los Angeles"): 2600, ("Boston", "San Francisco"): 2700,
        ("New York", "Los Angeles"): 2450, ("New York", "San Francisco"): 2570,
        ("Miami", "Portland"): 2700, ("Miami", "Seattle"): 2700,
        # Add more as needed, but for now use a heuristic
    }
    
    last_city = last_arena.get("city", "")
    current_city = current_arena.get("city", "")
    
    # Simple heuristic: timezone difference * 750 miles per hour of timezone
    distance = tz_change * 750
    
    return tz_change, coast_to_coast, distance

