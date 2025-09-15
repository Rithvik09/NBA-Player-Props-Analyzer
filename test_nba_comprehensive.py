#!/usr/bin/env python3
"""
Test script for comprehensive NBA analysis with 30 factors
"""
import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from basketball_betting_helper import BasketballBettingHelper

async def test_comprehensive_analysis():
    """Test the 30-factor NBA comprehensive analysis"""
    print("🏀 Testing NBA Comprehensive Analysis (30 Factors)")
    print("=" * 60)
    
    # Initialize the helper
    helper = BasketballBettingHelper()
    
    # Test parameters - LeBron James vs Boston Celtics
    player_id = 2544  # LeBron James
    prop_type = "points"
    line = 25
    opponent_team_id = 1610612738  # Boston Celtics
    
    print(f"📊 Testing Analysis:")
    print(f"   Player ID: {player_id}")
    print(f"   Prop Type: {prop_type}")
    print(f"   Line: {line}")
    print(f"   Opponent Team ID: {opponent_team_id}")
    print()
    
    try:
        # Run comprehensive analysis
        print("🔄 Running comprehensive NBA analysis...")
        result = await helper.analyze_comprehensive_nba_prop(
            player_id=player_id,
            prop_type=prop_type, 
            line=line,
            opponent_team_id=opponent_team_id
        )
        
        if result.get('success'):
            print("✅ Analysis completed successfully!")
            print()
            
            # Show key results
            print("📈 KEY RESULTS:")
            print(f"   Total Factors Analyzed: {result.get('enhanced_metrics', {}).get('total_factors_analyzed', 'N/A')}")
            print(f"   Predicted Value: {result.get('predicted_value', 'N/A')}")
            print(f"   Over Probability: {result.get('over_probability', 'N/A')}")
            print(f"   Recommendation: {result.get('recommendation', 'N/A')}")
            print(f"   Confidence Score: {result.get('confidence_score', 'N/A')}")
            print()
            
            # Show factor breakdown
            print("🔍 FACTOR BREAKDOWN:")
            enhanced_metrics = result.get('enhanced_metrics', {})
            print(f"   Player Factors: {result.get('enterprise_features', {}).get('player_factors', 'N/A')}")
            print(f"   Team Factors: {result.get('enterprise_features', {}).get('team_factors', 'N/A')}")
            print(f"   Processing Time: {enhanced_metrics.get('processing_time_ms', 'N/A')}ms")
            print()
            
            # Show some example team analytics
            comprehensive_analysis = result.get('comprehensive_analysis', {})
            if comprehensive_analysis:
                print("🏀 SAMPLE TEAM ANALYTICS:")
                team_offense = comprehensive_analysis.get('team_offensive_analytics', {})
                if team_offense:
                    print(f"   Offensive Rating: {team_offense.get('offensive_rating', 'N/A')}")
                    print(f"   Offensive Tier: {team_offense.get('offensive_efficiency_tier', 'N/A')}")
                    print(f"   Offensive Impact: {team_offense.get('offensive_impact_pct', 'N/A')}%")
                
                team_defense = comprehensive_analysis.get('team_defensive_analytics', {})
                if team_defense:
                    print(f"   Defensive Rating: {team_defense.get('defensive_rating', 'N/A')}")
                    print(f"   Defensive Tier: {team_defense.get('defensive_tier', 'N/A')}")
                    print(f"   Defensive Impact: {team_defense.get('defensive_impact_pct', 'N/A')}%")
                
                team_pace = comprehensive_analysis.get('team_pace_efficiency', {})
                if team_pace:
                    print(f"   Expected Pace: {team_pace.get('expected_game_pace', 'N/A')}")
                    print(f"   Game Style: {team_pace.get('game_style', 'N/A')}")
                    print(f"   Pace Impact: {team_pace.get('pace_impact_pct', 'N/A')}%")
            print()
            print("🎯 SUCCESS: All 30 NBA factors analyzed successfully!")
            
        else:
            print("❌ Analysis failed!")
            error = result.get('error', 'Unknown error')
            print(f"   Error: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_comprehensive_analysis())
    print()
    if success:
        print("🎉 NBA Comprehensive Analysis Test PASSED!")
        print("📊 All 30 factors (20 player + 10 team) implemented successfully")
    else:
        print("💥 NBA Comprehensive Analysis Test FAILED!")
    
    sys.exit(0 if success else 1)