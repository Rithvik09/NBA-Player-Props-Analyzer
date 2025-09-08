"""
Live Odds Integration for NBA Props Analyzer
Connects to multiple sportsbook APIs for real-time odds data
"""
import asyncio
import aiohttp
import websocket
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
import hashlib
import hmac
import base64
import time
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class OddsData:
    """Data class for odds information"""
    sportsbook: str
    player_name: str
    prop_type: str
    line: float
    over_odds: int
    under_odds: int
    timestamp: datetime
    game_id: str
    player_id: Optional[str] = None
    confidence: Optional[float] = None

class OddsAggregator:
    """Aggregates odds from multiple sportsbooks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.odds_data: Dict[str, List[OddsData]] = {}
        self.websocket_connections = {}
        self.update_callbacks = []
        
        # API configurations
        self.api_configs = {
            'draftkings': {
                'base_url': 'https://sportsbook-nash.draftkings.com/sites/US-SB/api/v4',
                'websocket_url': 'wss://sportsbook-nash.draftkings.com/socket.io/',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json'
                }
            },
            'fanduel': {
                'base_url': 'https://pa.api.fanduel.com/api',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json'
                }
            },
            'caesars': {
                'base_url': 'https://api.williamhill.com/v2',
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json'
                }
            },
            'odds_api': {
                'base_url': 'https://api.the-odds-api.com/v4',
                'api_key': 'YOUR_ODDS_API_KEY',  # Replace with actual key
                'headers': {
                    'Accept': 'application/json'
                }
            }
        }
    
    async def fetch_all_odds(self, sport='basketball_nba'):
        """Fetch odds from all available sources"""
        tasks = []
        
        # Create tasks for each sportsbook
        for sportsbook in self.api_configs.keys():
            if hasattr(self, f'fetch_{sportsbook}_odds'):
                task = getattr(self, f'fetch_{sportsbook}_odds')(sport)
                tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_odds = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching from {list(self.api_configs.keys())[i]}: {result}")
            elif result:
                all_odds.extend(result)
        
        return all_odds
    
    async def fetch_odds_api_odds(self, sport='basketball_nba'):
        """Fetch odds from The Odds API"""
        try:
            config = self.api_configs['odds_api']
            
            async with aiohttp.ClientSession() as session:
                # Fetch games
                games_url = f"{config['base_url']}/sports/{sport}/odds"
                params = {
                    'apiKey': config['api_key'],
                    'regions': 'us',
                    'markets': 'player_props',
                    'oddsFormat': 'american',
                    'dateFormat': 'iso'
                }
                
                async with session.get(games_url, params=params, headers=config['headers']) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_odds_api_data(data)
                    else:
                        self.logger.error(f"Odds API error: {response.status}")
                        return []
        
        except Exception as e:
            self.logger.error(f"Error fetching Odds API data: {e}")
            return []
    
    async def fetch_draftkings_odds(self, sport='basketball_nba'):
        """Fetch odds from DraftKings (example implementation)"""
        try:
            config = self.api_configs['draftkings']
            
            async with aiohttp.ClientSession() as session:
                # Note: This is a simplified example. Real implementation would need
                # proper authentication and API endpoints
                url = f"{config['base_url']}/eventgroups/88670846/categories/1215/subcategories/4511"
                
                async with session.get(url, headers=config['headers']) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_draftkings_data(data)
                    else:
                        return []
        
        except Exception as e:
            self.logger.error(f"Error fetching DraftKings data: {e}")
            return []
    
    async def fetch_fanduel_odds(self, sport='basketball_nba'):
        """Fetch odds from FanDuel"""
        try:
            config = self.api_configs['fanduel']
            
            async with aiohttp.ClientSession() as session:
                # Example endpoint (would need real FanDuel API)
                url = f"{config['base_url']}/betting/nba/player-props"
                
                async with session.get(url, headers=config['headers']) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_fanduel_data(data)
                    else:
                        return []
        
        except Exception as e:
            self.logger.error(f"Error fetching FanDuel data: {e}")
            return []
    
    def _parse_odds_api_data(self, data: Dict) -> List[OddsData]:
        """Parse Odds API response"""
        odds_list = []
        
        try:
            for game in data:
                game_id = game.get('id')
                
                for bookmaker in game.get('bookmakers', []):
                    sportsbook = bookmaker.get('title', 'unknown')
                    
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'player_props':
                            
                            for outcome in market.get('outcomes', []):
                                player_name = outcome.get('description', '').split(' - ')[0]
                                prop_type = self._extract_prop_type(outcome.get('description', ''))
                                
                                odds_data = OddsData(
                                    sportsbook=sportsbook,
                                    player_name=player_name,
                                    prop_type=prop_type,
                                    line=float(outcome.get('point', 0)),
                                    over_odds=outcome.get('price') if outcome.get('name') == 'Over' else 0,
                                    under_odds=outcome.get('price') if outcome.get('name') == 'Under' else 0,
                                    timestamp=datetime.now(),
                                    game_id=game_id
                                )
                                
                                odds_list.append(odds_data)
        
        except Exception as e:
            self.logger.error(f"Error parsing Odds API data: {e}")
        
        return odds_list
    
    def _parse_draftkings_data(self, data: Dict) -> List[OddsData]:
        """Parse DraftKings response"""
        odds_list = []
        
        try:
            # Parse DraftKings specific format
            for event in data.get('eventGroup', {}).get('events', []):
                game_id = event.get('eventId')
                
                for offer_category in event.get('eventOfferCategories', []):
                    for offer_subcategory in offer_category.get('offerSubcategories', []):
                        for offer in offer_subcategory.get('offers', []):
                            
                            player_name = offer.get('label', '').split(' - ')[0]
                            prop_type = self._extract_prop_type(offer.get('label', ''))
                            
                            outcomes = offer.get('outcomes', [])
                            if len(outcomes) >= 2:
                                odds_data = OddsData(
                                    sportsbook='DraftKings',
                                    player_name=player_name,
                                    prop_type=prop_type,
                                    line=float(offer.get('line', 0)),
                                    over_odds=outcomes[0].get('oddsAmerican', 0),
                                    under_odds=outcomes[1].get('oddsAmerican', 0),
                                    timestamp=datetime.now(),
                                    game_id=str(game_id)
                                )
                                
                                odds_list.append(odds_data)
        
        except Exception as e:
            self.logger.error(f"Error parsing DraftKings data: {e}")
        
        return odds_list
    
    def _parse_fanduel_data(self, data: Dict) -> List[OddsData]:
        """Parse FanDuel response"""
        odds_list = []
        
        try:
            # Parse FanDuel specific format
            for market in data.get('markets', []):
                for runner in market.get('runners', []):
                    
                    player_name = runner.get('runnerName', '').split(' - ')[0]
                    prop_type = self._extract_prop_type(runner.get('runnerName', ''))
                    
                    odds_data = OddsData(
                        sportsbook='FanDuel',
                        player_name=player_name,
                        prop_type=prop_type,
                        line=float(runner.get('handicap', 0)),
                        over_odds=runner.get('winRunnerOdds', {}).get('americanDisplayOdds', {}).get('americanOdds', 0),
                        under_odds=0,  # Would need to find corresponding under bet
                        timestamp=datetime.now(),
                        game_id=str(market.get('marketId'))
                    )
                    
                    odds_list.append(odds_data)
        
        except Exception as e:
            self.logger.error(f"Error parsing FanDuel data: {e}")
        
        return odds_list
    
    def _extract_prop_type(self, description: str) -> str:
        """Extract prop type from description"""
        description_lower = description.lower()
        
        prop_mappings = {
            'points': ['points', 'pts'],
            'assists': ['assists', 'ast'],
            'rebounds': ['rebounds', 'reb', 'total rebounds'],
            'steals': ['steals', 'stl'],
            'blocks': ['blocks', 'blk'],
            'turnovers': ['turnovers', 'to'],
            'three_pointers': ['3-pointers', 'threes', '3pm', '3-point'],
            'double_double': ['double-double', 'dd'],
            'triple_double': ['triple-double', 'td'],
            'pts_reb_ast': ['points + rebounds + assists', 'pra'],
            'pts_reb': ['points + rebounds'],
            'pts_ast': ['points + assists']
        }
        
        for prop_type, keywords in prop_mappings.items():
            if any(keyword in description_lower for keyword in keywords):
                return prop_type
        
        return 'unknown'

class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self, odds_aggregator: OddsAggregator):
        self.odds_aggregator = odds_aggregator
        self.logger = logging.getLogger(__name__)
        self.connections = {}
        self.running = False
    
    def start_websocket_connections(self):
        """Start WebSocket connections for real-time data"""
        self.running = True
        
        # Start WebSocket threads
        threading.Thread(target=self._start_draftkings_ws, daemon=True).start()
        threading.Thread(target=self._start_fanduel_ws, daemon=True).start()
    
    def _start_draftkings_ws(self):
        """Start DraftKings WebSocket connection"""
        try:
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    # Parse real-time updates and trigger callbacks
                    self._handle_realtime_update('draftkings', data)
                except Exception as e:
                    self.logger.error(f"Error parsing DraftKings message: {e}")
            
            def on_error(ws, error):
                self.logger.error(f"DraftKings WebSocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                self.logger.info("DraftKings WebSocket connection closed")
            
            ws_url = self.odds_aggregator.api_configs['draftkings']['websocket_url']
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            self.connections['draftkings'] = ws
            ws.run_forever()
            
        except Exception as e:
            self.logger.error(f"Error starting DraftKings WebSocket: {e}")
    
    def _start_fanduel_ws(self):
        """Start FanDuel WebSocket connection"""
        # Similar implementation for FanDuel WebSocket
        pass
    
    def _handle_realtime_update(self, source: str, data: Dict):
        """Handle real-time odds updates"""
        try:
            # Parse the update and create OddsData objects
            updated_odds = self._parse_realtime_data(source, data)
            
            # Trigger callbacks for subscribers
            for callback in self.odds_aggregator.update_callbacks:
                try:
                    callback(updated_odds)
                except Exception as e:
                    self.logger.error(f"Error in update callback: {e}")
        
        except Exception as e:
            self.logger.error(f"Error handling real-time update: {e}")
    
    def _parse_realtime_data(self, source: str, data: Dict) -> List[OddsData]:
        """Parse real-time data updates"""
        # Implementation would depend on the specific format from each sportsbook
        return []

class LineMovementTracker:
    """Tracks line movements and identifies betting opportunities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.line_history: Dict[str, List[Dict]] = {}
        self.movement_thresholds = {
            'significant': 1.0,  # 1 point movement
            'major': 2.0,        # 2 point movement
            'extreme': 3.0       # 3+ point movement
        }
    
    def track_line_movement(self, odds_data: List[OddsData]):
        """Track line movements for all props"""
        for odds in odds_data:
            key = f"{odds.player_name}_{odds.prop_type}_{odds.sportsbook}"
            
            current_data = {
                'line': odds.line,
                'over_odds': odds.over_odds,
                'under_odds': odds.under_odds,
                'timestamp': odds.timestamp
            }
            
            if key not in self.line_history:
                self.line_history[key] = []
            
            self.line_history[key].append(current_data)
            
            # Keep only recent history (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.line_history[key] = [
                entry for entry in self.line_history[key]
                if entry['timestamp'] > cutoff_time
            ]
            
            # Detect significant movements
            self._detect_movement(key, odds)
    
    def _detect_movement(self, key: str, current_odds: OddsData):
        """Detect significant line movements"""
        if len(self.line_history[key]) < 2:
            return
        
        previous_line = self.line_history[key][-2]['line']
        current_line = current_odds.line
        
        movement = abs(current_line - previous_line)
        
        if movement >= self.movement_thresholds['extreme']:
            self._trigger_movement_alert(key, current_odds, movement, 'EXTREME')
        elif movement >= self.movement_thresholds['major']:
            self._trigger_movement_alert(key, current_odds, movement, 'MAJOR')
        elif movement >= self.movement_thresholds['significant']:
            self._trigger_movement_alert(key, current_odds, movement, 'SIGNIFICANT')
    
    def _trigger_movement_alert(self, key: str, odds: OddsData, movement: float, level: str):
        """Trigger alert for significant line movement"""
        alert = {
            'type': 'line_movement',
            'level': level,
            'player': odds.player_name,
            'prop_type': odds.prop_type,
            'sportsbook': odds.sportsbook,
            'movement': movement,
            'new_line': odds.line,
            'timestamp': odds.timestamp
        }
        
        self.logger.info(f"{level} line movement detected: {key} moved {movement} points")
        
        # Here you would trigger notifications, webhooks, etc.
        return alert
    
    def get_movement_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of line movements in the last X hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        movements = []
        for key, history in self.line_history.items():
            recent_history = [entry for entry in history if entry['timestamp'] > cutoff_time]
            
            if len(recent_history) >= 2:
                total_movement = abs(recent_history[-1]['line'] - recent_history[0]['line'])
                if total_movement > 0:
                    movements.append({
                        'key': key,
                        'total_movement': total_movement,
                        'start_line': recent_history[0]['line'],
                        'current_line': recent_history[-1]['line'],
                        'direction': 'up' if recent_history[-1]['line'] > recent_history[0]['line'] else 'down'
                    })
        
        # Sort by total movement
        movements.sort(key=lambda x: x['total_movement'], reverse=True)
        
        return {
            'top_movements': movements[:10],
            'total_props_tracked': len(self.line_history),
            'time_period': f"{hours} hours"
        }

class ArbitrageDetector:
    """Detects arbitrage opportunities across sportsbooks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_profit_threshold = 0.02  # 2% minimum profit
    
    def find_arbitrage_opportunities(self, odds_data: List[OddsData]) -> List[Dict]:
        """Find arbitrage opportunities"""
        opportunities = []
        
        # Group odds by player and prop type
        grouped_odds = self._group_odds_by_prop(odds_data)
        
        for key, odds_list in grouped_odds.items():
            if len(odds_list) < 2:
                continue
            
            # Find best over and under odds
            best_over = max(odds_list, key=lambda x: x.over_odds) if any(x.over_odds for x in odds_list) else None
            best_under = max(odds_list, key=lambda x: x.under_odds) if any(x.under_odds for x in odds_list) else None
            
            if best_over and best_under and best_over.sportsbook != best_under.sportsbook:
                profit = self._calculate_arbitrage_profit(best_over.over_odds, best_under.under_odds)
                
                if profit > self.min_profit_threshold:
                    opportunity = {
                        'player': best_over.player_name,
                        'prop_type': best_over.prop_type,
                        'line': best_over.line,
                        'over_bet': {
                            'sportsbook': best_over.sportsbook,
                            'odds': best_over.over_odds
                        },
                        'under_bet': {
                            'sportsbook': best_under.sportsbook,
                            'odds': best_under.under_odds
                        },
                        'profit_percentage': profit * 100,
                        'timestamp': datetime.now()
                    }
                    
                    opportunities.append(opportunity)
        
        return sorted(opportunities, key=lambda x: x['profit_percentage'], reverse=True)
    
    def _group_odds_by_prop(self, odds_data: List[OddsData]) -> Dict[str, List[OddsData]]:
        """Group odds by player and prop type"""
        grouped = {}
        
        for odds in odds_data:
            key = f"{odds.player_name}_{odds.prop_type}_{odds.line}"
            
            if key not in grouped:
                grouped[key] = []
            
            grouped[key].append(odds)
        
        return grouped
    
    def _calculate_arbitrage_profit(self, over_odds: int, under_odds: int) -> float:
        """Calculate arbitrage profit percentage"""
        try:
            # Convert American odds to decimal
            over_decimal = self._american_to_decimal(over_odds)
            under_decimal = self._american_to_decimal(under_odds)
            
            # Calculate implied probabilities
            over_prob = 1 / over_decimal
            under_prob = 1 / under_decimal
            
            total_prob = over_prob + under_prob
            
            if total_prob < 1:
                return (1 - total_prob) / total_prob
            else:
                return 0
        
        except:
            return 0
    
    def _american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

# Main integration class
class LiveOddsIntegration:
    """Main class for live odds integration"""
    
    def __init__(self):
        self.odds_aggregator = OddsAggregator()
        self.websocket_manager = WebSocketManager(self.odds_aggregator)
        self.line_tracker = LineMovementTracker()
        self.arbitrage_detector = ArbitrageDetector()
        
        self.logger = logging.getLogger(__name__)
        
        # Set up update callbacks
        self.odds_aggregator.update_callbacks.append(self._handle_odds_update)
    
    async def start_live_monitoring(self):
        """Start live odds monitoring"""
        self.logger.info("Starting live odds monitoring...")
        
        # Start WebSocket connections
        self.websocket_manager.start_websocket_connections()
        
        # Start periodic polling for APIs that don't support WebSocket
        asyncio.create_task(self._periodic_odds_update())
    
    async def _periodic_odds_update(self):
        """Periodically fetch odds updates"""
        while True:
            try:
                odds_data = await self.odds_aggregator.fetch_all_odds()
                
                if odds_data:
                    # Track line movements
                    self.line_tracker.track_line_movement(odds_data)
                    
                    # Find arbitrage opportunities
                    arbitrage_opps = self.arbitrage_detector.find_arbitrage_opportunities(odds_data)
                    
                    if arbitrage_opps:
                        self.logger.info(f"Found {len(arbitrage_opps)} arbitrage opportunities")
                
                # Wait before next update (adjust based on API rate limits)
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in periodic update: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def _handle_odds_update(self, updated_odds: List[OddsData]):
        """Handle real-time odds updates"""
        try:
            # Process real-time updates
            self.line_tracker.track_line_movement(updated_odds)
            
            # Check for immediate arbitrage opportunities
            arbitrage_opps = self.arbitrage_detector.find_arbitrage_opportunities(updated_odds)
            
            if arbitrage_opps:
                # Send immediate notifications for time-sensitive opportunities
                self._send_arbitrage_alerts(arbitrage_opps)
        
        except Exception as e:
            self.logger.error(f"Error handling odds update: {e}")
    
    def _send_arbitrage_alerts(self, opportunities: List[Dict]):
        """Send alerts for arbitrage opportunities"""
        for opp in opportunities:
            self.logger.info(
                f"ARBITRAGE ALERT: {opp['player']} {opp['prop_type']} - "
                f"{opp['profit_percentage']:.2f}% profit"
            )
            
            # Here you would integrate with notification systems
            # (email, SMS, push notifications, webhooks, etc.)

# Example usage
async def main():
    # Initialize live odds integration
    live_odds = LiveOddsIntegration()
    
    # Start monitoring
    await live_odds.start_live_monitoring()

if __name__ == "__main__":
    asyncio.run(main())