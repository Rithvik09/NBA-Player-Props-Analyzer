from flask import Flask, render_template, request, jsonify, session
try:
    # Try relative imports first (when running as module)
    from .basketball_betting_helper import BasketballBettingHelper
    from .nfl_betting_helper import NFLBettingHelper
    from .nfl_weather import NFLWeatherSystem
except ImportError:
    # Fallback to absolute imports (when running directly)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from basketball_betting_helper import BasketballBettingHelper
    from nfl_betting_helper import NFLBettingHelper
    from nfl_weather import NFLWeatherSystem
import logging
from logging.handlers import RotatingFileHandler
import os
import asyncio
from functools import wraps
from nba_api.stats.static import players
# Enterprise imports
try:
    from .auth.simple_auth import simple_auth
    AUTH_AVAILABLE = True
    print("✅ Simple authentication system loaded")
except ImportError:
    try:
        from auth.simple_auth import simple_auth
        AUTH_AVAILABLE = True
        print("✅ Simple authentication system loaded")
    except ImportError as e:
        print(f"Warning: Auth not available: {e}")
        AUTH_AVAILABLE = False
        simple_auth = None

try:
    from .notifications.notification_system import NotificationManager
    NOTIFICATIONS_AVAILABLE = True
    print("✅ Notifications system loaded")
except ImportError as e:
    print(f"Warning: Notifications not available: {e}")
    NOTIFICATIONS_AVAILABLE = False
    NotificationManager = None

# Initialize Flask app
app = Flask(__name__, 
    static_url_path='',
    static_folder='../static',
    template_folder='../templates')

# Configure secret key for sessions
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize enterprise features
if AUTH_AVAILABLE:
    user_manager = simple_auth
else:
    user_manager = None

if NOTIFICATIONS_AVAILABLE:
    notification_manager = NotificationManager()
else:
    notification_manager = None

if not os.path.exists('logs'):
    os.mkdir('logs')

file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Basketball Betting Helper startup')

# Initialize betting helper with enterprise features
betting_helper = BasketballBettingHelper()
nfl_helper = None
nfl_weather = None

# Initialize NFL systems
try:
    nfl_helper = NFLBettingHelper()
    nfl_weather = NFLWeatherSystem()
    print("✅ NFL betting system initialized")
except Exception as e:
    print(f"Warning: NFL system not available: {e}")
    nfl_helper = None
    nfl_weather = None

def async_route(f):
    """Decorator to handle async routes"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not AUTH_AVAILABLE:
            return f(*args, **kwargs)  # Skip auth if not available
        
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            token = token[7:]
            user_data = user_manager.verify_token(token) if user_manager else None
            if user_data:
                request.current_user = user_data
                return f(*args, **kwargs)
        
        return jsonify({'error': 'Authentication required'}), 401
    return wrapper


@app.route('/test_api')
def test_api():
    #Test route to check NBA API functionality
    try:
        all_players = players.get_players()
        active_players = [p for p in all_players if p['is_active']]
        return jsonify({
            'total_players': len(all_players),
            'active_players': len(active_players),
            'sample_player': active_players[0] if active_players else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('enhanced_index.html')

@app.route('/enhanced')
def enhanced_home():
    return render_template('enhanced_index.html')

@app.route('/classic')
def classic_home():
    return render_template('index_backup.html')

@app.route('/search_players')
def search_players():
    try:
        query = request.args.get('q', '')
        
        if not query or len(query) < 2:
            return jsonify([])
            
        suggestions = betting_helper.get_player_suggestions(query)
        return jsonify(suggestions)
        
    except Exception as e:
        app.logger.error(f'Error searching players: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/get_player_stats/<int:player_id>')
def get_player_stats(player_id):
    #Get comprehensive player statistics
    try:
        stats = betting_helper.get_player_stats(player_id)
        if stats:
            return jsonify(stats)
        else:
            return jsonify({'error': 'Unable to retrieve player stats'}), 404
            
    except Exception as e:
        app.logger.error(f'Error getting player stats: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_prop', methods=['POST'])
@async_route
async def analyze_prop():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        required_fields = ['player_id', 'prop_type', 'line', 'opponent_team_id']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields. Required: {required_fields}'}), 400

        player_id = data['player_id']
        player_name = data.get('player_name', 'Unknown Player')
        prop_type = data['prop_type']
        line = float(data['line'])
        opponent_team_id = int(data['opponent_team_id'])  # Convert to int for NBA
        user_id = data.get('user_id', 'default')
        decimal_odds = data.get('decimal_odds', 1.91)
        
        # Get basic prop analysis
        include_situational = data.get('include_situational', True)
        
        # Use enterprise helper with user context
        user_id = data.get('user_id', 'anonymous')
        enterprise_helper = BasketballBettingHelper(user_id=user_id)
        
        analysis = await enterprise_helper.analyze_prop_bet(
            player_id=player_id,
            prop_type=prop_type,
            line=line,
            opponent_team_id=opponent_team_id
        )
        
        if not analysis or not analysis.get('success'):
            return jsonify({'error': 'Unable to perform analysis', 'success': False}), 500
        
        # Add player name to analysis
        analysis['player_name'] = player_name
        analysis['prop_type'] = prop_type
        analysis['line'] = line
        
        # Enhanced analysis is already included in the analyze_prop_bet method
        
        return jsonify(analysis)
            
    except Exception as e:
        app.logger.error(f'Error analyzing prop: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

# NBA Betting Analysis Endpoints
@app.route('/nba/analyze_moneyline', methods=['POST'])
def nba_analyze_moneyline():
    """Analyze NBA moneyline bet using comprehensive 30-factor analysis"""
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        home_odds = data.get('home_odds')
        away_odds = data.get('away_odds')
        
        if not all([home_team, away_team, home_odds, away_odds]):
            return jsonify({'error': 'home_team, away_team, home_odds, and away_odds required'}), 400
        
        analysis = betting_helper.analyze_nba_moneyline_bet(home_team, away_team, home_odds, away_odds)
        
        return jsonify({
            'success': True,
            'moneyline_analysis': analysis,
            'sport': 'NBA'
        })
        
    except Exception as e:
        app.logger.error(f'Error analyzing NBA moneyline: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/nba/analyze_spread', methods=['POST'])
def nba_analyze_spread():
    """Analyze NBA spread bet using comprehensive team analytics"""
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        spread_line = data.get('spread_line')
        home_odds = data.get('home_odds', -110)
        away_odds = data.get('away_odds', -110)
        
        if not all([home_team, away_team, spread_line]):
            return jsonify({'error': 'home_team, away_team, and spread_line required'}), 400
        
        analysis = betting_helper.analyze_nba_spread_bet(home_team, away_team, spread_line, home_odds, away_odds)
        
        return jsonify({
            'success': True,
            'spread_analysis': analysis,
            'sport': 'NBA'
        })
        
    except Exception as e:
        app.logger.error(f'Error analyzing NBA spread: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/nba/analyze_over_under', methods=['POST'])
def nba_analyze_over_under():
    """Analyze NBA over/under total bet using pace and efficiency factors"""
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        total_line = data.get('total_line')
        over_odds = data.get('over_odds', -110)
        under_odds = data.get('under_odds', -110)
        
        if not all([home_team, away_team, total_line]):
            return jsonify({'error': 'home_team, away_team, and total_line required'}), 400
        
        analysis = betting_helper.analyze_nba_over_under_total(home_team, away_team, total_line, over_odds, under_odds)
        
        return jsonify({
            'success': True,
            'over_under_analysis': analysis,
            'sport': 'NBA'
        })
        
    except Exception as e:
        app.logger.error(f'Error analyzing NBA over/under: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

# Enhanced Bankroll Management Endpoints
@app.route('/bankroll/dashboard')
def bankroll_dashboard():
    """Get comprehensive bankroll dashboard"""
    try:
        dashboard = betting_helper.get_bankroll_dashboard()
        return jsonify(dashboard)
    except Exception as e:
        app.logger.error(f'Error getting bankroll dashboard: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/bankroll/initialize', methods=['POST'])
def initialize_bankroll():
    """Initialize bankroll settings"""
    try:
        data = request.get_json()
        bankroll_amount = float(data.get('bankroll_amount', 1000))
        unit_size = data.get('unit_size')
        risk_tolerance = data.get('risk_tolerance', 'Medium')
        
        if unit_size:
            unit_size = float(unit_size)
        
        result = betting_helper.bankroll_manager.initialize_bankroll(
            bankroll_amount, unit_size, risk_tolerance
        )
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        app.logger.error(f'Error initializing bankroll: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/bankroll/log_bet', methods=['POST'])
def log_bet():
    """Log a bet to tracking system"""
    try:
        data = request.get_json()
        bet_id = betting_helper.bankroll_manager.log_bet(data)
        return jsonify({'success': True, 'bet_id': bet_id})
        
    except Exception as e:
        app.logger.error(f'Error logging bet: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

# Parlay Optimizer Endpoints
@app.route('/parlay/analyze_multiple', methods=['POST'])
def analyze_multiple_props():
    """Analyze multiple props and find parlay opportunities"""
    try:
        data = request.get_json()
        if not data or 'props' not in data:
            return jsonify({'error': 'No props data provided'}), 400
        
        props_list = data['props']
        if not isinstance(props_list, list) or len(props_list) < 1:
            return jsonify({'error': 'Invalid props list'}), 400
        
        analysis = betting_helper.analyze_multiple_props(props_list)
        return jsonify(analysis)
        
    except Exception as e:
        app.logger.error(f'Error analyzing multiple props: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/parlay/optimize', methods=['POST'])
def optimize_parlays():
    """Find optimal parlay combinations"""
    try:
        data = request.get_json()
        props_list = data.get('props', [])
        max_legs = data.get('max_legs', 4)
        min_ev = data.get('min_ev', 0.05)
        
        result = betting_helper.parlay_optimizer.find_optimal_same_game_parlays(
            props_list, max_legs, min_ev
        )
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f'Error optimizing parlays: {e}')
        return jsonify({'error': str(e)}), 500

# Player Information Endpoint
@app.route('/player/name/<int:player_id>')
def get_player_name(player_id):
    """Get player name by ID"""
    try:
        player_name = betting_helper.get_player_name(player_id)
        return jsonify({'success': True, 'player_name': player_name})
    except Exception as e:
        app.logger.error(f'Error getting player name: {e}')
        return jsonify({'error': str(e)}), 500

# Analytics Endpoints
@app.route('/analytics/performance')
def get_performance_analytics():
    """Get performance analytics"""
    try:
        days = int(request.args.get('days', 30))
        performance = betting_helper.bankroll_manager.calculate_performance_metrics(days)
        return jsonify({'success': True, 'performance': performance})
    except Exception as e:
        app.logger.error(f'Error getting performance analytics: {e}')
        return jsonify({'error': str(e)}), 500

# ==============================================================================
# ENTERPRISE ENDPOINTS
# ==============================================================================

@app.route('/enterprise/dashboard')
@async_route
@require_auth
async def enterprise_dashboard():
    """Get comprehensive enterprise analytics dashboard"""
    try:
        user_id = getattr(request, 'current_user', {}).get('user_id', 'anonymous')
        enterprise_helper = BasketballBettingHelper(user_id=user_id)
        
        dashboard = await enterprise_helper.get_enterprise_analytics_dashboard()
        return jsonify(dashboard)
        
    except Exception as e:
        app.logger.error(f'Error getting enterprise dashboard: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

# Authentication Endpoints
@app.route('/auth/register', methods=['POST'])
def register_user():
    """Register a new user"""
    if not AUTH_AVAILABLE:
        return jsonify({'error': 'Authentication not available'}), 501
    
    try:
        data = request.get_json()
        result = user_manager.register_user(
            email=data['email'],
            username=data['username'],
            password=data['password']
        )
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f'Error registering user: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/auth/login', methods=['POST'])
def login_user():
    """Login user and return JWT token"""
    if not AUTH_AVAILABLE:
        return jsonify({'error': 'Authentication not available'}), 501
    
    try:
        data = request.get_json()
        result = user_manager.login_user(
            email=data['email'],
            password=data['password']
        )
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f'Error logging in user: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

# Live Odds Endpoints
@app.route('/odds/live/<sport>')
@async_route
async def get_live_odds(sport):
    """Get live odds for specified sport (NBA or NFL)"""
    try:
        sport_upper = sport.upper()
        
        if sport_upper == 'NBA':
            enterprise_helper = BasketballBettingHelper()
        elif sport_upper == 'NFL':
            if not nfl_helper:
                return jsonify({
                    'success': False,
                    'error': 'NFL system not available'
                }), 501
            enterprise_helper = nfl_helper
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported sport: {sport}'
            }), 400
        
        if hasattr(enterprise_helper, 'odds_aggregator') and enterprise_helper.odds_aggregator:
            odds = await enterprise_helper.odds_aggregator.fetch_all_odds(sport_upper)
            return jsonify({
                'success': True,
                'sport': sport_upper,
                'odds': odds,
                'timestamp': odds.get('timestamp') if odds else None
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Live odds integration not available'
            }), 501
            
    except Exception as e:
        app.logger.error(f'Error getting live odds for {sport}: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/ai/models/status')
@require_auth
def ai_models_status():
    """Get status of AI models"""
    try:
        enterprise_helper = BasketballBettingHelper()
        
        status = {
            'tensorflow_available': enterprise_helper.tensorflow_predictor is not None,
            'pytorch_available': enterprise_helper.pytorch_predictor is not None,
            'cloud_integration': enterprise_helper.aws_integration is not None,
            'live_odds': enterprise_helper.odds_aggregator is not None,
            'user_management': enterprise_helper.user_manager is not None
        }
        
        if enterprise_helper.tensorflow_predictor:
            status['tensorflow_models'] = ['LSTM', 'Transformer', 'CNN']
        
        if enterprise_helper.pytorch_predictor:
            status['pytorch_models'] = ['GNN', 'VAE', 'GAN', 'Advanced_LSTM']
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        app.logger.error(f'Error getting AI models status: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

# System Status Endpoint
@app.route('/system/status')
def system_status():
    """Get comprehensive system status"""
    try:
        enterprise_helper = BasketballBettingHelper()
        
        status = {
            'api_version': '2.0.0-enterprise',
            'timestamp': os.popen('date').read().strip(),
            'features': {
                'standard_ml': True,
                'tensorflow': enterprise_helper.tensorflow_predictor is not None,
                'pytorch': enterprise_helper.pytorch_predictor is not None,
                'cloud_aws': enterprise_helper.aws_integration is not None,
                'live_odds': enterprise_helper.odds_aggregator is not None,
                'authentication': AUTH_AVAILABLE,
                'notifications': notification_manager is not None
            },
            'database': {
                'connected': True,
                'type': 'SQLite',
                'location': enterprise_helper.db_name
            }
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        app.logger.error(f'Error getting system status: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

# ==============================================================================
# NFL ENDPOINTS
# ==============================================================================

@app.route('/nfl/search_players')
def nfl_search_players():
    """Search NFL players"""
    if not nfl_helper:
        return jsonify({'error': 'NFL system not available'}), 501
    
    try:
        query = request.args.get('q', '')
        
        if not query or len(query) < 2:
            return jsonify([])
            
        suggestions = nfl_helper.get_player_suggestions(query)
        return jsonify(suggestions)
        
    except Exception as e:
        app.logger.error(f'Error searching NFL players: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/nfl/get_player_stats/<player_id>')
def nfl_get_player_stats(player_id):
    """Get comprehensive NFL player statistics"""
    if not nfl_helper:
        return jsonify({'error': 'NFL system not available'}), 501
    
    try:
        stats = nfl_helper.get_player_stats(player_id)
        if stats:
            return jsonify(stats)
        else:
            return jsonify({'error': 'Unable to retrieve NFL player stats'}), 404
            
    except Exception as e:
        app.logger.error(f'Error getting NFL player stats: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/nfl/analyze_prop', methods=['POST'])
@async_route
async def nfl_analyze_prop():
    """Analyze NFL prop bet with enterprise AI models"""
    if not nfl_helper:
        return jsonify({'error': 'NFL system not available'}), 501
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        required_fields = ['player_id', 'prop_type', 'line', 'opponent_team_id']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields. Required: {required_fields}'}), 400

        player_id = data['player_id']
        player_name = data.get('player_name', 'Unknown Player')
        prop_type = data['prop_type']
        line = float(data['line'])
        opponent_team_id = data['opponent_team_id']  # Keep as string for NFL (team abbreviation)
        user_id = data.get('user_id', 'anonymous')
        decimal_odds = data.get('decimal_odds', 1.91)
        
        # Use enterprise NFL helper with user context
        enterprise_nfl_helper = NFLBettingHelper(user_id=user_id)
        
        analysis = await enterprise_nfl_helper.analyze_prop_bet(
            player_id=player_id,
            prop_type=prop_type,
            line=line,
            opponent_team_id=opponent_team_id  # Pass team abbreviation string
        )
        
        if not analysis or not analysis.get('success'):
            # Get the actual error message from the analysis if available
            error_msg = analysis.get('error', 'Unable to perform NFL analysis') if analysis else 'NFL analysis service unavailable'
            return jsonify({'error': error_msg, 'success': False}), 500
        
        # Add player and sport info to analysis
        analysis['player_name'] = player_name
        analysis['prop_type'] = prop_type
        analysis['line'] = line
        analysis['sport'] = 'NFL'
        
        return jsonify(analysis)
            
    except Exception as e:
        app.logger.error(f'Error analyzing NFL prop: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/nfl/weather/analysis', methods=['POST'])
def nfl_weather_analysis():
    """Get NFL weather impact analysis for outdoor games"""
    if not nfl_weather:
        return jsonify({'error': 'NFL weather system not available'}), 501
    
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        game_date = data.get('game_date')
        prop_type = data.get('prop_type', 'passing_yards')
        
        if not home_team or not game_date:
            return jsonify({'error': 'home_team and game_date required'}), 400
        
        weather_analysis = nfl_weather.get_weather_impact_analysis(
            home_team=home_team,
            game_date=game_date,
            prop_type=prop_type
        )
        
        return jsonify({
            'success': True,
            'weather_analysis': weather_analysis,
            'sport': 'NFL'
        })
        
    except Exception as e:
        app.logger.error(f'Error getting NFL weather analysis: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/nfl/weather/forecast')
def nfl_weather_forecast():
    """Get current NFL weather forecast for all outdoor games"""
    if not nfl_weather:
        return jsonify({'error': 'NFL weather system not available'}), 501
    
    try:
        forecast = nfl_weather.get_weekly_weather_outlook()
        
        return jsonify({
            'success': True,
            'forecast': forecast,
            'sport': 'NFL'
        })
        
    except Exception as e:
        app.logger.error(f'Error getting NFL weather forecast: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/nfl/teams/outdoor')
def nfl_outdoor_teams():
    """Get list of NFL teams with outdoor stadiums"""
    if not nfl_weather:
        return jsonify({'error': 'NFL weather system not available'}), 501
    
    try:
        outdoor_teams = nfl_weather.get_outdoor_stadium_teams()
        
        return jsonify({
            'success': True,
            'outdoor_teams': outdoor_teams,
            'sport': 'NFL'
        })
        
    except Exception as e:
        app.logger.error(f'Error getting NFL outdoor teams: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/nfl/situational/divisional', methods=['POST'])
def nfl_divisional_analysis():
    """Analyze NFL divisional game impact"""
    if not nfl_helper:
        return jsonify({'error': 'NFL system not available'}), 501
    
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        if not home_team or not away_team:
            return jsonify({'error': 'home_team and away_team required'}), 400
        
        analysis = nfl_helper.analyze_divisional_matchup(home_team, away_team)
        
        return jsonify({
            'success': True,
            'divisional_analysis': analysis,
            'sport': 'NFL'
        })
        
    except Exception as e:
        app.logger.error(f'Error analyzing NFL divisional matchup: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/nfl/analyze_moneyline', methods=['POST'])
def nfl_analyze_moneyline():
    """Analyze NFL moneyline bet"""
    if not nfl_helper:
        return jsonify({'error': 'NFL system not available'}), 501
    
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        home_odds = data.get('home_odds')
        away_odds = data.get('away_odds')
        
        if not all([home_team, away_team, home_odds, away_odds]):
            return jsonify({'error': 'home_team, away_team, home_odds, and away_odds required'}), 400
        
        analysis = nfl_helper.analyze_moneyline_bet(home_team, away_team, home_odds, away_odds)
        
        return jsonify({
            'success': True,
            'moneyline_analysis': analysis,
            'sport': 'NFL'
        })
        
    except Exception as e:
        app.logger.error(f'Error analyzing NFL moneyline: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/nfl/analyze_spread', methods=['POST'])
def nfl_analyze_spread():
    """Analyze NFL spread bet"""
    if not nfl_helper:
        return jsonify({'error': 'NFL system not available'}), 501
    
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        spread_line = data.get('spread_line')
        home_odds = data.get('home_odds', -110)
        away_odds = data.get('away_odds', -110)
        
        if not all([home_team, away_team, spread_line]):
            return jsonify({'error': 'home_team, away_team, and spread_line required'}), 400
        
        analysis = nfl_helper.analyze_spread_bet(home_team, away_team, spread_line, home_odds, away_odds)
        
        return jsonify({
            'success': True,
            'spread_analysis': analysis,
            'sport': 'NFL'
        })
        
    except Exception as e:
        app.logger.error(f'Error analyzing NFL spread: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/nfl/analyze_over_under', methods=['POST'])
def nfl_analyze_over_under():
    """Analyze NFL over/under total bet"""
    if not nfl_helper:
        return jsonify({'error': 'NFL system not available'}), 501
    
    try:
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        total_line = data.get('total_line')
        over_odds = data.get('over_odds', -110)
        under_odds = data.get('under_odds', -110)
        
        if not all([home_team, away_team, total_line]):
            return jsonify({'error': 'home_team, away_team, and total_line required'}), 400
        
        analysis = nfl_helper.analyze_over_under_total(home_team, away_team, total_line, over_odds, under_odds)
        
        return jsonify({
            'success': True,
            'over_under_analysis': analysis,
            'sport': 'NFL'
        })
        
    except Exception as e:
        app.logger.error(f'Error analyzing NFL over/under: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

# Multi-Sport Endpoints
@app.route('/sports/supported')
def supported_sports():
    """Get list of supported sports and their status"""
    try:
        sports_status = {
            'NBA': {
                'available': True,
                'features': ['prop_analysis', 'live_odds', 'moneyline_betting', 'spread_betting', 'over_under_betting', 'enterprise_ai', 'bankroll_management']
            },
            'NFL': {
                'available': nfl_helper is not None,
                'features': ['prop_analysis', 'weather_analysis', 'divisional_analysis', 'moneyline_betting', 'spread_betting', 'over_under_betting', 'enterprise_ai'] if nfl_helper else []
            }
        }
        
        return jsonify({
            'success': True,
            'sports': sports_status
        })
        
    except Exception as e:
        app.logger.error(f'Error getting supported sports: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/sports/switch', methods=['POST'])
def switch_sport():
    """Switch active sport context"""
    try:
        data = request.get_json()
        sport = data.get('sport', 'NBA').upper()
        
        if sport not in ['NBA', 'NFL']:
            return jsonify({'error': 'Unsupported sport'}), 400
        
        if sport == 'NFL' and not nfl_helper:
            return jsonify({'error': 'NFL system not available'}), 501
        
        # Store sport preference in session if needed
        session['active_sport'] = sport
        
        return jsonify({
            'success': True,
            'active_sport': sport,
            'message': f'Switched to {sport} analysis'
        })
        
    except Exception as e:
        app.logger.error(f'Error switching sport: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)