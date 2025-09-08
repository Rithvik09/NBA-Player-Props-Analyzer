from flask import Flask, render_template, request, jsonify, session
from .basketball_betting_helper import BasketballBettingHelper
import logging
from logging.handlers import RotatingFileHandler
import os
import asyncio
from functools import wraps
from nba_api.stats.static import players
# Enterprise imports
try:
    from .auth.user_management import UserManager
    from .notifications.notification_system import NotificationManager
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__, 
    static_url_path='',
    static_folder='../static',
    template_folder='../templates')

# Configure secret key for sessions
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize enterprise features
if AUTH_AVAILABLE:
    user_manager = UserManager()
    notification_manager = NotificationManager()
else:
    user_manager = None
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
            user_data = user_manager.verify_jwt_token(token)
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
        opponent_team_id = int(data['opponent_team_id'])
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
    """Get live odds for specified sport"""
    try:
        enterprise_helper = BasketballBettingHelper()
        
        if enterprise_helper.odds_aggregator:
            odds = await enterprise_helper.odds_aggregator.fetch_all_odds(sport)
            return jsonify({
                'success': True,
                'odds': odds,
                'timestamp': odds.get('timestamp') if odds else None
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Live odds integration not available'
            }), 501
            
    except Exception as e:
        app.logger.error(f'Error getting live odds: {e}')
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

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)