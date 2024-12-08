from flask import Flask, render_template, request, jsonify
from src.basketball_betting_helper import BasketballBettingHelper
import logging
from logging.handlers import RotatingFileHandler
import os
from nba_api.stats.static import players

# Initialize Flask app
app = Flask(__name__, 
    static_url_path='',
    static_folder='../static',
    template_folder='../templates')

# Setup logging
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

# Initialize betting helper
betting_helper = BasketballBettingHelper()


@app.route('/test_api')
def test_api():
    """Test route to check NBA API functionality"""
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
    """Render the home page"""
    return render_template('index.html')

@app.route('/search_players')
def search_players():
    """Handle player search/autocomplete"""
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
    """Get comprehensive player statistics"""
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
def analyze_prop():
    """Analyze prop bet for given player and line"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        required_fields = ['player_id', 'prop_type', 'line']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        player_id = data['player_id']
        prop_type = data['prop_type']
        line = float(data['line'])
        
        analysis = betting_helper.analyze_prop_bet(
            player_id=player_id,
            prop_type=prop_type,
            line=line
        )
        
        if analysis:
            return jsonify({
                'success': True,
                'probability': analysis['hit_rate'],
                'average': analysis['average'],
                'last5_average': analysis['last5_average'],
                'times_hit': analysis['times_hit'],
                'total_games': analysis['total_games'],
                'trend': analysis['trend'],
                'recommendation': analysis['recommendation'],
                'confidence': analysis['confidence'],
                'edge': analysis['edge'],
                'values': analysis['values'],
                'predicted_value': analysis.get('predicted_value'),
                'over_probability': analysis.get('over_probability', 0)
            })
            
    except Exception as e:
        app.logger.error(f'Error analyzing prop: {e}')
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)