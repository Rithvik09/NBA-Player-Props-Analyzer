from flask import Flask, render_template, request, jsonify
from .basketball_betting_helper import BasketballBettingHelper
from .game_predictor import GamePredictor
import logging
from logging.handlers import RotatingFileHandler
import os
from nba_api.stats.static import players

# Initialize Flask app
app = Flask(__name__, 
    static_url_path='',
    static_folder='../static',
    template_folder='../templates')

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

betting_helper = BasketballBettingHelper()
game_predictor = GamePredictor(betting_helper)


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
    return render_template('index.html')


@app.route('/game-predictions')
def game_predictions_page():
    return render_template('game_predictions.html')

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
def analyze_prop():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        required_fields = ['player_id', 'prop_type', 'line', 'opponent_team_id']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields. Required: {required_fields}'}), 400

        player_id = data['player_id']
        prop_type = data['prop_type']
        line = float(data['line'])
        opponent_team_id = int(data['opponent_team_id'])
        game_location = data.get('game_location', 'auto')
        if game_location not in ('auto', 'home', 'away'):
            return jsonify({'error': 'Invalid game_location. Use auto|home|away', 'success': False}), 400

        # Non-blocking: keep daily datasets fresh in the background.
        betting_helper.kick_precompute_if_stale()
        
        # Fetch stats once; analysis will reuse them (avoids a second expensive nba_api call)
        stats = betting_helper.get_player_stats(player_id)
        if not stats:
            return jsonify({'error': 'Unable to retrieve player stats', 'success': False}), 500

        analysis = betting_helper.analyze_prop_bet(
            player_id=player_id,
            prop_type=prop_type,
            line=line,
            opponent_team_id=opponent_team_id,
            stats=stats,
            game_location=game_location
        )
        
        if analysis:
            return jsonify(analysis)  # The success flag is now included in the analysis dict
        else:
            return jsonify({'error': 'Unable to perform analysis', 'success': False}), 500
            
    except Exception as e:
        app.logger.error(f'Error analyzing prop: {e}')
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/predict_game', methods=['POST'])
def predict_game():
    """
    Predict game outcome (moneyline, spread, total) using player projections.
    
    Request JSON:
        {
            "home_team_id": 1610612744,  # Golden State Warriors
            "away_team_id": 1610612747,  # Los Angeles Lakers
            "home_roster": [201939, 203110, ...],  # Top 8-10 player IDs
            "away_roster": [2544, 1629029, ...],   # Top 8-10 player IDs
            "game_location": "home"  # optional: 'home', 'away', or 'neutral'
        }
    
    Response JSON:
        {
            "success": true,
            "moneyline": {
                "winner": 1610612744,
                "winner_team": "home",
                "win_probability": 0.652,
                "expected_margin": 5.2,
                "confidence": "MEDIUM"
            },
            "spread": {
                "spread": 5.2,
                "favorite": 1610612744,
                "favorite_team": "home",
                "line": 5.2,
                "confidence": "MEDIUM",
                "home_implied_score": 115.3,
                "away_implied_score": 110.1
            },
            "total": {
                "total": 225.4,
                "over_under": 225.4,
                "confidence": "HIGH",
                "pace_note": "Fast pace game",
                "avg_pace": 102.5
            },
            "team_projections": {
                "home": {...},
                "away": {...}
            }
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'success': False}), 400
        
        required_fields = ['home_team_id', 'away_team_id', 'home_roster', 'away_roster']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': f'Missing required fields. Required: {required_fields}',
                'success': False
            }), 400
        
        home_team_id = int(data['home_team_id'])
        away_team_id = int(data['away_team_id'])
        home_roster = data['home_roster']
        away_roster = data['away_roster']
        game_location = data.get('game_location', 'home')
        
        if not isinstance(home_roster, list) or not isinstance(away_roster, list):
            return jsonify({
                'error': 'Rosters must be lists of player IDs',
                'success': False
            }), 400
        
        if len(home_roster) == 0 or len(away_roster) == 0:
            return jsonify({
                'error': 'Rosters cannot be empty',
                'success': False
            }), 400
        
        # Predict game
        prediction = game_predictor.predict_game(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_roster=home_roster,
            away_roster=away_roster,
            game_location=game_location
        )
        
        # Add success flag
        prediction['success'] = True
        
        return jsonify(prediction)
    
    except Exception as e:
        app.logger.error(f'Error predicting game: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/job_status')
def job_status():
    """
    Lightweight status endpoint for background jobs (auto precompute/train).
    """
    try:
        auto = None
        if getattr(betting_helper, "_auto_jobs", None):
            auto = betting_helper._auto_jobs.status()
        return jsonify({
            "success": True,
            "auto_jobs": auto,
            "in_process_precompute": {
                "in_flight": bool(getattr(betting_helper, "_precompute_in_flight", False)),
                "last_attempt": int(getattr(betting_helper, "_precompute_last_attempt", 0) or 0),
            }
        })
    except Exception as e:
        app.logger.error(f'Error getting job status: {e}')
        return jsonify({'error': str(e), 'success': False}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)