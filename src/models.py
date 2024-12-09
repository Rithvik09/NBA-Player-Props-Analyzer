from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.stats
import numpy as np
import pandas as pd
import joblib
import os

class MLPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.classification_model = None
        self.regression_model = None
        self._load_or_create_models()

    def _load_or_create_models(self):
        #Load existing models or create new ones
        try:
            self.classification_model = joblib.load(f'{self.model_dir}/classification_model.joblib')
            self.regression_model = joblib.load(f'{self.model_dir}/regression_model.joblib')
            self.scaler = joblib.load(f'{self.model_dir}/scaler.joblib')
        except:
            self.classification_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
            self.regression_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

    def prepare_features(self, player_stats, opponent_data=None, matchup_history=None, team_data=None):
        features = {}
    
        features.update({
            'recent_avg': float(player_stats.get('last5_avg', 0)),
            'season_avg': float(player_stats.get('avg', 0)),
            'max_recent': float(max(player_stats.get('values', [0]))),
            'min_recent': float(min(player_stats.get('values', [0]))),
            'stddev': float(np.std(player_stats.get('values', [0]))),
            'games_played': len(player_stats.get('values', [])),
        })

        if team_data:
            features.update({
                'team_pace': float(team_data.get('pace', 0)),
                'team_offensive_rating': float(team_data.get('offensive_rating', 0)),
                'team_three_point_rate': float(team_data.get('three_point_rate', 0)),
                'team_possessions': float(team_data.get('possessions', 0)),
                'team_fastbreak_points': float(team_data.get('fastbreak_points', 0))
            })

        if matchup_history and opponent_data:
            features.update({
                'vs_team_avg': float(np.mean(matchup_history.get('values', [0]))),
                'vs_team_last_game': float(matchup_history.get('values', [0])[0] if matchup_history.get('values') else 0),
                'defensive_rating_against': float(opponent_data.get('defensive_rating', 0)),
                'matchup_pace': float((team_data.get('pace', 0) + opponent_data.get('pace', 0)) / 2),
                'strength_of_opponent': float(opponent_data.get('net_rating', 0))
            })

        if team_data and opponent_data:
            features.update({
                'team_injuries_impact': float(team_data.get('injuries_impact', 0)),
                'opponent_injuries_impact': float(opponent_data.get('injuries_impact', 0)),
                'available_rotation_players': int(team_data.get('available_players', 0)),
                'opponent_available_players': int(opponent_data.get('available_players', 0))
            })

        return features

    def train(self, training_data):
        if not training_data:
            raise ValueError("No training data provided")

        X = pd.DataFrame([data['features'] for data in training_data])
        y_class = [1 if data['result'] > data['line'] else 0 for data in training_data]
        y_reg = [data['result'] for data in training_data]

        X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2)
        _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.classification_model.fit(X_train_scaled, y_class_train)
        self.regression_model.fit(X_train_scaled, y_reg_train)

        joblib.dump(self.classification_model, f'{self.model_dir}/classification_model.joblib')
        joblib.dump(self.regression_model, f'{self.model_dir}/regression_model.joblib')
        joblib.dump(self.scaler, f'{self.model_dir}/scaler.joblib')

    def predict(self, features, line):
       try:
           features_df = pd.DataFrame([features])

           for col in features_df.columns:
               mean = features_df[col].mean() 
               std = features_df[col].std()
               if std != 0:
                   features_df[col] = (features_df[col] - mean) / std
               else:
                   features_df[col] = 0

           recent_avg = features.get('recent_avg', 0)
           season_avg = features.get('season_avg', 0)
           std_dev = features.get('stddev', 0)

           predicted_value = (0.7 * recent_avg + 0.3 * season_avg)

           z_score = (line - predicted_value) / (std_dev + 1e-6)
           over_prob = 1 - scipy.stats.norm.cdf(z_score)

           edge = ((predicted_value - line) / line) if line > 0 else 0
           
           return {
               'over_probability': float(over_prob),
               'predicted_value': float(predicted_value),
               'recommendation': self._generate_recommendation(over_prob, predicted_value, line),
               'confidence': self._calculate_confidence(over_prob, predicted_value, line),
               'edge': edge
           }
           
       except Exception as e:
           print(f"Prediction error: {e}")
           return {
               'over_probability': 0.6 if features.get('season_avg', line) > line else 0.4,
               'predicted_value': float(features.get('season_avg', line)),
               'recommendation': 'OVER' if features.get('season_avg', line) > line else 'UNDER',
               'confidence': 'LOW',
               'edge': 0.0
           }

    def _calculate_confidence(self, prob, pred_value, line):
        #Calculate prediction confidence
        prob_distance = abs(prob - 0.5)
        value_distance = abs(pred_value - line) / line if line > 0 else 0
        
        if prob_distance > 0.2 and value_distance > 0.1:
            return 'HIGH'
        elif prob_distance > 0.15 or value_distance > 0.07:
            return 'MEDIUM'
        return 'LOW'

    def _generate_recommendation(self, prob, pred_value, line):
        #Generate betting recommendation
        if prob > 0.65 or (pred_value - line) / line > 0.1:
            return 'OVER'
        elif prob < 0.35 or (line - pred_value) / line > 0.1:
            return 'UNDER'
        return 'PASS'