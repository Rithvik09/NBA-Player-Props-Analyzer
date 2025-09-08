"""
Advanced TensorFlow Deep Learning Models for NBA Props Prediction
Implements LSTM, Transformer, and CNN models for time-series prediction
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime, timedelta

class TensorFlowPredictor:
    def __init__(self, model_dir='ai_models/tensorflow'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self.lstm_model = None
        self.transformer_model = None
        self.cnn_model = None
        self.ensemble_model = None
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # Model configurations
        self.sequence_length = 10  # Games to look back
        self.feature_dim = 50  # Number of features
        self.prediction_horizon = 1  # Games to predict ahead
        
    def build_lstm_model(self, sequence_length=10, feature_dim=50):
        """Build LSTM model for time-series prediction"""
        model = models.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(sequence_length, feature_dim),
                       dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(64, return_sequences=True, 
                       dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # Third LSTM layer
            layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')  # For binary classification (over/under)
        ])
        
        # Advanced optimizer with learning rate scheduling
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        return model
    
    def build_transformer_model(self, sequence_length=10, feature_dim=50):
        """Build Transformer model with multi-head attention"""
        
        # Input layer
        inputs = layers.Input(shape=(sequence_length, feature_dim))
        
        # Positional encoding
        x = self._positional_encoding(inputs)
        
        # Multi-head attention layers
        for _ in range(3):  # 3 transformer blocks
            # Multi-head attention
            attention = layers.MultiHeadAttention(
                num_heads=8, key_dim=64, dropout=0.1
            )(x, x)
            
            # Add & Norm
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization()(x)
            
            # Feed-forward network
            ffn = layers.Dense(256, activation='relu')(x)
            ffn = layers.Dropout(0.1)(ffn)
            ffn = layers.Dense(feature_dim)(ffn)
            
            # Add & Norm
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization()(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs, outputs)
        
        # Compile with custom learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.96
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        return model
    
    def _positional_encoding(self, inputs):
        """Add positional encoding to inputs"""
        sequence_length = tf.shape(inputs)[1]
        feature_dim = inputs.shape[-1]
        
        # Create position encodings
        position = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, feature_dim, 2, dtype=tf.float32) * 
                         -(np.log(10000.0) / feature_dim))
        
        # Apply sin/cos positional encoding
        pos_encoding = tf.zeros((sequence_length, feature_dim))
        pos_encoding = tf.concat([
            tf.sin(position * div_term),
            tf.cos(position * div_term)
        ], axis=1)[:, :feature_dim]
        
        return inputs + pos_encoding
    
    def build_cnn_model(self, sequence_length=10, feature_dim=50):
        """Build CNN model for pattern recognition in game sequences"""
        model = models.Sequential([
            # Reshape for CNN
            layers.Reshape((sequence_length, feature_dim, 1), 
                          input_shape=(sequence_length, feature_dim)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        return model
    
    def build_ensemble_model(self, models_list):
        """Build ensemble model combining LSTM, Transformer, and CNN"""
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.feature_dim))
        
        # Get predictions from each model
        predictions = []
        for i, model in enumerate(models_list):
            # Create a new model without the final dense layer
            base_model = models.Model(
                inputs=model.input,
                outputs=model.layers[-2].output  # Second to last layer
            )
            base_model.trainable = False  # Freeze the base models
            
            pred = base_model(inputs)
            predictions.append(pred)
        
        # Concatenate all predictions
        concatenated = layers.Concatenate()(predictions)
        
        # Meta-learner layers
        x = layers.Dense(128, activation='relu')(concatenated)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        ensemble = models.Model(inputs, outputs)
        
        ensemble.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        return ensemble
    
    def prepare_sequences(self, data, target_column, sequence_length=10):
        """Prepare sequential data for training"""
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            # Features for the sequence
            sequence_features = data.iloc[i:i+sequence_length].drop(columns=[target_column])
            X.append(sequence_features.values)
            
            # Target for the next time step
            y.append(data.iloc[i+sequence_length][target_column])
        
        return np.array(X), np.array(y)
    
    def train_models(self, training_data, validation_data=None):
        """Train all models with advanced techniques"""
        print("🚀 Starting advanced ML training pipeline...")
        
        # Prepare data
        X_train, y_train = self.prepare_sequences(
            training_data, 'target', self.sequence_length
        )
        
        if validation_data is not None:
            X_val, y_val = self.prepare_sequences(
                validation_data, 'target', self.sequence_length
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        # Scale features
        X_train_scaled = self._scale_features(X_train, fit=True)
        X_val_scaled = self._scale_features(X_val, fit=False)
        
        # Advanced callbacks
        callbacks = self._get_callbacks()
        
        # Train LSTM model
        print("Training LSTM model...")
        self.lstm_model = self.build_lstm_model()
        lstm_history = self.lstm_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100, batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train Transformer model
        print("Training Transformer model...")
        self.transformer_model = self.build_transformer_model()
        transformer_history = self.transformer_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100, batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train CNN model
        print("Training CNN model...")
        self.cnn_model = self.build_cnn_model()
        cnn_history = self.cnn_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100, batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train Ensemble model
        print("Training Ensemble model...")
        self.ensemble_model = self.build_ensemble_model([
            self.lstm_model, self.transformer_model, self.cnn_model
        ])
        ensemble_history = self.ensemble_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=50, batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save models
        self.save_models()
        
        return {
            'lstm_history': lstm_history,
            'transformer_history': transformer_history,
            'cnn_history': cnn_history,
            'ensemble_history': ensemble_history
        }
    
    def _scale_features(self, X, fit=False):
        """Scale features for training"""
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit:
            X_scaled = self.feature_scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.feature_scaler.transform(X_reshaped)
        
        return X_scaled.reshape(original_shape)
    
    def _get_callbacks(self):
        """Get training callbacks"""
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            ),
            
            # Model checkpointing
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, 'best_model_{epoch:02d}.h5'),
                monitor='val_accuracy', save_best_only=True
            )
        ]
        
        return callbacks
    
    def predict_ensemble(self, X):
        """Make predictions using ensemble of all models"""
        if not all([self.lstm_model, self.transformer_model, 
                   self.cnn_model, self.ensemble_model]):
            raise ValueError("Models not trained yet!")
        
        # Scale features
        X_scaled = self._scale_features(X, fit=False)
        
        # Get predictions from individual models
        lstm_pred = self.lstm_model.predict(X_scaled, verbose=0)
        transformer_pred = self.transformer_model.predict(X_scaled, verbose=0)
        cnn_pred = self.cnn_model.predict(X_scaled, verbose=0)
        
        # Get ensemble prediction
        ensemble_pred = self.ensemble_model.predict(X_scaled, verbose=0)
        
        # Weighted combination (ensemble gets highest weight)
        final_pred = (
            0.4 * ensemble_pred +
            0.2 * lstm_pred +
            0.2 * transformer_pred +
            0.2 * cnn_pred
        )
        
        return {
            'ensemble_prediction': ensemble_pred,
            'lstm_prediction': lstm_pred,
            'transformer_prediction': transformer_pred,
            'cnn_prediction': cnn_pred,
            'final_prediction': final_pred,
            'confidence': self._calculate_prediction_confidence(
                [lstm_pred, transformer_pred, cnn_pred, ensemble_pred]
            )
        }
    
    def _calculate_prediction_confidence(self, predictions):
        """Calculate confidence based on model agreement"""
        predictions_array = np.array(predictions)
        
        # Calculate variance across models
        variance = np.var(predictions_array, axis=0)
        
        # Convert to confidence (lower variance = higher confidence)
        confidence = 1 / (1 + variance * 10)  # Scale appropriately
        
        return confidence
    
    def save_models(self):
        """Save all trained models"""
        if self.lstm_model:
            self.lstm_model.save(os.path.join(self.model_dir, 'lstm_model.h5'))
        
        if self.transformer_model:
            self.transformer_model.save(os.path.join(self.model_dir, 'transformer_model.h5'))
        
        if self.cnn_model:
            self.cnn_model.save(os.path.join(self.model_dir, 'cnn_model.h5'))
        
        if self.ensemble_model:
            self.ensemble_model.save(os.path.join(self.model_dir, 'ensemble_model.h5'))
        
        # Save scalers
        joblib.dump(self.feature_scaler, os.path.join(self.model_dir, 'feature_scaler.joblib'))
        joblib.dump(self.target_scaler, os.path.join(self.model_dir, 'target_scaler.joblib'))
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.lstm_model = keras.models.load_model(
                os.path.join(self.model_dir, 'lstm_model.h5')
            )
            self.transformer_model = keras.models.load_model(
                os.path.join(self.model_dir, 'transformer_model.h5')
            )
            self.cnn_model = keras.models.load_model(
                os.path.join(self.model_dir, 'cnn_model.h5')
            )
            self.ensemble_model = keras.models.load_model(
                os.path.join(self.model_dir, 'ensemble_model.h5')
            )
            
            # Load scalers
            self.feature_scaler = joblib.load(
                os.path.join(self.model_dir, 'feature_scaler.joblib')
            )
            self.target_scaler = joblib.load(
                os.path.join(self.model_dir, 'target_scaler.joblib')
            )
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False