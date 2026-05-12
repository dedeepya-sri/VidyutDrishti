"""
LSTM Model for VidyutDrishti Temporal Forecasting
Deep learning component for capturing temporal dependencies
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class LSTMTemporalModel:
    """LSTM model for temporal pattern learning in renewable energy forecasting"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_scaler = StandardScaler()

    def _default_config(self) -> Dict:
        return {
            'sequence_length': 24,  # 24 hours of history
            'forecast_horizon': 1,   # Predict next hour
            'lstm_units': [64, 32], # LSTM layers
            'dropout_rate': 0.2,
            'dense_units': [32, 16],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 10,
            'validation_split': 0.2
        }

    def create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences for LSTM"""

        X_seq, y_seq = [], []

        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM model architecture"""

        inputs = Input(shape=input_shape)

        # LSTM layers
        x = inputs
        for i, units in enumerate(self.config['lstm_units']):
            return_sequences = i < len(self.config['lstm_units']) - 1
            x = LSTM(units, return_sequences=return_sequences, dropout=self.config['dropout_rate'])(x)

        # Dense layers
        for units in self.config['dense_units']:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.config['dropout_rate'])(x)

        # Output layer
        outputs = Dense(1)(x)  # Single output for point prediction

        model = Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'generation_mw') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for LSTM modeling"""

        # Select feature columns
        exclude_cols = [
            'timestamp', 'plant_name', 'plant_type', 'latitude', 'longitude',
            'capacity_mw', target_col
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])

        # Handle missing values
        X = X.fillna(X.mean())

        # Store feature names
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()

        return X, y

    def train(self, df: pd.DataFrame, target_col: str = 'generation_mw') -> Dict:
        """Train LSTM model"""

        print("Training LSTM temporal model...")
        print(f"Sequence length: {self.config['sequence_length']} hours")
        print(f"Forecast horizon: {self.config['forecast_horizon']} hour(s)")

        # Prepare features
        X, y = self.prepare_features(df, target_col)

        # Scale features and target
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled, self.config['sequence_length'])

        print(f"Training sequences: {X_seq.shape}")
        print(f"Target shape: {y_seq.shape}")

        # Build model
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self.model = self.build_model(input_shape)

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['patience'],
            restore_best_weights=True
        )

        model_checkpoint = ModelCheckpoint(
            '../models/deep/lstm_model.h5',
            monitor='val_loss',
            save_best_only=True
        )

        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            validation_split=self.config['validation_split'],
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )

        # Training results
        training_results = {
            'history': history.history,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_mae': history.history['mae'][-1],
            'final_val_mae': history.history['val_mae'][-1],
            'epochs_trained': len(history.history['loss'])
        }

        print("
Training completed!")
        print(".4f")
        print(".4f")

        # Save model components
        self.save_model('../models/deep/lstm_model.pkl')

        return training_results

    def predict(self, X: pd.DataFrame, sequence_length: Optional[int] = None) -> np.ndarray:
        """Generate predictions using LSTM"""

        if sequence_length is None:
            sequence_length = self.config['sequence_length']

        # Prepare features
        X_prepared, _ = self.prepare_features(X)

        # Scale features
        X_scaled = self.scaler.transform(X_prepared)

        # Create sequences for prediction
        if len(X_scaled) < sequence_length:
            raise ValueError(f"Not enough data for sequence length {sequence_length}")

        # Use last sequence_length points for prediction
        X_seq = X_scaled[-sequence_length:].reshape(1, sequence_length, -1)

        # Predict
        y_pred_scaled = self.model.predict(X_seq, verbose=0)

        # Inverse transform
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        return y_pred

    def predict_multi_step(self, X: pd.DataFrame, steps: int = 24) -> np.ndarray:
        """Generate multi-step predictions"""

        predictions = []
        current_X = X.copy()

        for step in range(steps):
            # Predict next step
            pred = self.predict(current_X)
            predictions.append(pred[0])

            # Update input with prediction (simplified - in practice you'd update all features)
            # This is a basic implementation; production would need proper feature updating
            if len(current_X) > 0:
                current_X = current_X.iloc[1:].copy()  # Remove oldest
                # Add new prediction as latest generation
                new_row = current_X.iloc[-1].copy()
                new_row['generation_mw'] = pred[0]
                current_X = pd.concat([current_X, new_row.to_frame().T], ignore_index=True)

        return np.array(predictions)

    def save_model(self, filepath: str):
        """Save LSTM model and components"""

        model_data = {
            'model_path': '../models/deep/lstm_model.h5',
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'config': self.config
        }

        joblib.dump(model_data, filepath)
        print(f"LSTM model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load LSTM model and components"""

        model_data = joblib.load(filepath)

        self.model = tf.keras.models.load_model(model_data['model_path'])
        self.scaler = model_data['scaler']
        self.target_scaler = model_data['target_scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']

        print(f"LSTM model loaded from {filepath}")

    def plot_training_history(self, history: Dict):
        """Plot training history"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MAE
        ax2.plot(history['mae'], label='Training MAE')
        ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Loss difference
        loss_diff = np.array(history['val_loss']) - np.array(history['loss'])
        ax3.plot(loss_diff)
        ax3.set_title('Validation - Training Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.grid(True, alpha=0.3)

        # Learning curves
        ax4.plot(np.log10(history['loss']), label='Log Training Loss')
        ax4.plot(np.log10(history['val_loss']), label='Log Validation Loss')
        ax4.set_title('Log Loss Curves')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Log Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

class UncertaintyLSTM(LSTMTemporalModel):
    """LSTM with uncertainty quantification using dropout"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.mc_samples = 50  # Monte Carlo samples for uncertainty

    def enable_dropout(self):
        """Enable dropout at prediction time for uncertainty estimation"""

        def monte_carlo_dropout(model):
            """Apply Monte Carlo dropout"""
            return tf.keras.layers.Dropout(0.2)(model.output, training=True)

        # Create new model with dropout enabled
        dropout_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=monte_carlo_dropout(self.model)
        )

        return dropout_model

    def predict_with_uncertainty(self, X: pd.DataFrame, n_samples: int = 50) -> Dict[str, np.ndarray]:
        """Predict with uncertainty using Monte Carlo dropout"""

        dropout_model = self.enable_dropout()

        predictions = []

        for _ in range(n_samples):
            pred_scaled = dropout_model.predict(X, verbose=0)
            pred = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # Percentiles for uncertainty intervals
        p10 = np.percentile(predictions, 10, axis=0)
        p50 = np.percentile(predictions, 50, axis=0)
        p90 = np.percentile(predictions, 90, axis=0)

        return {
            'mean': mean_pred,
            'std': std_pred,
            'P10': p10,
            'P50': p50,
            'P90': p90,
            'all_samples': predictions
        }

def train_karnataka_lstm():
    """Train LSTM model for Karnataka data"""

    # Load processed data
    data_file = "../data/processed/karnataka_features.csv"

    try:
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by time
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"Loaded data: {df.shape}")
        print(f"Plants: {df['plant_name'].unique()}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Initialize model
        model = LSTMTemporalModel()

        # Train model
        training_results = model.train(df)

        # Plot training history
        model.plot_training_history(training_results['history'])

        return model

    except FileNotFoundError:
        print(f"Data file not found: {data_file}")
        print("Please run preprocessing first.")
        return None

if __name__ == "__main__":
    train_karnataka_lstm()