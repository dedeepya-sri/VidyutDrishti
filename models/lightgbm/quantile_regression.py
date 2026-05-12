"""
LightGBM Quantile Regression Model for VidyutDrishti
Implements probabilistic forecasting with uncertainty quantification
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class QuantileLightGBM:
    """LightGBM model with quantile regression for probabilistic forecasting"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def _default_config(self) -> Dict:
        return {
            'quantiles': [0.1, 0.5, 0.9],  # P10, P50, P90
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 50,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'early_stopping_rounds': 50,
            'verbose': -1
        }

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'generation_mw') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling"""

        # Select feature columns (exclude metadata and target)
        exclude_cols = [
            'timestamp', 'plant_name', 'plant_type', 'latitude', 'longitude',
            'capacity_mw', target_col
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('generation_mw')]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                # Handle unseen categories
                try:
                    X[col] = self.label_encoders[col].transform(X[col])
                except ValueError:
                    # For unseen categories, assign a new label
                    X[col] = self.label_encoders[col].transform(X[col].fillna('unknown'))

        # Handle missing values
        X = X.fillna(X.mean())

        # Store feature names
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()

        return X, y

    def train_quantile_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        quantile: float
    ) -> LGBMRegressor:
        """Train a single quantile model"""

        # LightGBM parameters for quantile regression
        params = self.config.copy()
        params['objective'] = 'quantile'
        params['alpha'] = quantile

        model = LGBMRegressor(**params)

        # Fit model with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='quantile',
            callbacks=[
                # Early stopping callback
            ]
        )

        return model

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'generation_mw',
        test_size: int = 0.2
    ) -> Dict:
        """Train quantile regression models"""

        print("Training LightGBM quantile regression models...")
        print(f"Quantiles: {self.config['quantiles']}")

        # Prepare features
        X, y = self.prepare_features(df, target_col)

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)

        # Train models for each quantile
        training_results = {}

        for quantile in self.config['quantiles']:
            print(f"\nTraining quantile {quantile} model...")

            # Use last split for final model
            train_indices, val_indices = list(tscv.split(X_scaled))[-1]

            X_train, X_val = X_scaled.iloc[train_indices], X_scaled.iloc[val_indices]
            y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

            # Train model
            model = self.train_quantile_model(X_train, y_train, X_val, y_val, quantile)

            # Store model
            self.models[quantile] = model

            # Evaluate
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            training_results[quantile] = {
                'mae': mae,
                'rmse': rmse,
                'model': model
            }

            print(".3f"
        return training_results

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate probabilistic predictions"""

        # Prepare features
        X_prepared, _ = self.prepare_features(X)

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_prepared),
            columns=X_prepared.columns,
            index=X_prepared.index
        )

        # Generate predictions for each quantile
        predictions = {}
        for quantile, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions[f'P{int(quantile*100)}'] = pred

        return predictions

    def predict_single(self, features: Dict) -> Dict[str, float]:
        """Predict for a single input"""

        # Convert to DataFrame
        X = pd.DataFrame([features])

        # Get predictions
        preds = self.predict(X)

        # Return as dictionary
        return {key: float(value[0]) for key, value in preds.items()}

    def calculate_uncertainty_metrics(self, y_true: pd.Series, predictions: Dict) -> Dict:
        """Calculate uncertainty quantification metrics"""

        y_pred_p50 = predictions['P50']
        y_pred_p10 = predictions['P10']
        y_pred_p90 = predictions['P90']

        # Prediction Interval Coverage Probability (PICP)
        picp = np.mean((y_true >= y_pred_p10) & (y_true <= y_pred_p90))

        # Mean Prediction Interval Width (MPIW)
        mpiw = np.mean(y_pred_p90 - y_pred_p10)

        # Normalized MPIW
        y_range = y_true.max() - y_true.min()
        nmpiw = mpiw / y_range

        # Prediction Interval Normalized Average Width (PINAW)
        pinaw = mpiw / (y_true.max() - y_true.min())

        # Coverage Width-based Criterion (CWC)
        # Penalizes both coverage and width
        gamma = 0.95  # Target coverage
        eta = 10     # Weight for width penalty
        cwc = nmpiw * (1 + eta * max(0, gamma - picp) ** 2)

        return {
            'picp': picp,
            'mpiw': mpiw,
            'nmpiw': nmpiw,
            'pinaw': pinaw,
            'cwc': cwc
        }

    def save_model(self, filepath: str):
        """Save trained models"""

        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'config': self.config
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained models"""

        model_data = joblib.load(filepath)

        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']

        print(f"Model loaded from {filepath}")

    def get_feature_importance(self, quantile: float = 0.5) -> pd.DataFrame:
        """Get feature importance for a specific quantile model"""

        if quantile not in self.models:
            raise ValueError(f"Model for quantile {quantile} not found")

        model = self.models[quantile]
        importance = model.feature_importances_

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def plot_feature_importance(self, quantile: float = 0.5, top_n: int = 20):
        """Plot feature importance"""

        importance_df = self.get_feature_importance(quantile)

        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=importance_df.head(top_n),
            x='importance',
            y='feature',
            palette='viridis'
        )
        plt.title(f'Feature Importance - P{int(quantile*100)} Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    def plot_prediction_intervals(self, y_true: pd.Series, predictions: Dict, n_samples: int = 100):
        """Plot prediction intervals with actual values"""

        plt.figure(figsize=(15, 8))

        # Sample for plotting
        indices = np.random.choice(len(y_true), n_samples, replace=False)
        indices = np.sort(indices)

        y_sample = y_true.iloc[indices]
        p10_sample = predictions['P10'][indices]
        p50_sample = predictions['P50'][indices]
        p90_sample = predictions['P90'][indices]

        # Plot prediction intervals
        plt.fill_between(range(n_samples), p10_sample, p90_sample, alpha=0.3, color='blue', label='80% Prediction Interval')
        plt.plot(range(n_samples), p50_sample, 'b-', linewidth=2, label='P50 Prediction')
        plt.plot(range(n_samples), y_sample.values, 'r-', linewidth=1, alpha=0.7, label='Actual')

        plt.xlabel('Sample Index')
        plt.ylabel('Generation (MW)')
        plt.title('Probabilistic Forecast with Prediction Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def train_karnataka_model():
    """Train quantile regression model for Karnataka data"""

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
        model = QuantileLightGBM()

        # Train model
        training_results = model.train(df, test_size=0.2)

        # Print results
        print("\nTraining Results:")
        for quantile, results in training_results.items():
            print(f"P{int(quantile*100)}: MAE={results['mae']:.3f}, RMSE={results['rmse']:.3f}")

        # Save model
        model.save_model("../models/lightgbm/quantile_model.pkl")

        # Plot feature importance
        model.plot_feature_importance(quantile=0.5)

        return model

    except FileNotFoundError:
        print(f"Data file not found: {data_file}")
        print("Please run preprocessing first.")
        return None

if __name__ == "__main__":
    train_karnataka_model()