"""
Ensemble Model for VidyutDrishti
Combines multiple models for improved probabilistic forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class HybridEnsemble:
    """Hybrid ensemble combining tree-based and deep learning models"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.models = {}
        self.weights = None
        self.is_trained = False

    def _default_config(self) -> Dict:
        return {
            'models': ['lightgbm', 'lstm'],  # Models to include
            'optimization_method': 'SLSQP',  # Weight optimization method
            'uncertainty_method': 'weighted_variance',  # How to combine uncertainties
            'validation_size': 0.2
        }

    def add_model(self, name: str, model_instance, model_type: str = 'point'):
        """Add a trained model to the ensemble"""

        self.models[name] = {
            'instance': model_instance,
            'type': model_type  # 'point', 'quantile', or 'uncertainty'
        }

    def optimize_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> np.ndarray:
        """Optimize ensemble weights using validation data"""

        def objective(weights):
            """Objective function to minimize (MAE)"""
            ensemble_pred = self._weighted_prediction(X_val, weights)
            return mean_absolute_error(y_val, ensemble_pred)

        def constraint(weights):
            """Weights must sum to 1"""
            return np.sum(weights) - 1.0

        # Initial weights (equal weighting)
        n_models = len(self.models)
        initial_weights = np.ones(n_models) / n_models

        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_models)]

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method=self.config['optimization_method'],
            bounds=bounds,
            constraints={'type': 'eq', 'fun': constraint}
        )

        if result.success:
            optimized_weights = result.x
            print(f"Optimized weights: {dict(zip(self.models.keys(), optimized_weights))}")
            return optimized_weights
        else:
            print("Weight optimization failed, using equal weights")
            return initial_weights

    def _weighted_prediction(self, X: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        """Generate weighted ensemble prediction"""

        predictions = []
        weights_list = []

        for i, (name, model_info) in enumerate(self.models.items()):
            model = model_info['instance']

            if model_info['type'] == 'point':
                # Point prediction model
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    if isinstance(pred, dict):
                        # Handle quantile predictions, use P50
                        pred = pred.get('P50', pred.get('P50', pred[list(pred.keys())[0]]))
                    predictions.append(pred)
                    weights_list.append(weights[i])

            elif model_info['type'] == 'quantile':
                # Quantile model - use P50 for point estimate
                pred_dict = model.predict(X)
                pred = pred_dict.get('P50', pred_dict[list(pred_dict.keys())[0]])
                predictions.append(pred)
                weights_list.append(weights[i])

        if not predictions:
            raise ValueError("No valid predictions from models")

        # Weighted average
        weights_array = np.array(weights_list)
        pred_array = np.array(predictions)

        return np.average(pred_array, weights=weights_array, axis=0)

    def _combine_uncertainties(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Combine uncertainties from different models"""

        all_predictions = {}

        for name, model_info in self.models.items():
            model = model_info['instance']

            if model_info['type'] == 'quantile':
                # Use quantile predictions directly
                preds = model.predict(X)
                for quantile_key, pred_values in preds.items():
                    if quantile_key not in all_predictions:
                        all_predictions[quantile_key] = []
                    all_predictions[quantile_key].append(pred_values)

            elif model_info['type'] == 'uncertainty':
                # Model provides uncertainty estimates
                if hasattr(model, 'predict_with_uncertainty'):
                    uncertainty_preds = model.predict_with_uncertainty(X)
                    for key, values in uncertainty_preds.items():
                        if key not in ['all_samples']:  # Skip raw samples
                            quantile_key = f"P{key}" if key.startswith('P') else key
                            if quantile_key not in all_predictions:
                                all_predictions[quantile_key] = []
                            all_predictions[quantile_key].append(values)

        # Combine predictions using different methods
        combined_preds = {}

        for quantile_key, pred_list in all_predictions.items():
            if not pred_list:
                continue

            pred_array = np.array(pred_list)

            if self.config['uncertainty_method'] == 'weighted_variance':
                # Weight by model performance (simplified - equal weights for now)
                weights = np.ones(len(pred_list)) / len(pred_list)
                combined_pred = np.average(pred_array, weights=weights, axis=0)

            elif self.config['uncertainty_method'] == 'median':
                # Use median of predictions
                combined_pred = np.median(pred_array, axis=0)

            elif self.config['uncertainty_method'] == 'mean':
                # Simple average
                combined_pred = np.mean(pred_array, axis=0)

            combined_preds[quantile_key] = combined_pred

        return combined_preds

    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series):
        """Train the ensemble by optimizing weights"""

        print("Training hybrid ensemble...")

        # Optimize weights using validation data
        self.weights = self.optimize_weights(X_val, y_val)

        # Store model weights
        self.model_weights = dict(zip(self.models.keys(), self.weights))

        self.is_trained = True
        print("Ensemble training completed!")

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate ensemble predictions"""

        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")

        # Generate point prediction
        point_pred = self._weighted_prediction(X, self.weights)

        # Generate uncertainty estimates
        uncertainty_preds = self._combine_uncertainties(X)

        # Combine results
        predictions = {'P50': point_pred}

        # Add uncertainty quantiles if available
        for key, values in uncertainty_preds.items():
            if key != 'P50':  # P50 already added
                predictions[key] = values

        # If no uncertainty estimates, create simple ones
        if len(predictions) == 1:
            # Simple uncertainty based on point prediction
            predictions['P10'] = point_pred * 0.9
            predictions['P90'] = point_pred * 1.1

        return predictions

    def predict_single(self, features: Dict) -> Dict[str, float]:
        """Predict for a single input"""

        X = pd.DataFrame([features])
        preds = self.predict(X)

        return {key: float(value[0]) if hasattr(value, '__len__') else float(value)
                for key, value in preds.items()}

    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate ensemble performance"""

        predictions = self.predict(X_test)

        # Point prediction metrics
        y_pred_p50 = predictions['P50']
        mae = mean_absolute_error(y_test, y_pred_p50)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_p50))

        # Uncertainty metrics (if available)
        uncertainty_metrics = {}
        if 'P10' in predictions and 'P90' in predictions:
            y_pred_p10 = predictions['P10']
            y_pred_p90 = predictions['P90']

            # Prediction Interval Coverage Probability
            picp = np.mean((y_test >= y_pred_p10) & (y_test <= y_pred_p90))

            # Mean Prediction Interval Width
            mpiw = np.mean(y_pred_p90 - y_pred_p10)

            uncertainty_metrics = {
                'picp': picp,
                'mpiw': mpiw,
                'picp_percent': picp * 100
            }

        results = {
            'mae': mae,
            'rmse': rmse,
            'mape': np.mean(np.abs((y_test - y_pred_p50) / y_test)) * 100,
            **uncertainty_metrics
        }

        return results

    def plot_ensemble_comparison(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Plot comparison of individual models vs ensemble"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Individual model predictions
        individual_preds = {}
        for name, model_info in self.models.items():
            model = model_info['instance']
            if hasattr(model, 'predict'):
                pred = model.predict(X_test)
                if isinstance(pred, dict):
                    pred = pred.get('P50', pred[list(pred.keys())[0]])
                individual_preds[name] = pred

        # Ensemble prediction
        ensemble_pred = self.predict(X_test)['P50']

        # Sample for plotting
        n_samples = min(200, len(y_test))
        indices = np.random.choice(len(y_test), n_samples, replace=False)
        indices = np.sort(indices)

        y_sample = y_test.iloc[indices]

        # Plot 1: Time series comparison
        ax1 = axes[0, 0]
        ax1.plot(range(n_samples), y_sample.values, 'k-', linewidth=2, label='Actual', alpha=0.8)

        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (name, pred) in enumerate(individual_preds.items()):
            color = colors[i % len(colors)]
            ax1.plot(range(n_samples), pred[indices], '--', color=color, alpha=0.7, label=f'{name} Pred')

        ax1.plot(range(n_samples), ensemble_pred[indices], 'r-', linewidth=2, label='Ensemble Pred')
        ax1.set_title('Model Predictions Comparison')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Generation (MW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error distributions
        ax2 = axes[0, 1]
        errors = {}
        for name, pred in individual_preds.items():
            errors[name] = y_test - pred

        errors['Ensemble'] = y_test - ensemble_pred

        for name, error in errors.items():
            sns.kdeplot(error, label=name, ax=ax2, alpha=0.7)

        ax2.set_title('Prediction Error Distributions')
        ax2.set_xlabel('Error (MW)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Scatter plots
        ax3 = axes[1, 0]
        ax3.scatter(y_sample, ensemble_pred[indices], alpha=0.6, color='red', label='Ensemble')
        ax3.plot([y_sample.min(), y_sample.max()], [y_sample.min(), y_sample.max()],
                'k--', alpha=0.8, label='Perfect Prediction')
        ax3.set_title('Ensemble: Predicted vs Actual')
        ax3.set_xlabel('Actual Generation (MW)')
        ax3.set_ylabel('Predicted Generation (MW)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Uncertainty intervals (if available)
        ax4 = axes[1, 1]
        ensemble_preds = self.predict(X_test)

        if 'P10' in ensemble_preds and 'P90' in ensemble_preds:
            p10_sample = ensemble_preds['P10'][indices]
            p90_sample = ensemble_preds['P90'][indices]
            p50_sample = ensemble_preds['P50'][indices]

            ax4.fill_between(range(n_samples), p10_sample, p90_sample,
                           alpha=0.3, color='blue', label='80% Prediction Interval')
            ax4.plot(range(n_samples), p50_sample, 'b-', linewidth=2, label='P50 Prediction')
            ax4.plot(range(n_samples), y_sample.values, 'r-', linewidth=1, alpha=0.7, label='Actual')
            ax4.set_title('Ensemble Uncertainty Intervals')
            ax4.set_xlabel('Sample Index')
            ax4.set_ylabel('Generation (MW)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No uncertainty\nestimates available',
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Uncertainty Intervals (N/A)')

        plt.tight_layout()
        plt.show()

    def save_ensemble(self, filepath: str):
        """Save ensemble model"""

        ensemble_data = {
            'models': {name: info for name, info in self.models.items()},
            'weights': self.weights,
            'model_weights': self.model_weights,
            'config': self.config,
            'is_trained': self.is_trained
        }

        joblib.dump(ensemble_data, filepath)
        print(f"Ensemble saved to {filepath}")

    def load_ensemble(self, filepath: str):
        """Load ensemble model"""

        ensemble_data = joblib.load(filepath)

        self.models = ensemble_data['models']
        self.weights = ensemble_data['weights']
        self.model_weights = ensemble_data.get('model_weights', {})
        self.config = ensemble_data['config']
        self.is_trained = ensemble_data['is_trained']

        print(f"Ensemble loaded from {filepath}")

def create_karnataka_ensemble():
    """Create and train ensemble for Karnataka data"""

    try:
        # Load LightGBM model
        from models.lightgbm.quantile_regression import QuantileLightGBM
        lgbm_model = QuantileLightGBM()
        lgbm_model.load_model("../models/lightgbm/quantile_model.pkl")

        # Load LSTM model (if available)
        try:
            from models.deep.lstm_model import LSTMTemporalModel
            lstm_model = LSTMTemporalModel()
            lstm_model.load_model("../models/deep/lstm_model.pkl")
            lstm_available = True
        except:
            print("LSTM model not available, using LightGBM only")
            lstm_available = False

        # Create ensemble
        ensemble = HybridEnsemble()

        # Add models
        ensemble.add_model('lightgbm', lgbm_model, 'quantile')

        if lstm_available:
            ensemble.add_model('lstm', lstm_model, 'point')

        # Load data for training ensemble
        data_file = "../data/processed/karnataka_features.csv"
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Split for ensemble training
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.2)

        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]

        # Prepare features
        X_train, y_train = lgbm_model.prepare_features(train_df)
        X_val, y_val = lgbm_model.prepare_features(val_df)

        # Train ensemble
        ensemble.train_ensemble(X_train, y_train, X_val, y_val)

        # Save ensemble
        ensemble.save_ensemble("../models/ensemble/hybrid_model.pkl")

        # Evaluate
        test_df = df[train_size + val_size:]
        X_test, y_test = lgbm_model.prepare_features(test_df)

        eval_results = ensemble.evaluate_ensemble(X_test, y_test)

        print("\nEnsemble Evaluation Results:")
        print(f"MAE: {eval_results['mae']:.3f} MW")
        print(f"RMSE: {eval_results['rmse']:.3f} MW")
        print(f"MAPE: {eval_results['mape']:.2f}%")

        if 'picp' in eval_results:
            print(f"PICP: {eval_results['picp']:.3f} ({eval_results['picp_percent']:.1f}%)")
            print(f"MPIW: {eval_results['mpiw']:.3f} MW")

        return ensemble

    except Exception as e:
        print(f"Error creating ensemble: {e}")
        return None

if __name__ == "__main__":
    create_karnataka_ensemble()