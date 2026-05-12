"""
Evaluation Metrics for VidyutDrishti
Comprehensive evaluation including probabilistic metrics and baseline comparisons
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class RenewableEnergyEvaluator:
    """Comprehensive evaluation for renewable energy forecasting models"""

    def __init__(self):
        self.metrics = {}
        self.baseline_results = {}

    def calculate_point_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate point prediction metrics"""

        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Normalized metrics (common in energy forecasting)
        y_mean = y_true.mean()
        nmae = mae / y_mean  # Normalized MAE
        nrmse = rmse / y_mean  # Normalized RMSE

        # Capacity factor aware metrics
        capacity = y_true.max()  # Assume max observed is capacity
        cf_mae = mae / capacity  # Capacity factor MAE

        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'nmae': nmae,
            'nrmse': nrmse,
            'cf_mae': cf_mae
        }

    def calculate_probabilistic_metrics(self, y_true: pd.Series,
                                      y_pred_p10: np.ndarray,
                                      y_pred_p50: np.ndarray,
                                      y_pred_p90: np.ndarray) -> Dict[str, float]:
        """Calculate probabilistic forecasting metrics"""

        # Prediction Interval Coverage Probability (PICP)
        picp_80 = np.mean((y_true >= y_pred_p10) & (y_true <= y_pred_p90))

        # Mean Prediction Interval Width (MPIW)
        mpiw_80 = np.mean(y_pred_p90 - y_pred_p10)

        # Normalized MPIW
        y_range = y_true.max() - y_true.min()
        nmpiw_80 = mpiw_80 / y_range

        # Prediction Interval Normalized Average Width (PINAW)
        pinaw_80 = mpiw_80 / (y_true.max() - y_true.min())

        # Coverage Width-based Criterion (CWC)
        # Penalizes both coverage and width
        gamma = 0.8  # Target coverage for 80% interval
        eta = 10     # Weight for width penalty
        cwc_80 = nmpiw_80 * (1 + eta * max(0, gamma - picp_80) ** 2)

        # Pinball Loss (Quantile Loss)
        def pinball_loss(y_true, y_pred, quantile):
            errors = y_true - y_pred
            return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))

        pinball_p10 = pinball_loss(y_true, y_pred_p10, 0.1)
        pinball_p50 = pinball_loss(y_true, y_pred_p50, 0.5)
        pinball_p90 = pinball_loss(y_true, y_pred_p90, 0.9)
        pinball_avg = (pinball_p10 + pinball_p50 + pinball_p90) / 3

        # Continuous Ranked Probability Score (CRPS) approximation
        # Simplified CRPS for normal distribution assumption
        pred_mean = y_pred_p50
        pred_std = (y_pred_p90 - y_pred_p10) / (2 * 1.645)  # Assuming 80% interval ~ 1.645 sigma

        crps = np.mean([
            ((y_t - pred_mean) ** 2 - (pred_std ** 2) *
             (1/np.sqrt(np.pi) - 1)) * (2 * norm.cdf(y_t, pred_mean, pred_std) - 1) -
            2 * pred_std * norm.pdf(y_t, pred_mean, pred_std)
            for y_t in y_true
        ])

        return {
            'picp_80': picp_80,
            'mpiw_80': mpiw_80,
            'nmpiw_80': nmpiw_80,
            'pinaw_80': pinaw_80,
            'cwc_80': cwc_80,
            'pinball_p10': pinball_p10,
            'pinball_p50': pinball_p50,
            'pinball_p90': pinball_p90,
            'pinball_avg': pinball_avg,
            'crps': crps
        }

    def calculate_baseline_persistence(self, y: pd.Series, horizon: int = 1) -> np.ndarray:
        """Calculate persistence baseline (use previous value as prediction)"""

        return y.shift(horizon).fillna(method='bfill').values

    def calculate_baseline_climatology(self, y: pd.Series, freq: str = 'D') -> np.ndarray:
        """Calculate climatology baseline (average of same hour/day)"""

        # Group by time of day/week and calculate mean
        if freq == 'H':
            # Hourly climatology
            hourly_means = y.groupby(y.index.hour).mean()
            return y.index.hour.map(hourly_means).values
        elif freq == 'D':
            # Daily climatology
            daily_means = y.groupby(y.index.dayofyear).mean()
            return y.index.dayofyear.map(daily_means).values
        else:
            # Simple overall mean
            return np.full(len(y), y.mean())

    def calculate_baseline_seasonal_naive(self, y: pd.Series, season_length: int = 24) -> np.ndarray:
        """Calculate seasonal naive baseline (same hour yesterday)"""

        return y.shift(season_length).fillna(method='bfill').values

    def evaluate_model(self, y_true: pd.Series, predictions: Dict[str, np.ndarray],
                      model_name: str = "Model") -> Dict[str, float]:
        """Comprehensive model evaluation"""

        results = {}

        # Point prediction metrics (using P50)
        if 'P50' in predictions:
            point_metrics = self.calculate_point_metrics(y_true, predictions['P50'])
            results.update(point_metrics)

        # Probabilistic metrics (if quantiles available)
        if all(q in predictions for q in ['P10', 'P50', 'P90']):
            prob_metrics = self.calculate_probabilistic_metrics(
                y_true, predictions['P10'], predictions['P50'], predictions['P90']
            )
            results.update(prob_metrics)

        # Store results
        self.metrics[model_name] = results

        return results

    def evaluate_baselines(self, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate baseline models"""

        baselines = {}

        # Persistence
        pred_persistence = self.calculate_baseline_persistence(y)
        baselines['Persistence'] = self.calculate_point_metrics(y, pred_persistence)

        # Climatology
        pred_climatology = self.calculate_baseline_climatology(y)
        baselines['Climatology'] = self.calculate_point_metrics(y, pred_climatology)

        # Seasonal Naive
        pred_seasonal = self.calculate_baseline_seasonal_naive(y)
        baselines['Seasonal_Naive'] = self.calculate_point_metrics(y, pred_seasonal)

        self.baseline_results = baselines
        return baselines

    def compare_models(self, models_to_compare: List[str] = None) -> pd.DataFrame:
        """Compare multiple models"""

        if models_to_compare is None:
            models_to_compare = list(self.metrics.keys()) + list(self.baseline_results.keys())

        comparison_data = {}

        for model_name in models_to_compare:
            if model_name in self.metrics:
                comparison_data[model_name] = self.metrics[model_name]
            elif model_name in self.baseline_results:
                comparison_data[model_name] = self.baseline_results[model_name]

        return pd.DataFrame(comparison_data).T

    def plot_metrics_comparison(self, metrics_to_plot: List[str] = None):
        """Plot comparison of key metrics across models"""

        if metrics_to_plot is None:
            metrics_to_plot = ['mae', 'rmse', 'mape', 'picp_80', 'mpiw_80']

        comparison_df = self.compare_models()

        # Filter to available metrics
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

        if not available_metrics:
            print("No metrics available for plotting")
            return

        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                ax = axes[i]

                # Get values for this metric
                values = comparison_df[metric].dropna()

                if not values.empty:
                    bars = ax.bar(range(len(values)), values.values)
                    ax.set_xticks(range(len(values)))
                    ax.set_xticklabels(values.index, rotation=45, ha='right')
                    ax.set_title(f'{metric.upper()}')
                    ax.set_ylabel(metric.upper())
                    ax.grid(True, alpha=0.3)

                    # Add value labels
                    for bar, val in zip(bars, values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               '.2f', ha='center', va='bottom', fontsize=9)

        # Hide empty subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_prediction_intervals(self, y_true: pd.Series, predictions: Dict[str, np.ndarray],
                                model_name: str = "Model", n_samples: int = 200):
        """Plot prediction intervals with actual values"""

        if not all(q in predictions for q in ['P10', 'P50', 'P90']):
            print("Quantile predictions not available for uncertainty plotting")
            return

        # Sample for plotting
        indices = np.random.choice(len(y_true), min(n_samples, len(y_true)), replace=False)
        indices = np.sort(indices)

        y_sample = y_true.iloc[indices]
        p10_sample = predictions['P10'][indices]
        p50_sample = predictions['P50'][indices]
        p90_sample = predictions['P90'][indices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Time series plot
        ax1.fill_between(range(len(indices)), p10_sample, p90_sample,
                        alpha=0.3, color='blue', label='80% Prediction Interval')
        ax1.plot(range(len(indices)), p50_sample, 'b-', linewidth=2, label='P50 Prediction')
        ax1.plot(range(len(indices)), y_sample.values, 'r-', linewidth=1, alpha=0.7, label='Actual')
        ax1.set_title(f'{model_name}: Prediction Intervals')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Generation (MW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scatter plot
        ax2.scatter(y_sample, p50_sample, alpha=0.6, color='blue', label='Predictions')
        ax2.fill_between([y_sample.min(), y_sample.max()],
                        [y_sample.min(), y_sample.max()], alpha=0.2, color='gray', label='Perfect Prediction')
        ax2.set_title(f'{model_name}: Predicted vs Actual')
        ax2.set_xlabel('Actual Generation (MW)')
        ax2.set_ylabel('Predicted Generation (MW)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def generate_evaluation_report(self, output_file: str = None) -> str:
        """Generate comprehensive evaluation report"""

        report = f"""
VidyutDrishti Model Evaluation Report
{'='*50}

"""

        # Model comparison
        comparison_df = self.compare_models()

        if not comparison_df.empty:
            report += "Model Comparison:\n"
            report += comparison_df.round(4).to_string()
            report += "\n\n"

        # Best models for each metric
        if not comparison_df.empty:
            report += "Best Models by Metric:\n"
            for metric in comparison_df.columns:
                best_model = comparison_df[metric].idxmin()
                best_value = comparison_df[metric].min()
                report += f"{metric.upper()}: {best_model} ({best_value:.4f})\n"

        # Probabilistic metrics analysis
        prob_metrics = ['picp_80', 'mpiw_80', 'pinball_avg']
        available_prob = [m for m in prob_metrics if m in comparison_df.columns]

        if available_prob:
            report += f"\nProbabilistic Forecasting Analysis:\n"
            for metric in available_prob:
                values = comparison_df[metric].dropna()
                if not values.empty:
                    best = values.idxmin() if 'pinball' in metric else values.idxmax()
                    best_val = values.min() if 'pinball' in metric else values.max()
                    report += f"{metric.upper()}: {best} ({best_val:.4f})\n"

        # Recommendations
        report += f"\nRecommendations:\n"

        if not comparison_df.empty:
            # Best overall model (lowest MAE)
            if 'mae' in comparison_df.columns:
                best_overall = comparison_df['mae'].idxmin()
                report += f"- Best overall model: {best_overall}\n"

            # Best probabilistic model
            if 'pinball_avg' in comparison_df.columns:
                best_prob = comparison_df['pinball_avg'].idxmin()
                report += f"- Best probabilistic model: {best_prob}\n"

            # Reliability check
            if 'picp_80' in comparison_df.columns:
                picp_values = comparison_df['picp_80'].dropna()
                reliable_models = picp_values[picp_values >= 0.75]  # At least 75% coverage
                if not reliable_models.empty:
                    report += f"- Reliable models (PICP ≥ 75%): {', '.join(reliable_models.index)}\n"

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Evaluation report saved to {output_file}")

        return report

def evaluate_karnataka_models():
    """Evaluate all models on Karnataka data"""

    evaluator = RenewableEnergyEvaluator()

    try:
        # Load test data
        data_file = "../data/processed/karnataka_features.csv"
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Split into train/val/test
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.2)
        test_df = df[train_size + val_size:]

        print(f"Test data: {len(test_df)} samples")
        print(f"Date range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

        # Evaluate baselines
        print("\nEvaluating baselines...")
        baseline_results = evaluator.evaluate_baselines(test_df.set_index('timestamp')['generation_mw'])
        print("Baselines evaluated!")

        # Evaluate LightGBM
        try:
            from models.lightgbm.quantile_regression import QuantileLightGBM
            lgbm_model = QuantileLightGBM()
            lgbm_model.load_model("../models/lightgbm/quantile_model.pkl")

            X_test, y_test = lgbm_model.prepare_features(test_df)
            lgbm_preds = lgbm_model.predict(X_test)

            lgbm_results = evaluator.evaluate_model(y_test, lgbm_preds, "LightGBM_Quantile")
            print("LightGBM evaluated!")

        except Exception as e:
            print(f"LightGBM evaluation failed: {e}")

        # Evaluate Ensemble
        try:
            from models.ensemble.hybrid_ensemble import HybridEnsemble
            ensemble = HybridEnsemble()
            ensemble.load_ensemble("../models/ensemble/hybrid_model.pkl")

            # Need to prepare features using LightGBM's preprocessor
            X_test, y_test = lgbm_model.prepare_features(test_df)
            ensemble_preds = ensemble.predict(X_test)

            ensemble_results = evaluator.evaluate_model(y_test, ensemble_preds, "Hybrid_Ensemble")
            print("Ensemble evaluated!")

        except Exception as e:
            print(f"Ensemble evaluation failed: {e}")

        # Generate comparison
        comparison = evaluator.compare_models()

        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(comparison[['mae', 'rmse', 'mape']].round(3))

        if 'picp_80' in comparison.columns:
            print("\nProbabilistic Metrics:")
            print(comparison[['picp_80', 'mpiw_80', 'pinball_avg']].round(3))

        # Plot comparisons
        evaluator.plot_metrics_comparison()

        # Generate report
        evaluator.generate_evaluation_report("../docs/model_evaluation_report.txt")

        return evaluator

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None

if __name__ == "__main__":
    evaluate_karnataka_models()