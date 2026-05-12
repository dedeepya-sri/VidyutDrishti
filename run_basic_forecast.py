#!/usr/bin/env python3
"""
VidyutDrishti - Basic Renewable Energy Forecasting System
==========================================================

A simple working forecasting system for Karnataka SLDC.
"""

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """Load and prepare the data for forecasting"""
    print("Loading data...")

    # Load generation and weather data
    gen_df = pd.read_csv("data/processed/karnataka_generation_data.csv")
    weather_df = pd.read_csv("data/synthetic_weather_data.csv")

    # Merge on timestamp
    df = pd.merge(gen_df, weather_df, on='timestamp', how='inner')
    print(f"Data merged successfully. Shape: {df.shape}")

    # Basic preprocessing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month

    return df

def train_basic_model(df, plant_name='Pavagada_Solar_Park_Phase1'):
    """Train a basic forecasting model for a specific plant"""
    print(f"Training model for {plant_name}...")

    # Select plant data
    plant_df = df[df['plant_name'] == plant_name].copy()
    print(f"Plant data shape: {plant_df.shape}")

    # Features and target
    feature_cols = ['hour', 'month', 'temperature_2m', 'wind_speed',
                    'surface_solar_radiation_downwards', 'total_cloud_cover', 'capacity_mw']
    X = plant_df[feature_cols]
    y = plant_df['generation_mw']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\nModel Performance:")
    print(".2f")
    print(".2f")

    return model, X_test, y_test, y_pred

def plot_results(y_test, y_pred, plant_name):
    """Plot the forecasting results"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:200], label="Actual", alpha=0.7)
    plt.plot(y_pred[:200], label="Predicted", alpha=0.7)
    plt.legend()
    plt.title(f"Generation Forecast - {plant_name}")
    plt.xlabel("Time Steps")
    plt.ylabel("Generation (MW)")
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Main function to run the basic forecasting system"""
    print("VidyutDrishti - Basic Renewable Energy Forecasting")
    print("=" * 50)

    try:
        # Load and prepare data
        df = load_and_prepare_data()

        # Train model for solar plant
        model, X_test, y_test, y_pred = train_basic_model(df, 'Pavagada_Solar_Park_Phase1')

        # Plot results
        plot_results(y_test, y_pred, 'Pavagada Solar Park Phase 1')

        print("\n✅ Basic forecasting system completed successfully!")
        print("\nNext steps:")
        print("- Run the notebook: jupyter notebook notebooks/baseline_forecasting.ipynb")
        print("- For advanced features, see the docs/README.md")

    except FileNotFoundError as e:
        print(f"❌ Error: Data files not found. Please run data generation first:")
        print("   python data_pipeline/ingestion/generate_scada_data.py")
        print(f"   Error details: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check the data files and try again.")

if __name__ == "__main__":
    main()