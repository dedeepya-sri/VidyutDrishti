"""
Data Preprocessing Pipeline for VidyutDrishti
Handles missing values, time alignment, and feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class RenewableEnergyPreprocessor:
    """Preprocessing pipeline for renewable energy forecasting data"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()

    def _default_config(self) -> Dict:
        """Default preprocessing configuration"""
        return {
            'time_col': 'timestamp',
            'freq': 'H',  # Hourly data
            'max_gap_hours': 6,  # Maximum gap to interpolate
            'rolling_windows': [3, 6, 12, 24, 168],  # Hours for rolling stats
            'lag_features': [1, 2, 3, 6, 12, 24, 48, 168],  # Lag hours
            'weather_cols': [
                'temperature_2m', 'wind_speed', 'surface_solar_radiation_downwards',
                'total_cloud_cover', 'total_precipitation', 'surface_pressure'
            ],
            'generation_col': 'generation_mw'
        }

    def load_and_merge_data(
        self,
        weather_file: str,
        generation_file: str,
        plant_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load and merge weather and generation data"""

        print("Loading weather data...")
        weather_df = pd.read_csv(weather_file)
        weather_df[self.config['time_col']] = pd.to_datetime(weather_df[self.config['time_col']])

        print("Loading generation data...")
        generation_df = pd.read_csv(generation_file)
        generation_df[self.config['time_col']] = pd.to_datetime(generation_df[self.config['time_col']])

        # Filter plants if specified
        if plant_filter:
            generation_df = generation_df[generation_df['plant_name'].isin(plant_filter)]

        # Merge on timestamp (find nearest weather station for each plant)
        print("Merging weather and generation data...")

        # For simplicity, assume single weather location or take average
        # In production, you'd use spatial interpolation
        weather_agg = weather_df.groupby(self.config['time_col'])[self.config['weather_cols']].mean().reset_index()

        # Merge generation with weather
        merged_df = pd.merge_asof(
            generation_df.sort_values(self.config['time_col']),
            weather_agg.sort_values(self.config['time_col']),
            on=self.config['time_col'],
            direction='nearest'
        )

        print(f"Merged data shape: {merged_df.shape}")
        return merged_df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""

        print("Handling missing values...")

        # Count missing values before
        missing_before = df.isnull().sum()
        print(f"Missing values before processing:\n{missing_before[missing_before > 0]}")

        # Time-based interpolation for weather data
        weather_cols = [col for col in self.config['weather_cols'] if col in df.columns]
        df[weather_cols] = df[weather_cols].interpolate(method='time', limit=self.config['max_gap_hours'])

        # Forward fill remaining gaps
        df[weather_cols] = df[weather_cols].fillna(method='ffill', limit=24)

        # Backward fill for any remaining
        df[weather_cols] = df[weather_cols].fillna(method='bfill', limit=24)

        # For generation data, use plant-specific logic
        if self.config['generation_col'] in df.columns:
            # Group by plant and interpolate
            df[self.config['generation_col']] = df.groupby('plant_name')[self.config['generation_col']].transform(
                lambda x: x.interpolate(method='linear', limit=6)
            )

        # Drop rows with critical missing data
        critical_cols = [self.config['generation_col'], 'plant_name', self.config['time_col']]
        df = df.dropna(subset=critical_cols)

        # Count missing values after
        missing_after = df.isnull().sum()
        print(f"Missing values after processing:\n{missing_after[missing_after > 0]}")

        return df

    def align_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align time series to regular intervals"""

        print("Aligning time series...")

        # Set timestamp as index
        df = df.set_index(self.config['time_col'])

        # Resample to regular frequency
        # Group by plant and resample
        resampled_dfs = []

        for plant_name, plant_df in df.groupby('plant_name'):
            resampled = plant_df.resample(self.config['freq']).agg({
                'generation_mw': 'mean',
                'capacity_mw': 'first',
                'plant_type': 'first',
                'latitude': 'first',
                'longitude': 'first',
                **{col: 'mean' for col in self.config['weather_cols'] if col in plant_df.columns}
            })

            # Interpolate missing values created by resampling
            resampled = resampled.interpolate(method='time', limit=6)

            resampled['plant_name'] = plant_name
            resampled_dfs.append(resampled.reset_index())

        # Combine all plants
        aligned_df = pd.concat(resampled_dfs, ignore_index=True)

        print(f"Aligned data shape: {aligned_df.shape}")
        return aligned_df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features"""

        print("Creating temporal features...")

        # Ensure timestamp is datetime
        df[self.config['time_col']] = pd.to_datetime(df[self.config['time_col']])

        # Basic time features
        df['hour'] = df[self.config['time_col']].dt.hour
        df['day_of_year'] = df[self.config['time_col']].dt.dayofyear
        df['month'] = df[self.config['time_col']].dt.month
        df['weekday'] = df[self.config['time_col']].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

        return df

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for time series forecasting"""

        print("Creating lag features...")

        lag_cols = []

        for lag in self.config['lag_features']:
            lag_col = f'{self.config["generation_col"]}_lag_{lag}h'
            df[lag_col] = df.groupby('plant_name')[self.config['generation_col']].shift(lag)
            lag_cols.append(lag_col)

            # Also create weather lags for key variables
            for weather_col in ['temperature_2m', 'wind_speed', 'surface_solar_radiation_downwards']:
                if weather_col in df.columns:
                    weather_lag_col = f'{weather_col}_lag_{lag}h'
                    df[weather_lag_col] = df.groupby('plant_name')[weather_col].shift(lag)
                    lag_cols.append(weather_lag_col)

        print(f"Created {len(lag_cols)} lag features")
        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics"""

        print("Creating rolling window features...")

        rolling_cols = []

        for window in self.config['rolling_windows']:
            # Rolling statistics for generation
            gen_col = self.config['generation_col']
            if gen_col in df.columns:
                roll_mean = f'{gen_col}_roll_mean_{window}h'
                roll_std = f'{gen_col}_roll_std_{window}h'
                roll_min = f'{gen_col}_roll_min_{window}h'
                roll_max = f'{gen_col}_roll_max_{window}h'

                df[roll_mean] = df.groupby('plant_name')[gen_col].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
                df[roll_std] = df.groupby('plant_name')[gen_col].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
                df[roll_min] = df.groupby('plant_name')[gen_col].rolling(window=window, min_periods=1).min().reset_index(0, drop=True)
                df[roll_max] = df.groupby('plant_name')[gen_col].rolling(window=window, min_periods=1).max().reset_index(0, drop=True)

                rolling_cols.extend([roll_mean, roll_std, roll_min, roll_max])

            # Rolling statistics for key weather variables
            for weather_col in ['temperature_2m', 'wind_speed', 'surface_solar_radiation_downwards']:
                if weather_col in df.columns:
                    roll_mean_weather = f'{weather_col}_roll_mean_{window}h'
                    df[roll_mean_weather] = df.groupby('plant_name')[weather_col].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
                    rolling_cols.append(roll_mean_weather)

        print(f"Created {len(rolling_cols)} rolling features")
        return df

    def create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for renewable energy"""

        print("Creating domain-specific features...")

        # Solar-specific features
        solar_mask = df['plant_type'] == 'solar'
        if solar_mask.any():
            # Clear sky index (actual irradiance / clear sky irradiance)
            if 'surface_solar_radiation_downwards' in df.columns:
                # Estimate clear sky irradiance (simplified model)
                solar_df = df[solar_mask].copy()
                hour = solar_df['hour']
                day_of_year = solar_df['day_of_year']

                # Simplified clear sky model
                solar_elevation = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))  # Rough approximation
                clear_sky_irradiance = 1000 * solar_elevation * (1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365))

                df.loc[solar_mask, 'clear_sky_index'] = (
                    df.loc[solar_mask, 'surface_solar_radiation_downwards'] / clear_sky_irradiance.replace(0, 1)
                ).clip(0, 2)

            # Temperature-adjusted irradiance
            if 'temperature_2m' in df.columns and 'surface_solar_radiation_downwards' in df.columns:
                df.loc[solar_mask, 'irradiance_temp_adjusted'] = (
                    df.loc[solar_mask, 'surface_solar_radiation_downwards'] *
                    (1 - 0.005 * (df.loc[solar_mask, 'temperature_2m'] - 25))
                )

        # Wind-specific features
        wind_mask = df['plant_type'] == 'wind'
        if wind_mask.any():
            # Wind power density
            if 'wind_speed' in df.columns and 'surface_pressure' in df.columns and 'temperature_2m' in df.columns:
                # Air density calculation (simplified)
                air_density = df.loc[wind_mask, 'surface_pressure'] / (287.05 * (df.loc[wind_mask, 'temperature_2m'] + 273.15))
                df.loc[wind_mask, 'wind_power_density'] = 0.5 * air_density * (df.loc[wind_mask, 'wind_speed'] ** 3)

            # Wind speed categories
            if 'wind_speed' in df.columns:
                df.loc[wind_mask, 'wind_speed_category'] = pd.cut(
                    df.loc[wind_mask, 'wind_speed'],
                    bins=[0, 3, 8, 12, 25, 100],
                    labels=['low', 'moderate', 'high', 'very_high', 'extreme']
                )

        # Capacity factor
        if 'capacity_mw' in df.columns and self.config['generation_col'] in df.columns:
            df['capacity_factor'] = df[self.config['generation_col']] / df['capacity_mw']

        return df

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or cap outliers"""

        print("Removing outliers...")

        gen_col = self.config['generation_col']

        if gen_col in df.columns:
            # Cap generation at 1.1 * capacity (allowing some overgeneration)
            df[gen_col] = df[gen_col].clip(upper=df['capacity_mw'] * 1.1)

            # Remove negative generation
            df[gen_col] = df[gen_col].clip(lower=0)

        # Cap weather variables at reasonable ranges
        if 'temperature_2m' in df.columns:
            df['temperature_2m'] = df['temperature_2m'].clip(-10, 60)

        if 'wind_speed' in df.columns:
            df['wind_speed'] = df['wind_speed'].clip(0, 50)

        if 'surface_solar_radiation_downwards' in df.columns:
            df['surface_solar_radiation_downwards'] = df['surface_solar_radiation_downwards'].clip(0, 1500)

        return df

    def preprocess_pipeline(
        self,
        weather_file: str,
        generation_file: str,
        output_file: str,
        plant_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Complete preprocessing pipeline"""

        print("Starting VidyutDrishti preprocessing pipeline...")
        print("=" * 60)

        # Load and merge
        df = self.load_and_merge_data(weather_file, generation_file, plant_filter)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Align time series
        df = self.align_time_series(df)

        # Create features
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_domain_features(df)

        # Remove outliers
        df = self.remove_outliers(df)

        # Sort by timestamp and plant
        df = df.sort_values([self.config['time_col'], 'plant_name']).reset_index(drop=True)

        # Save processed data
        df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")

        # Summary
        print("\nPreprocessing Summary:")
        print(f"Final shape: {df.shape}")
        print(f"Plants: {df['plant_name'].nunique()}")
        print(f"Date range: {df[self.config['time_col']].min()} to {df[self.config['time_col']].max()}")
        print(f"Features created: {len(df.columns) - 8}")  # Subtracting base columns

        return df

def main():
    """Main preprocessing function"""

    preprocessor = RenewableEnergyPreprocessor()

    # File paths
    weather_file = "../data/raw/synthetic_era5_karnataka_20200101_20231231.csv"
    generation_file = "../data/processed/karnataka_generation_data.csv"
    output_file = "../data/processed/karnataka_features.csv"

    # Run preprocessing
    processed_df = preprocessor.preprocess_pipeline(
        weather_file=weather_file,
        generation_file=generation_file,
        output_file=output_file
    )

if __name__ == "__main__":
    main()