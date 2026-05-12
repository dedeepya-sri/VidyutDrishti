"""
ERA5 Data Ingestion for VidyutDrishti
Downloads high-resolution weather reanalysis data for Karnataka region

Note: Requires CDS API key. Sign up at: https://cds.climate.copernicus.eu/
"""

import cdsapi
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Tuple

class ERA5Client:
    """Client for ERA5 reanalysis data via CDS API"""

    def __init__(self, api_key: str = None):
        if api_key:
            # Set up CDS API credentials
            os.environ['CDSAPI_RC'] = f'url: https://cds.climate.copernicus.eu/api/v2\nkey: {api_key}'

        self.client = cdsapi.Client()

    def download_weather_data(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        start_date: str,
        end_date: str,
        output_file: str
    ) -> None:
        """
        Download ERA5 data for specified region and time period

        Parameters:
        - north, south, east, west: Bounding box coordinates
        - start_date, end_date: Date range in YYYY-MM-DD format
        - output_file: Path to save the downloaded .nc file
        """

        # Convert dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # ERA5 parameters relevant for renewable energy forecasting
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '10m_u_component_of_wind',     # Wind speed U component
                '10m_v_component_of_wind',     # Wind speed V component
                '2m_temperature',              # Temperature
                '2m_dewpoint_temperature',     # Dewpoint
                'surface_pressure',            # Pressure
                'total_precipitation',         # Precipitation
                'surface_solar_radiation_downwards',  # Solar irradiance
                'total_cloud_cover',           # Cloud cover
            ],
            'year': [str(year) for year in range(start.year, end.year + 1)],
            'month': [f"{month:02d}" for month in range(1, 13)],
            'day': [f"{day:02d}" for day in range(1, 32)],
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
            ],
            'area': [north, west, south, east],  # North, West, South, East
        }

        # Filter months and days based on date range
        if start.year == end.year:
            request['month'] = [f"{month:02d}" for month in range(start.month, end.month + 1)]
            if start.month == end.month:
                request['day'] = [f"{day:02d}" for day in range(start.day, end.day + 1)]

        print(f"Downloading ERA5 data to {output_file}...")
        self.client.retrieve('reanalysis-era5-single-levels', request, output_file)
        print("Download completed!")

def process_era5_netcdf(file_path: str, location_name: str) -> pd.DataFrame:
    """Process ERA5 netCDF file into pandas DataFrame"""

    # Open the netCDF file
    ds = xr.open_dataset(file_path)

    # Calculate wind speed from U and V components
    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)

    # Convert to DataFrame
    df = ds.to_dataframe().reset_index()

    # Add calculated fields
    df['wind_speed'] = wind_speed.values
    df['wind_direction'] = np.arctan2(ds['v10'], ds['u10']) * 180 / np.pi  # Convert to degrees

    # Rename columns to match our convention
    column_mapping = {
        't2m': 'temperature_2m',
        'd2m': 'dewpoint_2m',
        'sp': 'surface_pressure',
        'tp': 'total_precipitation',
        'ssrd': 'surface_solar_radiation_downwards',
        'tcc': 'total_cloud_cover',
        'u10': 'wind_u_component',
        'v10': 'wind_v_component'
    }

    df = df.rename(columns=column_mapping)

    # Convert temperature from Kelvin to Celsius
    if 'temperature_2m' in df.columns:
        df['temperature_2m'] = df['temperature_2m'] - 273.15

    if 'dewpoint_2m' in df.columns:
        df['dewpoint_2m'] = df['dewpoint_2m'] - 273.15

    # Add location info
    df['location'] = location_name

    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)

    return df

def get_karnataka_bbox() -> Tuple[float, float, float, float]:
    """Get bounding box for Karnataka state"""
    # Karnataka approximate bounds
    north = 18.5
    south = 11.5
    east = 78.5
    west = 74.0

    return north, south, east, west

def download_karnataka_era5_data(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    output_dir: str = "../data/raw",
    api_key: str = None
) -> None:
    """Download ERA5 data for Karnataka region"""

    os.makedirs(output_dir, exist_ok=True)

    north, south, east, west = get_karnataka_bbox()

    # Download data
    client = ERA5Client(api_key=api_key)
    raw_file = f"{output_dir}/era5_karnataka_{start_date.replace('-', '')}_{end_date.replace('-', '')}.nc"

    try:
        client.download_weather_data(
            north=north,
            south=south,
            east=east,
            west=west,
            start_date=start_date,
            end_date=end_date,
            output_file=raw_file
        )

        # Process the data
        df = process_era5_netcdf(raw_file, "Karnataka")

        # Save as CSV
        csv_file = raw_file.replace('.nc', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"Processed data saved to {csv_file}")

        # Summary
        print(f"\nData Summary:")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Total records: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")

    except Exception as e:
        print(f"Error downloading ERA5 data: {e}")
        print("Make sure you have a valid CDS API key")
        print("Sign up at: https://cds.climate.copernicus.eu/")

def create_synthetic_era5_fallback(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    output_dir: str = "../data/raw"
) -> None:
    """Create synthetic ERA5-like data when API access is not available"""

    os.makedirs(output_dir, exist_ok=True)

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')

    # Karnataka locations
    locations = [
        ("Bangalore", 12.9716, 77.5946),
        ("Mysore", 12.2958, 76.6394),
        ("Hubli", 15.3647, 75.1240),
        ("Mangalore", 12.9141, 74.8560),
    ]

    all_data = []

    for location_name, lat, lon in locations:
        print(f"Generating synthetic data for {location_name}...")

        np.random.seed(42)  # For reproducible results

        n_points = len(date_range)

        # Generate realistic weather patterns
        data = {
            'time': date_range,
            'location': location_name,
            'latitude': lat,
            'longitude': lon,
            'temperature_2m': 25 + 10 * np.sin(2 * np.pi * np.arange(n_points) / 24) + np.random.normal(0, 2, n_points),
            'dewpoint_2m': 20 + 8 * np.sin(2 * np.pi * np.arange(n_points) / 24) + np.random.normal(0, 1.5, n_points),
            'surface_pressure': 1013 + np.random.normal(0, 5, n_points),
            'total_precipitation': np.maximum(0, np.random.exponential(0.001, n_points)),
            'surface_solar_radiation_downwards': np.maximum(0, 800 * np.sin(np.pi * (np.arange(n_points) % 24) / 12) + np.random.normal(0, 50, n_points)),
            'total_cloud_cover': np.random.beta(2, 5, n_points) * 100,
            'wind_u_component': np.random.normal(0, 3, n_points),
            'wind_v_component': np.random.normal(0, 3, n_points),
        }

        # Calculate derived fields
        data['wind_speed'] = np.sqrt(data['wind_u_component']**2 + data['wind_v_component']**2)
        data['wind_direction'] = np.arctan2(data['wind_v_component'], data['wind_u_component']) * 180 / np.pi

        df = pd.DataFrame(data)
        all_data.append(df)

    # Combine all locations
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save data
    filename = f"{output_dir}/synthetic_era5_karnataka_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
    combined_df.to_csv(filename, index=False)
    print(f"Synthetic ERA5 data saved to {filename}")

    print(f"\nSynthetic Data Summary:")
    print(f"Locations: {len(locations)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Total records: {len(combined_df)}")
    print(f"Frequency: Hourly")

if __name__ == "__main__":
    # Try to download real ERA5 data (requires API key)
    try:
        download_karnataka_era5_data()
    except Exception as e:
        print(f"ERA5 download failed: {e}")
        print("Falling back to synthetic data...")

        # Create synthetic fallback data
        create_synthetic_era5_fallback()