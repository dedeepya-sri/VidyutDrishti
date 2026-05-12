"""
NASA POWER Data Ingestion for VidyutDrishti
Downloads solar irradiance and meteorological data for Karnataka region
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from typing import List, Tuple

class NASAPowerClient:
    """Client for NASA POWER API"""

    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def __init__(self):
        self.session = requests.Session()

    def download_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        parameters: List[str]
    ) -> pd.DataFrame:
        """
        Download weather data from NASA POWER

        Parameters:
        - latitude, longitude: Location coordinates
        - start_date, end_date: Date range in YYYYMMDD format
        - parameters: List of parameters to download
        """

        params = {
            "start": start_date,
            "end": end_date,
            "latitude": latitude,
            "longitude": longitude,
            "community": "RE",
            "parameters": ",".join(parameters),
            "format": "JSON",
            "header": "true"
        }

        response = self.session.get(self.BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        # Extract time series data
        timeseries = []
        for date_str, values in data['properties']['parameter'].items():
            if date_str != 'T2M':  # Skip parameter names
                continue

            # Get all parameters for this date
            row = {'date': date_str}
            for param in parameters:
                if param in data['properties']['parameter']:
                    param_data = data['properties']['parameter'][param]
                    if date_str in param_data:
                        row[param] = param_data[date_str]
                    else:
                        row[param] = None
                else:
                    row[param] = None

            timeseries.append(row)

        df = pd.DataFrame(timeseries)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        return df

def get_karnataka_locations() -> List[Tuple[str, float, float]]:
    """Get key locations in Karnataka for solar/wind plants"""
    return [
        ("Bangalore", 12.9716, 77.5946),
        ("Mysore", 12.2958, 76.6394),
        ("Hubli", 15.3647, 75.1240),
        ("Mangalore", 12.9141, 74.8560),
        ("Gulbarga", 17.3297, 76.8343),
        ("Belgaum", 15.8497, 74.4977),
        ("Davangere", 14.4644, 75.9218),
        ("Bellary", 15.1394, 76.9214),
        ("Tumkur", 13.3409, 77.1010),
        ("Raichur", 16.2120, 77.3439)
    ]

def download_karnataka_weather_data(
    start_date: str = "20200101",
    end_date: str = "20231231",
    output_dir: str = "../data/raw"
) -> None:
    """Download weather data for Karnataka locations"""

    os.makedirs(output_dir, exist_ok=True)

    # Parameters for solar forecasting
    solar_params = [
        "ALLSKY_SFC_SW_DWN",  # All sky surface shortwave downward irradiance
        "CLRSKY_SFC_SW_DWN",  # Clear sky surface shortwave downward irradiance
        "T2M",                # Temperature at 2m
        "RH2M",               # Relative humidity at 2m
        "WS2M",               # Wind speed at 2m
        "WD2M",               # Wind direction at 2m
        "PRECTOTCORR",        # Precipitation
        "PS",                 # Surface pressure
    ]

    client = NASAPowerClient()
    locations = get_karnataka_locations()

    all_data = []

    for location_name, lat, lon in locations:
        print(f"Downloading data for {location_name}...")

        try:
            df = client.download_weather_data(
                latitude=lat,
                longitude=lon,
                start_date=start_date,
                end_date=end_date,
                parameters=solar_params
            )

            df['location'] = location_name
            df['latitude'] = lat
            df['longitude'] = lon

            all_data.append(df)

            # Save individual location data
            filename = f"{output_dir}/nasa_power_{location_name.lower().replace(' ', '_')}_{start_date}_{end_date}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {filename}")

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"Error downloading {location_name}: {e}")
            continue

    # Combine all locations
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_filename = f"{output_dir}/nasa_power_karnataka_{start_date}_{end_date}.csv"
        combined_df.to_csv(combined_filename, index=False)
        print(f"Saved combined data: {combined_filename}")

        # Summary statistics
        print(f"\nData Summary:")
        print(f"Locations: {len(locations)}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Total records: {len(combined_df)}")
        print(f"Columns: {list(combined_df.columns)}")

if __name__ == "__main__":
    # Download 3 years of data
    download_karnataka_weather_data(
        start_date="20200101",
        end_date="20231231"
    )