"""
SCADA-like Generation Data Generation for VidyutDrishti
Creates realistic solar and wind power generation data based on weather conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple

class PlantSimulator:
    """Simulates renewable energy plant generation based on weather data"""

    def __init__(self, plant_type: str, capacity_mw: float):
        self.plant_type = plant_type
        self.capacity_mw = capacity_mw

        # Plant-specific parameters
        if plant_type == "solar":
            self.efficiency = np.random.uniform(0.15, 0.22)  # Solar panel efficiency
            self.temperature_coefficient = -0.004  # Power loss per degree C
            self.base_temperature = 25  # STC temperature
        elif plant_type == "wind":
            self.cut_in_speed = 3.0  # m/s
            self.rated_speed = 12.0  # m/s
            self.cut_out_speed = 25.0  # m/s
            self.rated_power = capacity_mw

    def calculate_solar_generation(
        self,
        irradiance: float,  # W/m²
        temperature: float,  # °C
        cloud_cover: float = 0  # 0-100%
    ) -> float:
        """Calculate solar power generation"""

        if irradiance <= 0:
            return 0.0

        # Adjust for cloud cover (reduce irradiance)
        clear_sky_factor = (100 - cloud_cover) / 100
        effective_irradiance = irradiance * clear_sky_factor

        # Basic PV model
        base_power = effective_irradiance * self.efficiency

        # Temperature effect
        temp_loss = self.temperature_coefficient * (temperature - self.base_temperature)
        power = base_power * (1 + temp_loss)

        # Convert to MW and cap at capacity
        generation_mw = min(power / 1e6 * 1000, self.capacity_mw)  # Assuming 1000m² per MW

        return max(0, generation_mw)

    def calculate_wind_generation(self, wind_speed: float) -> float:
        """Calculate wind power generation using power curve"""

        if wind_speed < self.cut_in_speed or wind_speed > self.cut_out_speed:
            return 0.0
        elif wind_speed <= self.rated_speed:
            # Cubic power curve approximation
            power_ratio = (wind_speed ** 3 - self.cut_in_speed ** 3) / (self.rated_speed ** 3 - self.cut_in_speed ** 3)
            return self.rated_power * power_ratio
        else:
            return self.rated_power

def create_karnataka_power_plants() -> List[Dict]:
    """Create realistic power plant configurations for Karnataka"""

    plants = [
        # Solar plants
        {
            "name": "Pavagada_Solar_Park_Phase1",
            "type": "solar",
            "capacity_mw": 500,
            "latitude": 14.1,
            "longitude": 77.3,
            "commission_date": "2020-01-01"
        },
        {
            "name": "Yelesandra_Solar_Park",
            "type": "solar",
            "capacity_mw": 50,
            "latitude": 12.8,
            "longitude": 77.1,
            "commission_date": "2021-06-01"
        },
        {
            "name": "Ramanagara_Solar_Park",
            "type": "solar",
            "capacity_mw": 75,
            "latitude": 12.7,
            "longitude": 77.3,
            "commission_date": "2022-03-01"
        },
        # Wind plants
        {
            "name": "Chitradurga_Wind_Farm",
            "type": "wind",
            "capacity_mw": 100,
            "latitude": 14.2,
            "longitude": 76.4,
            "commission_date": "2019-08-01"
        },
        {
            "name": "Kundapur_Wind_Farm",
            "type": "wind",
            "capacity_mw": 150,
            "latitude": 13.6,
            "longitude": 74.7,
            "commission_date": "2020-11-01"
        },
        {
            "name": "Dharwad_Wind_Farm",
            "type": "wind",
            "capacity_mw": 80,
            "latitude": 15.5,
            "longitude": 75.0,
            "commission_date": "2021-04-01"
        }
    ]

    return plants

def generate_generation_data(
    weather_df: pd.DataFrame,
    plants: List[Dict],
    output_dir: str = "data/processed"
) -> pd.DataFrame:
    """Generate generation data based on weather conditions"""

    os.makedirs(output_dir, exist_ok=True)

    all_generation = []

    for plant in plants:
        print(f"Generating data for {plant['name']}...")

        plant_sim = PlantSimulator(plant['type'], plant['capacity_mw'])

        # Filter weather data for plant location (simple nearest neighbor)
        plant_weather = weather_df.copy()

        # Add some spatial variation
        np.random.seed(hash(plant['name']) % 2**32)
        temp_noise = np.random.normal(0, 1, len(plant_weather))
        wind_noise = np.random.normal(0, 0.5, len(plant_weather))
        irr_noise = np.random.normal(0, 20, len(plant_weather))

        plant_weather['temperature_2m'] += temp_noise
        plant_weather['wind_speed'] += wind_noise

        if 'surface_solar_radiation_downwards' in plant_weather.columns:
            plant_weather['surface_solar_radiation_downwards'] += irr_noise
            plant_weather['surface_solar_radiation_downwards'] = plant_weather['surface_solar_radiation_downwards'].clip(lower=0)

        # Generate generation data
        generations = []

        for idx, row in plant_weather.iterrows():
            timestamp = row['time'] if 'time' in row else row['date']

            if plant['type'] == 'solar':
                # Extract solar parameters
                irradiance = row.get('surface_solar_radiation_downwards', row.get('ALLSKY_SFC_SW_DWN', 800))
                temperature = row.get('temperature_2m', row.get('T2M', 25))
                cloud_cover = row.get('total_cloud_cover', 30)

                generation = plant_sim.calculate_solar_generation(
                    irradiance=irradiance,
                    temperature=temperature,
                    cloud_cover=cloud_cover
                )

            elif plant['type'] == 'wind':
                wind_speed = row.get('wind_speed', row.get('WS2M', 5))
                generation = plant_sim.calculate_wind_generation(wind_speed)

            # Add operational variations
            # Plant availability (95% uptime)
            availability = 0.95
            if np.random.random() < (1 - availability):
                generation *= np.random.uniform(0, 0.5)  # Partial or full outage

            # Curtailment (occasional)
            if np.random.random() < 0.02:  # 2% chance
                generation *= np.random.uniform(0.1, 0.9)

            # Measurement noise
            noise_factor = np.random.normal(1, 0.02)
            generation *= noise_factor

            # Ensure non-negative
            generation = max(0, generation)

            generations.append({
                'timestamp': timestamp,
                'plant_name': plant['name'],
                'plant_type': plant['type'],
                'capacity_mw': plant['capacity_mw'],
                'generation_mw': round(generation, 3),
                'capacity_factor': round(generation / plant['capacity_mw'], 4),
                'latitude': plant['latitude'],
                'longitude': plant['longitude']
            })

        plant_df = pd.DataFrame(generations)
        all_generation.append(plant_df)

        # Save individual plant data
        filename = f"{output_dir}/generation_{plant['name'].lower()}_{plant['type']}.csv"
        plant_df.to_csv(filename, index=False)

    # Combine all plants
    combined_df = pd.concat(all_generation, ignore_index=True)

    # Save combined data
    combined_filename = f"{output_dir}/karnataka_generation_data.csv"
    print(f"Saving to: {os.path.abspath(combined_filename)}")
    combined_df.to_csv(combined_filename, index=False)
    print(f"File exists after save: {os.path.exists(combined_filename)}")
    print(f"Combined generation data saved to {combined_filename}")

    return combined_df

def create_synthetic_weather_fallback(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    freq: str = "H"
) -> pd.DataFrame:
    """Create synthetic weather data when real data is not available"""

    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    np.random.seed(42)

    n_points = len(date_range)

    # Generate realistic weather patterns
    weather_data = {
        'time': date_range,
        'temperature_2m': 25 + 8 * np.sin(2 * np.pi * (np.arange(n_points) % 24) / 24) + np.random.normal(0, 2, n_points),
        'wind_speed': 3 + 2 * np.random.weibull(2, n_points),
        'surface_solar_radiation_downwards': np.maximum(0, 600 * np.sin(np.pi * (np.arange(n_points) % 24) / 12) + np.random.normal(0, 100, n_points)),
        'total_cloud_cover': np.random.beta(2, 5, n_points) * 100,
        'total_precipitation': np.maximum(0, np.random.exponential(0.001, n_points)),
    }

    return pd.DataFrame(weather_data)

def main():
    """Main function to generate Karnataka generation data"""

    print("VidyutDrishti: Generating Karnataka Power Generation Data")
    print("=" * 60)

    # Create plants
    plants = create_karnataka_power_plants()
    print(f"Created {len(plants)} power plants:")
    for plant in plants:
        print(f"  - {plant['name']}: {plant['capacity_mw']} MW {plant['type']}")

    # Try to load real weather data
    weather_file = "data/raw/synthetic_era5_karnataka_20200101_20231231.csv"

    if os.path.exists(weather_file):
        print(f"\nLoading weather data from {weather_file}")
        weather_df = pd.read_csv(weather_file)
        if 'time' not in weather_df.columns and 'date' in weather_df.columns:
            weather_df['time'] = pd.to_datetime(weather_df['date'])
    else:
        print("\nNo real weather data found, creating synthetic weather data...")
        weather_df = create_synthetic_weather_fallback()

    print(f"Weather data shape: {weather_df.shape}")
    print(f"Date range: {weather_df['time'].min()} to {weather_df['time'].max()}")

    # Generate generation data
    generation_df = generate_generation_data(weather_df, plants)

    # Summary statistics
    print("\nGeneration Data Summary:")
    print(f"Total records: {len(generation_df)}")
    print(f"Plants: {generation_df['plant_name'].nunique()}")
    print(f"Plant types: {generation_df['plant_type'].unique()}")
    print(f"Total capacity: {generation_df.groupby('plant_name')['capacity_mw'].first().sum():.0f} MW")

    # Capacity factor by plant type
    cf_by_type = generation_df.groupby('plant_type')['capacity_factor'].mean()
    print(f"\nAverage capacity factors:")
    for plant_type, cf in cf_by_type.items():
        print(".1%")

    # Monthly patterns
    generation_df['month'] = pd.to_datetime(generation_df['timestamp']).dt.month
    monthly_cf = generation_df.groupby(['plant_type', 'month'])['capacity_factor'].mean().unstack()

    print(f"\nMonthly capacity factor patterns:")
    print(monthly_cf.round(3))

if __name__ == "__main__":
    main()