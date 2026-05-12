#!/usr/bin/env python3
"""
VidyutDrishti Dashboard Launcher
================================

Easy way to start the Streamlit dashboard for Karnataka renewable energy forecasting.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    print("🚀 Starting VidyutDrishti Dashboard...")
    print("=" * 50)

    # Check if data exists
    data_file = "data/processed/karnataka_generation_data.csv"
    if not os.path.exists(data_file):
        print("❌ Data not found. Please generate data first:")
        print("   python data_pipeline/ingestion/generate_scada_data.py")
        return

    # Check if weather data exists
    weather_file = "data/synthetic_weather_data.csv"
    if not os.path.exists(weather_file):
        print("❌ Weather data not found. Please generate data first:")
        print("   python data_pipeline/ingestion/generate_scada_data.py")
        return

    print("✅ Data files found")
    print("🌐 Starting dashboard on http://localhost:8502")

    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "dashboard/app.py",
            "--server.headless", "true",
            "--server.port", "8503"
        ]

        subprocess.run(cmd, cwd=os.getcwd())

    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main()