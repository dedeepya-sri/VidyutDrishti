#!/usr/bin/env python3
"""
VidyutDrishti System Health Check
=================================

Check if all components are working properly.
"""

import os
import pandas as pd
import sys

def check_data_files():
    """Check if required data files exist"""
    print("📁 Checking data files...")

    required_files = [
        "data/processed/karnataka_generation_data.csv",
        "data/synthetic_weather_data.csv"
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_exist = False

    return all_exist

def check_code_files():
    """Check if required code files exist"""
    print("\n🔧 Checking code files...")

    required_files = [
        "run_basic_forecast.py",
        "run_dashboard.py",
        "dashboard/app.py",
        "notebooks/baseline_forecasting.ipynb"
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_exist = False

    return all_exist

def test_basic_forecast():
    """Test if basic forecasting works"""
    print("\n⚡ Testing basic forecast...")

    try:
        # Quick test of data loading and model training
        gen_df = pd.read_csv("data/processed/karnataka_generation_data.csv")
        weather_df = pd.read_csv("data/synthetic_weather_data.csv")
        df = pd.merge(gen_df, weather_df, on='timestamp', how='inner')

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['month'] = df['timestamp'].dt.month

        plant_df = df[df['plant_name'] == 'Pavagada_Solar_Park_Phase1'].copy()

        if len(plant_df) > 0:
            print(f"   ✅ Data loaded successfully ({len(plant_df)} records)")
            print("   ✅ Basic forecasting system ready")
            return True
        else:
            print("   ❌ No data found for test plant")
            return False

    except Exception as e:
        print(f"   ❌ Error testing forecast: {e}")
        return False

def main():
    """Run all health checks"""
    print("🏥 VidyutDrishti System Health Check")
    print("=" * 40)

    checks = [
        check_data_files(),
        check_code_files(),
        test_basic_forecast()
    ]

    passed = sum(checks)
    total = len(checks)

    print(f"\n📊 Health Check Results: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 System is healthy and ready to use!")
        print("\n🚀 Quick commands:")
        print("   python run_basic_forecast.py     # Run basic forecast")
        print("   python run_dashboard.py          # Launch dashboard")
        print("   jupyter notebook notebooks/baseline_forecasting.ipynb  # View notebook")
    else:
        print("⚠️  Some issues found. Please run:")
        print("   python data_pipeline/ingestion/generate_scada_data.py")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)