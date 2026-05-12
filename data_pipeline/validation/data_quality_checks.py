"""
Data Validation for VidyutDrishti
Ensures data quality and integrity for renewable energy forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class DataValidator:
    """Data validation and quality assurance for renewable energy data"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.validation_results = {}

    def _default_config(self) -> Dict:
        return {
            'time_col': 'timestamp',
            'generation_col': 'generation_mw',
            'capacity_col': 'capacity_mw',
            'weather_cols': [
                'temperature_2m', 'wind_speed', 'surface_solar_radiation_downwards',
                'total_cloud_cover', 'total_precipitation'
            ],
            'max_missing_pct': 5.0,  # Maximum allowed missing data percentage
            'min_data_points': 1000,  # Minimum data points per plant
            'expected_freq': 'H'  # Expected frequency
        }

    def validate_data_structure(self, df: pd.DataFrame) -> Dict:
        """Validate basic data structure and required columns"""

        results = {
            'passed': True,
            'issues': [],
            'warnings': []
        }

        required_cols = [
            self.config['time_col'],
            'plant_name',
            'plant_type',
            self.config['generation_col'],
            self.config['capacity_col']
        ]

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['passed'] = False
            results['issues'].append(f"Missing required columns: {missing_cols}")

        # Check data types
        if self.config['time_col'] in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[self.config['time_col']]):
                results['issues'].append(f"{self.config['time_col']} is not datetime type")

        # Check for empty dataframe
        if df.empty:
            results['passed'] = False
            results['issues'].append("DataFrame is empty")

        return results

    def validate_data_completeness(self, df: pd.DataFrame) -> Dict:
        """Validate data completeness and missing values"""

        results = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'missing_summary': {}
        }

        # Overall missing percentage
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (total_missing / total_cells) * 100

        results['missing_summary']['overall_missing_pct'] = missing_pct

        if missing_pct > self.config['max_missing_pct']:
            results['passed'] = False
            results['issues'].append(".1f")

        # Missing by column
        missing_by_col = df.isnull().sum() / len(df) * 100
        high_missing_cols = missing_by_col[missing_by_col > self.config['max_missing_pct']]

        if not high_missing_cols.empty:
            results['warnings'].append(f"High missing data columns: {high_missing_cols.to_dict()}")

        # Missing by plant
        if 'plant_name' in df.columns:
            missing_by_plant = df.groupby('plant_name').apply(lambda x: x.isnull().sum().sum() / (x.shape[0] * x.shape[1]) * 100)
            high_missing_plants = missing_by_plant[missing_by_plant > self.config['max_missing_pct']]

            if not high_missing_plants.empty:
                results['warnings'].append(f"Plants with high missing data: {high_missing_plants.to_dict()}")

        return results

    def validate_temporal_consistency(self, df: pd.DataFrame) -> Dict:
        """Validate temporal consistency and gaps"""

        results = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'temporal_summary': {}
        }

        if self.config['time_col'] not in df.columns:
            results['passed'] = False
            results['issues'].append(f"Time column {self.config['time_col']} not found")
            return results

        # Ensure sorted by time
        df_sorted = df.sort_values(self.config['time_col'])

        # Check for duplicates
        duplicates = df_sorted.duplicated(subset=[self.config['time_col'], 'plant_name']).sum()
        if duplicates > 0:
            results['warnings'].append(f"Found {duplicates} duplicate timestamp-plant combinations")

        # Check time gaps by plant
        gap_summary = {}
        for plant_name, plant_df in df_sorted.groupby('plant_name'):
            time_diffs = plant_df[self.config['time_col']].diff().dropna()
            expected_diff = pd.Timedelta(f'1{self.config["expected_freq"]}')

            gaps = time_diffs[time_diffs > expected_diff]
            gap_summary[plant_name] = {
                'total_points': len(plant_df),
                'gaps_count': len(gaps),
                'max_gap_hours': gaps.max().total_seconds() / 3600 if len(gaps) > 0 else 0
            }

        results['temporal_summary'] = gap_summary

        # Flag plants with too many gaps
        problematic_plants = [plant for plant, stats in gap_summary.items()
                            if stats['gaps_count'] > len(df_sorted) * 0.01]  # More than 1% gaps

        if problematic_plants:
            results['warnings'].append(f"Plants with excessive gaps: {problematic_plants}")

        return results

    def validate_physical_constraints(self, df: pd.DataFrame) -> Dict:
        """Validate physical constraints and realistic ranges"""

        results = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'constraint_violations': {}
        }

        # Generation constraints
        if self.config['generation_col'] in df.columns and self.config['capacity_col'] in df.columns:
            # Negative generation
            negative_gen = (df[self.config['generation_col']] < 0).sum()
            if negative_gen > 0:
                results['issues'].append(f"Found {negative_gen} negative generation values")

            # Over-generation (more than 110% of capacity)
            over_gen = (df[self.config['generation_col']] > df[self.config['capacity_col']] * 1.1).sum()
            if over_gen > 0:
                results['warnings'].append(f"Found {over_gen} over-generation instances (>110% capacity)")

        # Weather constraints
        weather_ranges = {
            'temperature_2m': (-50, 60),  # Celsius
            'wind_speed': (0, 50),       # m/s
            'surface_solar_radiation_downwards': (0, 1500),  # W/m²
            'total_cloud_cover': (0, 100),  # %
            'total_precipitation': (0, 500)  # mm
        }

        for col, (min_val, max_val) in weather_ranges.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    results['warnings'].append(f"{col}: {out_of_range} values outside range [{min_val}, {max_val}]")

        # Solar-specific validations
        solar_mask = df['plant_type'] == 'solar'
        if solar_mask.any() and 'surface_solar_radiation_downwards' in df.columns:
            # Solar generation should correlate with irradiance during daylight
            daylight_mask = solar_mask & (df['hour'].between(6, 18))
            if daylight_mask.any():
                solar_daylight = df[daylight_mask]
                correlation = solar_daylight[self.config['generation_col']].corr(
                    solar_daylight['surface_solar_radiation_downwards']
                )
                if correlation < 0.3:  # Weak correlation
                    results['warnings'].append(".3f")

        # Wind-specific validations
        wind_mask = df['plant_type'] == 'wind'
        if wind_mask.any() and 'wind_speed' in df.columns:
            # Wind generation should correlate with wind speed
            correlation = df.loc[wind_mask, self.config['generation_col']].corr(
                df.loc[wind_mask, 'wind_speed']
            )
            if correlation < 0.5:  # Moderate correlation expected
                results['warnings'].append(".3f")

        return results

    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Run all validation checks"""

        print("Running VidyutDrishti data validation...")
        print("=" * 50)

        all_results = {
            'structure': self.validate_data_structure(df),
            'completeness': self.validate_data_completeness(df),
            'temporal': self.validate_temporal_consistency(df),
            'physical': self.validate_physical_constraints(df)
        }

        # Overall assessment
        overall_passed = all(result['passed'] for result in all_results.values())

        # Compile all issues and warnings
        all_issues = []
        all_warnings = []

        for check_name, result in all_results.items():
            if result['issues']:
                all_issues.extend([f"{check_name}: {issue}" for issue in result['issues']])
            if result['warnings']:
                all_warnings.extend([f"{check_name}: {warning}" for warning in result['warnings']])

        validation_summary = {
            'overall_passed': overall_passed,
            'total_issues': len(all_issues),
            'total_warnings': len(all_warnings),
            'issues': all_issues,
            'warnings': all_warnings,
            'detailed_results': all_results
        }

        self.validation_results = validation_summary

        # Print summary
        print(f"Overall validation: {'PASSED' if overall_passed else 'FAILED'}")
        print(f"Issues found: {len(all_issues)}")
        print(f"Warnings: {len(all_warnings)}")

        if all_issues:
            print("\nIssues:")
            for issue in all_issues[:5]:  # Show first 5
                print(f"  - {issue}")
            if len(all_issues) > 5:
                print(f"  ... and {len(all_issues) - 5} more")

        if all_warnings:
            print("\nWarnings:")
            for warning in all_warnings[:5]:  # Show first 5
                print(f"  - {warning}")
            if len(all_warnings) > 5:
                print(f"  ... and {len(all_warnings) - 5} more")

        return validation_summary

    def generate_validation_report(self, df: pd.DataFrame, output_file: str = None):
        """Generate detailed validation report"""

        if not self.validation_results:
            self.validate_data_quality(df)

        report = f"""
VidyutDrishti Data Validation Report
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Summary:
- Shape: {df.shape}
- Plants: {df['plant_name'].nunique() if 'plant_name' in df.columns else 'N/A'}
- Date range: {df[self.config['time_col']].min() if self.config['time_col'] in df.columns else 'N/A'} to {df[self.config['time_col']].max() if self.config['time_col'] in df.columns else 'N/A'}
- Plant types: {df['plant_type'].unique() if 'plant_type' in df.columns else 'N/A'}

Validation Results:
- Overall: {'PASSED' if self.validation_results['overall_passed'] else 'FAILED'}
- Issues: {self.validation_results['total_issues']}
- Warnings: {self.validation_results['total_warnings']}

"""

        if self.validation_results['issues']:
            report += "\nIssues:\n"
            for issue in self.validation_results['issues']:
                report += f"- {issue}\n"

        if self.validation_results['warnings']:
            report += "\nWarnings:\n"
            for warning in self.validation_results['warnings']:
                report += f"- {warning}\n"

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Validation report saved to {output_file}")

        return report

def main():
    """Main validation function"""

    # Load processed data
    data_file = "../data/processed/karnataka_features.csv"

    try:
        df = pd.read_csv(data_file)
        print(f"Loaded data from {data_file}")

        validator = DataValidator()
        validation_results = validator.validate_data_quality(df)

        # Generate report
        validator.generate_validation_report(df, "../docs/data_validation_report.txt")

        if validation_results['overall_passed']:
            print("\n✅ Data validation passed! Ready for modeling.")
        else:
            print("\n❌ Data validation failed. Please address issues before proceeding.")

    except FileNotFoundError:
        print(f"Data file not found: {data_file}")
        print("Please run preprocessing first.")

if __name__ == "__main__":
    main()