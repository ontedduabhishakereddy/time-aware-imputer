"""Quickstart example for Time-Aware Imputer."""

import numpy as np
import pandas as pd
from time_aware_imputer import SplineImputer, GapAnalyzer


def main() -> None:
    """Run a simple quickstart example."""
    # Create sample data
    print("Creating sample time series data...")
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="h"),
            "sensor_value": [
                1.0,
                2.1,
                np.nan,
                3.9,
                5.2,
                np.nan,
                np.nan,
                8.1,
                9.0,
                10.2,
                11.1,
                np.nan,
                13.0,
                14.2,
                15.1,
                16.0,
                17.2,
                18.1,
                19.0,
                20.1,
            ],
        }
    )

    print(f"Created dataset with {len(df)} time points")
    print(f"Missing values: {df['sensor_value'].isna().sum()}")
    print()

    # Analyze gaps
    print("Step 1: Analyze missing data gaps")
    print("-" * 40)
    analyzer = GapAnalyzer()
    stats = analyzer.analyze(df)

    print(f"Number of gaps: {stats['sensor_value']['n_gaps']}")
    print(
        f"Missing percentage: {stats['sensor_value']['missing_percentage']:.1f}%"
    )
    print(f"Longest gap: {stats['sensor_value']['max_gap_duration']/3600:.1f} hours")
    print()

    # Impute missing values
    print("Step 2: Impute missing values")
    print("-" * 40)
    imputer = SplineImputer(method="cubic")
    df_imputed = imputer.fit_transform(df)

    print("Imputation complete!")
    print(f"Remaining NaN values: {df_imputed['sensor_value'].isna().sum()}")
    print()

    # Show results
    print("Step 3: Results")
    print("-" * 40)
    print("\nOriginal vs Imputed values:")
    print("Index | Original | Imputed  | Was Missing")
    print("-" * 45)

    for idx in range(len(df)):
        original = df.loc[idx, "sensor_value"]
        imputed = df_imputed.loc[idx, "sensor_value"]
        was_missing = "Yes" if pd.isna(original) else "No"

        if pd.isna(original):
            print(f"{idx:5d} | {' '*8} | {imputed:8.2f} | {was_missing}")
        else:
            print(f"{idx:5d} | {original:8.2f} | {imputed:8.2f} | {was_missing}")

    print()
    print("✓ Quickstart example complete!")


if __name__ == "__main__":
    main()