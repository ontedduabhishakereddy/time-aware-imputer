"""Example: IoT Sensor Data Imputation.

This example demonstrates using the Time-Aware Imputer library to handle
missing data from IoT temperature sensors with irregular timestamps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time_aware_imputer import SplineImputer, GapAnalyzer


def create_iot_sensor_data() -> pd.DataFrame:
    """Create simulated IoT sensor data with realistic patterns and gaps."""
    # Create irregular timestamps (sensors don't always report on schedule)
    base_timestamps = pd.date_range("2024-01-01", periods=200, freq="5min")

    # Add some jitter to make timestamps irregular
    jitter = pd.to_timedelta(np.random.randint(-60, 60, size=200), unit="s")
    timestamps = base_timestamps + jitter

    # Remove some timestamps to create gaps (network issues, sensor failures)
    keep_mask = np.random.random(200) > 0.15  # 15% data loss
    timestamps = timestamps[keep_mask]

    # Generate temperature data with daily pattern
    hours = np.arange(len(timestamps)) * (5 / 60)  # hours
    temperature = (
        20  # base temperature
        + 5 * np.sin(2 * np.pi * hours / 24)  # daily cycle
        + np.random.randn(len(timestamps)) * 0.5  # noise
    )

    # Create DataFrame
    df = pd.DataFrame({"timestamp": timestamps, "temperature": temperature})

    # Introduce some longer gaps (maintenance, power outage)
    df.loc[50:58, "temperature"] = np.nan  # ~45 minute gap
    df.loc[120:125, "temperature"] = np.nan  # ~30 minute gap
    df.loc[170:172, "temperature"] = np.nan  # ~15 minute gap

    return df


def main() -> None:
    """Run the IoT sensor data imputation example."""
    print("=" * 70)
    print("Time-Aware Imputer: IoT Sensor Data Example")
    print("=" * 70)
    print()

    # Create sample data
    print("Creating simulated IoT sensor data...")
    df = create_iot_sensor_data()
    print(f"Dataset: {len(df)} readings")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()

    # Analyze gaps
    print("Analyzing data gaps...")
    analyzer = GapAnalyzer()
    stats = analyzer.analyze(df)

    temp_stats = stats["temperature"]
    print(f"Number of gaps: {temp_stats['n_gaps']}")
    print(f"Total missing values: {temp_stats['total_missing']}")
    print(f"Missing percentage: {temp_stats['missing_percentage']:.1f}%")
    print(
        f"Mean gap duration: {temp_stats['mean_gap_duration']/60:.1f} minutes"
    )
    print(f"Max gap duration: {temp_stats['max_gap_duration']/60:.1f} minutes")
    print()

    # Compare different imputation methods
    methods = ["linear", "cubic", "pchip"]
    results = {}

    print("Testing different imputation methods...")
    for method in methods:
        imputer = SplineImputer(method=method)
        df_imputed = imputer.fit_transform(df.copy())
        results[method] = df_imputed
        print(f"  ✓ {method.capitalize()} interpolation complete")

    print()

    # Visualize results
    print("Creating visualizations...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Original data with gaps
    ax = axes[0]
    mask_present = ~df["temperature"].isna()
    ax.plot(
        df.loc[mask_present, "timestamp"],
        df.loc[mask_present, "temperature"],
        "o-",
        label="Observed",
        markersize=3,
        alpha=0.7,
    )

    # Highlight gaps
    for gap in temp_stats["gap_details"]:
        start_time, end_time, _ = gap[0], gap[1], gap[2]
        ax.axvspan(start_time, end_time, alpha=0.3, color="red", label="Gap")

    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Original Data with Missing Values")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Plot 2: Comparison of imputation methods
    ax = axes[1]
    for method, df_imputed in results.items():
        mask_imputed = df["temperature"].isna()
        ax.plot(
            df_imputed.loc[mask_imputed, "timestamp"],
            df_imputed.loc[mask_imputed, "temperature"],
            "o",
            label=f"{method.capitalize()}",
            markersize=5,
            alpha=0.7,
        )

    # Also plot original data
    ax.plot(
        df.loc[mask_present, "timestamp"],
        df.loc[mask_present, "temperature"],
        "k.",
        label="Observed",
        markersize=2,
        alpha=0.5,
    )

    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Comparison of Imputation Methods (Imputed Values Only)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Plot 3: Full imputed data (using cubic)
    ax = axes[2]
    df_cubic = results["cubic"]
    ax.plot(
        df_cubic["timestamp"],
        df_cubic["temperature"],
        "-",
        label="Imputed (Cubic)",
        linewidth=1,
        alpha=0.8,
    )
    ax.plot(
        df.loc[mask_present, "timestamp"],
        df.loc[mask_present, "temperature"],
        "o",
        label="Observed",
        markersize=3,
        alpha=0.7,
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Complete Time Series After Cubic Spline Imputation")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("iot_sensor_imputation.png", dpi=150)
    print("  ✓ Saved visualization to 'iot_sensor_imputation.png'")
    print()

    # Create gap analysis plots
    fig_gaps = analyzer.plot_gaps(df)
    fig_gaps.savefig("iot_sensor_gaps.png", dpi=150)
    print("  ✓ Saved gap analysis to 'iot_sensor_gaps.png'")
    print()

    print("Example complete!")
    print()
    print("Key Takeaways:")
    print("  • Linear interpolation is fastest but may miss smooth trends")
    print("  • Cubic spline provides smooth interpolation for most cases")
    print("  • PCHIP preserves monotonicity (no overshoot)")
    print("  • Choice depends on data characteristics and requirements")


if __name__ == "__main__":
    main()