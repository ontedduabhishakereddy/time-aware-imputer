# Use Cases for Time-Aware Missing Data Imputer

This document provides detailed use cases and examples for the Time-Aware Missing Data Imputer library.

## Table of Contents

1. [IoT & Sensor Networks](#1-iot--sensor-networks)
2. [Medical & Healthcare](#2-medical--healthcare)
3. [Financial Markets](#3-financial-markets)
4. [Environmental Monitoring](#4-environmental-monitoring)
5. [Smart Buildings](#5-smart-buildings)
6. [Industrial Automation](#6-industrial-automation)

---

## 1. IoT & Sensor Networks

### Use Case: Smart Home Temperature Monitoring

**Scenario:**
- Multiple temperature sensors throughout a house
- Sensors report every 5 minutes
- Network issues cause irregular data gaps
- Power outages create longer gaps

**Challenges:**
- Irregular reporting intervals
- Varying gap durations (seconds to hours)
- Need to preserve daily temperature patterns
- Multiple correlated sensors

**Solution:**
```python
import pandas as pd
import numpy as np
from time_aware_imputer import SplineImputer, GapAnalyzer

# Load sensor data (irregular timestamps)
df = pd.read_csv('home_sensors.csv', parse_dates=['timestamp'])

# Analyze gaps before imputation
analyzer = GapAnalyzer()
stats = analyzer.analyze(df)

# Visualize gap patterns
analyzer.plot_gaps(df, column='living_room_temp')
analyzer.plot_missing_heatmap(df)  # All rooms

# Impute using cubic spline (preserves smooth daily patterns)
imputer = SplineImputer(method='cubic')
df_clean = imputer.fit_transform(df)

# Get confidence in imputed values
imputed_mask = imputer.get_imputed_mask()
```

**Benefits:**
- Preserves natural temperature curves
- Handles irregular sensor intervals automatically
- Provides gap statistics for system monitoring
- No GPU or cloud processing needed

---

## 2. Medical & Healthcare

### Use Case: Continuous Glucose Monitoring (CGM)

**Scenario:**
- Diabetic patient wears CGM device
- Device measures glucose every 5 minutes
- Sensor failures or patient movement cause gaps
- Critical for insulin dosing decisions

**Challenges:**
- Missing data during critical periods
- Need physiological constraints (glucose can't be negative)
- Regulatory requirements for data quality
- Uncertainty quantification needed

**Solution:**
```python
from time_aware_imputer import SplineImputer, GapAnalyzer

# Load CGM data
df = pd.read_csv('cgm_data.csv', parse_dates=['timestamp'])

# Analyze data quality
analyzer = GapAnalyzer()
stats = analyzer.analyze(df)

print(f"Data completeness: {100 - stats['glucose']['missing_percentage']:.1f}%")
print(f"Longest gap: {stats['glucose']['max_gap_duration']/60:.0f} minutes")

# Impute with PCHIP (prevents unrealistic oscillations)
imputer = SplineImputer(method='pchip')
df_imputed = imputer.fit_transform(df)

# Flag low-confidence imputations for review
mask = imputer.get_imputed_mask()
long_gaps = df_imputed[mask['glucose'] & (df_imputed['timestamp'].diff() > pd.Timedelta('15min'))]

# Review these with healthcare provider
print(f"Periods requiring clinical review: {len(long_gaps)}")
```

**Benefits:**
- PCHIP method prevents overshoot (no negative glucose values)
- Gap analysis helps identify device issues
- Uncertainty tracking for clinical decisions
- Audit trail for regulatory compliance

---

## 3. Financial Markets

### Use Case: High-Frequency Trading Data

**Scenario:**
- Tick-by-tick price data
- Exchange outages or network issues
- Irregular timestamps (trades happen when they happen)
- Backtesting requires complete data

**Challenges:**
- Microsecond-level timestamps
- Extreme irregularity
- Cannot introduce artificial price movements
- Need to preserve bid-ask spreads

**Solution:**
```python
from time_aware_imputer import SplineImputer

# Load tick data
df = pd.read_csv('tick_data.csv', parse_dates=['timestamp'])

# For prices, use linear interpolation (conservative)
price_imputer = SplineImputer(
    method='linear',
    value_columns=['bid_price', 'ask_price']
)
df_prices = price_imputer.fit_transform(df)

# For volume, forward fill might be more appropriate
# But for demonstration, cubic for smooth estimates
volume_imputer = SplineImputer(
    method='cubic',
    value_columns=['volume']
)
df_complete = volume_imputer.fit_transform(df_prices)

# Verify no look-ahead bias introduced
imputed_mask = price_imputer.get_imputed_mask()
```

**Benefits:**
- Respects actual time intervals between trades
- Linear method for prices (conservative, no overshoot)
- Preserves market microstructure
- Suitable for backtesting validation

---

## 4. Environmental Monitoring

### Use Case: Weather Station Network

**Scenario:**
- Network of weather stations
- Measure temperature, humidity, pressure
- Stations go offline for maintenance
- Seasonal patterns important

**Challenges:**
- Correlated variables (temp affects humidity)
- Strong daily and seasonal cycles
- Multi-day gaps during maintenance
- Multiple stations need consistent approach

**Solution:**
```python
from time_aware_imputer import SplineImputer, GapAnalyzer

# Load multi-station data
df = pd.read_csv('weather_stations.csv', parse_dates=['timestamp'])

# Analyze gap patterns across all variables
analyzer = GapAnalyzer()
stats = analyzer.analyze(df)
summary = analyzer.get_summary()
print(summary)

# Impute all meteorological variables
# Cubic spline preserves smooth daily cycles
imputer = SplineImputer(
    method='cubic',
    value_columns=['temperature', 'humidity', 'pressure', 'wind_speed']
)
df_complete = imputer.fit_transform(df)

# Validate results
fig = analyzer.plot_gaps(df_complete, column='temperature')
```

**Benefits:**
- Preserves daily and seasonal patterns
- Handles all variables consistently
- Gap analysis identifies problematic stations
- Suitable for climate research

---

## 5. Smart Buildings

### Use Case: Energy Consumption Monitoring

**Scenario:**
- Smart meters record power usage every 15 minutes
- Communication failures create gaps
- Need accurate billing and analytics
- Cumulative consumption must be monotonic

**Challenges:**
- Consumption must never decrease (cumulative)
- Gaps affect billing accuracy
- Need to detect anomalies vs. gaps
- Regulatory requirements for utilities

**Solution:**
```python
from time_aware_imputer import SplineImputer, GapAnalyzer

# Load smart meter data
df = pd.read_csv('smart_meter.csv', parse_dates=['timestamp'])

# Analyze gaps for quality reporting
analyzer = GapAnalyzer()
stats = analyzer.analyze(df)

# For instantaneous power, use cubic spline
power_imputer = SplineImputer(
    method='cubic',
    value_columns=['power_kw']
)
df_imputed = power_imputer.fit_transform(df)

# Calculate energy from imputed power
df_imputed['energy_kwh'] = df_imputed['power_kw'] * 0.25  # 15-min intervals

# Verify cumulative energy is monotonic
cumulative = df_imputed['energy_kwh'].cumsum()
assert cumulative.is_monotonic_increasing

# Flag imputed periods in billing
mask = power_imputer.get_imputed_mask()
df_imputed['imputed_flag'] = mask['power_kw']
```

**Benefits:**
- Accurate billing with imputed data flagged
- Preserves load patterns for analytics
- Audit trail for regulatory compliance
- Anomaly detection remains effective

---

## 6. Industrial Automation

### Use Case: Manufacturing Process Monitoring

**Scenario:**
- Sensors monitor temperature, pressure, flow rate
- Machine vibrations measured continuously
- Sensor failures during production runs
- Need complete data for quality control

**Challenges:**
- Multiple correlated process variables
- Real-time requirements
- Cannot afford production delays
- Safety-critical measurements

**Solution:**
```python
from time_aware_imputer import SplineImputer, GapAnalyzer
import pandas as pd

# Load process data
df = pd.read_csv('process_data.csv', parse_dates=['timestamp'])

# Quick gap analysis
analyzer = GapAnalyzer()
stats = analyzer.analyze(df)

# Alert if gaps are too long for safe imputation
for var, var_stats in stats.items():
    max_gap_min = var_stats['max_gap_duration'] / 60
    if max_gap_min > 5:  # 5-minute threshold
        print(f"WARNING: {var} has {max_gap_min:.1f} min gap!")

# Impute process variables
imputer = SplineImputer(
    method='cubic',
    value_columns=['temperature', 'pressure', 'flow_rate']
)
df_complete = imputer.fit_transform(df)

# For safety: flag any imputed values
mask = imputer.get_imputed_mask()
df_complete['needs_verification'] = mask.any(axis=1)

# Real-time monitoring
recent_gaps = stats['temperature']['n_gaps']
if recent_gaps > 10:
    print("ALERT: Frequent sensor failures detected!")
```

**Benefits:**
- Continuous quality monitoring
- Fast enough for real-time use
- Safety flags for imputed data
- Early warning of sensor degradation

---

## Comparison Table

| Use Case | Best Method | Key Requirement | Gap Tolerance |
|----------|-------------|-----------------|---------------|
| IoT Sensors | Cubic | Smooth patterns | Hours |
| Medical CGM | PCHIP | No overshoot | 15-30 min |
| Finance | Linear | Conservative | Seconds |
| Weather | Cubic | Seasonal cycles | Days |
| Energy | Cubic | Monotonic cumulative | 1-2 hours |
| Industrial | Cubic | Real-time | 5-10 min |

---

## Best Practices by Domain

### IoT / Sensors
- **Method**: Cubic spline
- **Validation**: Check for sensor degradation
- **Monitoring**: Track gap frequency over time

### Healthcare
- **Method**: PCHIP (no overshoot)
- **Validation**: Clinical review of long gaps
- **Compliance**: Maintain audit trails

### Finance
- **Method**: Linear (conservative)
- **Validation**: No look-ahead bias
- **Testing**: Backtest with and without imputation

### Environmental
- **Method**: Cubic (seasonal patterns)
- **Validation**: Cross-check with nearby stations
- **Reporting**: Document data quality

### Energy / Utilities
- **Method**: Cubic for instantaneous
- **Validation**: Cumulative checks
- **Billing**: Flag imputed periods

### Industrial
- **Method**: Cubic with safety flags
- **Validation**: Real-time alerts
- **Safety**: Operator verification

---

## Getting Help

For specific use case questions:
- GitHub Discussions
- Email: ontedduabhishakereddy@gmail.com
- Documentation: https://github.com/ontedduabhishakereddy/time-aware-imputer

### `LICENSE`

MIT License

Copyright (c) 2026 Abhishake Reddy O 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.