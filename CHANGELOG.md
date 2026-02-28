# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-19

### Added
- Initial release of Time-Aware Imputer
- `SplineImputer` class for time-aware spline interpolation
  - Support for linear, cubic, PCHIP, and Akima interpolation methods
  - Handles irregular time intervals
  - Multivariate time series support
- `GapAnalyzer` class for missing data analysis
  - Gap statistics (count, duration, percentage)
  - Visualization tools (gap plots, heatmaps)
  - Summary table generation
- `TimeAwareImputer` base class following sklearn transformer API
- Comprehensive test suite with 53 tests
- Example scripts demonstrating usage
- Full documentation (README, CONTRIBUTING, USE_CASES)

### Fixed
- All Python type annotation errors resolved
- Pandas Series boolean operation errors in tests
- Test assertion for matplotlib colorbar axes

### Documentation
- Created comprehensive README.md
- Added MIT LICENSE
- Added py.typed marker for type checking support
- Included quickstart and IoT sensor examples
