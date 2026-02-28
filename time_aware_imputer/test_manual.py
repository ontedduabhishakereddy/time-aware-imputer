"""Manual test script to verify all modules work correctly."""

import sys
import numpy as np
import pandas as pd
from time_aware_imputer import TimeAwareImputer, SplineImputer, GapAnalyzer


def test_time_aware_imputer() -> None:
    """Test TimeAwareImputer base class."""
    print("Testing TimeAwareImputer...")
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
        'value': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
    })
    
    imputer = TimeAwareImputer()
    imputer.fit(df)
    
    assert hasattr(imputer, 'is_fitted_')
    assert imputer.is_fitted_ is True
    assert imputer.n_features_in_ == 2
    
    print("  ✓ Initialization works")
    print("  ✓ Fit method works")
    print("  ✓ Attributes are set correctly")
    print()


def test_spline_imputer() -> None:
    """Test SplineImputer."""
    print("Testing SplineImputer...")
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
        'value': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0]
    })
    
    # Test linear interpolation
    imputer = SplineImputer(method='linear')
    result = imputer.fit_transform(df)
    assert not result['value'].isna().any()
    assert abs(result.loc[2, 'value'] - 3.0) < 0.01
    print("  ✓ Linear interpolation works")
    
    # Test cubic interpolation
    imputer = SplineImputer(method='cubic')
    result = imputer.fit_transform(df)
    assert not result['value'].isna().any()
    print("  ✓ Cubic interpolation works")
    
    # Test PCHIP interpolation
    imputer = SplineImputer(method='pchip')
    result = imputer.fit_transform(df)
    assert not result['value'].isna().any()
    print("  ✓ PCHIP interpolation works")
    
    # Test Akima interpolation
    imputer = SplineImputer(method='akima')
    result = imputer.fit_transform(df)
    assert not result['value'].isna().any()
    print("  ✓ Akima interpolation works")
    
    # Test multivariate
    df_multi = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
        'value1': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
        'value2': [10.0, 9.0, 8.0, np.nan, 6.0, 5.0, np.nan, 3.0, 2.0, 1.0]
    })
    
    imputer = SplineImputer(method='linear')
    result = imputer.fit_transform(df_multi)
    assert not result['value1'].isna().any()
    assert not result['value2'].isna().any()
    print("  ✓ Multivariate imputation works")
    
    # Test get_imputed_mask
    mask = imputer.get_imputed_mask()
    assert isinstance(mask, pd.DataFrame)
    print("  ✓ get_imputed_mask works")
    print()


def test_gap_analyzer() -> None:
    """Test GapAnalyzer."""
    print("Testing GapAnalyzer...")
    
    timestamps = pd.date_range('2024-01-01', periods=20, freq='h')
    values = np.ones(20)
    values[5:8] = np.nan  # 3-hour gap
    values[15:17] = np.nan  # 2-hour gap
    
    df = pd.DataFrame({'timestamp': timestamps, 'value': values})
    
    analyzer = GapAnalyzer()
    stats = analyzer.analyze(df)
    
    assert stats['value']['n_gaps'] == 2
    assert stats['value']['total_missing'] == 5
    assert stats['value']['missing_percentage'] == 25.0
    print("  ✓ Gap analysis works")
    
    # Test summary
    summary = analyzer.get_summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 1
    print("  ✓ Summary table works")
    
    # Test multivariate
    df_multi = pd.DataFrame({
        'timestamp': timestamps,
        'value1': values.copy(),
        'value2': np.ones(20)
    })
    df_multi.loc[10:12, 'value2'] = np.nan
    
    stats = analyzer.analyze(df_multi)
    assert 'value1' in stats
    assert 'value2' in stats
    print("  ✓ Multivariate analysis works")
    
    # Test gap finding
    missing_mask = pd.Series([False, True, True, False, False])
    gaps = analyzer._find_gaps(pd.date_range('2024-01-01', periods=5, freq='h'), missing_mask)
    assert len(gaps) == 1
    print("  ✓ Gap finding works")
    print()


def test_irregular_timestamps() -> None:
    """Test with irregular time intervals."""
    print("Testing irregular timestamps...")
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime([
            '2024-01-01 00:00:00',
            '2024-01-01 01:00:00',
            '2024-01-01 03:00:00',  # 2-hour gap
            '2024-01-01 03:30:00',  # 30-min gap
            '2024-01-01 06:00:00',  # 2.5-hour gap
        ]),
        'value': [1.0, 2.0, np.nan, 4.0, 5.0]
    })
    
    imputer = SplineImputer(method='linear')
    result = imputer.fit_transform(df)
    assert not result['value'].isna().any()
    print("  ✓ Handles irregular timestamps correctly")
    print()


def test_edge_cases() -> None:
    """Test edge cases."""
    print("Testing edge cases...")
    
    # No missing values
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=5, freq='h'),
        'value': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    
    imputer = SplineImputer(method='linear')
    result = imputer.fit_transform(df)
    pd.testing.assert_frame_equal(result, df)
    print("  ✓ Handles data with no missing values")
    
    # Insufficient data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=2, freq='h'),
        'value': [np.nan, np.nan]
    })
    
    result = imputer.fit_transform(df)
    assert result['value'].isna().all()
    print("  ✓ Handles insufficient data gracefully")
    
    # Single known point
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=3, freq='h'),
        'value': [np.nan, 5.0, np.nan]
    })
    
    result = imputer.fit_transform(df)
    assert result['value'].isna().sum() == 2
    print("  ✓ Handles single known point correctly")
    print()


def test_sklearn_compatibility() -> None:
    """Test sklearn compatibility."""
    print("Testing sklearn compatibility...")
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
        'value': [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0]
    })
    
    imputer = SplineImputer(method='cubic')
    
    # Test fit then transform
    imputer.fit(df)
    result = imputer.transform(df)
    assert not result['value'].isna().any()
    print("  ✓ Separate fit() and transform() work")
    
    # Test fit_transform
    imputer2 = SplineImputer(method='cubic')
    result2 = imputer2.fit_transform(df)
    assert not result2['value'].isna().any()
    print("  ✓ fit_transform() works")
    
    # Test get_params
    params = imputer.get_params()
    assert 'method' in params
    assert params['method'] == 'cubic'
    print("  ✓ get_params() works")
    
    # Test set_params
    imputer.set_params(method='linear')
    assert imputer.method == 'linear'
    print("  ✓ set_params() works")
    
    # Test get_feature_names_out
    names = imputer.get_feature_names_out()
    assert names == list(df.columns)
    print("  ✓ get_feature_names_out() works")
    print()


def main() -> int:
    """Run all tests."""
    print("=" * 70)
    print("Time-Aware Imputer - Manual Test Suite")
    print("=" * 70)
    print()
    
    try:
        test_time_aware_imputer()
        test_spline_imputer()
        test_gap_analyzer()
        test_irregular_timestamps()
        test_edge_cases()
        test_sklearn_compatibility()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("Summary:")
        print("  • TimeAwareImputer base class: Working")
        print("  • SplineImputer (all methods): Working")
        print("  • GapAnalyzer: Working")
        print("  • Edge cases: Handled correctly")
        print("  • sklearn compatibility: Full")
        print()
        return 0
        
    except Exception as e:
        print("=" * 70)
        print("✗ TEST FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)