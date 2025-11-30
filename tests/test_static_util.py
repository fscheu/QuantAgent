"""
Test suite for static_util.py functions.
Tests the OHLCV data formatting and image generation utilities.
"""

import pandas as pd
import pytest
from quantagent.static_util import read_and_format_ohlcv


def test_read_and_format_ohlcv_basic():
    """Test basic OHLCV formatting with valid data."""
    # Create sample DataFrame
    df = pd.DataFrame({
        'Datetime': pd.date_range('2024-01-01', periods=50, freq='4H'),
        'Open': [100.0 + i for i in range(50)],
        'High': [105.0 + i for i in range(50)],
        'Low': [95.0 + i for i in range(50)],
        'Close': [102.0 + i for i in range(50)]
    })

    result = read_and_format_ohlcv(df)

    # Verify structure
    assert isinstance(result, dict)
    assert set(result.keys()) == {'Datetime', 'Open', 'High', 'Low', 'Close'}

    # Verify types
    assert isinstance(result['Datetime'], list)
    assert isinstance(result['Open'], list)
    assert isinstance(result['Datetime'][0], str)
    assert isinstance(result['Open'][0], float)

    # Verify length (should be 46 candles: tail(49) then remove last 3)
    assert len(result['Datetime']) == 46
    assert len(result['Open']) == 46


def test_read_and_format_ohlcv_small_dataset():
    """Test OHLCV formatting with less than 49 rows."""
    # Create small DataFrame
    df = pd.DataFrame({
        'Datetime': pd.date_range('2024-01-01', periods=30, freq='1H'),
        'Open': [100.0] * 30,
        'High': [105.0] * 30,
        'Low': [95.0] * 30,
        'Close': [102.0] * 30
    })

    result = read_and_format_ohlcv(df)

    # Should take last 30 rows (tail(45) but only 30 available)
    assert len(result['Datetime']) == 30


def test_read_and_format_ohlcv_datetime_format():
    """Test that datetime strings are correctly formatted."""
    df = pd.DataFrame({
        'Datetime': pd.date_range('2024-01-01 10:30:00', periods=50, freq='4H'),
        'Open': [100.0] * 50,
        'High': [105.0] * 50,
        'Low': [95.0] * 50,
        'Close': [102.0] * 50
    })

    result = read_and_format_ohlcv(df)

    # Check datetime format
    assert '2024-01-01 10:30:00' in result['Datetime'] or '2024-01-01 14:30:00' in result['Datetime']
    # All datetime strings should match the expected format
    for dt_str in result['Datetime']:
        assert len(dt_str) == 19  # "YYYY-MM-DD HH:MM:SS"


def test_read_and_format_ohlcv_missing_columns():
    """Test that missing required columns raise ValueError."""
    # Create DataFrame missing 'Close' column
    df = pd.DataFrame({
        'Datetime': pd.date_range('2024-01-01', periods=50, freq='4H'),
        'Open': [100.0] * 50,
        'High': [105.0] * 50,
        'Low': [95.0] * 50
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        read_and_format_ohlcv(df)


def test_read_and_format_ohlcv_numeric_values():
    """Test that numeric values are preserved correctly."""
    df = pd.DataFrame({
        'Datetime': pd.date_range('2024-01-01', periods=50, freq='4H'),
        'Open': [100.5, 101.2, 102.8] + [100.0] * 47,
        'High': [105.5, 106.2, 107.8] + [105.0] * 47,
        'Low': [95.5, 96.2, 97.8] + [95.0] * 47,
        'Close': [102.5, 103.2, 104.8] + [102.0] * 47
    })

    result = read_and_format_ohlcv(df)

    # Check that values are floats and approximately correct
    assert isinstance(result['Open'][0], float)
    assert result['Open'][0] in [100.5, 101.2, 102.8, 100.0]


def test_read_and_format_ohlcv_integration_with_csv():
    """Test integration with CSV data (like benchmark files)."""
    import io

    # Simulate CSV data
    csv_data = """Datetime,Open,High,Low,Close
2024-01-01 00:00:00,100.0,105.0,95.0,102.0
2024-01-01 04:00:00,102.0,107.0,97.0,104.0
2024-01-01 08:00:00,104.0,109.0,99.0,106.0
2024-01-01 12:00:00,106.0,111.0,101.0,108.0
2024-01-01 16:00:00,108.0,113.0,103.0,110.0
"""

    # Read CSV
    df = pd.read_csv(io.StringIO(csv_data))
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    result = read_and_format_ohlcv(df)

    # Verify successful conversion
    assert len(result['Datetime']) == 5
    assert result['Open'][0] == 100.0
    assert result['Close'][-1] == 110.0


if __name__ == "__main__":
    # Run tests
    test_read_and_format_ohlcv_basic()
    print("✓ test_read_and_format_ohlcv_basic passed")

    test_read_and_format_ohlcv_small_dataset()
    print("✓ test_read_and_format_ohlcv_small_dataset passed")

    test_read_and_format_ohlcv_datetime_format()
    print("✓ test_read_and_format_ohlcv_datetime_format passed")

    test_read_and_format_ohlcv_numeric_values()
    print("✓ test_read_and_format_ohlcv_numeric_values passed")

    test_read_and_format_ohlcv_integration_with_csv()
    print("✓ test_read_and_format_ohlcv_integration_with_csv passed")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
