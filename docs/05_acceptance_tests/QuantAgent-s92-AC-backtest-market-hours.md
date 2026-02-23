# Acceptance Criteria: Backtest Market Hours Filtering

**Issue ID:** QuantAgent-s92
**Type:** AC (Acceptance Criteria)
**Created:** 2026-01-05
**Related:**
- [Requirements](../01_requirements/QuantAgent-s92-RQ-backtest-market-hours.md)
- [Design](../03_design/QuantAgent-s92-DS-backtest-market-hours.md)

---

## Overview

This document defines verifiable acceptance criteria (oracles) for the market hours filtering feature. Each criterion follows Given/When/Then format and includes specific, measurable outcomes.

---

## AC1: Asset Type Classification

### AC1.1: Crypto Assets Classified Correctly

**Given:** Symbol is "BTC" or "ETH" or ends with "-USD"
**When:** `get_asset_type(symbol)` is called
**Then:** Returns `AssetType.CRYPTO`

```python
def test_crypto_classification():
    assert get_asset_type("BTC") == AssetType.CRYPTO
    assert get_asset_type("ETH") == AssetType.CRYPTO
    assert get_asset_type("BTC-USD") == AssetType.CRYPTO
    assert get_asset_type("SOL-USD") == AssetType.CRYPTO
```

### AC1.2: US Equity Assets Classified Correctly

**Given:** Symbol is "SPX", "QQQ", or starts with "^"
**When:** `get_asset_type(symbol)` is called
**Then:** Returns `AssetType.US_EQUITY`

```python
def test_equity_classification():
    assert get_asset_type("SPX") == AssetType.US_EQUITY
    assert get_asset_type("QQQ") == AssetType.US_EQUITY
    assert get_asset_type("^GSPC") == AssetType.US_EQUITY
    assert get_asset_type("^VIX") == AssetType.US_EQUITY
```

### AC1.3: US Futures Assets Classified Correctly

**Given:** Symbol is "ES", "NQ", "CL", "GC", or ends with "=F"
**When:** `get_asset_type(symbol)` is called
**Then:** Returns `AssetType.US_FUTURES`

```python
def test_futures_classification():
    assert get_asset_type("ES") == AssetType.US_FUTURES
    assert get_asset_type("ES=F") == AssetType.US_FUTURES
    assert get_asset_type("CL") == AssetType.US_FUTURES
    assert get_asset_type("GC=F") == AssetType.US_FUTURES
```

### AC1.4: Unknown Assets Default to Unknown

**Given:** Symbol is not in mapping and doesn't match known patterns
**When:** `get_asset_type(symbol)` is called
**Then:** Returns `AssetType.UNKNOWN`

```python
def test_unknown_classification():
    assert get_asset_type("CUSTOM") == AssetType.UNKNOWN
    assert get_asset_type("RANDOM123") == AssetType.UNKNOWN
```

---

## AC2: Market Calendar Integration

### AC2.1: Calendar Loads Successfully

**Given:** Exchange name is "NYSE" or "CME_Equity"
**When:** `MarketCalendar.get_calendar(exchange)` is called
**Then:** Returns valid calendar object (not None)

```python
def test_calendar_loads():
    calendar = MarketCalendar()
    nyse = calendar.get_calendar("NYSE")
    assert nyse is not None

    cme = calendar.get_calendar("CME_Equity")
    assert cme is not None
```

### AC2.2: Schedule Returns Valid DataFrame

**Given:** Valid calendar and date range
**When:** `get_schedule(exchange, start, end)` is called
**Then:** Returns DataFrame with 'market_open' and 'market_close' columns

```python
def test_schedule_structure():
    calendar = MarketCalendar()
    schedule = calendar.get_schedule(
        "NYSE",
        datetime(2024, 1, 2),
        datetime(2024, 1, 5),
    )

    assert schedule is not None
    assert "market_open" in schedule.columns
    assert "market_close" in schedule.columns
    assert len(schedule) > 0  # Has trading days
```

### AC2.3: Schedule Caching Works

**Given:** Same exchange and date range requested twice
**When:** `get_schedule()` is called twice
**Then:** Second call returns cached result (faster)

```python
def test_schedule_caching():
    calendar = MarketCalendar()

    # First call
    schedule1 = calendar.get_schedule("NYSE", start, end)

    # Second call should be cached
    schedule2 = calendar.get_schedule("NYSE", start, end)

    # Same object (cached)
    assert schedule1 is schedule2
```

### AC2.4: Graceful Fallback on Calendar Error

**Given:** Invalid or unavailable exchange name
**When:** `get_calendar()` is called
**Then:** Returns None (no exception raised)

```python
def test_calendar_fallback():
    calendar = MarketCalendar()
    result = calendar.get_calendar("INVALID_EXCHANGE_XYZ")
    assert result is None  # No exception, returns None
```

---

## AC3: Market Hours Filtering

### AC3.1: Crypto Timestamps Not Filtered

**Given:** List of timestamps and `AssetType.CRYPTO`
**When:** `filter_to_trading_hours()` is called
**Then:** Returns all original timestamps (no filtering)

```python
def test_crypto_no_filtering():
    calendar = MarketCalendar()
    timestamps = [
        datetime(2024, 1, 1, 3, 0),   # 3 AM
        datetime(2024, 1, 6, 12, 0),  # Saturday noon
        datetime(2024, 1, 7, 23, 0),  # Sunday 11 PM
    ]

    filtered = calendar.filter_to_trading_hours(timestamps, AssetType.CRYPTO)

    assert len(filtered) == len(timestamps)
    assert filtered == timestamps
```

### AC3.2: US Equity Weekends Filtered Out

**Given:** List of timestamps including Saturday and Sunday
**When:** `filter_to_trading_hours()` with `AssetType.US_EQUITY`
**Then:** Weekend timestamps are excluded

```python
def test_equity_weekend_filtering():
    calendar = MarketCalendar()
    timestamps = [
        datetime(2024, 1, 5, 14, 0),   # Friday 2 PM ET
        datetime(2024, 1, 6, 12, 0),   # Saturday noon
        datetime(2024, 1, 7, 12, 0),   # Sunday noon
        datetime(2024, 1, 8, 14, 0),   # Monday 2 PM ET
    ]

    filtered = calendar.filter_to_trading_hours(timestamps, AssetType.US_EQUITY)

    # Only weekdays included
    assert datetime(2024, 1, 5, 14, 0) in filtered
    assert datetime(2024, 1, 8, 14, 0) in filtered
    assert datetime(2024, 1, 6, 12, 0) not in filtered
    assert datetime(2024, 1, 7, 12, 0) not in filtered
```

### AC3.3: US Equity Outside Hours Filtered

**Given:** List of timestamps outside NYSE trading hours (9:30-16:00 ET)
**When:** `filter_to_trading_hours()` with `AssetType.US_EQUITY`
**Then:** Outside-hours timestamps are excluded

```python
def test_equity_outside_hours_filtering():
    calendar = MarketCalendar()
    timestamps = [
        datetime(2024, 1, 3, 8, 0),    # 8 AM (before open)
        datetime(2024, 1, 3, 10, 0),   # 10 AM (during session)
        datetime(2024, 1, 3, 15, 0),   # 3 PM (during session)
        datetime(2024, 1, 3, 18, 0),   # 6 PM (after close)
    ]

    filtered = calendar.filter_to_trading_hours(timestamps, AssetType.US_EQUITY)

    # Only trading hours included
    assert datetime(2024, 1, 3, 10, 0) in filtered  # 10 AM OK
    assert datetime(2024, 1, 3, 15, 0) in filtered  # 3 PM OK
    assert datetime(2024, 1, 3, 8, 0) not in filtered   # Too early
    assert datetime(2024, 1, 3, 18, 0) not in filtered  # Too late
```

### AC3.4: Market Holidays Filtered

**Given:** Timestamp on US market holiday (e.g., July 4th)
**When:** `filter_to_trading_hours()` with `AssetType.US_EQUITY`
**Then:** Holiday timestamp is excluded

```python
def test_holiday_filtering():
    calendar = MarketCalendar()
    timestamps = [
        datetime(2024, 7, 3, 14, 0),   # July 3 (trading day)
        datetime(2024, 7, 4, 14, 0),   # July 4 (holiday)
        datetime(2024, 7, 5, 14, 0),   # July 5 (trading day)
    ]

    filtered = calendar.filter_to_trading_hours(timestamps, AssetType.US_EQUITY)

    assert datetime(2024, 7, 3, 14, 0) in filtered
    assert datetime(2024, 7, 5, 14, 0) in filtered
    assert datetime(2024, 7, 4, 14, 0) not in filtered  # Holiday excluded
```

### AC3.5: Unknown Assets Not Filtered

**Given:** List of timestamps and `AssetType.UNKNOWN`
**When:** `filter_to_trading_hours()` is called
**Then:** Returns all original timestamps (safe default)

```python
def test_unknown_no_filtering():
    calendar = MarketCalendar()
    timestamps = [datetime(2024, 1, 6, 12, 0)]  # Saturday

    filtered = calendar.filter_to_trading_hours(timestamps, AssetType.UNKNOWN)

    assert len(filtered) == len(timestamps)  # No filtering
```

---

## AC4: Backtest Integration

### AC4.1: Backtest with Filtering Enabled (Default)

**Given:** Backtest for US equity (SPX) with 4h timeframe, 30 days
**When:** `backtest.run()` is called with default config
**Then:** Analysis periods are ~40-60% fewer than 24/7 calculation

```python
def test_backtest_filtering_enabled():
    # Without filtering: 30 days * 6 periods/day = 180 periods
    # With filtering (NYSE hours only): ~30 days * 0.69 * ~2 periods = ~42 periods

    backtest = Backtest(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        assets=["SPX"],
        timeframe="4h",
        initial_capital=100000.0,
        # market_hours_filter=True is default
    )

    # Count periods that would be analyzed
    periods = backtest._get_date_range_for_asset("SPX")

    # Should be significantly fewer than unfiltered
    all_periods = backtest._get_date_range()

    assert len(periods) < len(all_periods) * 0.7  # At least 30% reduction
    assert len(periods) > 0  # Some periods remain
```

### AC4.2: Backtest with Filtering Disabled

**Given:** Backtest for US equity with `config={'market_hours_filter': False}`
**When:** `backtest.run()` is called
**Then:** All periods are analyzed (backwards compatible)

```python
def test_backtest_filtering_disabled():
    backtest = Backtest(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        assets=["SPX"],
        timeframe="4h",
        initial_capital=100000.0,
        config={"market_hours_filter": False},
    )

    # Should have all periods (no filtering)
    periods = backtest._get_date_range_for_asset("SPX")
    all_periods = backtest._get_date_range()

    assert len(periods) == len(all_periods)
```

### AC4.3: Backtest with Crypto Asset

**Given:** Backtest for BTC with filtering enabled
**When:** `backtest.run()` is called
**Then:** All periods are analyzed (crypto is 24/7)

```python
def test_backtest_crypto_no_change():
    backtest = Backtest(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        assets=["BTC"],
        timeframe="4h",
        initial_capital=100000.0,
        # market_hours_filter=True (default)
    )

    periods = backtest._get_date_range_for_asset("BTC")
    all_periods = backtest._get_date_range()

    # Crypto should have all periods
    assert len(periods) == len(all_periods)
```

### AC4.4: Mixed Asset Backtest

**Given:** Backtest with [BTC, SPX] assets
**When:** `backtest.run()` is called
**Then:** BTC has all periods, SPX has filtered periods

```python
def test_backtest_mixed_assets():
    backtest = Backtest(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        assets=["BTC", "SPX"],
        timeframe="4h",
        initial_capital=100000.0,
    )

    btc_periods = backtest._get_date_range_for_asset("BTC")
    spx_periods = backtest._get_date_range_for_asset("SPX")

    # BTC should have more periods than SPX
    assert len(btc_periods) > len(spx_periods)
```

### AC4.5: Backtest Logging Shows Filtering

**Given:** Backtest with filtering enabled
**When:** `backtest.run()` is called
**Then:** Logs show asset types and filtered period counts

```python
def test_backtest_logging(caplog):
    backtest = Backtest(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 7),
        assets=["SPX"],
        timeframe="4h",
        initial_capital=100000.0,
    )

    with caplog.at_level(logging.INFO):
        backtest.run()

    # Should log asset type and period count
    assert "us_equity" in caplog.text.lower()
    assert "analysis periods" in caplog.text.lower()
```

---

## AC5: Performance Requirements

### AC5.1: Filtering Overhead Under 100ms

**Given:** 1-year backtest with filtering enabled
**When:** Date range generation and filtering executed
**Then:** Total time < 100ms

```python
def test_filtering_performance():
    import time

    backtest = Backtest(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
        assets=["SPX"],
        timeframe="1h",
        initial_capital=100000.0,
    )

    start = time.time()
    periods = backtest._get_date_range_for_asset("SPX")
    elapsed = time.time() - start

    assert elapsed < 0.1  # Under 100ms
```

### AC5.2: Calendar Caching Effective

**Given:** Multiple assets using same exchange (NYSE)
**When:** Date ranges generated for each asset
**Then:** Calendar loaded only once (cached)

```python
def test_calendar_caching_effectiveness():
    calendar = MarketCalendar()

    # First asset
    calendar.filter_to_trading_hours([...], AssetType.US_EQUITY)
    initial_cache_size = len(calendar._schedule_cache)

    # Second asset (same exchange)
    calendar.filter_to_trading_hours([...], AssetType.US_EQUITY)
    final_cache_size = len(calendar._schedule_cache)

    # Cache should be reused, not doubled
    assert final_cache_size == initial_cache_size
```

---

## AC6: Edge Cases

### AC6.1: Empty Date Range

**Given:** Start date equals end date
**When:** `_get_date_range_for_asset()` is called
**Then:** Returns list with single timestamp (if valid trading period)

```python
def test_empty_date_range():
    backtest = Backtest(
        start_date=datetime(2024, 1, 3, 12, 0),  # Wednesday noon
        end_date=datetime(2024, 1, 3, 12, 0),
        assets=["SPX"],
        timeframe="1h",
    )

    periods = backtest._get_date_range_for_asset("SPX")

    assert len(periods) <= 1  # At most one period
```

### AC6.2: Start Date on Weekend

**Given:** Start date is Saturday
**When:** `_get_date_range_for_asset()` is called for US equity
**Then:** First period is Monday (or next trading day)

```python
def test_start_on_weekend():
    backtest = Backtest(
        start_date=datetime(2024, 1, 6, 12, 0),  # Saturday
        end_date=datetime(2024, 1, 10, 12, 0),
        assets=["SPX"],
        timeframe="1d",
    )

    periods = backtest._get_date_range_for_asset("SPX")

    # First period should be Monday or later
    if periods:
        assert periods[0].weekday() < 5  # Not weekend
```

### AC6.3: Entire Range is Non-Trading

**Given:** Date range is Saturday to Sunday only
**When:** `_get_date_range_for_asset()` is called for US equity
**Then:** Returns empty list

```python
def test_entire_range_non_trading():
    backtest = Backtest(
        start_date=datetime(2024, 1, 6, 12, 0),  # Saturday
        end_date=datetime(2024, 1, 7, 12, 0),    # Sunday
        assets=["SPX"],
        timeframe="4h",
    )

    periods = backtest._get_date_range_for_asset("SPX")

    assert len(periods) == 0  # No trading periods
```

### AC6.4: DST Transition Handling

**Given:** Date range spans DST transition (March or November)
**When:** `filter_to_trading_hours()` is called
**Then:** Trading hours are correctly adjusted for DST

```python
def test_dst_transition():
    calendar = MarketCalendar()

    # March DST transition (spring forward)
    timestamps = [
        datetime(2024, 3, 8, 15, 0),   # Before DST
        datetime(2024, 3, 11, 15, 0),  # After DST
    ]

    filtered = calendar.filter_to_trading_hours(timestamps, AssetType.US_EQUITY)

    # Both should be valid trading hours (calendar handles DST)
    assert len(filtered) == 2
```

---

## AC7: Backwards Compatibility

### AC7.1: Existing Tests Pass

**Given:** Existing backtest tests in test suite
**When:** Tests are run after implementation
**Then:** All existing tests pass without modification

```bash
# Validation command
pytest tests/test_backtest.py tests/test_backtest_integration.py -v
# Expected: All tests pass
```

### AC7.2: Opt-Out Restores Old Behavior

**Given:** Backtest with `config={'market_hours_filter': False}`
**When:** Compared to pre-implementation behavior
**Then:** Results are identical

```python
def test_opt_out_backwards_compatible():
    # Run with filtering disabled
    backtest = Backtest(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        assets=["SPX"],
        timeframe="4h",
        config={"market_hours_filter": False},
    )

    periods = backtest._get_date_range_for_asset("SPX")
    all_periods = backtest._get_date_range()

    # Should match original behavior exactly
    assert periods == all_periods
```

---

## Validation Commands

### Run Unit Tests

```bash
# Test asset types
pytest tests/test_asset_types.py -v

# Test market calendar
pytest tests/test_market_calendar.py -v

# Test backtest integration
pytest tests/test_backtest_market_hours.py -v
```

### Run All Backtest Tests

```bash
pytest tests/test_backtest*.py -v
```

### Manual Validation

```python
# Compare analysis counts
from quantagent.backtesting.backtest import Backtest
from datetime import datetime

# With filtering (default)
bt_filtered = Backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    assets=["SPX"],
    timeframe="4h",
)

# Without filtering
bt_unfiltered = Backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    assets=["SPX"],
    timeframe="4h",
    config={"market_hours_filter": False},
)

filtered_count = len(bt_filtered._get_date_range_for_asset("SPX"))
unfiltered_count = len(bt_unfiltered._get_date_range_for_asset("SPX"))

print(f"Filtered: {filtered_count} periods")
print(f"Unfiltered: {unfiltered_count} periods")
print(f"Reduction: {100 * (1 - filtered_count/unfiltered_count):.1f}%")

# Expected output:
# Filtered: ~126 periods (trading hours only)
# Unfiltered: ~540 periods (all hours)
# Reduction: ~75%
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Period reduction for US equities (4h) | 40-60% fewer | Compare filtered vs unfiltered counts |
| Period reduction for crypto | 0% | Verify no change |
| Filtering overhead | < 100ms | Time date range generation |
| Test coverage | > 80% | `pytest --cov` on new modules |
| Existing tests passing | 100% | `pytest tests/test_backtest*.py` |

---

## References

- [Requirements Document](../01_requirements/QuantAgent-s92-RQ-backtest-market-hours.md)
- [Design Document](../03_design/QuantAgent-s92-DS-backtest-market-hours.md)
