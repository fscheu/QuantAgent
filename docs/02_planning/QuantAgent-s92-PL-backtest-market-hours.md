# Implementation Plan: Backtest Market Hours Filtering

**Issue ID:** QuantAgent-s92
**Type:** PL (Planning)
**Created:** 2026-01-05
**Related:**
- [Requirements](../01_requirements/QuantAgent-s92-RQ-backtest-market-hours.md)
- [Design](../03_design/QuantAgent-s92-DS-backtest-market-hours.md)
- [Acceptance Criteria](../05_acceptance_tests/QuantAgent-s92-AC-backtest-market-hours.md)

---

## Summary

Add market hours awareness to the backtest engine to filter out non-trading periods, reducing unnecessary LLM agent executions for traditional market assets while preserving 24/7 behavior for crypto.

**Estimated Effort:** 4-6 hours
**Risk Level:** Low (additive feature with opt-out)

---

## Tasks

### Phase 1: Core Infrastructure (2-2.5 hours)

#### Task 1.1: Add pandas_market_calendars Dependency

**File:** `requirements.txt`
**Effort:** 5 min
**Dependencies:** None

**Changes:**
```
# Add to requirements.txt
pandas_market_calendars>=4.0.0,<5.0.0
```

**Validation:**
```bash
pip install pandas_market_calendars
python -c "import pandas_market_calendars as mcal; print(mcal.get_calendar('NYSE'))"
```

---

#### Task 1.2: Create Asset Types Module

**File:** `quantagent/data/asset_types.py` (new)
**Effort:** 30 min
**Dependencies:** None

**Implementation:**
1. Define `AssetType` enum with values: `CRYPTO`, `US_EQUITY`, `US_FUTURES`, `EUROPEAN`, `UNKNOWN`
2. Create `ASSET_TYPE_MAPPING` dictionary with known symbols
3. Implement `get_asset_type(symbol: str) -> AssetType` function with pattern inference

**Key logic:**
```python
def get_asset_type(symbol: str) -> AssetType:
    # 1. Direct lookup in mapping
    if symbol in ASSET_TYPE_MAPPING:
        return ASSET_TYPE_MAPPING[symbol]

    # 2. Pattern-based inference
    if symbol.endswith("-USD") or symbol in ("BTC", "ETH", "SOL"):
        return AssetType.CRYPTO
    if symbol.endswith("=F"):
        return AssetType.US_FUTURES
    if symbol.startswith("^"):
        return AssetType.US_EQUITY

    # 3. Default
    return AssetType.UNKNOWN
```

**Tests to write:** `tests/test_asset_types.py`
- `test_crypto_classification`
- `test_equity_classification`
- `test_futures_classification`
- `test_unknown_defaults`
- `test_pattern_inference`

---

#### Task 1.3: Create Market Calendar Module

**File:** `quantagent/data/market_calendar.py` (new)
**Effort:** 1 hour
**Dependencies:** Task 1.1, Task 1.2

**Implementation:**
1. Create `MarketCalendar` class with calendar caching
2. Implement `get_calendar(exchange: str)` with error handling
3. Implement `get_schedule(exchange, start, end)` with caching
4. Implement `filter_to_trading_hours(timestamps, asset_type)` main filtering logic
5. Create singleton accessor `get_market_calendar()`

**Key components:**
```python
class MarketCalendar:
    def __init__(self):
        self._calendars: dict[str, mcal.MarketCalendar] = {}
        self._schedule_cache: dict[tuple, pd.DataFrame] = {}

    def get_calendar(self, exchange: str) -> Optional[mcal.MarketCalendar]:
        # Load and cache calendar, handle errors gracefully

    def get_schedule(self, exchange, start, end) -> Optional[pd.DataFrame]:
        # Get trading schedule with caching

    def filter_to_trading_hours(self, timestamps, asset_type) -> list[datetime]:
        # Main filtering logic
```

**Tests to write:** `tests/test_market_calendar.py`
- `test_calendar_load_nyse`
- `test_calendar_load_cme`
- `test_calendar_fallback_invalid`
- `test_schedule_caching`
- `test_filter_crypto_no_change`
- `test_filter_equity_weekends`
- `test_filter_equity_hours`
- `test_filter_holiday`
- `test_filter_unknown_no_change`

---

#### Task 1.4: Add Configuration Options

**File:** `quantagent/default_config.py`
**Effort:** 10 min
**Dependencies:** None

**Changes:**
```python
# Add to existing file:

# Market hours filtering for backtests
MARKET_HOURS_CONFIG = {
    "enabled": True,
    "fallback_to_24_7": True,
    "include_extended_hours": False,
}
```

---

### Phase 2: Backtest Integration (1.5-2 hours)

#### Task 2.1: Modify Backtest.__init__

**File:** `quantagent/backtesting/backtest.py`
**Effort:** 20 min
**Dependencies:** Phase 1 complete

**Changes:**
1. Add imports for asset_types and market_calendar modules
2. Add `self.market_hours_filter` from config (default: True)
3. Initialize `self._market_calendar` if filtering enabled
4. Cache asset types: `self._asset_types = {asset: get_asset_type(asset) for asset in assets}`

**Code location:** After line 148 (after `self.equity_curve`)

```python
# Market hours filtering (new)
self.market_hours_filter = self.config.get("market_hours_filter", True)
self._market_calendar = get_market_calendar() if self.market_hours_filter else None
self._asset_types: Dict[str, AssetType] = {
    asset: get_asset_type(asset) for asset in assets
}
```

---

#### Task 2.2: Add _get_date_range_for_asset Method

**File:** `quantagent/backtesting/backtest.py`
**Effort:** 30 min
**Dependencies:** Task 2.1

**New method after `_get_date_range()` (after line 292):**

```python
def _get_date_range_for_asset(self, asset: str) -> List[datetime]:
    """
    Get date range filtered by market hours for specific asset.

    Args:
        asset: Asset symbol

    Returns:
        List of valid trading timestamps for this asset
    """
    all_dates = self._get_date_range()

    if not self.market_hours_filter or self._market_calendar is None:
        return all_dates

    asset_type = self._asset_types.get(asset, AssetType.UNKNOWN)
    return self._market_calendar.filter_to_trading_hours(all_dates, asset_type)
```

---

#### Task 2.3: Modify run() Method

**File:** `quantagent/backtesting/backtest.py`
**Effort:** 45 min
**Dependencies:** Task 2.2

**Changes to `run()` method:**

1. Add logging for market_hours_filter status (after line 167)
2. Replace single date_range loop with per-asset date range
3. Update progress logging to use total periods across all assets

**Key changes:**

```python
def run(self, name: Optional[str] = None) -> BacktestMetrics:
    logger.info(f"Starting backtest: {self.start_date} to {self.end_date}")
    logger.info(f"Assets: {self.assets}, Timeframe: {self.timeframe}")
    logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
    logger.info(f"Market hours filter: {self.market_hours_filter}")  # NEW

    self._create_backtest_run(name)

    # Calculate total periods with per-asset filtering
    total_periods = 0
    asset_date_ranges = {}
    for asset in self.assets:
        asset_dates = self._get_date_range_for_asset(asset)
        asset_date_ranges[asset] = asset_dates
        total_periods += len(asset_dates)

        asset_type = self._asset_types.get(asset, AssetType.UNKNOWN)
        logger.info(f"Asset {asset} ({asset_type.value}): {len(asset_dates)} periods")

    logger.info(f"Backtesting {total_periods} total analysis periods")

    periods_completed = 0

    for asset in self.assets:
        asset_dates = asset_date_ranges[asset]

        for i, current_date in enumerate(asset_dates):
            self.current_date = current_date

            if i == 0 or current_date.date() != asset_dates[i - 1].date():
                self.risk_manager.reset_daily_tracker()

            try:
                self._analyze_and_trade(asset, current_date)
            except Exception as e:
                logger.error(f"Error analyzing {asset} at {current_date}: {e}")
                continue

            self._record_equity(current_date)
            periods_completed += 1

            if periods_completed % 100 == 0 or periods_completed == total_periods:
                progress = (periods_completed / total_periods) * 100
                logger.info(f"Progress: {progress:.1f}% ({periods_completed}/{total_periods})")

    # ... rest of method unchanged
```

---

### Phase 3: Testing (1-1.5 hours)

#### Task 3.1: Unit Tests for Asset Types

**File:** `tests/test_asset_types.py` (new)
**Effort:** 20 min
**Dependencies:** Task 1.2

**Test cases:**
```python
class TestAssetTypes:
    def test_crypto_direct_lookup(self):
        assert get_asset_type("BTC") == AssetType.CRYPTO

    def test_crypto_pattern_usd_suffix(self):
        assert get_asset_type("SOL-USD") == AssetType.CRYPTO

    def test_equity_direct_lookup(self):
        assert get_asset_type("SPX") == AssetType.US_EQUITY

    def test_equity_pattern_caret_prefix(self):
        assert get_asset_type("^AAPL") == AssetType.US_EQUITY

    def test_futures_direct_lookup(self):
        assert get_asset_type("ES") == AssetType.US_FUTURES

    def test_futures_pattern_f_suffix(self):
        assert get_asset_type("ZN=F") == AssetType.US_FUTURES

    def test_unknown_default(self):
        assert get_asset_type("RANDOM123") == AssetType.UNKNOWN
```

---

#### Task 3.2: Unit Tests for Market Calendar

**File:** `tests/test_market_calendar.py` (new)
**Effort:** 30 min
**Dependencies:** Task 1.3

**Test cases (with mocks):**
```python
class TestMarketCalendar:
    def test_calendar_loads_nyse(self):
        cal = MarketCalendar()
        assert cal.get_calendar("NYSE") is not None

    def test_calendar_fallback_invalid(self):
        cal = MarketCalendar()
        assert cal.get_calendar("INVALID_XYZ") is None

    def test_schedule_caching(self):
        cal = MarketCalendar()
        s1 = cal.get_schedule("NYSE", start, end)
        s2 = cal.get_schedule("NYSE", start, end)
        assert s1 is s2  # Same cached object

    @patch('pandas_market_calendars.get_calendar')
    def test_filter_crypto_no_change(self, mock_cal):
        cal = MarketCalendar()
        timestamps = [datetime(2024, 1, 6, 12, 0)]  # Saturday
        result = cal.filter_to_trading_hours(timestamps, AssetType.CRYPTO)
        assert result == timestamps

    @patch('pandas_market_calendars.get_calendar')
    def test_filter_equity_weekends(self, mock_cal):
        # Configure mock to return schedule without weekends
        # Assert weekend timestamps are filtered out
```

---

#### Task 3.3: Integration Tests for Backtest

**File:** `tests/test_backtest_market_hours.py` (new)
**Effort:** 30 min
**Dependencies:** Phase 2 complete

**Test cases:**
```python
class TestBacktestMarketHours:
    def test_filtering_reduces_equity_periods(self):
        bt = Backtest(start, end, ["SPX"], "4h", 100000)
        filtered = bt._get_date_range_for_asset("SPX")
        unfiltered = bt._get_date_range()
        assert len(filtered) < len(unfiltered) * 0.7

    def test_filtering_no_change_crypto(self):
        bt = Backtest(start, end, ["BTC"], "4h", 100000)
        filtered = bt._get_date_range_for_asset("BTC")
        unfiltered = bt._get_date_range()
        assert len(filtered) == len(unfiltered)

    def test_filtering_disabled_opt_out(self):
        bt = Backtest(start, end, ["SPX"], "4h", 100000,
                      config={"market_hours_filter": False})
        filtered = bt._get_date_range_for_asset("SPX")
        unfiltered = bt._get_date_range()
        assert len(filtered) == len(unfiltered)

    def test_mixed_assets(self):
        bt = Backtest(start, end, ["BTC", "SPX"], "4h", 100000)
        btc_periods = bt._get_date_range_for_asset("BTC")
        spx_periods = bt._get_date_range_for_asset("SPX")
        assert len(btc_periods) > len(spx_periods)
```

---

#### Task 3.4: Verify Existing Tests Pass

**Effort:** 10 min
**Dependencies:** All implementation complete

**Validation:**
```bash
pytest tests/test_backtest.py tests/test_backtest_integration.py -v
```

All existing tests must pass without modification.

---

### Phase 4: Documentation & Cleanup (30 min)

#### Task 4.1: Update Module Docstrings

**Files:** All new modules
**Effort:** 15 min

Add comprehensive docstrings to:
- `quantagent/data/asset_types.py`
- `quantagent/data/market_calendar.py`

---

#### Task 4.2: Update Backtesting Engine Docs

**File:** `docs/03_design/backtesting_engine.md`
**Effort:** 15 min

Add section:
```markdown
## Market Hours Filtering

The backtest engine supports filtering analysis periods to only include trading hours for traditional markets. This reduces unnecessary LLM agent executions when markets are closed.

### Configuration

```python
config = {
    "market_hours_filter": True,  # Enable filtering (default)
}
```

### Asset Type Classification

| Asset Type | Examples | Trading Hours |
|------------|----------|---------------|
| CRYPTO | BTC, ETH | 24/7 (no filtering) |
| US_EQUITY | SPX, QQQ | NYSE hours (9:30-16:00 ET) |
| US_FUTURES | ES, CL | CME extended hours |

### Efficiency Gains

For intraday timeframes on US equities:
- ~40-60% reduction in analysis periods
- Proportional reduction in LLM token costs
```

---

## Task Summary Table

| # | Task | File(s) | Effort | Dependencies |
|---|------|---------|--------|--------------|
| 1.1 | Add dependency | requirements.txt | 5 min | - |
| 1.2 | Asset types module | asset_types.py (new) | 30 min | - |
| 1.3 | Market calendar module | market_calendar.py (new) | 1 hr | 1.1, 1.2 |
| 1.4 | Configuration | default_config.py | 10 min | - |
| 2.1 | Backtest init | backtest.py | 20 min | Phase 1 |
| 2.2 | Date range method | backtest.py | 30 min | 2.1 |
| 2.3 | Run method changes | backtest.py | 45 min | 2.2 |
| 3.1 | Asset type tests | test_asset_types.py | 20 min | 1.2 |
| 3.2 | Calendar tests | test_market_calendar.py | 30 min | 1.3 |
| 3.3 | Integration tests | test_backtest_market_hours.py | 30 min | Phase 2 |
| 3.4 | Verify existing tests | - | 10 min | All |
| 4.1 | Module docstrings | *.py | 15 min | All |
| 4.2 | Update docs | backtesting_engine.md | 15 min | All |

**Total estimated: 4-6 hours**

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Calendar library API changes | Low | Medium | Pin version, add tests |
| Timezone handling bugs | Medium | High | Comprehensive UTC handling, DST tests |
| Performance regression | Low | Low | Caching at multiple levels |
| Unknown asset misclassification | Medium | Low | Conservative fallback (24/7) |

---

## Rollout Strategy

### 1. Feature Flag (Default On)

```python
config = {"market_hours_filter": True}  # Default
```

### 2. Gradual Validation

1. Run existing backtests with filtering enabled
2. Compare results (period counts, not metrics)
3. Validate no unexpected behavior

### 3. Opt-Out Available

```python
config = {"market_hours_filter": False}  # Disable
```

---

## Success Criteria

- [ ] All acceptance criteria in AC document pass
- [ ] Existing backtest tests pass
- [ ] New tests have >80% coverage
- [ ] Period reduction for US equities: 40-60%
- [ ] No change for crypto assets
- [ ] Performance overhead < 100ms

---

## Post-Implementation

### Monitoring

After deployment, monitor:
- Backtest execution times
- Error rates in calendar lookups
- User feedback on filtering behavior

### Future Enhancements (Out of Scope)

1. Pre-market/after-hours support for equities
2. Custom exchange calendars
3. Real-time market status checking
4. Forex market support

---

## References

- [Requirements](../01_requirements/QuantAgent-s92-RQ-backtest-market-hours.md)
- [Design](../03_design/QuantAgent-s92-DS-backtest-market-hours.md)
- [Acceptance Criteria](../05_acceptance_tests/QuantAgent-s92-AC-backtest-market-hours.md)
- pandas_market_calendars: https://github.com/rsheftel/pandas_market_calendars
