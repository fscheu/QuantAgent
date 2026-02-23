# Requirements: Backtest Market Hours Filtering

**Issue ID:** QuantAgent-s92
**Type:** RQ (Requirements)
**Level:** STANDARD
**Created:** 2026-01-05

---

## Objective

Improve backtest efficiency by filtering out analysis periods when markets are closed, reducing unnecessary LLM agent executions and associated token costs.

---

## Context

### Current State

The backtest engine (`quantagent/backtesting/backtest.py`) generates time periods for analysis at regular intervals based on timeframe:

```python
# _get_date_range() at lines 266-292
def _get_date_range(self) -> List[datetime]:
    dates = []
    current = self.start_date
    # Determine step size based on timeframe
    if self.timeframe in ["1h", "4h"]:
        step_hours = int(self.timeframe.replace("h", ""))
        step = timedelta(hours=step_hours)
    elif self.timeframe == "1d":
        step = timedelta(days=1)
    # ... generates ALL intervals regardless of market hours
```

**Problem**: For intraday timeframes (1h, 4h) over extended periods:
- A 90-day backtest at 4h generates ~540 periods per asset
- Many of these periods fall outside market hours (weekends, overnight for US markets)
- Each period triggers a full agent analysis (LLM calls)
- This wastes execution time and token costs when prices haven't changed

### Asset Type Differences

| Asset Type | Trading Hours | Examples |
|------------|---------------|----------|
| Crypto | 24/7/365 | BTC, ETH |
| US Equities | Mon-Fri 9:30-16:00 ET | SPX, QQQ, individual stocks |
| US Futures | Sun 18:00 - Fri 17:00 ET (with breaks) | ES, NQ, CL, GC |
| European | Mon-Fri, varies by exchange | DAX |

### Professional Systems Approach

- **QuantConnect**: Uses `TradingCalendar` and `market-hours` API
- **pandas_market_calendars**: Python library with 50+ exchange calendars
- **exchange_calendars**: Zipline's calendar library
- **Backtrader**: Supports per-data-feed trading calendars

---

## Scope (In-Scope)

### 1. Asset Type Classification

Add metadata to classify assets by trading schedule:
- `CRYPTO`: 24/7 trading (no filtering)
- `US_EQUITY`: NYSE/NASDAQ hours (Mon-Fri 9:30-16:00 ET)
- `US_FUTURES`: Extended hours with overnight breaks
- Allow custom/override per symbol

### 2. Market Calendar Integration

Integrate `pandas_market_calendars` library for exchange schedules:
- Support NYSE, NASDAQ, CME exchanges
- Filter `_get_date_range()` output to trading hours only
- Handle timezone conversions properly (user timezone -> exchange timezone)

### 3. Filtering Logic in Backtest

Modify backtest loop to skip non-trading periods:
- For daily timeframe: skip weekends/holidays for traditional markets
- For intraday: skip outside regular trading hours
- Preserve full iteration for crypto assets

### 4. Configuration Options

Allow user control over filtering behavior:
- Enable/disable market hours filtering (default: enabled)
- Override asset type classification
- Extend to pre-market/after-hours for equities (optional)

---

## Non-Scope (Out-of-Scope)

### Explicitly Excluded

1. **Half-day trading sessions** - Treat as regular trading days for MVP
2. **Market-specific holidays** - Use calendar library defaults, no custom holidays
3. **Partial period analysis** - If period spans market open/close, include entirely
4. **After-hours/pre-market for equities** - Excluded in MVP (regular hours only)
5. **Foreign exchange (forex)** - Not currently supported in system

### Deferred to Future

1. **Real-time market status checking** - Only historical calendars
2. **Custom exchange calendars** - Use library defaults
3. **Minute-level timeframe optimizations** - Focus on 1h+ timeframes

---

## Use Cases

### UC1: Stock Backtest with Intraday Timeframe

**Given:** User runs backtest for SPX with 4h timeframe over 30 days
**When:** Backtest generates date range
**Then:**
- Only periods during NYSE trading hours (9:30-16:00 ET) are included
- Weekends and holidays are skipped
- Approximately 60% fewer analysis periods compared to current behavior

### UC2: Crypto Backtest (No Change)

**Given:** User runs backtest for BTC with 4h timeframe over 30 days
**When:** Backtest generates date range
**Then:**
- All periods are included (24/7 market)
- No filtering applied
- Behavior identical to current implementation

### UC3: Mixed Asset Backtest

**Given:** User runs backtest for [BTC, SPX, CL] with 1h timeframe
**When:** Backtest iterates through assets at each period
**Then:**
- BTC: analyzed at all hours
- SPX: analyzed only during NYSE hours
- CL: analyzed during CME futures hours
- Each asset filtered independently

### UC4: Daily Timeframe Backtest

**Given:** User runs backtest for SPX with 1d timeframe over 90 days
**When:** Backtest generates date range
**Then:**
- Weekends and market holidays are skipped
- Approximately 252/365 (~69%) of days are included

### UC5: User Disables Filtering

**Given:** User runs backtest with `config={'market_hours_filter': False}`
**When:** Backtest generates date range
**Then:**
- All periods included regardless of market hours
- Behavior matches current implementation (opt-out)

---

## Constraints & Non-Functional Requirements

### Performance

- Calendar lookups must be cached (not re-computed per period)
- Filtering overhead should be < 100ms for 1-year backtest
- Memory usage should not increase significantly

### Accuracy

- Must correctly handle timezone conversions (user input -> exchange timezone)
- Must handle daylight saving time transitions correctly
- Must not filter out valid trading periods

### Reliability

- If calendar lookup fails, fall back to no filtering (graceful degradation)
- Log warnings when fallback is used
- Handle unknown symbols by defaulting to 24/7 schedule

### Maintainability

- Asset type mapping should be easily extensible
- Calendar library version should be pinned to avoid breaking changes
- Clear separation between calendar logic and backtest logic

---

## Edge Cases

### EC1: Market Opens Mid-Period

**Scenario:** 4h period from 8:00-12:00, market opens at 9:30
**Behavior:** Include the period (partial overlap counts)

### EC2: Market Closes Mid-Period

**Scenario:** 4h period from 14:00-18:00, market closes at 16:00
**Behavior:** Include the period (partial overlap counts)

### EC3: Holiday During Backtest Range

**Scenario:** Backtest spans July 4th for US equity
**Behavior:** Skip July 4th entirely for US assets, include for crypto

### EC4: Unknown Symbol

**Scenario:** User backtests symbol not in asset mapping (e.g., "CUSTOM")
**Behavior:** Log warning, treat as 24/7 asset (no filtering)

### EC5: Start/End Date Outside Trading Hours

**Scenario:** Start date is Saturday
**Behavior:** First analysis period is next valid trading period

### EC6: Timezone Edge Case

**Scenario:** User in different timezone than exchange
**Behavior:** Convert user times to exchange timezone for filtering, display in user timezone

---

## Dependencies

### New External Dependencies

```
pandas_market_calendars>=4.0.0
```

**Rationale:** Well-maintained, supports 50+ exchanges, used by professional quant systems

### Alternative Considered

`exchange_calendars` - More comprehensive but heavier, less actively maintained for our use case

### Changes to Existing Modules

| File | Change Type | Description |
|------|-------------|-------------|
| `quantagent/backtesting/backtest.py` | Modify | Add filtering to `_get_date_range()` |
| `quantagent/data/provider.py` | Modify | Add asset type classification |
| `quantagent/default_config.py` | Modify | Add market hours configuration |
| `requirements.txt` | Modify | Add `pandas_market_calendars` |

---

## Definition of "Done"

The change is complete when:

### 1. Code

- [ ] Asset type classification implemented (CRYPTO, US_EQUITY, US_FUTURES)
- [ ] `pandas_market_calendars` integrated for exchange schedules
- [ ] `_get_date_range()` filters periods by market hours
- [ ] Configuration option to enable/disable filtering
- [ ] Fallback to 24/7 schedule for unknown assets

### 2. Tests

- [ ] Unit tests for asset type classification
- [ ] Unit tests for market hours filtering logic
- [ ] Integration test: backtest with filtering enabled vs disabled
- [ ] Test coverage >80% for new code

### 3. Documentation

- [ ] Docstrings for new functions
- [ ] Update `docs/03_design/backtesting_engine.md` with market hours section
- [ ] Planning/design documents completed

### 4. Validation

- [ ] Existing tests pass (backwards compatibility)
- [ ] New tests pass
- [ ] Manual validation: compare analysis counts with/without filtering

### 5. Metrics (Quantifiable Success)

- [ ] Reduction in analysis periods for US equities: ~40-60% for intraday timeframes
- [ ] No change in analysis periods for crypto assets
- [ ] Performance overhead < 100ms for 1-year backtest

---

## References

- Current code: `quantagent/backtesting/backtest.py:_get_date_range()`
- DataProvider symbols: `quantagent/data/provider.py:SYMBOL_MAPPING`
- pandas_market_calendars: https://github.com/rsheftel/pandas_market_calendars
- QuantConnect market hours: https://www.quantconnect.com/docs/v2/writing-algorithms/securities/asset-classes/us-equity/market-hours

---

## Notes

- **Level STANDARD**: Meaningful efficiency improvement, requires new dependency, but not architectural change
- **Backwards compatible**: Filtering can be disabled via config
- **Graceful degradation**: Unknown assets default to 24/7 (no filtering)
