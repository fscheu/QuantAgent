# QuantAgent-4w4 — Implementation Notes: Backtest Lookback Windows

**Issue:** QuantAgent-4w4  
**Type:** Implementation Spec (planner output → implementer input)  
**Created:** 2026-05-06  

---

## Overview

Replace the hardcoded `lookback_days = 30` in `Backtest._analyze_and_trade()` with a strategy-driven
value. Add a `required_history_bars` property to `TradingStrategy` (default 30, backward-compatible)
that strategies override to declare their minimum data requirement. The engine converts this bar count
to calendar days and requests the correct window from the data provider.

---

## Files to Change

| File | Change |
|------|--------|
| `quantagent/strategy/base.py` | Add `required_history_bars` property |
| `quantagent/backtesting/backtest.py` | Use property; add `_bars_to_calendar_days` helper |
| `tests/test_backtest_lookback_window_4w4.py` | New test file (ACs 1–8) |

**Not changing:** `fifty_two_week_high_strategy.py` (that override is b8r's responsibility post-merge).

---

## Change 1 — `quantagent/strategy/base.py`

Add a concrete (non-abstract) property to `TradingStrategy` after the existing `get_default_exit_policy` method:

```python
@property
def required_history_bars(self) -> int:
    """Minimum number of OHLCV bars the backtest engine must supply.

    Override in subclasses that need longer lookback windows.
    Default: 30 — matches the historical engine behavior.
    """
    return 30
```

No changes to any abstract methods or existing logic.

---

## Change 2 — `quantagent/backtesting/backtest.py`

### 2a — `_analyze_and_trade()` (lines 400–412 area)

Replace:
```python
lookback_days = 30
data_start = current_date - timedelta(days=lookback_days)

df = self.data_provider.get_ohlc(
    symbol=asset,
    timeframe=self.timeframe,
    start_date=data_start,
    end_date=current_date,
)

if df.empty or len(df) < 30:
    logger.warning(
        f"Insufficient data for {asset} at {current_date} (got {len(df)} records)",
        ...
    )
    return
```

With:
```python
lookback_bars = self.strategy.required_history_bars
data_start = current_date - timedelta(days=self._bars_to_calendar_days(lookback_bars))

df = self.data_provider.get_ohlc(
    symbol=asset,
    timeframe=self.timeframe,
    start_date=data_start,
    end_date=current_date,
)

if df.empty or len(df) < lookback_bars:
    logger.warning(
        f"Insufficient data for {asset} at {current_date} "
        f"(got {len(df)}, need {lookback_bars})",
        extra={"event_type": "backtest_data_warning", "symbol": asset},
    )
    return
```

### 2b — New `_bars_to_calendar_days` method

Add to `Backtest` class (e.g. after `_get_periods_per_year`):

```python
def _bars_to_calendar_days(self, bars: int) -> int:
    """Convert trading bars to calendar days to request from the data provider.

    Applies a timeframe-specific multiplier to account for weekends and holidays,
    ensuring the provider window always contains at least `bars` usable candles.
    """
    bars = max(bars, 1)
    if self.timeframe == "1d":
        import math
        return math.ceil(bars * 365 / 252)
    elif self.timeframe == "1h":
        import math
        return math.ceil(bars / 6.5 * 7 / 5)
    elif self.timeframe == "4h":
        import math
        return math.ceil(bars * 4 / 6.5 * 7 / 5)
    else:
        return bars * 2
```

Add `import math` at module level (top of file) if not already present.

---

## Change 3 — `tests/test_backtest_lookback_window_4w4.py` (new file)

Cover ACs 1–8 from the acceptance criteria doc. Key test cases:

### Tests

```
test_4w4_required_history_bars_default
  - Instantiate TripleScreenStrategy, assert .required_history_bars == 30

test_4w4_required_history_bars_override
  - Define inline MockHighBarsStrategy(required_history_bars=300)
  - Assert property returns 300

test_4w4_bars_to_calendar_days
  - Instantiate Backtest with timeframe="1d"
  - Assert _bars_to_calendar_days(252) == 365
  - Assert _bars_to_calendar_days(303) == 439
  - Assert _bars_to_calendar_days(30) == 44 (ceil(30*365/252))

test_4w4_engine_requests_sufficient_bars
  - Mock data_provider.get_ohlc to return empty DataFrame
  - Patch strategy.required_history_bars = 300, timeframe = "1d"
  - Call _analyze_and_trade() for an arbitrary current_date
  - Capture start_date argument to get_ohlc
  - Assert (current_date - start_date).days >= ceil(300 * 365 / 252)

test_4w4_insufficient_data_guard
  - Mock data_provider.get_ohlc returning a 10-row DataFrame
  - Mock strategy.required_history_bars = 300
  - Call _analyze_and_trade(), assert it returns without executing a trade
  - Assert warning log contains "need 300"

test_4w4_no_spurious_warnings (regression)
  - Mock data_provider.get_ohlc returning a 35-row DataFrame
  - Default strategy (required_history_bars=30)
  - Assert no "Insufficient data" warning logged
```

Use `unittest.mock` for data_provider and strategy; avoid DB dependencies (use existing
`conftest.py` fixtures where possible).

---

## b8r Dependency Note

After `QuantAgent-4w4` is merged to main, the b8r feature branch
(`feature/QuantAgent-b8r-m1-strategy-3-52-week-high-momentum-brea`) must:

1. Rebase on main
2. Add the following property to `FiftyTwoWeekHighStrategy`:

```python
@property
def required_history_bars(self) -> int:
    return self.lookback_days + max(self.trend_ma_period, self.volume_ma_period) + 1
```

With defaults (`lookback_days=252`, `trend_ma_period=50`, `volume_ma_period=20`) this returns `303`.

This is what makes the b8r backtest reference run pass AC9 in the acceptance criteria.

---

## Risks and Constraints

| Risk | Mitigation |
|------|-----------|
| Daily strategies with `required_history_bars=252` will fetch ~439 calendar days of data per analysis step — a potentially large query | Data provider has caching; this is expected behavior. No optimization needed for M1. |
| `_bars_to_calendar_days` for `1h` / `4h` uses 6.5 trading hours — correct for US equities, may be off for crypto or forex | Existing strategies only use `1h` or `4h` for crypto/futures; they will keep default 30 bars. No immediate impact. |
| b8r feature branch needs manual rebase | Documented here and in AC9. b8r tech lead is aware. |

---

## Validation

After implementation:

```bash
# Static check
grep -n "lookback_days = 30" quantagent/backtesting/backtest.py  # must return empty

# Unit tests
pytest tests/test_backtest_lookback_window_4w4.py -v

# Regression
pytest tests/test_backtest.py tests/test_backtest_integration.py \
       tests/test_backtest_market_hours.py tests/test_backtest_phase4_metrics.py -v

# Optional compile check
python -m compileall -q quantagent/strategy/base.py quantagent/backtesting/backtest.py
```
