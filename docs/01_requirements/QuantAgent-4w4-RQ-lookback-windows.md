# QuantAgent-4w4 — Requirements: Backtest Must Honor Strategy-Specific Lookback Windows

**Issue:** QuantAgent-4w4  
**Unblocks:** QuantAgent-b8r (M1 Strategy 3 — 52-week high momentum)  
**Type:** Requirements  
**Created:** 2026-05-06  

---

## Objective

Fix the `Backtest` engine so it fetches enough historical data to satisfy the lookback requirement
of the active strategy instead of using a hardcoded 30-day window.

---

## Problem Statement

`Backtest._analyze_and_trade()` (line 402 of `quantagent/backtesting/backtest.py`) hardcodes:

```python
lookback_days = 30
data_start = current_date - timedelta(days=lookback_days)
```

For `FiftyTwoWeekHighStrategy`, which requires `min_candles = 252 + max(50, 20) + 1 = 303` daily bars,
the engine only supplies ~21–22 candles (30 calendar days minus weekends/holidays). The strategy's
internal guard (`len(kline_data) < min_candles`) returns `None` every call, producing
`TOTAL_TRADES = 0` for the wrong reason — engine limitation, not a real "no signal" condition.

---

## Scope In

- New `required_history_bars` property on `TradingStrategy` base class (default `30`)
- `Backtest._analyze_and_trade()` reads `self.strategy.required_history_bars` instead of `30`
- Helper `Backtest._bars_to_calendar_days(bars)` converts trading bars to calendar days per timeframe
- Update minimum-data guard in `_analyze_and_trade()` to use `required_history_bars`
- Unit tests verifying bar-to-day conversion and engine data request logic
- `FiftyTwoWeekHighStrategy.required_history_bars` override **documented as required for b8r** (implementer of b8r must add it)

## Scope Out

- Changes to `FiftyTwoWeekHighStrategy` signal logic (b8r scope)
- Data provider performance optimisation
- New datasets or data sources
- Changes to any strategy other than the base class property addition
- Changes to `StrategyAssembler` or `PositionMonitor`

---

## Functional Requirements

### FR1 — `TradingStrategy.required_history_bars` property

A concrete read-only property `required_history_bars: int` must exist on `TradingStrategy`.

- Default return value: `30` (backward-compatible with all existing strategies)
- Subclasses may override to return any positive integer
- The property must be accessible without a running instance (i.e., accessible on `self.strategy` inside `Backtest`)

### FR2 — Backtest reads `required_history_bars`

`Backtest._analyze_and_trade()` must replace the hardcoded `lookback_days = 30` with:

```python
lookback_bars = self.strategy.required_history_bars
lookback_days = self._bars_to_calendar_days(lookback_bars)
data_start = current_date - timedelta(days=lookback_days)
```

The rest of the data-fetch and guard logic must use `lookback_bars` (not the old `30`) as the
minimum candle count.

### FR3 — `_bars_to_calendar_days` conversion

`Backtest` must implement a private helper that converts trading bars to calendar days:

| Timeframe | Formula | Example (303 bars) |
|-----------|---------|---------------------|
| `1d` | `ceil(bars × 365 / 252)` | 439 calendar days |
| `1h` | `ceil(bars / 6.5 × 7 / 5)` | 66 calendar days |
| `4h` | `ceil(bars × 4 / 6.5 × 7 / 5)` | 262 calendar days |
| default | `bars × 2` | 606 calendar days (safe upper bound) |

The multiplier accounts for weekends and market holidays so the data provider always receives at
least `lookback_bars` usable trading bars within the window.

### FR4 — Minimum data guard update

The existing guard:
```python
if df.empty or len(df) < 30:
```
must become:
```python
if df.empty or len(df) < lookback_bars:
```

The log message must include the actual `len(df)` and the expected `lookback_bars` so
`Insufficient data` warnings are self-explanatory.

### FR5 — Backward compatibility

All existing strategies (`LLMAgentStrategy`, `RSIMeanReversionStrategy`, `TripleScreenStrategy`)
inherit the default `required_history_bars = 30`. No behavioral change occurs for these strategies.

### FR6 — FiftyTwoWeekHighStrategy override (b8r dependency note)

`FiftyTwoWeekHighStrategy` must override `required_history_bars` to return:
```python
self.lookback_days + max(self.trend_ma_period, self.volume_ma_period) + 1
```
With defaults: `252 + 50 + 1 = 303`.

**This override is the responsibility of the QuantAgent-b8r implementer.** It is documented
here so the b8r implementer knows what to add when rebasing on the 4w4 fix.

---

## Edge Cases

- Strategy with `required_history_bars = 0` or negative → treat as `30` (clamp to minimum)
- Timeframe string not recognised by `_bars_to_calendar_days` → use `bars × 2` fallback
- `data_provider.get_ohlc` returns fewer bars than expected (sparse data) → existing warning path is preserved; warning now includes the required count

---

## Definition of Done

- `TradingStrategy.required_history_bars` property exists in `quantagent/strategy/base.py`
- `Backtest._analyze_and_trade()` no longer contains `lookback_days = 30`
- `Backtest._bars_to_calendar_days()` exists and is tested
- Acceptance criteria in `docs/05_acceptance_tests/QuantAgent-4w4-AC-lookback-windows.md` all pass
- No regression in existing backtest tests (test_backtest.py, test_backtest_integration.py, etc.)
