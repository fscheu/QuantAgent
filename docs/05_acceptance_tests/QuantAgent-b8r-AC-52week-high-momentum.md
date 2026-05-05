# QuantAgent-b8r — Acceptance Criteria: M1 Strategy 3 — 52-Week High Momentum / Breakout

**Issue:** QuantAgent-b8r  
**Parent:** QuantAgent-l0h (AC3)  
**Type:** Acceptance Tests  
**Created:** 2026-05-05  

---

## AC1 — Strategy class exists and satisfies the `TradingStrategy` ABC

**Given** `quantagent/strategy/fifty_two_week_high_strategy.py` is present in the repo  
**When** `from quantagent.strategy.fifty_two_week_high_strategy import FiftyTwoWeekHighStrategy` is executed  
**Then**:
- No import errors
- `issubclass(FiftyTwoWeekHighStrategy, TradingStrategy)` is `True`
- Instantiation with default parameters succeeds: `FiftyTwoWeekHighStrategy()`
- `FiftyTwoWeekHighStrategy` appears in `quantagent/strategy/__init__.py` exports

**Testable:** `pytest tests/test_fifty_two_week_high_strategy.py::test_class_is_valid_strategy`

---

## AC2 — 52-week high calculation is correct

### AC2.1 — Rolling max over lookback window

**Given** a DataFrame where the `high` column has a known maximum value at position `-(lookback_days+1)` to `-1` (excluding current bar)  
**When** `_compute_52w_high()` is called  
**Then** the returned value equals the known maximum

### AC2.2 — Current bar excluded from lookback

**Given** a kline series where the current (last) candle has an artificially high `high` value  
**When** `_compute_52w_high()` is called  
**Then** the current candle's high is NOT included in the 52w high calculation

**Testable:** `pytest tests/test_fifty_two_week_high_strategy.py -k "52w_high"`

---

## AC3 — Trend filter activates correctly

### AC3.1 — Price above SMA → trend filter passes

**Given** a kline series where the closing prices trend upward and `current_price > SMA(close, 50)[-1]`  
**When** `generate_signal()` evaluates the trend condition  
**Then** the trend filter does not block a breakout signal

### AC3.2 — Price below SMA → trend filter blocks signal

**Given** a kline series where `current_price < SMA(close, 50)[-1]` even though price breaks the 52w high  
**When** `generate_signal()` is called  
**Then** return value is `None`

**Testable:** `pytest tests/test_fifty_two_week_high_strategy.py -k "trend_filter"`

---

## AC4 — Volume filter activates correctly

### AC4.1 — Volume above threshold → volume filter passes

**Given** a kline series where the last candle's volume is `> volume_factor × volume_MA`  
**When** `generate_signal()` evaluates the volume condition  
**Then** the volume filter does not block a breakout signal

### AC4.2 — Low volume breakout → volume filter blocks signal

**Given** a kline series where price breaks above 52w high but last candle volume is `< volume_factor × volume_MA`  
**When** `generate_signal()` is called  
**Then** return value is `None`

**Testable:** `pytest tests/test_fifty_two_week_high_strategy.py -k "volume_filter"`

---

## AC5 — Breakout condition

### AC5.1 — No breakout → no signal

**Given** `current_price <= high_52w` (even if ratio is close, e.g., 0.99)  
**When** `generate_signal()` is called with trend and volume filters passing  
**Then** return value is `None`

### AC5.2 — Exact equality → no signal

**Given** `current_price == high_52w`  
**When** `generate_signal()` is called  
**Then** return value is `None` (strict `>` comparison required)

**Testable:** `pytest tests/test_fifty_two_week_high_strategy.py -k "breakout"`

---

## AC6 — Combined: `generate_signal` produces correct LONG decision

### AC6.1 — All conditions pass → LONG signal

**Given** deterministic kline_data (≥ 303 candles) where:
- `current_price > high_52w` (breakout)
- `current_price > SMA(close, 50)[-1]` (uptrend)
- `volume[-1] > 1.5 × vol_MA[-1]` (volume confirmation)

**When** `strategy.generate_signal(kline_data, "AAPL", "1d", current_price)` is called  
**Then**:
- Return value is a `TradingSignal` (not `None`)
- `signal.decision == "LONG"`
- `signal.confidence` is in `[0.1, 1.0]`
- `signal.entry_price == current_price`
- `signal.stop_loss == current_price * (1 - stop_loss_pct)` (within float tolerance)
- `signal.take_profit == current_price * (1 + take_profit_pct)` (within float tolerance)
- `signal.reasoning` contains "52w-high" (breakout is named in the reasoning string)
- `signal.exit_policy == ExitPolicy.TRAILING_STOP`

### AC6.2 — No SHORT signals ever produced

**Given** any kline_data and any `current_price`  
**When** `generate_signal()` returns a non-`None` value  
**Then** `signal.decision != "SHORT"` always

### AC6.3 — Trend filter fails alone → `None`

**Given** breakout and volume conditions pass but `current_price < SMA[-1]`  
**When** `generate_signal()` is called  
**Then** return value is `None`

### AC6.4 — Volume filter fails alone → `None`

**Given** breakout and trend conditions pass but volume is insufficient  
**When** `generate_signal()` is called  
**Then** return value is `None`

### AC6.5 — Fewer than min candles → `None` without exception

**Given** `len(kline_data) < lookback_days + max(trend_ma_period, volume_ma_period) + 1`  
**When** `generate_signal()` is called  
**Then** return value is `None` and no exception is raised

**Testable:** `pytest tests/test_fifty_two_week_high_strategy.py -k "generate_signal"`

---

## AC7 — Confidence calculation

**Given** a breakout where `current_price` is 5% above `high_52w`  
**When** `generate_signal()` returns a signal  
**Then** `signal.confidence` equals `max(0.1, min(1.0, 0.05 * 10))` = `0.5` (within float tolerance)

**Given** a breakout where `current_price` is 0.5% above `high_52w`  
**When** `generate_signal()` returns a signal  
**Then** `signal.confidence == 0.1` (floor applied)

**Given** a breakout where `current_price` is 15% above `high_52w`  
**When** `generate_signal()` returns a signal  
**Then** `signal.confidence == 1.0` (cap applied)

**Testable:** `pytest tests/test_fifty_two_week_high_strategy.py -k "confidence"`

---

## AC8 — Non-HOLD signal generated in realistic scenario

**Given** a synthetic 400-candle daily dataset designed with:
- A clear 6-month trend upward (SMA upward)
- Normal volume for most bars, with a spike on the last bar
- The last candle's close price slightly above the rolling 252-bar high

**When** `generate_signal()` is called on the last candle  
**Then** the return value is a `TradingSignal` with `decision == "LONG"`, proving the strategy is not permanently stuck in HOLD

**Testable:** `pytest tests/test_fifty_two_week_high_strategy.py::test_generates_signal_in_realistic_scenario`

---

## AC9 — `should_reevaluate` always returns False

**Given** any `ActivePosition` and `current_price`  
**When** `strategy.should_reevaluate(position, current_price)` is called  
**Then** the return value is `False`

**Testable:** `pytest tests/test_fifty_two_week_high_strategy.py::test_should_reevaluate_false`

---

## AC10 — Reference backtest completes without crash and PnL is calculated

**Given** a test environment with database configured and AAPL daily data available for 2022–2023

**When** the reference backtest from `QuantAgent-b8r-IM` is run:
```python
strategy = FiftyTwoWeekHighStrategy()
backtest = Backtest(
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2023, 12, 31),
    assets=["AAPL"],
    timeframe="1d",
    initial_capital=100_000.0,
    strategy=strategy,
)
metrics = backtest.run(name="QuantAgent-b8r-reference")
```

**Then**:
- `backtest.run()` completes without raising any exception
- `metrics.total_pnl` is a finite float (positive, negative, or zero)
- `metrics.total_trades >= 0`
- The run is logged with a name containing "QuantAgent-b8r"

**Testable:** Manual integration test or `pytest tests/test_fifty_two_week_high_strategy.py::test_backtest_integration -m integration`

---

## Non-goals (explicitly excluded)

- Performance targets (win rate, Sharpe ratio, drawdown)
- Short-selling signals or short-side momentum
- Proximity signal (without full breakout)
- Multi-symbol parallel validation
- Provider-specific behaviour or LLM integration
- Comparison against RSI, Triple Screen, or LLMAgentStrategy
