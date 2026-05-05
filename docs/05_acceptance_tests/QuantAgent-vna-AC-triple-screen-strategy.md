# QuantAgent-vna — Acceptance Criteria: M1 Strategy 1 — Triple Screen Strategy

**Issue:** QuantAgent-vna  
**Parent:** QuantAgent-l0h (AC3)  
**Type:** Acceptance Tests  
**Created:** 2026-05-05  

---

## AC1 — Strategy class exists and satisfies the `TradingStrategy` ABC

**Given** `quantagent/strategy/triple_screen_strategy.py` is present in the repo  
**When** `from quantagent.strategy.triple_screen_strategy import TripleScreenStrategy` is executed  
**Then**:
- No import errors
- `issubclass(TripleScreenStrategy, TradingStrategy)` is `True`
- Instantiation with default parameters succeeds: `TripleScreenStrategy()`

**Testable:** `pytest tests/test_triple_screen_strategy.py::test_class_is_valid_strategy`

---

## AC2 — Screen 1: EMA slope correctly identifies trend direction

### AC2.1 — Rising EMA slope → uptrend

**Given** synthetic weekly bars whose close prices strictly increase over the last `trend_ema_period + 1` bars  
**When** `_screen1_trend()` is called with those bars  
**Then** the return value is `"UP"`

### AC2.2 — Falling EMA slope → downtrend

**Given** synthetic weekly bars whose close prices strictly decrease  
**When** `_screen1_trend()` is called  
**Then** the return value is `"DOWN"`

### AC2.3 — Insufficient weekly bars → None

**Given** fewer than `trend_ema_period + 1` synthetic weekly bars  
**When** `_screen1_trend()` is called  
**Then** the return value is `None`

**Testable:** `pytest tests/test_triple_screen_strategy.py -k "screen1"`

---

## AC3 — Screen 2: Stochastic oscillator activates correctly

### AC3.1 — Uptrend + oversold %K → screen activated

**Given** a candle series where the last few closes are near the period low (deep pullback)  
  and trend is `"UP"`  
**When** `_screen2_oscillator()` is called  
**Then** the screen returns `True` (buy setup confirmed)

### AC3.2 — Uptrend + %K above oversold threshold → screen not activated

**Given** a candle series where %K is above `stoch_oversold`  
  and trend is `"UP"`  
**When** `_screen2_oscillator()` is called  
**Then** the screen returns `False`

### AC3.3 — Downtrend + overbought %K → screen activated

**Given** a candle series where the last few closes are near the period high (rally in downtrend)  
  and trend is `"DOWN"`  
**When** `_screen2_oscillator()` is called  
**Then** the screen returns `True` (short setup confirmed)

### AC3.4 — Downtrend + %K below overbought threshold → screen not activated

**Given** a candle series where %K is below `stoch_overbought`  
  and trend is `"DOWN"`  
**When** `_screen2_oscillator()` is called  
**Then** the screen returns `False`

**Testable:** `pytest tests/test_triple_screen_strategy.py -k "screen2"`

---

## AC4 — Screen 3: Breakout trigger fires correctly

### AC4.1 — Uptrend + breakout above prior high → LONG trigger

**Given** trend is `"UP"`, Screen 2 activated, and `current_price > kline_data[-2]["high"]`  
**When** `_screen3_trigger()` is called  
**Then** the return value is `True`

### AC4.2 — Uptrend + no breakout → no trigger

**Given** trend is `"UP"`, Screen 2 activated, but `current_price <= kline_data[-2]["high"]`  
**When** `_screen3_trigger()` is called  
**Then** the return value is `False`

### AC4.3 — Downtrend + breakout below prior low → SHORT trigger

**Given** trend is `"DOWN"`, Screen 2 activated, and `current_price < kline_data[-2]["low"]`  
**When** `_screen3_trigger()` is called  
**Then** the return value is `True`

### AC4.4 — Downtrend + no breakout → no trigger

**Given** trend is `"DOWN"`, Screen 2 activated, but `current_price >= kline_data[-2]["low"]`  
**When** `_screen3_trigger()` is called  
**Then** the return value is `False`

**Testable:** `pytest tests/test_triple_screen_strategy.py -k "screen3"`

---

## AC5 — Combined: `generate_signal` produces correct decisions

### AC5.1 — All three screens pass (uptrend) → LONG signal

**Given** deterministic kline_data where:
- EMA slope is rising (Screen 1 = UP)
- Stochastic %K < `stoch_oversold` (Screen 2 activated)
- `current_price > kline_data[-2]["high"]` (Screen 3 triggered)

**When** `strategy.generate_signal(kline_data, "BTC-USD", "4h", current_price)` is called  
**Then**:
- Return value is a `TradingSignal` (not None)
- `signal.decision == "LONG"`
- `signal.confidence` is in `[0.1, 1.0]`
- `signal.entry_price == current_price`
- `signal.stop_loss == current_price * (1 - stop_loss_pct)` (within float tolerance)
- `signal.take_profit == current_price * (1 + take_profit_pct)` (within float tolerance)
- `signal.reasoning` contains "Screen" (screens are named in the reasoning string)

### AC5.2 — All three screens pass (downtrend) → SHORT signal

**Given** deterministic kline_data where:
- EMA slope is falling (Screen 1 = DOWN)
- Stochastic %K > `stoch_overbought` (Screen 2 activated)
- `current_price < kline_data[-2]["low"]` (Screen 3 triggered)

**When** `generate_signal()` is called  
**Then**:
- `signal.decision == "SHORT"`
- `signal.stop_loss == current_price * (1 + stop_loss_pct)`
- `signal.take_profit == current_price * (1 - take_profit_pct)`

### AC5.3 — Screen 1 passes but Screen 2 does not activate → `None`

**Given** EMA slope rising (UP) but stochastic %K is well above `stoch_oversold`  
**When** `generate_signal()` is called  
**Then** return value is `None`

### AC5.4 — Screens 1 and 2 pass but Screen 3 does not fire → `None`

**Given** EMA slope rising (UP), %K below `stoch_oversold`, but `current_price <= kline_data[-2]["high"]`  
**When** `generate_signal()` is called  
**Then** return value is `None`

### AC5.5 — Fewer than min candles → `None` without exception

**Given** `len(kline_data) < weekly_bars * (trend_ema_period + 1) + stoch_k_period + stoch_d_period`  
**When** `generate_signal()` is called  
**Then** return value is `None` and no exception is raised

**Testable:** `pytest tests/test_triple_screen_strategy.py -k "generate_signal"`

---

## AC6 — Non-HOLD signals are generated in a realistic market scenario

**Given** a synthetic 200-candle 4h dataset designed with a clear trend + pullback + breakout pattern  
**When** `generate_signal()` is called on the last candle  
**Then** the return value is a `TradingSignal` (not `None`), proving the strategy is not permanently stuck in HOLD

**Testable:** `pytest tests/test_triple_screen_strategy.py::test_generates_signal_in_realistic_scenario`

---

## AC7 — Reference backtest completes without crash and PnL is calculated

**Given** a test environment with:
- PostgreSQL or SQLite database configured
- Market data available for BTCUSD, timeframe 4h, 2024-01-01 to 2024-03-31

**When** the reference backtest from `QuantAgent-vna-IM` is run:
```python
strategy = TripleScreenStrategy()
backtest = Backtest(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    assets=["BTC-USD"],
    timeframe="4h",
    initial_capital=100_000.0,
    strategy=strategy,
)
metrics = backtest.run(name="QuantAgent-vna-reference")
```

**Then**:
- `backtest.run()` completes without raising any exception
- `metrics.total_pnl` is a finite float (positive, negative, or zero)
- `metrics.total_trades >= 0`
- The run is logged with a name containing "QuantAgent-vna"

**Testable:** Manual integration test or `pytest tests/test_triple_screen_strategy.py::test_backtest_integration -m integration`

---

## AC8 — `should_reevaluate` always returns False

**Given** any `ActivePosition` and `current_price`  
**When** `strategy.should_reevaluate(position, current_price)` is called  
**Then** the return value is `False`

**Testable:** `pytest tests/test_triple_screen_strategy.py::test_should_reevaluate_false`

---

## Non-goals (explicitly excluded)

- Performance targets (win rate, Sharpe ratio, drawdown)
- Multi-symbol parallel backtesting validation
- Provider-specific behaviour or LLM integration
- Comparison against RSI or LLMAgentStrategy results
