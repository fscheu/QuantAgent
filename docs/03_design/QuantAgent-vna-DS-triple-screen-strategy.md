# QuantAgent-vna — Design: M1 Strategy 1 — Triple Screen Strategy (Alexander Elder)

**Issue:** QuantAgent-vna  
**Parent:** QuantAgent-l0h (AC3)  
**Type:** Design  
**Created:** 2026-05-05  

---

## Design Summary

Implement `TripleScreenStrategy` as a single Python class in `quantagent/strategy/triple_screen_strategy.py`. It subclasses `TradingStrategy` (no new abstractions) and synthesises multi-timeframe behaviour from a single `kline_data` list by aggregating bars.

---

## Architecture

### Integration point

```
quantagent/strategy/
├── base.py                     (unchanged)
├── rsi_strategy.py             (unchanged)
├── llm_agent_strategy.py       (unchanged)
├── assembler.py                (unchanged)
└── triple_screen_strategy.py   ← NEW
```

No changes to `backtest.py` or any shared module. The strategy is instantiated by the caller and passed as the `strategy=` argument to `Backtest` (already supported by the existing engine).

### Class skeleton

```python
class TripleScreenStrategy(TradingStrategy):
    def __init__(self, weekly_bars=5, trend_ema_period=13,
                 stoch_k_period=5, stoch_d_period=3,
                 stoch_oversold=20.0, stoch_overbought=80.0,
                 stop_loss_pct=0.02, take_profit_pct=0.04,
                 trailing_stop_pct=0.05): ...

    def generate_signal(self, kline_data, symbol, timeframe, current_price): ...

    def should_reevaluate(self, position, current_price) -> bool: ...

    # Private helpers
    def _aggregate_weekly_bars(self, kline_data) -> pd.DataFrame: ...
    def _ema(self, series: pd.Series, period: int) -> pd.Series: ...
    def _stochastic(self, df: pd.DataFrame) -> tuple[float, float]: ...
    def _screen1_trend(self, weekly_df: pd.DataFrame) -> str | None: ...
    def _screen2_oscillator(self, df: pd.DataFrame, trend: str) -> bool: ...
    def _screen3_trigger(self, kline_data, trend: str, current_price: float) -> bool: ...
    def _confidence(self, stoch_k: float, trend: str) -> float: ...
```

---

## Algorithm Detail

### Screen 1 — Trend filter (weekly bars)

**Purpose:** Identify dominant trend from a higher-TF perspective.

**Steps:**
1. Take `kline_data` as a list of OHLCV dicts.
2. Drop any trailing incomplete block (`len % weekly_bars` remainder).
3. Reshape into groups of `weekly_bars` candles; each group becomes one synthetic bar:
   - `open` = group[0]["open"]
   - `high` = max of all highs in group
   - `low` = min of all lows in group
   - `close` = group[-1]["close"]
   - `volume` = sum of group volumes
4. Compute EMA(`trend_ema_period`) on the synthetic bars' close prices using standard exponential smoothing (alpha = 2/(period+1)).
5. Slope = `ema[-1] - ema[-2]`.
6. Return `"UP"` if slope > 0, `"DOWN"` if slope ≤ 0.
7. Return `None` if there are fewer than `trend_ema_period + 1` synthetic bars.

**Rationale for EMA-slope trend (not MACD):** Single-indicator simplicity for M1; EMA slope on weekly bars is the minimal viable Screen 1 described by Elder himself as the most common replacement when MACD isn't available.

### Screen 2 — Stochastic oscillator (intermediate TF)

**Purpose:** Detect that price is in a pullback (uptrend) or a rally (downtrend).

**Steps:**
1. On the full `kline_data` DataFrame, compute Stochastic %K and %D:
   - `lowest_low = rolling_min(low, stoch_k_period)`
   - `highest_high = rolling_max(high, stoch_k_period)`
   - `%K = 100 * (close - lowest_low) / (highest_high - lowest_low + ε)`
   - `%D = SMA(%K, stoch_d_period)`
2. Read the last completed values `%K[-1]` and `%D[-1]`.
3. Activation rules:
   - Uptrend: activated if `%K[-1] <= stoch_oversold`
   - Downtrend: activated if `%K[-1] >= stoch_overbought`
4. Return the `%K[-1]` value alongside a boolean for downstream use.

**Rationale for Stochastic (not Force Index):** Stochastic is pure price-based, doesn't require reliable volume data, and is the more common Screen 2 choice in Elder's published examples. Force Index requires volume that may be noisy in crypto data.

### Screen 3 — Entry trigger (breakout)

**Purpose:** Precise entry timing after trend + oscillator confirmed.

**Steps (uptrend):**
- `trigger_high = kline_data[-2]["high"]` (prior completed candle)
- Signal LONG if `current_price > trigger_high`

**Steps (downtrend):**
- `trigger_low = kline_data[-2]["low"]` (prior completed candle)
- Signal SHORT if `current_price < trigger_low`

**Rationale:** This is Elder's original Screen 3: a buy-stop above the prior day's high / sell-stop below the prior day's low, adapted to intrabar current price.

### Confidence calculation

```python
if trend == "UP":
    # Deeper oversold → higher confidence
    confidence = (stoch_oversold - stoch_k) / stoch_oversold
else:
    # Deeper overbought → higher confidence
    confidence = (stoch_k - stoch_overbought) / (100 - stoch_overbought)

confidence = max(0.1, min(1.0, confidence))
```

### Exit policy

Default: `ExitPolicy.TRAILING_STOP` (same as RSI strategy). No custom `should_exit` override needed for M1.

---

## Parameter Defaults and Rationale

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `weekly_bars` | 5 | Maps roughly to 5 daily bars per week; on 4h data ≈ 30 bars |
| `trend_ema_period` | 13 | Elder's standard weekly EMA for Screen 1 |
| `stoch_k_period` | 5 | Elder's recommended fast stochastic for Screen 2 |
| `stoch_d_period` | 3 | Standard 3-bar smoothing |
| `stoch_oversold` | 20.0 | Standard oversold threshold |
| `stoch_overbought` | 80.0 | Standard overbought threshold |
| `stop_loss_pct` | 0.02 | 2% fixed SL, conservative for crypto volatility |
| `take_profit_pct` | 0.04 | 2:1 R/R ratio with default SL |
| `trailing_stop_pct` | 0.05 | Matches RSI strategy default |

---

## Data Requirements

Minimum input candles:
```
min_candles = weekly_bars * (trend_ema_period + 1) + stoch_k_period + stoch_d_period
            = 5 * 14 + 5 + 3 = 78 candles (with defaults)
```

The implementation should guard on `len(kline_data) < min_candles` and return `None`.

---

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `quantagent/strategy/triple_screen_strategy.py` | CREATE | Main strategy class |
| `tests/test_triple_screen_strategy.py` | CREATE | Unit + integration tests |

No other files require modification.

---

## Reference Backtest Profile (M1)

```python
from datetime import datetime
from quantagent.backtesting.backtest import Backtest
from quantagent.strategy.triple_screen_strategy import TripleScreenStrategy

strategy = TripleScreenStrategy(
    weekly_bars=5,
    trend_ema_period=13,
    stoch_k_period=5,
    stoch_d_period=3,
    stoch_oversold=20.0,
    stoch_overbought=80.0,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    trailing_stop_pct=0.05,
)

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

**Expected outcomes (not performance targets):**
- `metrics.total_pnl` is a finite float (any sign)
- `metrics.total_trades >= 0`
- No exceptions raised during the run
- At least one non-HOLD signal generated when backtesting 3 months of 4h BTC data

---

## Risk / Open Issues

| Risk | Mitigation |
|------|-----------|
| Bar aggregation when `len(kline_data) % weekly_bars != 0` drops the trailing remainder silently | Documented; acceptable for M1. Add assertion in debug logging. |
| Stochastic division by zero when `highest_high == lowest_low` (flat market) | Guard with `+ ε` (1e-10) |
| Screen 3 may never fire if trend + oscillator are only briefly aligned | Acceptable for M1; strategy may generate few signals on low-volatility assets |
| Backtest engine timeframe support | Implementer must verify the `Backtest` class accepts the `strategy=` kwarg with the chosen timeframe |
