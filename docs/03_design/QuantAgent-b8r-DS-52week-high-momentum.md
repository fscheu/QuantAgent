# QuantAgent-b8r — Design: M1 Strategy 3 — 52-Week High Momentum / Breakout

**Issue:** QuantAgent-b8r  
**Parent:** QuantAgent-l0h (AC3)  
**Type:** Design  
**Created:** 2026-05-05  

---

## Design Summary

Implement `FiftyTwoWeekHighStrategy` as a single Python class in `quantagent/strategy/fifty_two_week_high_strategy.py`. It subclasses `TradingStrategy` (no new abstractions). The strategy uses daily OHLCV data from yfinance (already supported by `DataProvider`), computes a rolling 52-week high, and generates LONG signals when price breaks above that level with trend and volume confirmation.

---

## Architecture

### Integration point

```
quantagent/strategy/
├── base.py                              (unchanged)
├── rsi_strategy.py                      (unchanged)
├── llm_agent_strategy.py               (unchanged)
├── triple_screen_strategy.py           (added by QuantAgent-vna)
├── assembler.py                        (unchanged)
└── fifty_two_week_high_strategy.py     ← NEW
```

`quantagent/strategy/__init__.py` → add `FiftyTwoWeekHighStrategy` export.

No changes to `backtest.py`, `assembler.py`, `base.py`, or any shared module. The strategy is instantiated by the caller and passed as the `strategy=` argument to `Backtest`.

---

## Class Skeleton

```python
class FiftyTwoWeekHighStrategy(TradingStrategy):
    def __init__(
        self,
        lookback_days: int = 252,
        proximity_threshold: float = 0.98,
        trend_ma_period: int = 50,
        volume_ma_period: int = 20,
        volume_factor: float = 1.5,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        trailing_stop_pct: float = 0.08,
    ): ...

    def generate_signal(self, kline_data, symbol, timeframe, current_price) -> Optional[TradingSignal]: ...

    def should_reevaluate(self, position, current_price) -> bool: ...

    def get_default_exit_policy(self) -> ExitPolicy: ...

    # Private helpers
    def _compute_52w_high(self, highs: pd.Series) -> float: ...
    def _compute_sma(self, series: pd.Series, period: int) -> pd.Series: ...
    def _confidence(self, current_price: float, high_52w: float) -> float: ...
```

---

## Algorithm Detail

### 1. Data ingestion

Convert `kline_data` (list of OHLCV dicts) to a DataFrame. Required columns: `open`, `high`, `low`, `close`, `volume`.

### 2. Minimum candle guard

```python
min_candles = self.lookback_days + max(self.trend_ma_period, self.volume_ma_period) + 1
if len(df) < min_candles:
    return None
```

With defaults: `252 + 50 + 1 = 303` candles.

### 3. 52-week high

```python
# Exclude the most recent candle (in-progress); use completed candles only
high_52w = df["high"].iloc[-(self.lookback_days + 1):-1].max()
```

This gives the rolling max of the `lookback_days` completed candles before the current bar, matching the George & Hwang formulation where the signal is computed at the open of the current bar.

### 4. Trend filter

```python
sma_close = df["close"].rolling(self.trend_ma_period).mean()
trend_ok = current_price > sma_close.iloc[-1]
```

If `sma_close.iloc[-1]` is NaN → return `None`.

### 5. Volume filter

```python
vol_ma = df["volume"].rolling(self.volume_ma_period).mean()
vol_ok = df["volume"].iloc[-1] > self.volume_factor * vol_ma.iloc[-1]
```

Guard: `vol_ma.iloc[-1]` denominator with `+ 1e-10` to avoid zero division.

### 6. Breakout condition

```python
breakout = current_price > high_52w
```

### 7. Combined entry

```python
if not (breakout and trend_ok and vol_ok):
    return None
```

### 8. Confidence

```python
raw = (current_price - high_52w) / (high_52w + 1e-10)
confidence = max(0.1, min(1.0, raw * 10))
# Semantics: 1% above 52w high → 0.1, 10%+ above → 1.0
```

### 9. Signal construction

```python
return TradingSignal(
    decision="LONG",
    confidence=confidence,
    entry_price=current_price,
    stop_loss=current_price * (1 - self.stop_loss_pct),
    take_profit=current_price * (1 + self.take_profit_pct),
    reasoning=(
        f"52w-high breakout: price={current_price:.2f} > high_52w={high_52w:.2f} "
        f"(+{raw*100:.1f}%), trend_ok={trend_ok}, vol_ok={vol_ok}"
    ),
    exit_policy=ExitPolicy.TRAILING_STOP,
    trailing_stop_pct=self.trailing_stop_pct,
)
```

### 10. Exit policy

Default: `ExitPolicy.TRAILING_STOP` via base class `_check_trailing_stop`. No custom `should_exit` override for M1.

---

## Parameter Defaults and Rationale

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `lookback_days` | 252 | Canonical 52-week window (252 US equity trading days) |
| `proximity_threshold` | 0.98 | Documented but unused in M1 breakout-only mode; reserved for future proximity signal |
| `trend_ma_period` | 50 | Standard medium-term trend filter for daily equity data |
| `volume_ma_period` | 20 | ≈1 month of trading days; standard volume normalization window |
| `volume_factor` | 1.5 | Moderate volume confirmation; filters low-conviction breakouts |
| `stop_loss_pct` | 0.05 | 5% SL appropriate for daily equity volatility (vs 2% for crypto) |
| `take_profit_pct` | 0.15 | 3:1 R/R with default SL; momentum trades are expected to run |
| `trailing_stop_pct` | 0.08 | 8% trailing stop allows equity momentum to run before cutting |

---

## Data Requirements

- **Timeframe:** `"1d"` (daily)
- **Assets:** US equities (e.g., `"AAPL"`, `"MSFT"`, `"SPY"`) via yfinance / `DataProvider`
- **Minimum candles:** 303 (with defaults); the backtest engine will not generate signals during the warmup period

---

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `quantagent/strategy/fifty_two_week_high_strategy.py` | CREATE | Main strategy class |
| `quantagent/strategy/__init__.py` | UPDATE | Add `FiftyTwoWeekHighStrategy` to exports |
| `tests/test_fifty_two_week_high_strategy.py` | CREATE | Unit + integration tests |

No other files require modification.

---

## Reference Backtest Profile (M1)

```python
from datetime import datetime
from quantagent.backtesting.backtest import Backtest
from quantagent.strategy.fifty_two_week_high_strategy import FiftyTwoWeekHighStrategy

strategy = FiftyTwoWeekHighStrategy(
    lookback_days=252,
    proximity_threshold=0.98,
    trend_ma_period=50,
    volume_ma_period=20,
    volume_factor=1.5,
    stop_loss_pct=0.05,
    take_profit_pct=0.15,
    trailing_stop_pct=0.08,
)

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

**Expected outcomes (not performance targets):**
- `metrics.total_pnl` is a finite float (any sign)
- `metrics.total_trades >= 0`
- No exceptions raised during the run
- The strategy generates at least one signal on a 2-year daily dataset of AAPL

**Note on warmup:** The first 303 bars will yield `None` signals. With `start_date=2022-01-01`, the first eligible signal date is roughly late 2022 / early 2023 (after 303 trading days). The backtest should be run on a date range long enough to include a live trading period after warmup.

---

## Risk / Open Issues

| Risk | Mitigation |
|------|-----------|
| `lookback_days` candles not available at backtest start | Minimum candle guard returns `None`; warmup period accepted |
| 52w high defined on `high` vs `close` | Using `high` column (per George & Hwang definition and common practice) |
| Volume data may be zero or NaN (pre-market / half days) | Guard with `+ 1e-10` and NaN-safe rolling mean |
| Breakout rarely fires for low-volatility equities | Expected for M1; fewer signals is acceptable |
| `proximity_threshold` parameter exists but unused in M1 breakout-only mode | Documented as reserved; implementer should not wire it up unless explicitly requested |
