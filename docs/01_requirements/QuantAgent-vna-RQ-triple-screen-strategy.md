# QuantAgent-vna — Requirements: M1 Strategy 1 — Triple Screen Strategy (Alexander Elder)

**Issue:** QuantAgent-vna  
**Parent:** QuantAgent-l0h (AC3)  
**Type:** Requirements  
**Created:** 2026-05-05  

---

## Objective

Implement the Triple Screen trading system by Alexander Elder as M1 Strategy 1. The strategy must integrate cleanly with the existing `TradingStrategy` abstraction and be exercisable by the current `Backtest` engine with no changes to shared infrastructure.

---

## Scope In

- New file `quantagent/strategy/triple_screen_strategy.py` implementing `TripleScreenStrategy(TradingStrategy)`
- Unit tests covering all three screens independently and their combined signal logic
- Reference backtest profile in the implementation doc (IM)
- Entry in `quantagent/strategy/__init__.py` (or equivalent export)

## Scope Out

- Changes to `backtest.py`, `assembler.py`, `base.py`, or any other shared module
- Multi-timeframe live data fetching (the strategy simulates multi-TF via bar aggregation)
- Parameter optimization per asset
- Comparison against other strategies

---

## Background — Triple Screen System

Triple Screen (Elder, 1986) filters entries through three sequential screens to reduce false signals:

| Screen | Purpose | Classic indicators |
|--------|---------|-------------------|
| 1 | Dominant trend on a higher timeframe | MACD histogram slope, EMA slope |
| 2 | Pullback/rally oscillator on intermediate TF | Stochastic, Force Index |
| 3 | Entry trigger on lowest TF | Breakout above/below prior bar high/low |

Only trades aligned with Screen 1's trend direction that also pass Screen 2's oscillator filter and Screen 3's trigger are executed.

---

## Functional Requirements

### FR1 — Strategy class
A `TripleScreenStrategy` class must exist in `quantagent/strategy/triple_screen_strategy.py` and implement the full `TradingStrategy` ABC (`generate_signal`, `should_reevaluate`).

### FR2 — Configurable parameters
The constructor must accept at minimum:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weekly_bars` | 5 | Number of input candles to aggregate into one "higher-TF bar" for Screen 1 |
| `trend_ema_period` | 13 | EMA period computed on the aggregated (weekly) bars |
| `stoch_k_period` | 5 | Stochastic %K lookback on the intermediate-TF candles |
| `stoch_d_period` | 3 | Signal line smoothing for %D |
| `stoch_oversold` | 20.0 | %K threshold below which a pullback is confirmed (uptrend entries) |
| `stoch_overbought` | 80.0 | %K threshold above which a rally is confirmed (downtrend entries) |
| `stop_loss_pct` | 0.02 | Fixed stop loss as a fraction of entry price |
| `take_profit_pct` | 0.04 | Fixed take profit as a fraction of entry price |
| `trailing_stop_pct` | 0.05 | Trailing stop fraction passed through to `TradingSignal` |

### FR3 — Screen 1: weekly trend filter
`generate_signal()` must aggregate the input `kline_data` into blocks of `weekly_bars` candles to form synthetic higher-TF bars (OHLCV aggregation: open=first, high=max, low=min, close=last, volume=sum). It then computes an EMA of `trend_ema_period` on the synthetic bars' close prices and determines trend direction from the slope of the last two EMA values:

- Slope > 0 → uptrend
- Slope < 0 → downtrend
- Insufficient data → return `None` (no signal)

### FR4 — Screen 2: stochastic pullback/rally filter
On the original input candles (not aggregated), compute Stochastic %K and %D over `stoch_k_period` / `stoch_d_period`:

- In uptrend: the screen is "activated" when the most recent %K is below `stoch_oversold`
- In downtrend: the screen is "activated" when the most recent %K is above `stoch_overbought`
- If the screen is not activated, return `None`

### FR5 — Screen 3: breakout entry trigger
If Screens 1 and 2 both pass, use the most recent completed candle as the trigger:

- Uptrend + oversold %K: signal `LONG` when `current_price > kline_data[-2]["high"]` (breakout above prior bar high)
- Downtrend + overbought %K: signal `SHORT` when `current_price < kline_data[-2]["low"]` (breakout below prior bar low)
- If the price has not broken out, return `None` (wait for trigger)

### FR6 — Minimum data guard
`generate_signal()` must return `None` without raising when `len(kline_data)` is less than the minimum required:

```
min_candles = weekly_bars * trend_ema_period + stoch_k_period + stoch_d_period
```

With defaults that is `5 × 13 + 5 + 3 = 73` candles.

### FR7 — Signal construction
When all three screens pass, produce a `TradingSignal` with:

- `decision`: "LONG" or "SHORT"
- `confidence`: normalised value derived from how extreme %K is (deeper oversold/overbought → higher confidence), clamped to [0.1, 1.0]
- `entry_price`: `current_price`
- `stop_loss`: entry ± `stop_loss_pct` (long: below, short: above)
- `take_profit`: entry ± `take_profit_pct` (long: above, short: below)
- `reasoning`: human-readable string naming active screens and indicator values
- `exit_policy`: `ExitPolicy.TRAILING_STOP`
- `trailing_stop_pct`: `trailing_stop_pct`

### FR8 — `should_reevaluate` behaviour
Return `False`; the strategy does not re-evaluate once a position is open (same contract as `RSIMeanReversionStrategy`).

---

## Edge Cases

- Fewer candles than `min_candles` → return `None`, no exception
- EMA slope exactly 0 → treat as downtrend (neutral → skip to avoid whipsaws)
- %K equal to threshold boundary → inclusive comparison (`<=` for oversold, `>=` for overbought)
- Aggregation remainder (candles not divisible by `weekly_bars`) → ignore the tail, use only complete blocks
- `kline_data[-2]` access: validated via the minimum candle guard (always safe when guard passes)

---

## Definition of Done

- `TripleScreenStrategy` class exists and all three screen methods are independently testable
- Default parameters chosen are documented here (above)
- Unit tests in `tests/test_triple_screen_strategy.py` exercise each screen's pass/fail independently and the combined logic
- A reference backtest profile is documented in the IM doc and a run confirms PnL is calculated and no crashes occur
- `docs/05_acceptance_tests/QuantAgent-vna-AC-triple-screen-strategy.md` is complete and aligned
