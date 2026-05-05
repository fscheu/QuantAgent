# QuantAgent-b8r — Requirements: M1 Strategy 3 — 52-Week High Momentum / Breakout

**Issue:** QuantAgent-b8r  
**Parent:** QuantAgent-l0h (AC3)  
**Type:** Requirements  
**Created:** 2026-05-05  

---

## Objective

Implement a 52-week high momentum/breakout strategy for US equities as M1 Strategy 3. The strategy must integrate cleanly with the existing `TradingStrategy` abstraction and be exercisable by the current `Backtest` engine with no changes to shared infrastructure.

**Academic reference:** George and Hwang, "The 52-Week High and Momentum Investing" (Journal of Finance, 2004). Core finding: stocks near their 52-week high exhibit positive momentum due to investor anchoring, generating persistent outperformance on breakout confirmation.

---

## Scope In

- New file `quantagent/strategy/fifty_two_week_high_strategy.py` implementing `FiftyTwoWeekHighStrategy(TradingStrategy)`
- Unit tests covering 52-week high calculation, breakout/proximity signal logic, filter conditions, and exit rules
- Reference backtest profile documented in the implementation doc (IM)
- Entry in `quantagent/strategy/__init__.py` exports

## Scope Out

- Changes to `backtest.py`, `assembler.py`, `base.py`, or any other shared module
- Short-selling or short-biased signals (long-only in M1; see FR12 for rationale)
- PEAD or earnings surprise datasets
- Intraday strategies or microstructure-dependent signals
- Sophisticated borrow constraints or short-selling mechanics
- Advanced universe selection or quantitative optimization

---

## Background — 52-Week High Momentum

George and Hwang (2004) documented that stocks whose prices are near their 52-week high tend to generate subsequent positive abnormal returns. The mechanism is investor anchoring: market participants treat the 52-week high as a salient reference point and are reluctant to push prices above it. Once price does break through, under-reaction resolves and momentum accelerates.

Key properties for an M1 implementation:
- Signal is price-only; no earnings, fundamentals, or alternative data required
- Works with standard OHLCV daily data available from yfinance
- Long-biased (the anomaly is documented on the long side)
- Complements Triple Screen (trend-following on higher TF) and LLMAgentStrategy (multi-agent pipeline) by providing a distinct momentum edge

---

## Functional Requirements

### FR1 — Strategy class

A `FiftyTwoWeekHighStrategy` class must exist in `quantagent/strategy/fifty_two_week_high_strategy.py` and implement the full `TradingStrategy` ABC (`generate_signal`, `should_reevaluate`).

### FR2 — Configurable parameters

The constructor must accept at minimum:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_days` | 252 | Rolling window (in candles) for the 52-week high calculation |
| `proximity_threshold` | 0.98 | Ratio `close / 52w_high` above which price is considered "near" the high |
| `trend_ma_period` | 50 | SMA period for trend filter |
| `volume_ma_period` | 20 | Rolling period for average volume |
| `volume_factor` | 1.5 | Minimum multiplier: breakout candle volume vs. average |
| `stop_loss_pct` | 0.05 | Fixed stop loss as a fraction of entry price |
| `take_profit_pct` | 0.15 | Fixed take profit (3:1 R/R with default SL) |
| `trailing_stop_pct` | 0.08 | Trailing stop fraction passed to `TradingSignal` |

### FR3 — 52-week high calculation

`generate_signal()` must compute a rolling maximum of the `high` column over the last `lookback_days` candles (excluding the current candle). This value is `high_52w`. The proximity ratio is defined as:

```
ratio = current_price / high_52w
```

### FR4 — Entry signal: breakout above 52-week high

A LONG signal is generated when **all** of the following hold:

1. `current_price > high_52w` — price breaks above the 52-week high
2. Trend filter passes: `current_price > sma_trend[-1]` (price above trend MA)
3. Volume filter passes: `volume[-1] > volume_factor × volume_ma[-1]` (above-average volume confirms the breakout)

If not all three conditions are met, `generate_signal()` returns `None`.

### FR5 — Proximity mode (alternative entry)

When `current_price > high_52w` is not yet satisfied but `ratio >= proximity_threshold`, the strategy is in a "setup" state. In M1 the strategy **does not enter on proximity alone** — breakout is required. This simplification is documented here to make the decision explicit and to allow a future implementer to add it without ambiguity.

### FR6 — Long-only constraint

The strategy generates only LONG decisions. It does not generate SHORT signals. See FR12 for justification.

### FR7 — Minimum data guard

`generate_signal()` must return `None` without raising when `len(kline_data)` is less than:

```
min_candles = lookback_days + max(trend_ma_period, volume_ma_period) + 1
```

With defaults: `252 + 50 + 1 = 303` candles.

### FR8 — Signal construction

When all entry conditions pass, produce a `TradingSignal` with:

- `decision`: `"LONG"`
- `confidence`: derived from how far above the 52-week high the breakout is, normalised to `[0.1, 1.0]`:
  ```
  raw = (current_price - high_52w) / high_52w
  confidence = max(0.1, min(1.0, raw * 10))  # 10x scale: 1% above = 0.1, 10% above = 1.0
  ```
- `entry_price`: `current_price`
- `stop_loss`: `current_price * (1 - stop_loss_pct)` (below entry)
- `take_profit`: `current_price * (1 + take_profit_pct)` (above entry)
- `reasoning`: human-readable string naming the 52w high, proximity ratio, and active filters
- `exit_policy`: `ExitPolicy.TRAILING_STOP`
- `trailing_stop_pct`: `trailing_stop_pct`

### FR9 — `should_reevaluate` behaviour

Return `False`; the strategy does not re-evaluate once a position is open (same contract as `RSIMeanReversionStrategy`).

### FR10 — Trend MA calculation

SMA computed on the `close` column over `trend_ma_period` candles using a standard rolling mean. Price above SMA → uptrend filter active.

### FR11 — Volume MA calculation

SMA computed on the `volume` column over `volume_ma_period` candles. The comparison uses the most recent completed candle's volume against this average.

### FR12 — Long-only rationale (M1)

The George & Hwang (2004) anomaly is documented on the **long side** only. Short-selling based on distance from 52-week lows is a different signal (54-week low momentum), requires borrow availability, and would expand scope beyond the referenced paper. M1 stays long-only.

---

## Edge Cases

- `high_52w` equals `current_price` exactly → `ratio == 1.0`, breakout condition `current_price > high_52w` is `False`, no signal
- Volume MA is zero (flat volume) → guard with `+ ε` (1e-10) in denominator
- Trend MA is flat or zero → guard with `+ ε`; if `sma_trend[-1]` is NaN (insufficient data for MA), return `None`
- Fewer candles than `min_candles` → return `None`, no exception
- NaN values in price/volume data → `pd.DataFrame` operations propagate NaN; final conditions evaluate to `False`, no signal

---

## Definition of Done

- `FiftyTwoWeekHighStrategy` class exists and all signal logic is independently testable
- Default parameters are documented here (above)
- Unit tests in `tests/test_fifty_two_week_high_strategy.py` exercise 52w high calculation, each filter, and combined entry logic
- A reference backtest profile is documented in the IM doc and a run confirms PnL is calculated with no crashes
- `docs/05_acceptance_tests/QuantAgent-b8r-AC-52week-high-momentum.md` is complete and aligned
- Strategy is documented as **long-only** (this doc serves as the planner's justification)
