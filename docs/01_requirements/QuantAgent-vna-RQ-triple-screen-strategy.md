# QuantAgent-vna — Requirements — Triple Screen Strategy (Alexander Elder)

## Objective
Implement a non-LLM trading strategy based on Alexander Elder’s **Triple Screen** method to run **comparative backtests** vs the existing LLM strategy over the **same assets + same period**.

## Scope (in)
- Add a new strategy module:
  - `quantagent/strategy/triple_screen_strategy.py`
- Provide a strategy class implementing `quantagent.strategy.base.TradingStrategy`.
- Strategy uses **3 screens / timeframes**:
  1. **Screen 1 (long-term):** determine primary trend using **MACD** on a higher timeframe.
  2. **Screen 2 (mid-term):** identify counter-trend corrections using a configurable **oscillator**.
  3. **Screen 3 (short-term):** generate entry timing using **breakout** rules and manage exits via trailing stop / SL/TP.
- Backtest integration:
  - Must be runnable via the existing `Backtest(..., strategy=...)` hook.
  - Must support running **LLM vs Triple Screen** backtests with identical parameters (dates/assets/timeframe) for comparison.
- Add a test file skeleton/coverage target:
  - `tests/test_triple_screen_strategy.py`
- Document all configurable parameters (timeframes, indicators, thresholds).

## Scope (out)
- No new UI (Streamlit/Flask) for selecting strategies (unless already trivial and required by existing workflow).
- No broker/live-trading integration changes.
- No portfolio/risk system redesign.

## Functional Requirements
### RQ1 — Strategy interface compatibility
- The new strategy must implement `TradingStrategy.generate_signal(...)` and return `TradingSignal` or `None` (HOLD).
- Must work inside `quantagent.backtesting.backtest.Backtest` without changing how results/metrics are persisted.

### RQ2 — Screen 1: trend filter (MACD)
- Determine trend direction from higher timeframe data using MACD:
  - Trend = **bullish** when MACD line > signal line (configurable rule).
  - Trend = **bearish** when MACD line < signal line.
  - If trend is undefined (insufficient data), strategy must return HOLD.

### RQ3 — Screen 2: correction filter (oscillator)
- On the mid timeframe, detect pullbacks **against** the Screen 1 trend using a configurable oscillator:
  - Default oscillator: RSI (configurable).
  - Bull trend: look for oversold condition.
  - Bear trend: look for overbought condition.
  - If correction condition is not met, strategy must return HOLD.

### RQ4 — Screen 3: entry timing (breakout)
- On the base/backtest timeframe, generate entry signals using a breakout rule consistent with the trend:
  - Bull trend: enter LONG on breakout above a configurable lookback high.
  - Bear trend: enter SHORT on breakdown below a configurable lookback low.
- If breakout condition is not met, strategy must return HOLD.

### RQ5 — Exits
- Strategy must output exit parameters via `TradingSignal`:
  - Stop loss and take profit (fixed % or derived, depending on minimal implementation choice).
  - Trailing stop percentage (default enabled).
- Default exit policy should align with existing `TradingStrategy` template behavior (`ExitPolicy.TRAILING_STOP`).

### RQ6 — Configurable parameters (must be documented)
At minimum:
- Timeframes:
  - `screen_1_timeframe` (e.g., `1w` or `1d`)
  - `screen_2_timeframe` (e.g., `1d` or `4h`)
  - Screen 3 timeframe = backtest timeframe (input `timeframe`)
- MACD parameters: fast/slow/signal periods.
- Oscillator type + parameters (at least RSI period + thresholds).
- Breakout lookback window.
- Risk/exit params: stop-loss %, take-profit %, trailing-stop %.

## Edge Cases / Constraints
- Insufficient candles for any screen → HOLD (no signal).
- Strategy must be deterministic (no network calls, no randomness).
- Multi-timeframe data must be derived from available candle history (no new external fetching inside the strategy).

## Definition of Done
- New strategy module exists and is importable.
- Backtest can run with `strategy=TripleScreenStrategy(...)`.
- A comparative backtest recipe is documented (LLM vs Triple Screen same period).
- Tests exist and cover at least:
  - HOLD when insufficient data
  - Trend filter gating (screen1)
  - Correction filter gating (screen2)
  - Breakout triggering (screen3)

## References
- Alexander Elder, *Trading for a Living* (1993)
