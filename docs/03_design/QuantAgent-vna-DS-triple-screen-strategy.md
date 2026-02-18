# QuantAgent-vna — Design — Triple Screen Strategy (Alexander Elder)

Links:
- Requirements: `../01_requirements/QuantAgent-vna-RQ-triple-screen-strategy.md`

## Level of Detail
STANDARD (new strategy + multi-timeframe logic + backtest comparability).

## Affected Components
- New:
  - `quantagent/strategy/triple_screen_strategy.py`
- Existing (integration point, no architectural change):
  - `quantagent/backtesting/backtest.py` (already supports `strategy: Optional[TradingStrategy]`)
  - `quantagent/strategy/base.py` (`TradingStrategy`, `TradingSignal`)

## Design Goals
- Keep strategy **pure + deterministic**.
- Avoid adding new dependencies (use pandas already present in repo).
- Derive Screen 1/2 timeframes from Screen 3 candle history (resample), to avoid changing data access / caching.

## Strategy Data Flow
Input (from Backtest loop):
- `kline_data`: list[dict] with OHLCV + timestamp for the **base timeframe** (Screen 3)
- `timeframe`: base timeframe string
- `current_price`: last close / current price

Processing:
1. Convert `kline_data` → DataFrame with datetime index.
2. Resample base candles to:
   - Screen 2 timeframe
   - Screen 1 timeframe
3. Compute:
   - Screen 1 MACD → primary trend
   - Screen 2 oscillator → correction/pullback gate
   - Screen 3 breakout → entry timing
4. If all gates pass, return `TradingSignal` with decision + exit params.

## Indicators / Rules (minimal, configurable)
### Screen 1 — MACD trend
- Compute MACD line and signal line from Screen 1 closes.
- Trend rule (default):
  - bullish if `macd > signal`
  - bearish if `macd < signal`
  - neutral/unknown if not enough points

### Screen 2 — Oscillator correction gate
- Default oscillator: RSI on Screen 2 closes.
- Gate rule (default):
  - bullish trend: require RSI <= `oversold_threshold`
  - bearish trend: require RSI >= `overbought_threshold`

### Screen 3 — Breakout trigger
- Use Donchian-style breakout on Screen 3:
  - bullish: close > rolling_high(lookback)
  - bearish: close < rolling_low(lookback)

### Exits
- Use existing `TradingStrategy` template exit behavior:
  - Provide `stop_loss`, `take_profit`, `trailing_stop_pct`, and set `exit_policy=TRAILING_STOP`.
- Minimal exit parameterization:
  - fixed SL/TP % off entry
  - fixed trailing-stop %

## Configuration Surface
Expose params via `TripleScreenStrategy.__init__(...)` (preferred to keep Backtest unchanged):
- `screen_1_timeframe: str`
- `screen_2_timeframe: str`
- `macd_fast: int`, `macd_slow: int`, `macd_signal: int`
- `oscillator: Literal['rsi']` (future extension but keep initial minimal)
- `rsi_period: int`, `oversold_threshold: float`, `overbought_threshold: float`
- `breakout_lookback: int`
- `stop_loss_pct: float`, `take_profit_pct: float`, `trailing_stop_pct: float`

## Timeframe Resampling
- Strategy must interpret timeframe strings supported by the repo’s DataProvider/backtest conventions (e.g., `1h`, `4h`, `1d`).
- Resampling approach:
  - Build `pd.DatetimeIndex` from candle timestamps.
  - Use `df.resample(rule).agg({open:'first', high:'max', low:'min', close:'last', volume:'sum'})`.
- Guardrails:
  - If resampled DF has insufficient rows for MACD/RSI/breakout windows, return HOLD.

## Comparative Backtest Recipe (design)
- LLM strategy: default `Backtest(..., strategy=None)` (uses `LLMAgentStrategy`).
- Triple Screen:
  - `Backtest(..., strategy=TripleScreenStrategy(...))`
- Same inputs:
  - start/end dates, assets, base timeframe, initial capital, risk config.
- Persisted outputs remain identical schema (BacktestRun + metrics), enabling side-by-side comparison by run id/name.

### Example (minimal)
```python
# illustrative only
llm_metrics = Backtest(..., timeframe="1h").run(name="LLM")
triple_metrics = Backtest(..., timeframe="1h", strategy=TripleScreenStrategy(
    screen_1_timeframe="1d",
    screen_2_timeframe="4h",
)).run(name="Triple Screen")
```

## Risks / Notes
- Resampling from base timeframe requires sufficiently long history; backtest window and the per-step lookback must ensure enough candles.
- Timeframe string → pandas resample rule mapping may need a minimal mapping table.
- MACD/RSI implementation must be consistent across tests (avoid external TA libs unless already used elsewhere).
