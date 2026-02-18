# QuantAgent-vna — Planning — Triple Screen Strategy (Alexander Elder)

Links:
- Requirements: `../01_requirements/QuantAgent-vna-RQ-triple-screen-strategy.md`
- Design: `../03_design/QuantAgent-vna-DS-triple-screen-strategy.md`
- Acceptance: `../05_acceptance_tests/QuantAgent-vna-AC-triple-screen-strategy.md`

## Level of Detail
STANDARD.

## Implementation Plan (task breakdown)

### 1) Repo scan + confirm conventions (0.5h)
- Review existing strategies under `quantagent/strategy/`.
- Confirm timeframe string conventions and candle schema (timestamp field name/type).

### 2) Implement `TripleScreenStrategy` skeleton (1h)
- Create `quantagent/strategy/triple_screen_strategy.py`.
- Implement `__init__` parameters (timeframes + indicators + exits).
- Implement `should_reevaluate(...)` (expected: `False`).

### 3) Implement multi-timeframe derivation via resampling (1–2h)
- Convert base candles into a DataFrame with a datetime index.
- Add minimal timeframe-to-resample-rule mapping (e.g., `1h -> '1H'`, `4h -> '4H'`, `1d -> '1D'`, `1w -> '1W'`).
- Compute Screen 1 and Screen 2 OHLCV frames.
- Guard: if any required frame lacks enough rows → HOLD.

### 4) Implement indicators + gating logic (1–2h)
- MACD implementation on Screen 1 closes.
- RSI implementation on Screen 2 closes (re-use or replicate minimal RSI logic consistent with `RSIMeanReversionStrategy`).
- Breakout logic on Screen 3 (rolling high/low).
- Return `TradingSignal` with exit params when all gates pass.

### 5) Backtest integration smoke (0.5–1h)
- Run two backtests with same inputs:
  - default LLM
  - Triple Screen
- Ensure both complete and persist.

### 6) Tests (1–2h)
Create `tests/test_triple_screen_strategy.py` covering:
- insufficient data → HOLD
- bullish trend + oversold correction + breakout → LONG
- bearish trend + overbought correction + breakdown → SHORT
- correction gate blocks
- breakout gate blocks

### 7) Docs / example usage (0.5–1h)
- Ensure parameters are documented in docstrings and/or in a short section within the new strategy module.
- Optionally add a minimal example snippet in an existing example script (only if needed for discoverability).

## Dependencies / Risks
- Timeframe parsing/resampling: ensure timestamps are timezone-safe and sorted.
- Backtest loop provides enough lookback candles per step; if not, strategy will HOLD frequently.

## Validation Checklist
- `pytest -q` passes.
- Comparative backtest can run on a small date range (e.g., 10 days) for quick iteration.
- Docs files are linked from the relevant `docs/*/README.md` indexes.
