# QuantAgent-vna — Acceptance Criteria — Triple Screen Strategy (Alexander Elder)

Links:
- Requirements: `../01_requirements/QuantAgent-vna-RQ-triple-screen-strategy.md`

## AC1 — Module + class existence
Given the repo is installed (editable or normal)
When importing `quantagent.strategy.triple_screen_strategy`
Then the import succeeds and exposes a `TripleScreenStrategy` implementing `TradingStrategy`.

## AC2 — Determinism / no external calls
Given the strategy is executed during backtests
When running on the same candle inputs twice
Then generated signals (decision + numeric fields) are identical.

## AC3 — HOLD on insufficient data (any screen)
Given the provided base timeframe candle history is insufficient to compute Screen 1 MACD OR Screen 2 oscillator OR Screen 3 breakout window
When `generate_signal(...)` is called
Then it returns `None` (HOLD).

## AC4 — Screen 1 trend gates entries
Given Screen 1 MACD indicates bullish trend
And Screen 2 correction gate is satisfied
And Screen 3 breakout is bullish
When `generate_signal(...)` is called
Then it returns a `TradingSignal` with `decision == 'LONG'`.

Given Screen 1 MACD indicates bearish trend
And Screen 2 correction gate is satisfied
And Screen 3 breakout is bearish
When `generate_signal(...)` is called
Then it returns a `TradingSignal` with `decision == 'SHORT'`.

Given Screen 1 MACD indicates bullish trend
When Screen 3 breakout is bearish (or not bullish)
Then the strategy returns HOLD.

## AC5 — Screen 2 correction gate blocks entries
Given Screen 1 trend is bullish
When Screen 2 oscillator does NOT indicate oversold correction (per configured threshold)
Then the strategy returns HOLD.

Given Screen 1 trend is bearish
When Screen 2 oscillator does NOT indicate overbought correction (per configured threshold)
Then the strategy returns HOLD.

## AC6 — Screen 3 breakout required
Given Screen 1 trend and Screen 2 correction gate both pass
When Screen 3 price action does not break the configured lookback high/low
Then the strategy returns HOLD.

## AC7 — Signal contains exit parameters
Given the strategy returns a LONG or SHORT signal
When inspecting the returned `TradingSignal`
Then:
- `entry_price` is set
- `stop_loss` is set
- `take_profit` is set
- `exit_policy == TRAILING_STOP`
- `trailing_stop_pct` is set (or explicitly documented as disabled)

## AC8 — Backtest comparative runs are possible
Given the same start/end dates, assets, timeframe and config
When running:
- a backtest with default strategy (LLM)
- a backtest with `strategy=TripleScreenStrategy(...)`
Then both runs complete and return `BacktestMetrics` enabling comparison (same metrics fields available).
