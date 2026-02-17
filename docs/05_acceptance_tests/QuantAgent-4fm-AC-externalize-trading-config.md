# QuantAgent-4fm — AC — Externalize hardcoded trading configuration

## Env-backed defaults
- Given `.env` sets `TRADING_BASE_POSITION_PCT=0.02`
  When a backtest is run without providing `base_position_pct` in any profile/override
  Then `PositionSizer` uses `base_position_pct=0.02`.

- Given `.env` sets `TRADING_MAX_DAILY_LOSS_PCT=0.03` and `TRADING_MAX_POSITION_PCT=0.08`
  When a backtest is run without providing risk overrides
  Then `RiskManager` is constructed with `max_daily_loss_pct=0.03` and `max_position_pct=0.08`.

- Given `.env` sets `TRADING_SLIPPAGE_PCT=0.005`
  When a backtest is run without providing `slippage_pct`
  Then `PaperBroker` is constructed with `slippage_pct=0.005`.

- Given `.env` sets `BACKTEST_MARKET_HOURS_FILTER=false`
  When a backtest is run without providing `market_hours_filter` in overrides
  Then market hours filtering is disabled.

## DB profile overrides env
- Given `.env` sets `TRADING_BASE_POSITION_PCT=0.02`
  And a persisted StrategyConfig profile sets `base_position_pct=0.06`
  When running a backtest using that profile (directly or via existing Streamlit workflow)
  Then `base_position_pct=0.06` is used.

## Explicit overrides override everything
- Given `.env` sets `TRADING_BASE_POSITION_PCT=0.02`
  And a StrategyConfig profile sets `base_position_pct=0.06`
  When a backtest run provides an explicit override `base_position_pct=0.01`
  Then `base_position_pct=0.01` is used.

## No scattered literals in targeted modules
- Given the codebase at HEAD for this issue
  When searching `quantagent/strategy/assembler.py` and `quantagent/backtesting/backtest.py` for the previous literal defaults (e.g., `0.05`, `0.10`, `0.01`, `100000.0`) used as fallbacks for the in-scope parameters
  Then those literals are not present as module-level defaults or `.get(..., <literal>)` fallbacks for those parameters.
