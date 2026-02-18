# QuantAgent-e4k: Backtest depends only on OrderManager (facade) — Acceptance Criteria

## AC-1: Backtest has no direct execution-component dependencies

```
Given the codebase at this change
When reviewing `quantagent/backtesting/backtest.py`
Then it does not import or store direct references to `PositionSizer`, `RiskManager`, or `PaperBroker`
And execution interactions occur via `OrderManager` only
```

## AC-2: Backtest daily reset routes through OrderManager

```
Given a backtest run iterating across multiple days
When the daily reset point is reached
Then Backtest calls `OrderManager.reset_daily_tracker()` (or equivalent facade method)
And Backtest does not call RiskManager directly
```

## AC-3: Equity curve recording does not reach into PortfolioManager directly

```
Given Backtest records an equity curve point each period
When equity is recorded
Then Backtest obtains cash/total_value via an OrderManager facade method (snapshot or delegated accessors)
And Backtest does not access `PortfolioManager` directly
```

## AC-4: Refactor-only: behavior is preserved

```
Given an existing backtest scenario that produces trades and metrics
When running backtest before vs after this change with identical inputs/config
Then the run completes without new errors
And trades are still created and persisted
And metrics calculation still completes
```

## AC-5: Regression checks pass

```
Given the repository test suite
When running the backtest-related tests
Then the suite passes without failures attributable to this change
```
