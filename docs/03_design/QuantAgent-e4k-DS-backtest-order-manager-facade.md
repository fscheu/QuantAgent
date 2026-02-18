# QuantAgent-e4k: Backtest depends only on OrderManager (facade) — Design

## Context

Per `docs/03_design/backtesting_engine.md`, `Backtest` is the orchestrator and `OrderManager` is the execution facade.

This change removes execution-layer component wiring from `Backtest` and makes `OrderManager` the single entry point for execution-related operations.

## Affected Components

- `quantagent/backtesting/backtest.py`
- `quantagent/trading/order_manager.py` (only if required to make facade complete)

## Design Summary

### 1) Backtest no longer stores execution components

In `Backtest.__init__`, keep only:
- `self.order_manager` (execution facade)
- `self.position_monitor` (position tracking; unchanged by this issue)
- `self.strategy` / `self.trading_graph` / `self.data_provider` / `self.db` (orchestration)

Remove from `Backtest` instance state:
- `self.position_sizer`
- `self.risk_manager`
- `self.broker`
- (optionally) `self.portfolio` (preferred to avoid any execution detail leaking; see below)

### 2) OrderManager facade additions (minimal)

Backtest currently needs two execution-adjacent capabilities outside open/close calls:

1. **Reset daily risk tracking**
   - Add `OrderManager.reset_daily_tracker()` delegating to `self.risk_manager.reset_daily_tracker()`

2. **Equity curve inputs**
   Preferred: add a small accessor returning a snapshot to avoid exposing the portfolio object.
   - Add `OrderManager.get_portfolio_snapshot()` returning `{total_value: float, cash: float}` (or a small dataclass)
   - Backtest uses this snapshot to record equity curve.

If the repo strongly prefers direct delegation over snapshots, acceptable alternative:
- Add `OrderManager.get_total_value()` and `OrderManager.get_cash()` delegating to portfolio.

### 3) Backtest flow changes (call-site routing)

- In the main run loop, replace:
  - `self.risk_manager.reset_daily_tracker()` → `self.order_manager.reset_daily_tracker()`
- In equity recording, replace portfolio access with facade access:
  - `self.portfolio.get_total_value()` / `self.portfolio.cash` → `self.order_manager.get_portfolio_snapshot()`

No changes to:
- `OrderManager.execute_decision(...)` usage in Backtest
- `PositionMonitor` interactions

### Example (minimal)

```python
# order_manager.py
class OrderManager:
    def reset_daily_tracker(self) -> None:
        self.risk_manager.reset_daily_tracker()

    def get_portfolio_snapshot(self) -> dict:
        return {
            "total_value": float(self.portfolio.get_total_value()),
            "cash": float(self.portfolio.cash),
        }
```

## Risks / Notes

- **Hidden dependencies**: If Backtest uses any other execution component methods indirectly (beyond current usages), those should be routed through `OrderManager` as well.
- **Type/contract drift**: Keep new `OrderManager` methods minimal and stable to avoid turning it into a “god object”.
