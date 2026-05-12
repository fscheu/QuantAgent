# QuantAgent-e4k — Requirements: Refactor Backtest to depend only on OrderManager

**Issue:** QuantAgent-e4k  
**Labels:** architecture, encapsulation, refactor, openclaw:design_approved  
**Type:** Refactor (architectural coupling reduction)

---

## Context

`Backtest` currently holds direct references to three trading sub-components it receives from `StrategyAssembler`:

| Attribute | Type | Current Usage |
|---|---|---|
| `self.position_sizer` | `PositionSizer` | Stored but never called directly from `Backtest` |
| `self.risk_manager` | `RiskManager` | `reset_daily_tracker()` called in `run()` and `run_replay()` |
| `self.broker` | `PaperBroker` | Stored but never called directly from `Backtest` |

`OrderManager` already internally owns all three components and acts as the execution facade for all trade decisions. The intent of this issue is to complete the encapsulation: `Backtest` should interact with the trading layer exclusively through `OrderManager`.

An additional gap exists: `Backtest._analyze_and_trade()` and `Backtest._replay_and_trade()` both call `self.order_manager.close_trade(trade_id, current_price, environment=...)`, but `OrderManager` has **no `close_trade` method** — this is a latent runtime error that this refactor must fix.

---

## Functional Requirements

### FR-1: Remove direct sub-component references from Backtest

`Backtest.__init__` must NOT store `self.position_sizer`, `self.risk_manager`, or `self.broker`. These components are implementation details of `OrderManager` and should not be accessible from `Backtest`.

### FR-2: Expose `reset_daily_tracker` on OrderManager

`OrderManager` must expose a `reset_daily_tracker()` method that delegates to its internal `RiskManager`. `Backtest` must use `self.order_manager.reset_daily_tracker()` instead of `self.risk_manager.reset_daily_tracker()`.

### FR-3: Expose `close_trade` on OrderManager

`OrderManager` must expose a `close_trade(trade_id: int, current_price: float, environment=None) -> Optional[Order]` method that:
- Looks up the `Trade` record by `trade_id`
- Creates a closing `Order` (opposite side, same quantity)
- Executes it through the broker
- Updates portfolio state via `portfolio.execute_trade()`
- Updates risk tracker via `risk_manager.on_trade_executed()`
- Returns the filled `Order`, or `None` if the trade is not found or execution fails

### FR-4: No behaviour change

The refactor is purely structural. No change in execution logic, P&L calculations, or test outcomes is expected.

---

## Out of Scope

- No changes to `StrategyAssembler.build_components()` — it still builds and wires all four components
- No changes to `PositionMonitor`, `PortfolioManager`, `DataProvider`, or strategy classes
- No changes to the `TradingComponents` dataclass (it still exposes all fields for other callers)
- No changes to database schema
