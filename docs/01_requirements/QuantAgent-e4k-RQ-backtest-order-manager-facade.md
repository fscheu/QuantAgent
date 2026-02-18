# QuantAgent-e4k: Backtest depends only on OrderManager (facade) — Requirements

## Problem Statement

`Backtest` currently stores and uses direct references to execution-layer components (e.g., `PositionSizer`, `RiskManager`, `PaperBroker`). This increases coupling and leaks execution details outside the trading-execution facade.

`OrderManager` already represents the end-to-end execution flow (size → validate → execute → persist). `Backtest` should interact with execution exclusively via `OrderManager`.

## Functional Requirements

### FR-1: Backtest execution uses OrderManager as the only execution dependency
- `Backtest` MUST NOT hold direct references to `PositionSizer`, `RiskManager`, or `PaperBroker`.
- `Backtest` MUST execute opens/closes via `OrderManager` methods only.

### FR-2: Facade completeness for Backtest needs
`OrderManager` MUST expose (directly, or via a minimal sub-facade) the minimum capabilities `Backtest` needs from execution components, including:
- resetting daily risk tracking (currently `RiskManager.reset_daily_tracker()`)
- retrieving portfolio totals needed for equity curve recording (today uses `PortfolioManager.get_total_value()` and `PortfolioManager.cash`)

> Note: Whether portfolio access is exposed as dedicated `OrderManager` methods (preferred) or via a returned snapshot object is an implementation detail (see DS).

### FR-3: No behavior change in backtest results (refactor-only)
- The refactor MUST preserve backtest behavior and outputs (trades created, metrics calculated, logs) for the same inputs/config.

## Non-Functional Requirements

### NFR-1: Minimal, scoped change
- Only changes required to remove direct dependencies and route through `OrderManager` are in scope.

### NFR-2: Backwards compatibility
- No behavior changes are introduced for PAPER/PROD execution flows.

## Scope

### In Scope
- Refactor `quantagent/backtesting/backtest.py` to remove direct component references
- Minimal `OrderManager` API additions required by Backtest (if needed)
- Documentation updates for this issue (RQ/DS/AC/PL)

### Out of Scope
- Changing sizing/risk/execution logic
- Refactoring StrategyAssembler beyond what’s required to pass the facade dependency
- New features or performance optimizations

## Definition of Done
- `Backtest` no longer imports/depends on `PositionSizer`, `RiskManager`, `PaperBroker`
- `Backtest` interacts with execution through `OrderManager` only
- Backtest-related tests and/or smoke runs pass (see AC)
