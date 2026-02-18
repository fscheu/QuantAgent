# QuantAgent-94d: Backtest Run ID Isolation - Requirements

## Problem Statement

Multiple backtest runs sharing the same database see each other's active positions, causing:
- Incorrect position state (positions from Run A visible in Run B)
- Corrupted metrics (MDA, close_reasons aggregated across runs)
- Non-reproducible results

## Functional Requirements

### FR-1: Position Isolation by Backtest Run
- Each ActivePosition created during a backtest MUST be associated with that backtest's `backtest_run_id`
- Position queries during backtest MUST filter by `backtest_run_id`
- Metrics calculations (MDA, close_reasons) MUST scope to current `backtest_run_id`

### FR-2: Parallel Execution Support
- Multiple concurrent backtests on same database MUST NOT interfere
- Each backtest run operates in complete isolation

## Non-Functional Requirements

### NFR-1: Performance
- Composite index on `(symbol, is_active, backtest_run_id, environment)` required
- No measurable regression in position query latency

### NFR-2: Data Integrity
- FK constraint to BacktestRun with **delete restricted** (do not allow deleting a BacktestRun while referenced by ActivePosition)

## Scope

### In Scope
- Alembic migration adding column, FK, index
- ActivePosition model changes
- PositionMonitor query filtering
- Backtest class context propagation
- Metrics query updates (_calculate_directional_accuracy, _calculate_close_reasons)

### Out of Scope
- Auto-closing positions at backtest end
- UI changes

## Definition of Done
- Migration runs without errors
- All backtest positions have non-NULL backtest_run_id
- Parallel backtests produce isolated results
- All relevant tests pass
