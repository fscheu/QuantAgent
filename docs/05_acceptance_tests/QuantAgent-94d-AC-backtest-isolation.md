# QuantAgent-94d: Backtest Run ID Isolation - Acceptance Criteria

## AC-1: Migration Execution

**Given** a database at current schema version
**When** the migration `add_backtest_run_id_to_active_positions` is applied
**Then**:
- Column `backtest_run_id` exists on `active_positions` table
- Column is nullable
- FK constraint to `backtest_runs(id)` exists and **prevents deleting a referenced BacktestRun** (delete restricted)
- Index `idx_active_position_isolation` exists on `(symbol, is_active, backtest_run_id, environment)`

## AC-2: Position Creation with Isolation

**Given** a backtest run with `id=42`
**When** PositionMonitor.open_position() is called
**Then** the created ActivePosition has `backtest_run_id = 42`

## AC-3: Position Query Isolation

**Given**:
- Backtest Run A (id=1) with position P1 for symbol BTC
- Backtest Run B (id=2) with position P2 for symbol BTC

**When** PositionMonitor(backtest_run_id=1).get_active_position("BTC")
**Then** returns P1 only (NOT P2)

**When** PositionMonitor(backtest_run_id=2).get_active_position("BTC")
**Then** returns P2 only (NOT P1)

## AC-4: Metrics Isolation

**Given**:
- Run A (id=1): 10 closed positions, 7 with accuracy >= 0.5
- Run B (id=2): 5 closed positions, 2 with accuracy >= 0.5

**When** `_calculate_directional_accuracy()` runs for Run A
**Then** returns MDA based on Run A's 10 positions only

**When** `_calculate_close_reasons()` runs for Run A
**Then** returns close reasons from Run A's 10 positions only

## AC-5: Parallel Backtest Independence

**Given** two concurrent backtest processes A and B on same database
**When** both create and query positions for the same symbol
**Then**:
- Process A sees only its positions
- Process B sees only its positions
- No cross-contamination of metrics

## AC-6: BacktestRun Deletion is Prevented When Referenced

**Given** a BacktestRun id=5 with at least one ActivePosition row referencing it
**When** attempting to delete BacktestRun id=5
**Then** the delete is rejected by the database (FK restriction) and no ActivePosition rows are modified

## Negative Cases

### NC-1: Cross-Run Visibility Prevention

**Given** Run A with an open BTC position
**When** Run B checks for open BTC position
**Then** returns None (not Run A's position)

### NC-2: Metrics Scope Enforcement

**Given** Run A with 0 positions, Run B with 10 positions
**When** Run A calculates MDA
**Then** returns 0.0 (not Run B's metrics)

## Performance Invariants

- Position lookup query uses `idx_active_position_isolation` index
- No full table scans for position queries with `backtest_run_id` filter
