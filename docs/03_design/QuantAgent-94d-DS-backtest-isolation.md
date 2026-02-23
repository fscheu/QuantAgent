# QuantAgent-94d: Backtest Run ID Isolation - Design

## Architecture Overview

```
BacktestRun (id)
     |
     | 1:N (nullable FK)
     v
ActivePosition (backtest_run_id)
```

## Schema Changes

### ActivePosition Table

| Column | Type | Constraint | Notes |
|--------|------|------------|-------|
| `backtest_run_id` | INTEGER | NULLABLE | FK to backtest_runs.id |

**FK Constraint:**
- **Do not allow deleting a BacktestRun** if there are ActivePosition rows referencing it.
- Implement as `ON DELETE RESTRICT` / `NO ACTION` (backend default), i.e. no orphaning.

**New Index:**
- `idx_active_position_isolation`: `(symbol, is_active, backtest_run_id, environment)`

## Component Changes

### 1. ActivePosition Model (`quantagent/models.py`)

Add after line 348:
- `backtest_run_id` column with FK
- `backtest_run` relationship
- Update `__table_args__` with new index

### 2. PositionMonitor (`quantagent/trading/position_monitor.py`)

**Constructor change:**
- Accept optional `backtest_run_id: Optional[int] = None`
- Store as instance attribute

**Query changes in `get_active_position()`:**
- When `backtest_run_id` is set: filter `backtest_run_id == self.backtest_run_id`
- When `backtest_run_id` is None: **do not add any run filter** (no special-casing to `IS NULL`).

**Position creation in `open_position()`:**
- Accept `backtest_run_id` parameter
- Set on new ActivePosition

### 3. Backtest Class (`quantagent/backtesting/backtest.py`)

**Initialization change (line 169):**
- Defer `PositionMonitor` creation to after `_create_backtest_run()`
- Pass `backtest_run_id` to PositionMonitor

**Alternative:** Create PositionMonitor with setter pattern:
- Create in `__init__` with `backtest_run_id=None`
- Call `position_monitor.set_backtest_run_id(self.backtest_run_id)` in `run()`

**Metrics queries (lines 908-916, 962-970):**
- Add filter: `ActivePosition.backtest_run_id == self.backtest_run_id`

### Example (minimal)

```python
# PositionMonitor query pattern
def get_active_position(self, symbol: str) -> Optional[ActivePosition]:
    query = self.db.query(ActivePosition).filter(
        ActivePosition.symbol == symbol,
        ActivePosition.is_active.is_(True),
    )
    if self.backtest_run_id is not None:
        query = query.filter(ActivePosition.backtest_run_id == self.backtest_run_id)
    return query.first()
```

## Query Patterns

| Context | Filter Logic |
|---------|--------------|
| Backtest (run_id=5) | `backtest_run_id == 5` |
| Non-backtest contexts (no run_id) | no `backtest_run_id` filter |

## Migration Strategy

1. Add nullable column (no default)
2. Add FK constraint with delete restricted (`RESTRICT`/`NO ACTION`)
3. Add composite index
4. No compatibility/backfill guarantees for pre-existing rows (system is not yet production)

## Risks

| Risk | Mitigation |
|------|------------|
| Index bloat | Composite index replaces multiple single-column indexes for this query pattern |
| Query plan regression | Test with realistic data volume |
| BacktestRun delete blocked | Enforce FK delete restriction |
