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
- `ON DELETE SET NULL` - preserves position data if run deleted

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
- When NULL: filter `backtest_run_id.is_(None)`

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
    else:
        query = query.filter(ActivePosition.backtest_run_id.is_(None))
    return query.first()
```

## Query Patterns

| Context | Filter Logic |
|---------|--------------|
| Backtest (run_id=5) | `backtest_run_id == 5` |
| PAPER/PROD | `backtest_run_id IS NULL` |
| Existing positions | Remain with `NULL`, queryable in PAPER |

## Migration Strategy

1. Add nullable column (no default)
2. Add FK constraint with `ON DELETE SET NULL`
3. Add composite index
4. Existing rows remain `NULL`

## Risks

| Risk | Mitigation |
|------|------------|
| Index bloat | Composite index replaces multiple single-column indexes for this query pattern |
| Query plan regression | Test with realistic data volume |
| Orphaned positions | `ON DELETE SET NULL` preserves data |
