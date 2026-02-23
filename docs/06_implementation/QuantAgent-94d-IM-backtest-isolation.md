# QuantAgent-94d: Backtest Run ID Isolation - Implementation Plan

## Task Breakdown

### Phase 1: Schema Migration (~1h)

**Task 1.1: Create Alembic Migration**
- File: `alembic/versions/XXXXX_add_backtest_run_id_to_active_positions.py`
- Commands:
  ```bash
  python -m alembic revision --autogenerate -m "add_backtest_run_id_to_active_positions"
  ```
- Manual edits required for:
  - FK constraint with delete restricted (`RESTRICT`/`NO ACTION`)
  - Composite index definition

**Task 1.2: Update ActivePosition Model**
- File: `quantagent/models.py`
- Location: Lines 311-349
- Changes:
  - Add `backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id"), nullable=True, index=True)` (delete restricted)
  - Add `backtest_run = relationship("BacktestRun", backref="active_positions")`
  - Add to `__table_args__`: `Index("idx_active_position_isolation", "symbol", "is_active", "backtest_run_id", "environment")`

**Task 1.3: Run Migration**
- Commands:
  ```bash
  python -m alembic upgrade head
  python -m alembic current  # verify
  ```

### Phase 2: PositionMonitor Update (~1h)

**Task 2.1: Update Constructor**
- File: `quantagent/trading/position_monitor.py`
- Location: Lines 18-19
- Add: `backtest_run_id: Optional[int] = None` parameter
- Store: `self.backtest_run_id = backtest_run_id`

**Task 2.2: Update get_active_position()**
- File: `quantagent/trading/position_monitor.py`
- Location: Lines 21-30
- Add conditional filter for `backtest_run_id`

**Task 2.3: Update open_position()**
- File: `quantagent/trading/position_monitor.py`
- Location: Lines 32-68
- Add `backtest_run_id` parameter
- Set on ActivePosition creation

### Phase 3: Backtest Class Updates (~1.5h)

**Task 3.1: Defer PositionMonitor Initialization**
- File: `quantagent/backtesting/backtest.py`
- Location: Line 169
- Change: Initialize `self.position_monitor = None`

**Task 3.2: Initialize PositionMonitor After Run Creation**
- File: `quantagent/backtesting/backtest.py`
- Location: After line 315 (in `_create_backtest_run`)
- Add: `self.position_monitor = PositionMonitor(self.db, backtest_run_id=self.backtest_run_id)`

**Task 3.3: Update Metrics Queries**
- File: `quantagent/backtesting/backtest.py`
- Location: Lines 908-916 (`_calculate_directional_accuracy`)
- Add filter: `ActivePosition.backtest_run_id == self.backtest_run_id`

- Location: Lines 962-970 (`_calculate_close_reasons`)
- Add filter: `ActivePosition.backtest_run_id == self.backtest_run_id`

### Phase 4: Testing (~1.5h)

**Task 4.1: Create Isolation Test File**
- File: `tests/test_backtest_isolation.py`
- Test cases:
  - `test_position_created_with_backtest_run_id`
  - `test_parallel_runs_isolated`
  - `test_metrics_scoped_to_run`
  - `test_backtest_run_delete_restricted`

**Task 4.2: Run Existing Tests**
- Commands:
  ```bash
  pytest tests/ -v --tb=short
  ```
- Verify no regressions

## Dependencies

```
Task 1.2 --> Task 1.1 (model before migration verification)
Task 1.3 --> Task 1.1, 1.2
Task 2.* --> Task 1.3 (migration must be applied)
Task 3.* --> Task 2.* (PositionMonitor changes first)
Task 4.* --> Task 3.* (all code changes complete)
```

## Validation Commands

```bash
# Migration
python -m alembic upgrade head
python -m alembic current

# Tests
pytest tests/test_backtest_isolation.py -v
pytest tests/ -v

# Manual verification
python -c "
from quantagent.models import ActivePosition
from sqlalchemy import inspect
cols = [c.name for c in inspect(ActivePosition).columns]
print('backtest_run_id' in cols)  # Should print True
"
```

## Rollback Plan

If issues arise:
```bash
python -m alembic downgrade -1  # Revert migration
```

No backward-compatibility guarantees required (system is not yet production).
