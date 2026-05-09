# QuantAgent-uzq — Implementation: Fix TradingScheduler Heartbeat and Scheduler Unit-Test Regressions

**Issue ID:** QuantAgent-uzq  
**Phase:** implementer  
**Acceptance tests:** `docs/05_acceptance_tests/QuantAgent-uzq-AC-fix-scheduler-heartbeat.md`

---

## Root Cause Analysis

### Problem 1 — Missing heartbeat methods in `TradingScheduler`

The `SchedulerHeartbeat` model (`quantagent/models.py:375`) and its tests (`tests/test_vje_scheduler_heartbeat_backend.py`) were written expecting two private methods on `TradingScheduler`:

- `_upsert_heartbeat_start(started_at: datetime) -> Optional[SchedulerHeartbeat]`
- `_upsert_heartbeat_complete(heartbeat: Optional[SchedulerHeartbeat], stats: dict) -> None`

These methods were never added to `quantagent/trading/scheduler.py`. The `analyze_and_trade()` method (line 148) also never calls them.

### Problem 2 — `DummySession.query()` returns an incomplete mock chain

`PositionMonitor.get_active_position()` (`quantagent/trading/position_monitor.py:26-36`) chains:
```python
self.db.query(ActivePosition).filter(...).filter(...).order_by(ActivePosition.id).first()
```

`DummySession.query()` in `tests/trading/test_scheduler.py` (line 48-53) configures:
```python
mock_query.filter.return_value.first.return_value = None
mock_query.filter.return_value.all.return_value = []
```

But does **not** configure `filter.return_value.order_by.return_value.first.return_value`. Because `order_by()` returns an unconfigured `Mock`, calling `.first()` on it returns a truthy `Mock` object instead of `None`.

This causes `if position:` in `_process_asset()` to be `True`, leading to `update_candle_tracking()` being called on the mock position, which fails with:
```
TypeError: unsupported operand type(s) for +=: 'Mock' and 'int'
```
at `position.candles_since_entry += 1`.

The exception is caught by the bare `except Exception:` handler in `analyze_and_trade()`, incrementing `errors` instead of `processed`.

---

## Changes Required

### Change 1: `quantagent/trading/scheduler.py`

#### 1a. Add import for `SchedulerHeartbeat`

Add `SchedulerHeartbeat` to the existing `from quantagent.models import ...` line (line 16).

**Before:**
```python
from quantagent.models import Environment, Signal, TradeSignal
```

**After:**
```python
from quantagent.models import Environment, SchedulerHeartbeat, Signal, TradeSignal
```

#### 1b. Add `_upsert_heartbeat_start` method

Add after the `_make_thread_id` method (after line 457):

```python
def _upsert_heartbeat_start(self, started_at: datetime) -> Optional["SchedulerHeartbeat"]:
    try:
        existing = (
            self.db.query(SchedulerHeartbeat)
            .filter(SchedulerHeartbeat.environment == self.environment)
            .order_by(SchedulerHeartbeat.id)
            .first()
        )
        if existing is not None:
            existing.timestamp = started_at
            existing.status = "running"
            existing.assets = list(self.config.assets)
            self.db.commit()
            self.db.refresh(existing)
            return existing

        hb = SchedulerHeartbeat(
            timestamp=started_at,
            status="running",
            environment=self.environment,
            assets=list(self.config.assets),
        )
        self.db.add(hb)
        self.db.commit()
        self.db.refresh(hb)
        return hb
    except Exception:
        logger.exception("Heartbeat start failed; continuing cycle")
        return None
```

**Contract:**
- One row per environment — if a row already exists for `self.environment`, update it in place (upsert).
- Returns `None` on any DB failure; callers must handle this gracefully.

#### 1c. Add `_upsert_heartbeat_complete` method

Add after `_upsert_heartbeat_start`:

```python
def _upsert_heartbeat_complete(
    self, heartbeat: Optional["SchedulerHeartbeat"], stats: Dict[str, float]
) -> None:
    if heartbeat is None:
        return
    try:
        from quantagent.models import Trade
        last_trade = (
            self.db.query(Trade)
            .order_by(Trade.id.desc())
            .first()
        )
        heartbeat.status = "completed"
        heartbeat.completed_at = datetime.utcnow()
        heartbeat.stats = stats
        heartbeat.last_trade_id = last_trade.id if last_trade else None
        self.db.commit()
    except Exception:
        logger.exception("Heartbeat complete failed")
```

**Contract:**
- No-op when `heartbeat is None` (graceful degradation when start failed).
- Sets `last_trade_id` to the globally most-recent trade's id (not filtered by environment — consistent with the test expectation in AC-2).

#### 1d. Modify `analyze_and_trade` to call heartbeat methods

Wrap the existing method body with heartbeat calls:

**Before (line 148-226):**
```python
def analyze_and_trade(self) -> Dict[str, float]:
    cycle_start = datetime.utcnow()
    processed = 0
    errors = 0

    for symbol in self.config.assets:
        ...

    duration = (datetime.utcnow() - cycle_start).total_seconds()
    stats = { ... }
    self.last_run_stats = stats

    logger.info(...)
    return stats
```

**After:**
```python
def analyze_and_trade(self) -> Dict[str, float]:
    cycle_start = datetime.utcnow()
    processed = 0
    errors = 0

    heartbeat = self._upsert_heartbeat_start(cycle_start)  # ADD

    for symbol in self.config.assets:
        ...

    duration = (datetime.utcnow() - cycle_start).total_seconds()
    stats = { ... }
    self.last_run_stats = stats

    self._upsert_heartbeat_complete(heartbeat, stats)  # ADD

    logger.info(...)
    return stats
```

The `heartbeat` call is made before the asset loop; `_upsert_heartbeat_complete` is called after `last_run_stats` is set but before the final `logger.info`.

---

### Change 2: `tests/trading/test_scheduler.py`

#### 2a. Fix `DummySession.query()` mock chain

The current mock only configures `.filter(...).first()`. It needs to also handle `.filter(...).order_by(...).first()` and `.order_by(...).first()` (used by `_upsert_heartbeat_complete`).

**Before:**
```python
def query(self, model):
    """Return a mock query object that returns no active positions."""
    mock_query = Mock()
    mock_query.filter.return_value.first.return_value = None
    mock_query.filter.return_value.all.return_value = []
    return mock_query
```

**After:**
```python
def query(self, model):
    """Return a mock query object that returns no results."""
    mock_q = Mock()
    mock_q.filter.return_value = mock_q
    mock_q.order_by.return_value = mock_q
    mock_q.first.return_value = None
    mock_q.all.return_value = []
    return mock_q
```

**Why this works:** The mock becomes self-referential for all chainable methods (`filter`, `order_by`), so any call sequence ending in `.first()` returns `None` and `.all()` returns `[]`. This covers:
- `query(M).filter(...).order_by(...).first()` → `None` (fixes AC-6, AC-7, AC-8)
- `query(M).filter(...).first()` → `None` (backwards compatible)
- `query(M).order_by(...).first()` → `None` (needed by `_upsert_heartbeat_complete`)

---

## Files to Modify

| File | Change type | Lines affected |
|------|------------|----------------|
| `quantagent/trading/scheduler.py` | Add 2 methods + modify `analyze_and_trade` | ~458–475 (new methods), ~148–227 (modify) |
| `tests/trading/test_scheduler.py` | Fix `DummySession.query()` | lines 48–53 |

## Files NOT to Touch

- `quantagent/models.py` — `SchedulerHeartbeat` model is correct as-is
- `quantagent/trading/position_monitor.py` — no changes needed
- `tests/test_vje_scheduler_heartbeat_backend.py` — tests are correct; implementation must match
- `.github/workflows/main-ci-deploy.yml` — out of scope

---

## Verification Commands

```bash
# Focused gate (8 failing tests)
DATABASE_URL=postgresql://test:test@localhost:5432/quantagent_test \
  /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python \
  -m pytest tests/test_vje_scheduler_heartbeat_backend.py tests/trading/test_scheduler.py \
  -v --tb=short --maxfail=10 -m "not integration and not slow"

# Quick check without PostgreSQL (SQLite fallback)
/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python \
  -m pytest tests/test_vje_scheduler_heartbeat_backend.py tests/trading/test_scheduler.py \
  -v --tb=short -m "not integration and not slow"
```

Expected: all previously-failing tests now show `PASSED`.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `_upsert_heartbeat_start` race condition on upsert | Low (single-threaded scheduler) | Acceptable; no unique constraint needed |
| DummySession change breaks currently-passing tests | Low | Self-referential mock is backwards-compatible for all prior patterns |
| `_upsert_heartbeat_complete` importing `Trade` inline causes circular import | Very low | `Trade` is in `models.py`, already imported elsewhere in scheduler |
| Heartbeat writes slow down scheduler cycle | Very low | Commit only 2 extra rows per cycle |
