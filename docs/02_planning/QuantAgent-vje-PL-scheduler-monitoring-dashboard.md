# QuantAgent-vje — Planning: Scheduler Status and Controls in Streamlit Dashboard

**Issue ID:** QuantAgent-vje  
**Title:** Scheduler status and controls in Streamlit Dashboard  
**Type:** Feature  
**Priority:** 2

---

## Objective

Add scheduler monitoring to Streamlit dashboard via heartbeat mechanism: scheduler writes execution records, dashboard reads and displays status.

---

## Tasks

### Task 1: Add SchedulerHeartbeat Model
**Estimate:** 1h (60 minutes)

**What:**
- Add `SchedulerHeartbeat` class to `quantagent/models.py`
- Define table schema with all required columns
- Add indexes for performance

**Code:**
```python
class SchedulerHeartbeat(Base):
    __tablename__ = "scheduler_heartbeats"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False)
    environment = Column(Enum(Environment), nullable=False, index=True)
    assets = Column(JSON, nullable=True)
    stats = Column(JSON, nullable=True)
    last_trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    error_message = Column(Text, nullable=True)
    
    __table_args__ = (
        Index("idx_heartbeat_env_ts", "environment", "timestamp"),
    )
```

**How to validate:**
```python
# Test import
from quantagent.models import SchedulerHeartbeat
assert SchedulerHeartbeat.__tablename__ == "scheduler_heartbeats"
```

**Dependencies:** None

---

### Task 2: Create Database Migration
**Estimate:** 0.5h (30 minutes)

**What:**
- Generate Alembic migration
- Review auto-generated script
- Test upgrade and downgrade

**Commands:**
```bash
# Generate migration
alembic revision --autogenerate -m "Add scheduler_heartbeats table"

# Review file
vim migrations/versions/XXXXXX_add_scheduler_heartbeats.py

# Test upgrade
alembic upgrade head

# Test downgrade
alembic downgrade -1

# Re-upgrade for development
alembic upgrade head
```

**How to validate:**
```sql
-- Verify table exists
\d scheduler_heartbeats

-- Verify indexes
\di scheduler_heartbeats*
```

**Dependencies:** Task 1

---

### Task 3: Add Heartbeat Writing to Scheduler
**Estimate:** 2h (120 minutes)

**What:**
- Add `_upsert_heartbeat_start()` method to TradingScheduler
- Add `_upsert_heartbeat_complete()` method
- Call methods in `analyze_and_trade()`
- Add error handling (continue on heartbeat failure)

**Code locations:**
- File: `quantagent/trading/scheduler.py`
- Method: `analyze_and_trade()`

**Implementation steps:**
1. Import SchedulerHeartbeat model
2. At cycle start: call `_upsert_heartbeat_start()`
3. At cycle end: call `_upsert_heartbeat_complete(stats)`
4. On exception: update heartbeat with error status
5. Wrap all heartbeat writes in try/except (log warning, continue)

**How to validate:**
```python
def test_scheduler_writes_heartbeat():
    scheduler.analyze_and_trade()
    hb = db.query(SchedulerHeartbeat).first()
    assert hb.status == "completed"
    assert hb.stats["processed"] >= 0
```

**Dependencies:** Task 2

---

### Task 4: Add DB Service Methods
**Estimate:** 1h (60 minutes)

**What:**
- Add `get_latest_heartbeat(environment)` to `apps/streamlit/services/db.py`
- Add `get_recent_heartbeats(environment, limit)` to same file
- Handle missing table gracefully (return None/empty list)

**Code:**
```python
def get_latest_heartbeat(self, environment: str) -> Optional[dict]:
    if not self.ok:
        return None
    try:
        from quantagent.models import SchedulerHeartbeat, Environment
        hb = self.session.query(SchedulerHeartbeat).filter_by(
            environment=Environment(environment)
        ).order_by(SchedulerHeartbeat.timestamp.desc()).first()
        # ... convert to dict and return ...
    except Exception:
        return None

def get_recent_heartbeats(self, environment: str, limit: int = 10) -> List[dict]:
    # Similar pattern
```

**How to validate:**
```python
def test_db_service_methods():
    create_test_heartbeat()
    latest = db.get_latest_heartbeat("paper")
    assert latest is not None
    recent = db.get_recent_heartbeats("paper", limit=5)
    assert len(recent) <= 5
```

**Dependencies:** Task 2

---

### Task 5: Create Paper Trading View
**Estimate:** 2.5h (150 minutes)

**What:**
- Create new file: `apps/streamlit/views/paper_trading.py`
- Implement `render(db, environment)` function
- Add status card rendering: `_render_status_card(heartbeat)`
- Add recent runs table: `_render_recent_runs(recent)`
- Add helper functions: `_calculate_status()`, `_humanize_time()`, `_calculate_duration()`

**Structure:**
```python
def render(db, environment: str) -> None:
    """Main render function."""
    st.header("📊 Paper Trading Scheduler")
    
    if not db.ok:
        st.error("Database not available")
        return
    
    heartbeat = db.get_latest_heartbeat(environment)
    
    if not heartbeat:
        st.warning("No scheduler heartbeat found")
        return
    
    _render_status_card(heartbeat)
    st.divider()
    
    recent = db.get_recent_heartbeats(environment, 10)
    _render_recent_runs(recent)
```

**How to validate:**
- Manual test: run Streamlit, view tab
- Unit tests for helper functions

**Dependencies:** Task 4

---

### Task 6: Integrate Tab into App.py
**Estimate:** 0.5h (30 minutes)

**What:**
- Import `paper_trading` view module
- Add "Paper Trading" to tabs list
- Add render call in appropriate tab

**Code changes:**
```python
# Import
from apps.streamlit.views.paper_trading import render as render_paper_trading

# Add to tabs
tabs = st.tabs([
    "Dashboard",
    "Configuration",
    "Analyses",
    "Backtesting",
    "Replay",
    "Orders & Positions",
    "Paper Trading",  # NEW
    "Logs",
])

# Render
with tabs[6]:
    render_paper_trading(db, environment)
```

**How to validate:**
- Run Streamlit
- Verify tab appears in UI

**Dependencies:** Task 5

---

### Task 7: Add Unit Tests
**Estimate:** 2h (120 minutes)

**What:**
- Create `tests/trading/test_scheduler_heartbeat.py`
- Test heartbeat writing methods
- Test status calculation logic
- Test graceful failure handling

**Tests to write:**
1. `test_upsert_heartbeat_start()` — Verify creation/update
2. `test_upsert_heartbeat_complete()` — Verify stats recording
3. `test_heartbeat_on_error()` — Verify error recording
4. `test_heartbeat_failure_graceful()` — Scheduler continues on DB error
5. `test_calculate_status_active()` — 30 min ago
6. `test_calculate_status_stale()` — 5 hours ago
7. `test_calculate_status_stopped()` — 25 hours ago
8. `test_db_service_methods()` — Query methods work

**How to validate:**
```bash
pytest tests/trading/test_scheduler_heartbeat.py -v
# Expected: All tests PASS
```

**Dependencies:** Tasks 3-5

---

### Task 8: Add Integration Tests
**Estimate:** 1.5h (90 minutes)

**What:**
- Create `tests/integration/test_scheduler_monitoring.py`
- Test full flow: scheduler write → DB → dashboard read

**Tests:**
1. `test_scheduler_writes_and_dashboard_reads()` — End-to-end
2. `test_multiple_cycles_upsert()` — Verify single row per env
3. `test_missing_table_graceful()` — Dashboard doesn't crash

**How to validate:**
```bash
pytest tests/integration/test_scheduler_monitoring.py -v
```

**Dependencies:** Tasks 3-6

---

### Task 9: Manual Testing and Documentation
**Estimate:** 1h (60 minutes)

**What:**
- Run manual test procedure (see AC document)
- Update README if needed
- Add screenshots to docs (optional)
- Verify all ACs pass

**Manual test checklist:**
1. Run scheduler, verify heartbeat in DB
2. View dashboard, verify status card
3. Stop scheduler 3h, verify stale status
4. Simulate error, verify error display
5. Test missing table, verify graceful handling

**How to validate:**
- All ACs checked off
- Screenshots captured

**Dependencies:** Tasks 1-8

---

## Total Estimate

**Total: 12 hours** (720 minutes)

**Breakdown:**
- Database layer: 3.5h (Tasks 1-2, 4)
- Scheduler integration: 2h (Task 3)
- UI implementation: 3h (Tasks 5-6)
- Testing: 3.5h (Tasks 7-9)

**Note:** Original estimate was 10h (600 min). Planning identified more comprehensive testing needs, increasing to 12h.

---

## Execution Order

### Phase 1: Database Layer (Day 1 AM)
1. **Task 1** — Model (1h)
2. **Task 2** — Migration (0.5h)

**Total Phase 1:** 1.5 hours

---

### Phase 2: Backend Integration (Day 1 PM)
3. **Task 3** — Scheduler heartbeat writing (2h)
4. **Task 4** — DB service methods (1h)

**Total Phase 2:** 3 hours

---

### Phase 3: UI Implementation (Day 2 AM)
5. **Task 5** — Paper Trading view (2.5h)
6. **Task 6** — App.py integration (0.5h)

**Total Phase 3:** 3 hours

---

### Phase 4: Testing & Validation (Day 2 PM)
7. **Task 7** — Unit tests (2h)
8. **Task 8** — Integration tests (1.5h)
9. **Task 9** — Manual testing (1h)

**Total Phase 4:** 4.5 hours

---

## Risks & Mitigations

### Risk 1: Migration Conflicts
**Description:** Alembic migration conflicts with pending migrations

**Mitigation:**
- Pull latest main before creating migration
- Review auto-generated script carefully
- Test upgrade/downgrade

**Probability:** Low  
**Impact:** Low (easy to fix)

---

### Risk 2: Heartbeat Write Performance
**Description:** DB writes slow down scheduler

**Mitigation:**
- Use upsert (single UPDATE, not INSERT)
- Wrap in try/except (continue on failure)
- Monitor latency in testing

**Probability:** Very Low  
**Impact:** Low

---

### Risk 3: Streamlit Tab Ordering
**Description:** Adding tab shifts other tab indexes in app.py

**Mitigation:**
- Carefully update all `with tabs[N]` blocks
- Test all tabs after change
- Consider using dict-based tab routing (future)

**Probability:** Medium  
**Impact:** Low (easy to spot and fix)

---

### Risk 4: Missing Alembic History
**Description:** Production DB has different migration state

**Mitigation:**
- Document migration ID
- Test on staging first
- Provide rollback instructions

**Probability:** Low  
**Impact:** Medium

---

## Testing Strategy Summary

### Unit Tests (Task 7)
- Test scheduler methods in isolation
- Mock DB to test error handling
- Test status calculation logic
- Fast, no real DB needed

### Integration Tests (Task 8)
- Test full flow with real DB (test fixture)
- Verify upsert behavior
- Test missing table gracefully

### Manual Tests (Task 9)
- Run scheduler and dashboard together
- Verify UI updates
- Test edge cases (stale, stopped)

---

## Rollback Plan

### Option 1: Feature Flag
```python
# settings.py
ENABLE_SCHEDULER_HEARTBEAT = os.getenv("ENABLE_SCHEDULER_HEARTBEAT", "true") == "true"

# scheduler.py
if settings.ENABLE_SCHEDULER_HEARTBEAT:
    self._upsert_heartbeat_start(...)
```

Set env var to disable:
```bash
export ENABLE_SCHEDULER_HEARTBEAT=false
```

---

### Option 2: Revert Migration
```bash
# Downgrade to remove table
alembic downgrade -1

# Or drop manually
psql -c "DROP TABLE scheduler_heartbeats;"
```

Dashboard will show "No heartbeat found" warning (graceful).

---

### Option 3: Remove Tab
Comment out tab in `app.py`:
```python
tabs = st.tabs([
    "Dashboard",
    # ... other tabs ...
    # "Paper Trading",  # Temporarily disabled
    "Logs",
])
```

---

## Success Criteria

- [ ] SchedulerHeartbeat model added ✓
- [ ] Migration created and tested ✓
- [ ] Scheduler writes heartbeat on each cycle ✓
- [ ] DB service methods implemented ✓
- [ ] Paper Trading tab added ✓
- [ ] Status card shows active/stale/stopped ✓
- [ ] Recent runs table displays ✓
- [ ] All 12 ACs pass ✓
- [ ] 8+ unit tests pass ✓
- [ ] 3+ integration tests pass ✓
- [ ] Manual testing complete ✓

---

## Dependencies

**External:**
- PostgreSQL database (SQLite not required for this feature)
- Alembic for migrations

**Internal:**
- TradingScheduler must be running for heartbeat to populate
- Streamlit dashboard must have DB access

---

## Post-Completion Tasks

1. Monitor scheduler logs for heartbeat write errors
2. Verify dashboard loads quickly with heartbeat queries
3. Gather user feedback on status thresholds (2h/24h)
4. Consider adding:
   - Historical charts (future)
   - Email alerts on "stopped" status (future)
   - Start/stop controls (separate issue)

---

## Final Checklist

**Before starting implementation:**
- [ ] Review current scheduler.py structure
- [ ] Review current app.py tab structure
- [ ] Verify Alembic setup is working
- [ ] Understand upsert pattern (single row per environment)

**During implementation:**
- [ ] Task 1: Model ✓
- [ ] Task 2: Migration ✓
- [ ] Task 3: Scheduler integration ✓
- [ ] Task 4: DB services ✓
- [ ] Task 5: UI view ✓
- [ ] Task 6: App integration ✓
- [ ] Task 7: Unit tests ✓
- [ ] Task 8: Integration tests ✓
- [ ] Task 9: Manual testing ✓

**After implementation:**
- [ ] All tests pass
- [ ] Code review complete
- [ ] Migration tested on staging
- [ ] Merge to main
- [ ] Monitor production
- [ ] Close Beads issue
