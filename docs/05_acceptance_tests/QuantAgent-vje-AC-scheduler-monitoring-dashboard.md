# QuantAgent-vje — Acceptance Criteria: Scheduler Status and Controls in Streamlit Dashboard

**Issue ID:** QuantAgent-vje  
**Title:** Scheduler status and controls in Streamlit Dashboard  
**Type:** Feature

---

## Acceptance Criteria

### AC-1: SchedulerHeartbeat Table Exists

**Given** database migrations have been run  
**When** querying the database schema  
**Then**:
- Table `scheduler_heartbeats` exists
- Columns: id, timestamp, completed_at, status, environment, assets, stats, last_trade_id, error_message
- Indexes on timestamp and environment exist

**Verification:**
```sql
\d scheduler_heartbeats  -- PostgreSQL
SELECT * FROM scheduler_heartbeats LIMIT 1;
```

---

### AC-2: Scheduler Writes Heartbeat on Cycle Start

**Given** TradingScheduler is running  
**When** `analyze_and_trade()` starts  
**Then**:
- Heartbeat row created or updated with status="running"
- `timestamp` = cycle start time
- `environment` = scheduler's environment

**Verification:**
```python
def test_scheduler_writes_heartbeat_start():
    scheduler.analyze_and_trade()
    hb = db.query(SchedulerHeartbeat).filter_by(
        environment=Environment.PAPER
    ).first()
    assert hb is not None
    assert hb.status == "running"
```

---

### AC-3: Scheduler Updates Heartbeat on Cycle Complete

**Given** TradingScheduler completes a cycle  
**When** `analyze_and_trade()` finishes successfully  
**Then**:
- Heartbeat updated with status="completed"
- `completed_at` timestamp set
- `stats` contains: processed, errors, duration_seconds, total
- `assets` list populated

**Verification:**
```python
def test_scheduler_updates_heartbeat_complete():
    scheduler.analyze_and_trade()
    hb = db.query(SchedulerHeartbeat).first()
    assert hb.status == "completed"
    assert hb.completed_at is not None
    assert hb.stats["processed"] >= 0
    assert hb.assets is not None
```

---

### AC-4: Scheduler Records Error in Heartbeat

**Given** TradingScheduler encounters an error  
**When** `analyze_and_trade()` raises exception  
**Then**:
- Heartbeat updated with status="error"
- `error_message` contains exception details
- `completed_at` timestamp set

**Verification:**
```python
def test_scheduler_records_error():
    # Mock data provider to raise error
    mock_data_provider.side_effect = Exception("Test error")
    
    with pytest.raises(Exception):
        scheduler.analyze_and_trade()
    
    hb = db.query(SchedulerHeartbeat).first()
    assert hb.status == "error"
    assert "Test error" in hb.error_message
```

---

### AC-5: Streamlit Shows Paper Trading Tab

**Given** Streamlit app is running  
**When** user opens the UI  
**Then**:
- "Paper Trading" tab is visible in the tab list
- Tab is positioned between "Orders & Positions" and "Logs"

**Verification:** Manual test — navigate to Streamlit, verify tab exists

---

### AC-6: Status Card Shows Active

**Given** scheduler last ran 30 minutes ago  
**And** heartbeat status="completed"  
**When** viewing Paper Trading tab  
**Then**:
- Status displays: 🟢 Active
- Last run shows: "30m ago" (or similar)
- Cycle stats displayed: "X/Y assets, Z errors (N.Ns)"

**Verification:**
```python
def test_status_card_shows_active():
    heartbeat = create_heartbeat(timestamp=datetime.utcnow() - timedelta(minutes=30))
    status, emoji, _ = _calculate_status(heartbeat)
    assert status == "active"
    assert emoji == "🟢"
```

---

### AC-7: Status Card Shows Stale

**Given** scheduler last ran 5 hours ago  
**When** viewing Paper Trading tab  
**Then**:
- Status displays: 🟡 Stale
- Last run shows: "5h ago"
- Warning message displayed

**Verification:**
```python
def test_status_card_shows_stale():
    heartbeat = create_heartbeat(timestamp=datetime.utcnow() - timedelta(hours=5))
    status, emoji, _ = _calculate_status(heartbeat)
    assert status == "stale"
    assert emoji == "🟡"
```

---

### AC-8: Status Card Shows Stopped

**Given** scheduler last ran 25 hours ago (or no heartbeat exists)  
**When** viewing Paper Trading tab  
**Then**:
- Status displays: 🔴 Stopped
- Message: "No recent scheduler activity"

**Verification:**
```python
def test_status_card_shows_stopped():
    heartbeat = create_heartbeat(timestamp=datetime.utcnow() - timedelta(hours=25))
    status, emoji, _ = _calculate_status(heartbeat)
    assert status == "stopped"
    assert emoji == "🔴"
```

---

### AC-9: Recent Runs Table Displays

**Given** 10 heartbeat records exist  
**When** viewing Paper Trading tab  
**Then**:
- Table shows up to 10 rows
- Columns: Timestamp, Duration, Assets, Errors, Status
- Rows sorted by timestamp (newest first)
- Duration calculated correctly (completed_at - timestamp)

**Verification:** Manual test — view dashboard with multiple heartbeats

---

### AC-10: Graceful Handling of Missing Table

**Given** `scheduler_heartbeats` table doesn't exist (migrations not run)  
**When** viewing Paper Trading tab  
**Then**:
- No Python exception/crash
- Warning message displayed: "No scheduler heartbeat found. Scheduler may not be running or table missing."
- Dashboard remains functional

**Verification:**
```python
def test_missing_table_graceful():
    # Drop table
    op.drop_table('scheduler_heartbeats')
    
    # Access dashboard
    heartbeat = db.get_latest_heartbeat("paper")
    assert heartbeat is None  # Not an exception
```

---

### AC-11: Heartbeat Write Failure Doesn't Crash Scheduler

**Given** database write fails (e.g., connection lost)  
**When** scheduler attempts to write heartbeat  
**Then**:
- Exception caught and logged
- Scheduler continues executing trading logic
- Trading is NOT affected by monitoring failure

**Verification:**
```python
def test_heartbeat_failure_graceful():
    # Mock db.commit to raise exception
    mock_db.commit.side_effect = Exception("DB error")
    
    # Should not raise
    stats = scheduler.analyze_and_trade()
    
    # Trading logic executed
    assert stats["processed"] > 0
```

---

### AC-12: DB Service Methods Return Correct Data

**Given** heartbeats exist in database  
**When** calling `db.get_latest_heartbeat("paper")`  
**Then** returns dict with all heartbeat fields

**When** calling `db.get_recent_heartbeats("paper", limit=10)`  
**Then** returns list of up to 10 heartbeat dicts

**Verification:**
```python
def test_db_service_methods():
    # Create heartbeats
    create_heartbeat(environment="paper")
    
    # Test get_latest
    latest = db.get_latest_heartbeat("paper")
    assert latest is not None
    assert latest["environment"] == "paper"
    
    # Test get_recent
    recent = db.get_recent_heartbeats("paper", limit=10)
    assert len(recent) <= 10
    assert recent[0]["timestamp"] >= recent[-1]["timestamp"]  # Sorted desc
```

---

## Edge Cases

### Edge Case 1: No Heartbeat for Environment
**Given** scheduler never ran for environment  
**When** querying heartbeat  
**Then** returns None, UI shows "No activity"

---

### Edge Case 2: Scheduler Running (Status=Running)
**Given** scheduler cycle in progress (status="running")  
**When** viewing dashboard  
**Then** shows "Running..." with in-progress indicator

---

### Edge Case 3: Multiple Environments
**Given** heartbeats exist for "paper" and "backtest"  
**When** filtering by environment="paper"  
**Then** only paper heartbeats returned

---

### Edge Case 4: Very Long Cycle Duration
**Given** cycle took 10 minutes (abnormal)  
**When** displaying in table  
**Then** shows "600.0s" (no crash on large values)

---

## Performance Criteria

### Perf-1: Heartbeat Write Latency
- Heartbeat write adds < 50ms per cycle
- Uses single UPDATE query (upsert pattern)

### Perf-2: Dashboard Query Latency
- `get_latest_heartbeat()` < 100ms (indexed query)
- `get_recent_heartbeats()` < 200ms (indexed, limited)

### Perf-3: Dashboard Load Time
- Paper Trading tab renders in < 1 second

---

## Manual Test Procedure

### Test 1: Full Cycle Monitoring

1. **Setup:**
   - Run migrations: `alembic upgrade head`
   - Clear scheduler_heartbeats: `DELETE FROM scheduler_heartbeats;`
   - Start scheduler: `python apps/paper_trading.py --interval-hours 0.1`

2. **Verify Heartbeat Writing:**
   - Wait for 1 cycle (6 minutes)
   - Query DB: `SELECT * FROM scheduler_heartbeats;`
   - Expected: 1 row, status="completed", stats populated

3. **Verify Dashboard Display:**
   - Open Streamlit: `streamlit run apps/streamlit/app.py`
   - Navigate to "Paper Trading" tab
   - Expected: Status card shows 🟢 Active, last run "X min ago"

4. **Test Stale Status:**
   - Stop scheduler
   - Wait 3 hours (or manually update timestamp)
   - Refresh dashboard
   - Expected: Status shows 🟡 Stale

5. **Test Stopped Status:**
   - Wait 25 hours (or manually update timestamp)
   - Refresh dashboard
   - Expected: Status shows 🔴 Stopped

---

### Test 2: Error Handling

1. **Simulate Scheduler Error:**
   - Modify data provider to raise exception
   - Run scheduler
   - Verify: Heartbeat status="error", error_message populated

2. **Simulate DB Write Failure:**
   - Disconnect database during cycle
   - Verify: Scheduler continues, logs warning

3. **Simulate Missing Table:**
   - Downgrade migration: `alembic downgrade -1`
   - Open dashboard
   - Verify: Warning message, no crash

---

## Definition of Done Checklist

- [ ] SchedulerHeartbeat model added to models.py ✓
- [ ] Migration created and tested ✓
- [ ] Scheduler writes heartbeat on cycle start ✓
- [ ] Scheduler updates heartbeat on cycle complete ✓
- [ ] Scheduler records errors in heartbeat ✓
- [ ] DB service methods implemented ✓
- [ ] Paper Trading tab added to app.py ✓
- [ ] Status card renders with correct status ✓
- [ ] Recent runs table displays ✓
- [ ] Graceful handling of missing table ✓
- [ ] All 12 ACs pass ✓
- [ ] Manual test procedure executed ✓
- [ ] Performance criteria met ✓
- [ ] Edge cases handled ✓
