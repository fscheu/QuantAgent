# QuantAgent-vje — Requirements: Scheduler Status and Controls in Streamlit Dashboard

**Issue ID:** QuantAgent-vje  
**Title:** Scheduler status and controls in Streamlit Dashboard  
**Type:** Feature  
**Priority:** 2  
**Labels:** monitoring, streamlit  
**Estimated:** 600 minutes (10 hours)

---

## Objective

Add scheduler monitoring to the Streamlit dashboard so operators can see when the paper trading scheduler last ran, its current status, and recent trading activity — without checking logs manually.

---

## Background

### Current State (Problem)

The TradingScheduler runs as a separate process (`apps/paper_trading.py`). The Streamlit dashboard has **zero visibility** into scheduler health:

**Gaps:**
1. **No status visibility**: Can't tell if scheduler is running, when it last executed, or if it's stale
2. **No recent activity**: Can't see recent trades or signals from the dashboard
3. **No shared state**: Scheduler and dashboard are isolated processes with no communication
4. **Manual log checking**: Operators must SSH and grep logs to understand scheduler state

**Current workaround:** Check logs manually:
```bash
grep "scheduler.cycle_complete" quantagent.log | tail -1
```

---

## Scope

### In Scope

1. **SchedulerHeartbeat Model**
   - New database table to track scheduler runs
   - Columns: timestamp, status, assets, last_trade_id, stats (JSON)
   - Written by scheduler on each tick

2. **Scheduler Heartbeat Writing**
   - Modify `TradingScheduler.analyze_and_trade()` to write heartbeat
   - Record: cycle start/end timestamp, stats (processed, errors, duration)
   - Update existing heartbeat row (or insert if none)

3. **Streamlit Paper Trading Tab**
   - New tab in `app.py`: "Paper Trading"
   - Status card: last run time, status (active/stale/stopped), next run estimate
   - Recent runs table: last 10 scheduler cycles with timestamps, stats
   - Quick link to latest trades

4. **DB Service Queries**
   - Add `get_latest_heartbeat()` to `services/db.py`
   - Add `get_recent_heartbeats(limit=10)` to `services/db.py`

5. **Status Logic**
   - **Active**: Last heartbeat < 2 hours ago
   - **Stale**: Last heartbeat 2-24 hours ago
   - **Stopped**: No heartbeat in > 24 hours

### Out of Scope

- Starting/stopping scheduler from UI (separate infra issue)
- Real-time streaming updates (polling is sufficient)
- Historical charts/metrics (future enhancement)
- Scheduler configuration editing from UI
- Multi-scheduler support (assumes single instance)

---

## Requirements

### FR-1: SchedulerHeartbeat Database Model

**Description:** New model to persist scheduler execution records

**Requirements:**
- Table: `scheduler_heartbeats`
- Columns:
  - `id` (Integer, primary key)
  - `timestamp` (DateTime, not null) — When cycle started
  - `completed_at` (DateTime, nullable) — When cycle finished
  - `status` (String, not null) — "running", "completed", "error"
  - `environment` (Enum, not null) — "backtest", "paper", "prod"
  - `assets` (JSON, nullable) — List of assets processed
  - `stats` (JSON, nullable) — Cycle stats: {processed, errors, duration_seconds, total}
  - `last_trade_id` (Integer, nullable, FK to trades) — Most recent trade created
  - `error_message` (Text, nullable) — Error details if status="error"

**Indexes:**
- Primary key on `id`
- Index on `timestamp` (for recent queries)
- Index on `environment`

---

### FR-2: Scheduler Writes Heartbeat

**Description:** TradingScheduler updates heartbeat on each cycle

**Requirements:**
- In `analyze_and_trade()`:
  - Before cycle: Create or update heartbeat with status="running"
  - After cycle: Update same heartbeat with status="completed", stats, completed_at
  - On error: Update with status="error", error_message
- Use upsert pattern: one heartbeat row per environment (update existing, not insert new)
- Commit after updating (separate transaction from trading logic)

**Heartbeat update flow:**
```python
# Start of cycle
heartbeat = db.query(SchedulerHeartbeat).filter_by(
    environment=self.environment
).first()
if not heartbeat:
    heartbeat = SchedulerHeartbeat(environment=self.environment, ...)
heartbeat.timestamp = datetime.utcnow()
heartbeat.status = "running"
db.add(heartbeat)
db.commit()

# ... trading logic ...

# End of cycle
heartbeat.completed_at = datetime.utcnow()
heartbeat.status = "completed"
heartbeat.stats = {"processed": ..., "errors": ..., "duration_seconds": ...}
db.commit()
```

---

### FR-3: Streamlit Paper Trading Tab

**Description:** New dashboard tab showing scheduler status and activity

**Requirements:**

**3.1 Status Card**
- Display:
  - Current status: 🟢 Active / 🟡 Stale / 🔴 Stopped
  - Last run: "2 minutes ago" (humanized timestamp)
  - Next run: Estimated based on interval_hours
  - Cycle stats: "5/5 assets, 2 trades, 0 errors (3.2s)"
- Status logic:
  - **Active (🟢)**: Last heartbeat < 2 hours ago
  - **Stale (🟡)**: Last heartbeat 2-24 hours ago
  - **Stopped (🔴)**: No heartbeat in > 24 hours OR no heartbeat row exists
- Refresh: Manual (Streamlit auto-refresh on interaction)

**3.2 Recent Runs Table**
- Show last 10 heartbeats
- Columns: Timestamp, Duration, Assets, Trades, Errors, Status
- Sortable by timestamp (newest first)
- Clickable trade IDs → navigate to Orders & Positions tab

**3.3 Quick Links**
- "View Latest Trades" → Orders & Positions tab, filtered by environment
- "View Logs" → Logs tab, filtered by "scheduler" events

---

### FR-4: DB Service Methods

**Description:** Add heartbeat queries to Streamlit backend

**Requirements:**

**4.1 `get_latest_heartbeat(environment: str) -> Optional[dict]`**
- Query: `SELECT * FROM scheduler_heartbeats WHERE environment=? ORDER BY timestamp DESC LIMIT 1`
- Return: dict with all columns or None

**4.2 `get_recent_heartbeats(environment: str, limit: int = 10) -> List[dict]`**
- Query: `SELECT * FROM scheduler_heartbeats WHERE environment=? ORDER BY timestamp DESC LIMIT ?`
- Return: list of dicts

**4.3 Handle missing table gracefully**
- If table doesn't exist: return None / empty list
- Show message in UI: "Scheduler monitoring not available (table missing)"

---

### FR-5: Status Calculation Logic

**Description:** Derive scheduler status from heartbeat timestamp

**Requirements:**

```python
def calculate_status(last_heartbeat: Optional[dict]) -> str:
    if not last_heartbeat:
        return "stopped"
    
    timestamp = last_heartbeat["timestamp"]
    age_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
    
    if age_hours < 2:
        return "active"
    elif age_hours < 24:
        return "stale"
    else:
        return "stopped"
```

**Status meanings:**
- **active**: Scheduler is healthy, running on schedule
- **stale**: Scheduler hasn't run recently (possible issue)
- **stopped**: Scheduler is down or not configured

---

## Acceptance Criteria

### AC-1: SchedulerHeartbeat Model Exists
**Given** the database schema is up to date  
**When** querying the database  
**Then** `scheduler_heartbeats` table exists with all required columns

---

### AC-2: Scheduler Writes Heartbeat
**Given** TradingScheduler is running  
**When** a cycle completes  
**Then**:
- Heartbeat row updated with status="completed"
- `completed_at` timestamp is set
- `stats` contains: processed, errors, duration_seconds, total

---

### AC-3: Streamlit Shows Paper Trading Tab
**Given** Streamlit app is running  
**When** user navigates to UI  
**Then** "Paper Trading" tab is visible in the tab list

---

### AC-4: Status Card Shows Active
**Given** scheduler last ran 30 minutes ago  
**When** viewing Paper Trading tab  
**Then**:
- Status shows: 🟢 Active
- Last run: "30 minutes ago"
- Cycle stats displayed

---

### AC-5: Status Card Shows Stale
**Given** scheduler last ran 5 hours ago  
**When** viewing Paper Trading tab  
**Then**:
- Status shows: 🟡 Stale
- Last run: "5 hours ago"
- Warning message: "Scheduler hasn't run recently"

---

### AC-6: Status Card Shows Stopped
**Given** no heartbeat in 25 hours (or no heartbeat row)  
**When** viewing Paper Trading tab  
**Then**:
- Status shows: 🔴 Stopped
- Message: "No recent scheduler activity"

---

### AC-7: Recent Runs Table Displays
**Given** 10 heartbeat records exist  
**When** viewing Paper Trading tab  
**Then**:
- Table shows 10 rows
- Columns: Timestamp, Duration, Assets, Trades, Errors, Status
- Rows sorted by timestamp (newest first)

---

### AC-8: Graceful Handling of Missing Table
**Given** `scheduler_heartbeats` table doesn't exist  
**When** viewing Paper Trading tab  
**Then**:
- No error/crash
- Message: "Scheduler monitoring not available (run migrations)"

---

## Constraints

- **Read-only UI**: No start/stop controls (out of scope)
- **Single scheduler assumption**: One scheduler per environment
- **Polling-based**: No real-time websockets (Streamlit limitation)
- **PostgreSQL only**: SQLite not required for this feature

---

## Non-Functional Requirements

### NFR-1: Performance
- Heartbeat write adds < 50ms per scheduler cycle
- Dashboard queries < 500ms (simple indexed queries)

### NFR-2: Reliability
- Heartbeat write failures don't crash scheduler
- Dashboard degradation doesn't affect trading

### NFR-3: Observability
- Heartbeat writes logged (debug level)
- Dashboard errors logged (error level)

---

## Definition of Done

- [ ] SchedulerHeartbeat model added to models.py
- [ ] Migration created for new table
- [ ] TradingScheduler writes heartbeat on each cycle
- [ ] Streamlit has "Paper Trading" tab
- [ ] Status card shows active/stale/stopped correctly
- [ ] Recent runs table displays last 10 heartbeats
- [ ] DB service methods implemented
- [ ] Graceful handling of missing table
- [ ] All ACs pass
- [ ] Manual testing confirms UI updates
