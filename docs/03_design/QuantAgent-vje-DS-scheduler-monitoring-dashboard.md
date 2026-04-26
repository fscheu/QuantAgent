# QuantAgent-vje — Design: Scheduler Status and Controls in Streamlit Dashboard

**Issue ID:** QuantAgent-vje  
**Title:** Scheduler status and controls in Streamlit Dashboard  
**Type:** Feature

---

## Design Overview

Add scheduler monitoring by creating a heartbeat mechanism: scheduler writes execution records to a database table, Streamlit reads and displays them with status indicators.

---

## Architecture Changes

### Current Architecture (Before)

```
┌─────────────────────┐       ┌──────────────────┐
│ TradingScheduler    │       │ Streamlit UI     │
│ (apps/paper_trading)│       │ (apps/streamlit) │
│                     │       │                  │
│ - Runs every Nh     │       │ - Shows trades   │
│ - Writes trades     │       │ - Shows signals  │
│ - Logs to file      │       │ - No scheduler   │
│                     │       │   visibility     │
└──────────┬──────────┘       └────────┬─────────┘
           │                           │
           └───────────┬───────────────┘
                       │
                  ┌────▼──────┐
                  │ Database  │
                  │ (trades,  │
                  │  signals) │
                  └───────────┘
```

**Gap:** No shared state between processes.

---

### New Architecture (After)

```
┌─────────────────────┐       ┌──────────────────┐
│ TradingScheduler    │       │ Streamlit UI     │
│                     │       │                  │
│ analyze_and_trade() │       │ Paper Trading tab│
│   ├─ Write heartbeat│       │   ├─ Read hb     │
│   ├─ Process assets │       │   ├─ Calculate   │
│   └─ Update stats   │       │   │   status     │
│                     │       │   └─ Display     │
└──────────┬──────────┘       └────────┬─────────┘
           │ Writes                     │ Reads
           │ heartbeat                  │ heartbeat
           └──────────┬─────────────────┘
                      │
              ┌───────▼───────────┐
              │ Database          │
              │ ┌───────────────┐ │
              │ │ scheduler_hb  │ │
              │ │ (new table)   │ │
              │ └───────────────┘ │
              │ ┌───────────────┐ │
              │ │ trades        │ │
              │ │ signals       │ │
              │ └───────────────┘ │
              └───────────────────┘
```

**Solution:** Shared database table for heartbeat.

---

## Implementation Details

### 1. SchedulerHeartbeat Model

**File:** `quantagent/models.py`

```python
class SchedulerHeartbeat(Base):
    """Track TradingScheduler execution cycles."""
    
    __tablename__ = "scheduler_heartbeats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False)  # running, completed, error
    environment = Column(Enum(Environment), nullable=False, index=True)
    assets = Column(JSON, nullable=True)  # List of symbols processed
    stats = Column(JSON, nullable=True)   # {processed, errors, duration_seconds, total}
    last_trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    error_message = Column(Text, nullable=True)
    
    __table_args__ = (
        Index("idx_heartbeat_env_ts", "environment", "timestamp"),
    )
```

**Design choice: Single row per environment (upsert pattern)**
- Rationale: Simplifies queries ("get latest heartbeat" = single row lookup)
- Alternative considered: Insert new row each cycle
  - Rejected: Grows table unbounded, requires cleanup job

**Design choice: Status field values**
- "running": Cycle started, not yet complete
- "completed": Cycle finished successfully
- "error": Cycle failed with exception
- Rationale: Simple state machine, easy to query

---

### 2. Scheduler Heartbeat Writing

**File:** `quantagent/trading/scheduler.py`

**Modification location:** `analyze_and_trade()` method

```python
def analyze_and_trade(self) -> Dict[str, float]:
    cycle_start = datetime.utcnow()
    
    # NEW: Write heartbeat at cycle start
    heartbeat = self._upsert_heartbeat_start(cycle_start)
    
    processed = 0
    errors = 0
    
    # Existing code: process assets
    for symbol in self.config.assets:
        try:
            self._process_asset(symbol)
            processed += 1
        except ...:
            errors += 1
            # ... existing error handling ...
    
    duration = (datetime.utcnow() - cycle_start).total_seconds()
    stats = {
        "processed": processed,
        "errors": errors,
        "duration_seconds": duration,
        "total": len(self.config.assets),
    }
    self.last_run_stats = stats
    
    # NEW: Update heartbeat at cycle end
    self._upsert_heartbeat_complete(heartbeat, stats)
    
    # Existing logging
    logger.info(...)
    return stats
```

**Helper method 1: Start heartbeat**
```python
def _upsert_heartbeat_start(self, timestamp: datetime) -> SchedulerHeartbeat:
    """Create or update heartbeat for cycle start."""
    from quantagent.models import SchedulerHeartbeat
    
    heartbeat = self.db.query(SchedulerHeartbeat).filter_by(
        environment=self.environment
    ).first()
    
    if not heartbeat:
        heartbeat = SchedulerHeartbeat(
            environment=self.environment,
            timestamp=timestamp,
            status="running",
        )
        self.db.add(heartbeat)
    else:
        heartbeat.timestamp = timestamp
        heartbeat.status = "running"
        heartbeat.completed_at = None
        heartbeat.error_message = None
    
    try:
        self.db.commit()
    except Exception as exc:
        logger.warning("Failed to write heartbeat: %s", exc)
        self.db.rollback()
        # Continue execution (heartbeat is nice-to-have)
    
    return heartbeat
```

**Helper method 2: Complete heartbeat**
```python
def _upsert_heartbeat_complete(
    self, heartbeat: SchedulerHeartbeat, stats: Dict[str, float]
) -> None:
    """Update heartbeat with cycle completion details."""
    heartbeat.completed_at = datetime.utcnow()
    heartbeat.status = "completed"
    heartbeat.stats = stats
    heartbeat.assets = self.config.assets
    
    # Optionally link last trade
    last_trade = self.db.query(Trade).filter_by(
        environment=self.environment
    ).order_by(Trade.created_at.desc()).first()
    if last_trade:
        heartbeat.last_trade_id = last_trade.id
    
    try:
        self.db.commit()
    except Exception as exc:
        logger.warning("Failed to update heartbeat: %s", exc)
        self.db.rollback()
```

**Error handling:**
```python
# In analyze_and_trade(), wrap with try/except:
try:
    # ... process assets ...
except Exception as exc:
    heartbeat.status = "error"
    heartbeat.error_message = str(exc)
    heartbeat.completed_at = datetime.utcnow()
    try:
        self.db.commit()
    except:
        pass  # Best effort
    raise  # Re-raise for existing error handling
```

---

### 3. Streamlit DB Service Methods

**File:** `apps/streamlit/services/db.py`

Add methods to DB handle class:

```python
def get_latest_heartbeat(self, environment: str) -> Optional[dict]:
    """Get most recent scheduler heartbeat for environment."""
    if not self.ok:
        return None
    
    try:
        from quantagent.models import SchedulerHeartbeat, Environment
        
        hb = self.session.query(SchedulerHeartbeat).filter_by(
            environment=Environment(environment)
        ).order_by(SchedulerHeartbeat.timestamp.desc()).first()
        
        if not hb:
            return None
        
        return {
            "id": hb.id,
            "timestamp": hb.timestamp,
            "completed_at": hb.completed_at,
            "status": hb.status,
            "environment": hb.environment.value,
            "assets": hb.assets or [],
            "stats": hb.stats or {},
            "last_trade_id": hb.last_trade_id,
            "error_message": hb.error_message,
        }
    except Exception as exc:
        # Table might not exist (migrations not run)
        logger.debug("Failed to query heartbeat: %s", exc)
        return None


def get_recent_heartbeats(
    self, environment: str, limit: int = 10
) -> List[dict]:
    """Get recent scheduler heartbeats."""
    if not self.ok:
        return []
    
    try:
        from quantagent.models import SchedulerHeartbeat, Environment
        
        heartbeats = self.session.query(SchedulerHeartbeat).filter_by(
            environment=Environment(environment)
        ).order_by(SchedulerHeartbeat.timestamp.desc()).limit(limit).all()
        
        return [
            {
                "id": hb.id,
                "timestamp": hb.timestamp,
                "completed_at": hb.completed_at,
                "status": hb.status,
                "stats": hb.stats or {},
                "error_message": hb.error_message,
            }
            for hb in heartbeats
        ]
    except Exception:
        return []
```

---

### 4. Streamlit Paper Trading Tab

**File:** `apps/streamlit/views/paper_trading.py` (NEW FILE)

```python
"""Paper Trading tab: scheduler status and recent activity."""

from datetime import datetime, timedelta
from typing import Optional

import streamlit as st


def render(db, environment: str) -> None:
    """Render Paper Trading monitoring tab."""
    st.header("📊 Paper Trading Scheduler")
    
    if not db.ok:
        st.error("Database not available. Cannot display scheduler status.")
        return
    
    # Get latest heartbeat
    heartbeat = db.get_latest_heartbeat(environment)
    
    if not heartbeat:
        st.warning("No scheduler heartbeat found. Scheduler may not be running or table missing.")
        return
    
    # Status card
    _render_status_card(heartbeat)
    
    st.divider()
    
    # Recent runs
    st.subheader("Recent Scheduler Runs")
    recent = db.get_recent_heartbeats(environment, limit=10)
    _render_recent_runs(recent)
    
    st.divider()
    
    # Quick links
    st.subheader("Quick Links")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 View Latest Trades"):
            # TODO: Navigate to Orders & Positions tab
            st.info("Navigate to 'Orders & Positions' tab")
    with col2:
        if st.button("📋 View Scheduler Logs"):
            # TODO: Navigate to Logs tab
            st.info("Navigate to 'Logs' tab")


def _render_status_card(heartbeat: dict) -> None:
    """Render scheduler status card."""
    status, emoji, color = _calculate_status(heartbeat)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Status", f"{emoji} {status.title()}")
    
    with col2:
        last_run = _humanize_time(heartbeat["timestamp"])
        st.metric("Last Run", last_run)
    
    with col3:
        next_run = _estimate_next_run(heartbeat)
        st.metric("Next Run (est.)", next_run)
    
    # Cycle stats
    stats = heartbeat.get("stats", {})
    if stats:
        st.caption(
            f"📊 Last cycle: {stats.get('processed', 0)}/{stats.get('total', 0)} assets, "
            f"{stats.get('errors', 0)} errors ({stats.get('duration_seconds', 0):.1f}s)"
        )
    
    # Error message if present
    if heartbeat.get("error_message"):
        st.error(f"⚠️ Last cycle error: {heartbeat['error_message']}")


def _render_recent_runs(recent: list[dict]) -> None:
    """Render table of recent scheduler runs."""
    if not recent:
        st.info("No recent runs recorded.")
        return
    
    # Format data for table
    rows = []
    for hb in recent:
        timestamp = hb["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        duration = _calculate_duration(hb)
        stats = hb.get("stats", {})
        processed = f"{stats.get('processed', 0)}/{stats.get('total', 0)}"
        errors = stats.get('errors', 0)
        status_emoji = "✅" if hb["status"] == "completed" else "❌"
        
        rows.append({
            "Timestamp": timestamp,
            "Duration": duration,
            "Assets": processed,
            "Errors": errors,
            "Status": f"{status_emoji} {hb['status']}",
        })
    
    st.table(rows)


def _calculate_status(heartbeat: dict) -> tuple[str, str, str]:
    """Calculate scheduler status from heartbeat age."""
    timestamp = heartbeat["timestamp"]
    age_seconds = (datetime.utcnow() - timestamp).total_seconds()
    age_hours = age_seconds / 3600
    
    if age_hours < 2:
        return ("active", "🟢", "green")
    elif age_hours < 24:
        return ("stale", "🟡", "orange")
    else:
        return ("stopped", "🔴", "red")


def _humanize_time(dt: datetime) -> str:
    """Convert datetime to human-readable relative time."""
    delta = datetime.utcnow() - dt
    seconds = delta.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)}s ago"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    elif seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    else:
        return f"{int(seconds / 86400)}d ago"


def _estimate_next_run(heartbeat: dict) -> str:
    """Estimate next scheduler run time."""
    # Assumes interval is available (would need to be stored in heartbeat or config)
    # For MVP: just show "Unknown" or estimate from past runs
    return "Unknown"  # TODO: Store interval in heartbeat


def _calculate_duration(hb: dict) -> str:
    """Calculate cycle duration."""
    if not hb.get("completed_at"):
        return "N/A"
    
    start = hb["timestamp"]
    end = hb["completed_at"]
    duration = (end - start).total_seconds()
    return f"{duration:.1f}s"
```

---

### 5. App.py Integration

**File:** `apps/streamlit/app.py`

**Changes:**

1. Import new view:
```python
from apps.streamlit.views.paper_trading import render as render_paper_trading
```

2. Add tab to list:
```python
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
```

3. Render in tab:
```python
with tabs[6]:  # Paper Trading
    render_paper_trading(db, environment)

with tabs[7]:  # Logs (shifted index)
    render_logs(db, environment)
```

---

## Design Decisions

### Decision 1: Upsert vs Insert Pattern

**Chosen:** Upsert (single row per environment, update on each cycle)

**Rationale:**
- Simpler queries: "latest heartbeat" = SELECT by environment
- No unbounded table growth
- No cleanup job needed

**Alternative considered:** Insert new row each cycle
- **Rejected:** Requires periodic cleanup, more complex "latest" query

---

### Decision 2: Heartbeat Failure Handling

**Chosen:** Log warning, continue scheduler execution

**Rationale:**
- Heartbeat is monitoring, not critical path
- Trading should not fail due to monitoring issues
- Graceful degradation

**Code:**
```python
try:
    self.db.commit()  # Write heartbeat
except Exception:
    logger.warning("Heartbeat write failed")
    self.db.rollback()
    # Continue execution
```

---

### Decision 3: Status Thresholds

**Chosen:**
- Active: < 2 hours
- Stale: 2-24 hours
- Stopped: > 24 hours

**Rationale:**
- 2 hours = reasonable for hourly scheduler with tolerance for delays
- 24 hours = clear indication of failure
- Trade-off: May show "stale" for daily schedulers (acceptable for MVP)

---

### Decision 4: Historical Runs Storage

**Chosen:** Don't store history (upsert pattern)

**Rationale:**
- MVP scope: Current status only
- Historical metrics can be added later
- Reduces complexity

**Future enhancement:**
- Separate `scheduler_runs_history` table
- Keep detailed logs for analysis

---

## Database Migration

**File:** `migrations/versions/YYYYMMDD_add_scheduler_heartbeat.py`

```python
"""Add scheduler_heartbeats table

Revision ID: XXXXXXXX
Revises: previous_revision
Create Date: YYYY-MM-DD HH:MM:SS
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'XXXXXXXX'
down_revision = 'previous_revision'


def upgrade():
    op.create_table(
        'scheduler_heartbeats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('environment', sa.Enum('backtest', 'paper', 'prod', name='environment'), nullable=False),
        sa.Column('assets', sa.JSON(), nullable=True),
        sa.Column('stats', sa.JSON(), nullable=True),
        sa.Column('last_trade_id', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['last_trade_id'], ['trades.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_heartbeat_env_ts', 'scheduler_heartbeats', ['environment', 'timestamp'])
    op.create_index(op.f('ix_scheduler_heartbeats_timestamp'), 'scheduler_heartbeats', ['timestamp'])


def downgrade():
    op.drop_index(op.f('ix_scheduler_heartbeats_timestamp'), table_name='scheduler_heartbeats')
    op.drop_index('idx_heartbeat_env_ts', table_name='scheduler_heartbeats')
    op.drop_table('scheduler_heartbeats')
```

---

## Testing Strategy

### Unit Tests
- `test_upsert_heartbeat_start()` — Verify heartbeat creation/update
- `test_upsert_heartbeat_complete()` — Verify stats recording
- `test_calculate_status()` — Verify status logic (active/stale/stopped)
- `test_heartbeat_failure_graceful()` — Verify scheduler continues on DB error

### Integration Tests
- `test_scheduler_writes_heartbeat()` — Full cycle with DB
- `test_streamlit_reads_heartbeat()` — DB service methods
- `test_status_card_rendering()` — Streamlit view logic

### Manual Tests
- Run scheduler, verify heartbeat appears in DB
- View dashboard, verify status card updates
- Stop scheduler, wait 3 hours, verify "stale" status
- Simulate DB error, verify scheduler continues

---

## Rollback Plan

If issues arise:

### Option 1: Feature Flag
```python
# Add to settings.py
ENABLE_SCHEDULER_HEARTBEAT = os.getenv("ENABLE_SCHEDULER_HEARTBEAT", "true").lower() == "true"

# In scheduler.py
if settings.ENABLE_SCHEDULER_HEARTBEAT:
    self._upsert_heartbeat_start(...)
```

### Option 2: Revert Migration
```bash
alembic downgrade -1  # Remove scheduler_heartbeats table
```

---

## Open Questions

None — design is straightforward.
