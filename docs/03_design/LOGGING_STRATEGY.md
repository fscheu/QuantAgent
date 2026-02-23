# Logging Strategy for QuantAgent

**Status**: Planned (Not yet implemented)
**Priority**: Low - Implement after MVP manual backtest validation
**Estimated Implementation Time**: ~2 hours
**Last Updated**: 2025-12-08

---

## Overview

This document defines the comprehensive logging strategy for the QuantAgent trading system. The strategy addresses current gaps in visibility, provides audit trail capabilities, and enables effective debugging through structured logging.

### Current State

**Existing Logging:**
- `quantagent/trading/order_manager.py` - Proper logging (info, warning, error)
- `quantagent/trading/paper_broker.py` - Order placement/cancellation logs
- `quantagent/backtesting/backtest.py` - Progress and metrics logging
- `quantagent/data/provider.py` - Data fetching and caching logs

**Gaps:**
- Agent nodes use `print()` statements instead of logging
- No centralized logging configuration
- No database persistence of logs
- Streamlit logs tab is a placeholder (not wired)
- Inconsistent log formats and levels
- No environment-based filtering

---

## Design Requirements

Based on user preferences and system needs:

| Requirement | Specification |
|-------------|---------------|
| **Detail Level** | Basic (agent start/finish only) - minimal performance impact |
| **Storage** | Database (PostgreSQL `logs` table) + Console (configurable) |
| **Format** | Hybrid (human-readable text for console, JSON/structured for DB) |
| **Filtering** | By environment (backtest/paper/prod), symbol, event_type, log level |
| **Performance** | <5% latency increase from logging overhead |
| **Configuration** | Environment variables (LOG_LEVEL, LOG_TO_CONSOLE, LOG_TO_DB) |

---

## Architecture

### 1. Database Schema

**New Table: `logs`**

```sql
CREATE TABLE logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    level VARCHAR(10) NOT NULL,  -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    module VARCHAR(100) NOT NULL,  -- e.g., "quantagent.indicator_agent"
    message TEXT NOT NULL,
    environment VARCHAR(20),  -- backtest, paper, prod
    symbol VARCHAR(20),  -- BTC, SPX, etc.
    event_type VARCHAR(50),  -- agent_start, agent_end, llm_call, order_placed, etc.
    metadata JSONB,  -- Additional structured data (agent params, results, errors)
    thread_id VARCHAR(100),  -- LangGraph thread_id for tracing
    checkpoint_id VARCHAR(100),  -- LangGraph checkpoint_id

    INDEX idx_logs_timestamp (timestamp),
    INDEX idx_logs_level (level),
    INDEX idx_logs_environment (environment),
    INDEX idx_logs_symbol (symbol),
    INDEX idx_logs_event_type (event_type),
    INDEX idx_logs_thread_id (thread_id)
);
```

**ORM Model** (`quantagent/models.py`):

```python
class Log(Base):
    """System event logs for debugging, audit trail, and monitoring."""
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    level = Column(String(10), nullable=False, index=True)
    module = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    environment = Column(String(20), index=True)
    symbol = Column(String(20), index=True)
    event_type = Column(String(50), index=True)
    metadata = Column(JSONB)  # Structured data (agent params, results, errors)
    thread_id = Column(String(100), index=True)  # LangGraph thread_id
    checkpoint_id = Column(String(100))  # LangGraph checkpoint_id
```

**Migration:**
```bash
python -m alembic revision --autogenerate -m "Add logs table for structured logging"
python -m alembic upgrade head
```

---

### 2. Logging Configuration Module

**New File**: `quantagent/logging_config.py`

#### DatabaseLogHandler

Custom logging handler that writes to PostgreSQL:

```python
import logging
import os
from typing import Optional
from quantagent.database import SessionLocal
from quantagent.models import Log

class DatabaseLogHandler(logging.Handler):
    """Custom logging handler that writes to PostgreSQL logs table."""

    def __init__(self, environment: Optional[str] = None, symbol: Optional[str] = None):
        super().__init__()
        self.environment = environment
        self.symbol = symbol

    def emit(self, record: logging.LogRecord):
        """Write log record to database."""
        try:
            db = SessionLocal()
            log_entry = Log(
                level=record.levelname,
                module=record.name,
                message=record.getMessage(),
                environment=self.environment,
                symbol=self.symbol,
                event_type=getattr(record, 'event_type', None),
                metadata=getattr(record, 'metadata', None),
                thread_id=getattr(record, 'thread_id', None),
                checkpoint_id=getattr(record, 'checkpoint_id', None),
            )
            db.add(log_entry)
            db.commit()
        except Exception:
            self.handleError(record)
        finally:
            db.close()
```

#### Setup Function

Configures dual handlers (console + database):

```python
def setup_logging(
    level: Optional[str] = None,
    log_to_console: bool = True,
    log_to_db: bool = True,
    environment: Optional[str] = None,
    symbol: Optional[str] = None
):
    """
    Configure logging with dual handlers (console + database).

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Enable console output (default: True)
        log_to_db: Enable database persistence (default: True)
        environment: Environment tag (backtest, paper, prod)
        symbol: Symbol tag for filtering

    Returns:
        Configured root logger

    Example:
        >>> setup_logging(level="INFO", log_to_console=True, environment="backtest")
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_to_console = log_to_console or os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Console handler (human-readable text)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Database handler (structured JSON)
    if log_to_db:
        db_handler = DatabaseLogHandler(environment=environment, symbol=symbol)
        root_logger.addHandler(db_handler)

    return root_logger
```

---

### 3. Environment Configuration

**Add to `quantagent/settings.py`:**

```python
# Logging settings
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_CONSOLE: bool = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
LOG_TO_DB: bool = os.getenv("LOG_TO_DB", "true").lower() == "true"
```

**Add to `.env` file:**

```bash
# Logging configuration
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_CONSOLE=true               # Show logs in terminal/stdout
LOG_TO_DB=true                    # Persist logs to PostgreSQL logs table
```

---

## Implementation Plan

### Phase 1: Infrastructure Setup (30 min)

1. **Create database schema** (5 min)
   - Add `Log` model to `quantagent/models.py`
   - Run Alembic migration: `python -m alembic revision --autogenerate -m "Add logs table"`
   - Apply migration: `python -m alembic upgrade head`

2. **Create logging config module** (20 min)
   - Create `quantagent/logging_config.py`
   - Implement `DatabaseLogHandler` class
   - Implement `setup_logging()` function

3. **Add settings** (5 min)
   - Add LOG_LEVEL, LOG_TO_CONSOLE, LOG_TO_DB to `quantagent/settings.py`
   - Update `.env.example` with logging variables

### Phase 2: Replace print() Statements (15 min)

Files to modify:

1. **`quantagent/trend_agent.py`** (lines 36, 48, 107)
   ```python
   # Before:
   print(f"Retrying {attempt}/{retries}...")

   # After:
   import logging
   logger = logging.getLogger(__name__)
   logger.info(
       f"Retrying {attempt}/{retries}...",
       extra={'event_type': 'trend_agent_retry', 'metadata': {'attempt': attempt}}
   )
   ```

2. **`quantagent/pattern_agent.py`** (lines 50, 62, 113)
   ```python
   # Before:
   print(f"Retrying {attempt}/{retries}...")

   # After:
   import logging
   logger = logging.getLogger(__name__)
   logger.info(
       f"Retrying {attempt}/{retries}...",
       extra={'event_type': 'pattern_agent_retry', 'metadata': {'attempt': attempt}}
   )
   ```

3. **`quantagent/graph_util.py`** (lines 286, 458)
   ```python
   # Before:
   print(f"Error calculating RSI: {e}")

   # After:
   import logging
   logger = logging.getLogger(__name__)
   logger.error(
       f"Error calculating RSI: {e}",
       exc_info=True,
       extra={'event_type': 'rsi_calculation_error'}
   )
   ```

4. **`quantagent/static_util.py`** (line 100)
   ```python
   # Before:
   print(f"Warning: {msg}")

   # After:
   import logging
   logger = logging.getLogger(__name__)
   logger.warning(msg, extra={'event_type': 'data_validation_warning'})
   ```

### Phase 3: Add Agent Logging (20 min)

**Requirement**: Basic level - agent start/finish only (minimal performance impact)

1. **`quantagent/indicator_agent.py`**
   ```python
   import logging
   logger = logging.getLogger(__name__)

   def indicator_node(state):
       logger.info(
           f"Starting indicator agent for {state.get('stock_name', 'unknown')}",
           extra={
               'event_type': 'agent_start',
               'symbol': state.get('stock_name'),
               'thread_id': state.get('thread_id')
           }
       )

       # ... existing analysis code ...

       logger.info(
           f"Indicator agent completed",
           extra={
               'event_type': 'agent_end',
               'symbol': state.get('stock_name'),
               'metadata': {'rsi': report.rsi, 'macd': report.macd}  # Summary only
           }
       )
       return state
   ```

2. **`quantagent/pattern_agent.py`**
   ```python
   import logging
   logger = logging.getLogger(__name__)

   def pattern_node(state):
       logger.info(
           f"Starting pattern agent for {state.get('stock_name')}",
           extra={'event_type': 'agent_start', 'symbol': state.get('stock_name')}
       )

       # ... existing analysis code ...

       logger.info(
           f"Pattern agent completed - detected: {report.candlestick_pattern}",
           extra={
               'event_type': 'agent_end',
               'metadata': {'pattern': report.candlestick_pattern}
           }
       )
       return state
   ```

3. **`quantagent/trend_agent.py`**
   ```python
   import logging
   logger = logging.getLogger(__name__)

   def trend_node(state):
       logger.info(
           f"Starting trend agent for {state.get('stock_name')}",
           extra={'event_type': 'agent_start', 'symbol': state.get('stock_name')}
       )

       # ... existing analysis code ...

       logger.info(
           f"Trend agent completed - trend: {report.trend}",
           extra={
               'event_type': 'agent_end',
               'metadata': {'trend': report.trend}
           }
       )
       return state
   ```

4. **`quantagent/decision_agent.py`**
   ```python
   import logging
   logger = logging.getLogger(__name__)

   def decision_node(state):
       logger.info(
           f"Starting decision agent for {state.get('stock_name')}",
           extra={'event_type': 'agent_start', 'symbol': state.get('stock_name')}
       )

       # ... existing analysis code ...

       logger.info(
           f"Decision agent completed - signal: {decision.signal.value}",
           extra={
               'event_type': 'agent_end',
               'metadata': {
                   'signal': decision.signal.value,
                   'confidence': decision.confidence,
                   'reasoning': decision.reasoning[:100]  # Truncate for brevity
               }
           }
       )
       return state
   ```

### Phase 4: Infrastructure Logging (15 min)

1. **`quantagent/trading_graph.py`**
   ```python
   import logging
   logger = logging.getLogger(__name__)

   def __init__(self, use_checkpointing: bool = False):
       logger.info("Initializing TradingGraph", extra={'event_type': 'graph_init'})

       # ... LLM setup ...
       logger.info(
           f"LLM configured: agent={settings.AGENT_LLM_PROVIDER}/{settings.AGENT_LLM_MODEL}",
           extra={'event_type': 'llm_config'}
       )

       # ... checkpointer setup ...
       if use_checkpointing:
           logger.info("PostgreSQL checkpointer enabled", extra={'event_type': 'checkpointer_enabled'})

       # ... graph compilation ...
       logger.info("TradingGraph initialized successfully", extra={'event_type': 'graph_ready'})
   ```

2. **`quantagent/trading/risk_manager.py`**
   ```python
   import logging
   logger = logging.getLogger(__name__)

   def validate_order(self, ...):
       # ... validation logic ...

       if not valid:
           logger.warning(
               f"Order rejected: {reason}",
               extra={
                   'event_type': 'order_rejected',
                   'symbol': symbol,
                   'metadata': {'reason': reason, 'side': side, 'quantity': quantity}
               }
           )
       return (valid, reason)
   ```

3. **`quantagent/backtesting/backtest.py`** (enhance existing logging)
   ```python
   # Add event_type to existing log calls:
   logger.info(
       f"Starting backtest: {self.start_date} to {self.end_date}",
       extra={'event_type': 'backtest_start', 'environment': 'backtest'}
   )
   ```

### Phase 5: Streamlit UI Integration (25 min)

**File**: `apps/streamlit/views/logs.py`

```python
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def render(db, environment: str) -> None:
    st.subheader("Logs - System Event Viewer")

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        level_filter = st.multiselect(
            "Log Level",
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default=["INFO", "WARNING", "ERROR"]
        )
    with col2:
        symbol_filter = st.text_input("Symbol", value="")
    with col3:
        event_type_filter = st.text_input("Event Type", value="")
    with col4:
        hours_back = st.number_input("Hours back", min_value=1, max_value=168, value=24)

    if not db.ok:
        st.info("Connect DB to view logs.")
        return

    # Query logs from database
    try:
        with db.SessionLocal() as s:
            q = s.query(db.models.Log).filter(db.models.Log.environment == environment)

            # Apply filters
            if level_filter:
                q = q.filter(db.models.Log.level.in_(level_filter))
            if symbol_filter:
                q = q.filter(db.models.Log.symbol.contains(symbol_filter))
            if event_type_filter:
                q = q.filter(db.models.Log.event_type.contains(event_type_filter))

            # Time window
            window_start = datetime.utcnow() - timedelta(hours=int(hours_back))
            q = q.filter(db.models.Log.timestamp >= window_start)

            # Execute query
            logs = q.order_by(db.models.Log.timestamp.desc()).limit(500).all()

            if not logs:
                st.info("No logs found for selected filters.")
                return

            # Display as dataframe
            rows = [
                {
                    "timestamp": log.timestamp,
                    "level": log.level,
                    "module": log.module,
                    "event_type": log.event_type,
                    "symbol": log.symbol,
                    "message": log.message[:100],  # Truncate
                    "thread_id": log.thread_id,
                }
                for log in logs
            ]
            st.dataframe(pd.DataFrame(rows), width='stretch')

            # Details expander for recent 10
            st.markdown("**Recent Log Details (Latest 10)**")
            for log in logs[:10]:
                with st.expander(f"{log.timestamp} - {log.level} - {log.event_type or 'N/A'}"):
                    st.write(f"**Module:** {log.module}")
                    st.write(f"**Message:** {log.message}")
                    if log.metadata:
                        st.json(log.metadata)
                    if log.thread_id:
                        st.write(f"**Thread ID:** {log.thread_id}")

    except Exception as e:
        st.error(f"Error loading logs: {e}")
```

### Phase 6: Initialize Logging in Entry Points (10 min)

1. **`examples/run_backtest.py`**
   ```python
   from quantagent.logging_config import setup_logging

   def main():
       # Setup logging (console + DB)
       setup_logging(
           level="INFO",
           log_to_console=True,  # Show in terminal
           log_to_db=True,       # Persist to DB
           environment="backtest"
       )

       # ... rest of backtest code ...
   ```

2. **`apps/streamlit/app.py`**
   ```python
   from quantagent.logging_config import setup_logging

   # Initialize logging once at app startup
   setup_logging(
       level="INFO",
       log_to_console=False,  # Don't clutter Streamlit terminal
       log_to_db=True,
       environment=st.session_state.get('environment', 'paper')
   )
   ```

3. **`apps/flask/web_interface.py`**
   ```python
   from quantagent.logging_config import setup_logging

   # Initialize logging
   setup_logging(level="INFO", log_to_console=True, log_to_db=False)
   ```

---

## Event Types Reference

Standardized event types for filtering and analysis:

| Event Type | Description | Module |
|------------|-------------|--------|
| `graph_init` | TradingGraph initialization started | trading_graph |
| `llm_config` | LLM provider/model configured | trading_graph |
| `checkpointer_enabled` | PostgreSQL checkpointer initialized | trading_graph |
| `graph_ready` | TradingGraph fully initialized | trading_graph |
| `agent_start` | Agent node execution started | *_agent |
| `agent_end` | Agent node execution completed | *_agent |
| `agent_retry` | Agent retrying after failure | *_agent |
| `backtest_start` | Backtest run started | backtest |
| `backtest_end` | Backtest run completed | backtest |
| `order_placed` | Order submitted to broker | order_manager |
| `order_rejected` | Order rejected by risk manager | risk_manager |
| `order_executed` | Order filled by broker | paper_broker |
| `data_fetch` | Market data fetched from provider | data_provider |
| `rsi_calculation_error` | Error calculating technical indicator | graph_util |
| `data_validation_warning` | Data quality issue detected | static_util |

---

## Testing Checklist

After implementation, verify:

- [ ] Run backtest and verify logs appear in console
- [ ] Query database: `SELECT * FROM logs ORDER BY timestamp DESC LIMIT 10;`
- [ ] Verify logs table populated with correct fields
- [ ] Check Streamlit logs tab displays logs from database
- [ ] Test environment filtering (backtest vs paper)
- [ ] Test symbol filtering (BTC, SPX, etc.)
- [ ] Test event_type filtering (agent_start, agent_end, etc.)
- [ ] Test log level filtering (INFO, WARNING, ERROR)
- [ ] Verify agent start/finish logs include correct metadata
- [ ] Confirm no `print()` statements remain in agent code
- [ ] Measure performance impact (<5% latency increase expected)
- [ ] Test with `LOG_TO_CONSOLE=false` - verify no console output
- [ ] Test with `LOG_TO_DB=false` - verify no database writes

---

## Performance Considerations

### Expected Impact

- **Latency**: <5% increase from logging overhead (estimated ~50-200ms per backtest iteration)
- **Database Growth**: ~1000-5000 log entries per backtest run (basic level)
  - Agent start/finish: 4 entries per analysis (indicator, pattern, trend, decision)
  - Infrastructure: ~10 entries per backtest run
  - Errors/warnings: Variable based on failures
- **Disk Space**: ~1KB per log entry → ~1-5MB per backtest run

### Optimization Strategies

1. **Async Database Writes** (future enhancement)
   - Buffer log entries in memory
   - Batch write to database every N seconds
   - Reduces database connection overhead

2. **Log Retention Policy** (future enhancement)
   - Auto-delete logs older than 30 days
   - Archive old logs to S3 or file storage
   - Implement cron job for cleanup

3. **Sampling** (if needed)
   - Log only every Nth iteration in long backtests
   - Configurable via `LOG_SAMPLING_RATE` environment variable

---

## Future Enhancements (Phase 3)

After basic logging is working, consider:

### 1. LLM Prompt/Response Logging

For debugging agent reasoning:

```python
logger.debug(
    "LLM prompt sent",
    extra={
        'event_type': 'llm_prompt',
        'metadata': {
            'prompt': prompt_text,
            'model': 'gpt-4o-mini',
            'temperature': 0.1
        }
    }
)

logger.debug(
    "LLM response received",
    extra={
        'event_type': 'llm_response',
        'metadata': {
            'response': response_text,
            'tokens': 150,
            'latency_ms': 1200
        }
    }
)
```

### 2. Audit Trail Middleware

Automatic logging of all state transitions:

```python
# LangGraph middleware pattern
def audit_middleware(state, next_node):
    logger.info(
        f"State transition: {current_node} → {next_node}",
        extra={
            'event_type': 'state_transition',
            'metadata': state.dict()
        }
    )
    return next_node(state)
```

### 3. Performance Metrics

Track execution time per agent:

```python
import time

start = time.time()
# ... agent execution ...
elapsed = time.time() - start

logger.info(
    f"Agent completed in {elapsed:.2f}s",
    extra={
        'event_type': 'performance_metric',
        'metadata': {'execution_time_ms': elapsed * 1000}
    }
)
```

### 4. Alert System

Send notifications on critical errors:

```python
if log.level == "CRITICAL":
    send_slack_notification(log.message)
    send_email_alert(log.message, log.metadata)
```

### 5. Log Rotation

Prevent unbounded database growth:

```python
# Cron job or scheduler task
def cleanup_old_logs():
    cutoff = datetime.utcnow() - timedelta(days=30)
    db.query(Log).filter(Log.timestamp < cutoff).delete()
    db.commit()
```

---

## Files to Modify

### New Files
1. `quantagent/logging_config.py` - DatabaseLogHandler, setup_logging()

### Database
2. `quantagent/models.py` - Add Log ORM model
3. `alembic/versions/<new>_add_logs_table.py` - Migration script

### Configuration
4. `quantagent/settings.py` - Add LOG_LEVEL, LOG_TO_CONSOLE, LOG_TO_DB
5. `.env.example` - Document logging environment variables

### Agent Nodes (replace print, add start/finish)
6. `quantagent/indicator_agent.py`
7. `quantagent/pattern_agent.py`
8. `quantagent/trend_agent.py`
9. `quantagent/decision_agent.py`

### Utilities (replace print)
10. `quantagent/graph_util.py`
11. `quantagent/static_util.py`

### Core Infrastructure
12. `quantagent/trading_graph.py` - Initialization logging
13. `quantagent/trading/risk_manager.py` - Rejection logging
14. `quantagent/backtesting/backtest.py` - Enhance with event_type tags

### UI
15. `apps/streamlit/views/logs.py` - Implement log viewer
16. `apps/streamlit/app.py` - Initialize logging

### Examples
17. `examples/run_backtest.py` - Initialize logging with console output

**Total**: 17 files (1 new, 3 database, 13 modified)

---

## Implementation Timeline

When ready to implement (post-MVP validation):

| Phase | Tasks | Time | Status |
|-------|-------|------|--------|
| **Phase 1** | Database schema + logging config | 30 min | Pending |
| **Phase 2** | Replace print() statements | 15 min | Pending |
| **Phase 3** | Add agent start/finish logging | 20 min | Pending |
| **Phase 4** | Infrastructure logging | 15 min | Pending |
| **Phase 5** | Streamlit UI integration | 25 min | Pending |
| **Phase 6** | Initialize in entry points | 10 min | Pending |
| **Testing** | Validation and performance check | 15 min | Pending |
| **Total** | | **~2 hours** | |

---

## Related Documentation

- `docs/03_technical/langgraph_improvements.md` - Phase 2 middleware planning
- `docs/01_requirements/trading_system_requirements.md` - Logging requirements
- `docs/03_technical/streamlit_app_architecture.md` - UI architecture
- `docs/03_technical/MIGRATIONS.md` - Database migration workflow

---

## Notes

- **Priority**: Low - implement AFTER manual backtest validation (Case 1-3 from `MVP_MANUAL_TEST_CASES.md`)
- **User Preference**: Database + console (configurable via environment variables)
- **Format**: Hybrid (text for console, JSON/structured for DB)
- **Detail Level**: Basic (agent start/finish only) to minimize performance impact
- **Database Impact**: Moderate (new table + 6 indexes, but low write volume with basic logging)
- **Backward Compatibility**: No breaking changes - existing code continues to work without logging
