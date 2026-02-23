# Planning: Comprehensive Structured Logging System

**Issue ID:** QuantAgent-yuk
**Type:** Epic
**Priority:** P3 (Low - pending MVP validation)
**Status:** Open
**Created:** 2026-01-04
**Estimated Total Effort:** ~2 hours

---

## 1. Executive Summary

This document provides the comprehensive implementation plan for the structured logging system Epic. The plan is organized into 6 sequential phases, with clear dependencies, task breakdowns, and effort estimates.

### 1.1 Implementation Overview

| Phase | Description | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 1 | Infrastructure Setup | 30 min | None |
| 2 | Replace print() Statements | 15 min | Phase 1 |
| 3 | Agent Logging Instrumentation | 20 min | Phase 1 |
| 4 | Infrastructure Logging | 15 min | Phase 1 |
| 5 | Streamlit UI Integration | 25 min | Phases 1-4 |
| 6 | Entry Point Initialization | 10 min | Phases 1-4 |
| - | Testing & Validation | 15 min | All phases |
| **Total** | | **~2 hours** | |

### 1.2 Critical Path

```
Phase 1 (Infrastructure) ─┬─► Phase 2 (Print Replacement)
                          ├─► Phase 3 (Agent Logging)
                          ├─► Phase 4 (Infrastructure Logging)
                          └─► Phases 5 & 6 (UI & Entry Points)
```

Phase 1 is the critical blocker; Phases 2-4 can be executed in parallel after Phase 1 completes.

---

## 2. Phase 1: Infrastructure Setup (30 min)

### 2.1 Objectives
- Create database schema for log storage
- Implement centralized logging configuration module
- Add environment variable support

### 2.2 Tasks

| Task ID | Description | File(s) | Effort | Acceptance |
|---------|-------------|---------|--------|------------|
| P1-T1 | Add Log ORM model | `quantagent/models.py` | 5 min | AC-1.1 |
| P1-T2 | Create Alembic migration | `alembic/versions/*.py` | 5 min | AC-1.2 |
| P1-T3 | Create logging_config.py | `quantagent/logging_config.py` | 15 min | AC-1.3, AC-1.4, AC-1.5 |
| P1-T4 | Add settings variables | `quantagent/settings.py` | 3 min | AC-1.6 |
| P1-T5 | Update .env.example | `.env.example` | 2 min | AC-DOC-1 |

### 2.3 Task Details

#### P1-T1: Add Log ORM Model

Add to `quantagent/models.py`:

```python
from sqlalchemy.dialects.postgresql import JSONB

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
    metadata = Column(JSONB)
    thread_id = Column(String(100), index=True)
    checkpoint_id = Column(String(100))
```

#### P1-T2: Create Alembic Migration

```bash
python -m alembic revision --autogenerate -m "Add logs table for structured logging"
python -m alembic upgrade head
```

#### P1-T3: Create logging_config.py

New file at `quantagent/logging_config.py` containing:
- `DatabaseLogHandler` class extending `logging.Handler`
- `setup_logging()` function configuring dual handlers
- Error handling in `emit()` to prevent crashes

Key implementation notes:
- Use `SessionLocal()` for database access
- Close session in `finally` block
- Use `handleError()` for graceful error recovery

#### P1-T4: Add Settings Variables

Add to `quantagent/settings.py`:

```python
# Logging settings
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_CONSOLE: bool = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
LOG_TO_DB: bool = os.getenv("LOG_TO_DB", "true").lower() == "true"
```

#### P1-T5: Update .env.example

Add documentation block:

```bash
# Logging configuration
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_CONSOLE=true               # Show logs in terminal/stdout
LOG_TO_DB=true                    # Persist logs to PostgreSQL logs table
```

### 2.4 Verification

```bash
# Run migration
python -m alembic upgrade head

# Verify table exists
psql $DATABASE_URL -c "SELECT * FROM logs LIMIT 1;"

# Test logging config
python -c "from quantagent.logging_config import setup_logging; setup_logging()"
```

---

## 3. Phase 2: Replace print() Statements (15 min)

### 3.1 Objectives
- Replace all print() calls in agent code with logger calls
- Maintain original message content
- Add appropriate log levels and event types

### 3.2 Tasks

| Task ID | Description | File(s) | Effort | Acceptance |
|---------|-------------|---------|--------|------------|
| P2-T1 | Replace prints in trend_agent.py | `quantagent/trend_agent.py` | 4 min | AC-2.1 |
| P2-T2 | Replace prints in pattern_agent.py | `quantagent/pattern_agent.py` | 4 min | AC-2.1 |
| P2-T3 | Replace prints in graph_util.py | `quantagent/graph_util.py` | 4 min | AC-2.1 |
| P2-T4 | Replace prints in static_util.py | `quantagent/static_util.py` | 3 min | AC-2.1 |

### 3.3 Replacement Reference

| File | Line | Original | Replacement |
|------|------|----------|-------------|
| trend_agent.py | 36 | `print("No precomputed trend image found...")` | `logger.info("No precomputed trend image found...", extra={'event_type': 'trend_image_generation'})` |
| trend_agent.py | 48 | `print(f"Failed to generate trend image: {e}")` | `logger.error(f"Failed to generate trend image: {e}", exc_info=True, extra={'event_type': 'trend_image_error'})` |
| trend_agent.py | 104 | `print("Retrying without system message...")` | `logger.info("Retrying without system message for Anthropic compatibility...", extra={'event_type': 'trend_agent_retry'})` |
| pattern_agent.py | 50 | `print("No precomputed pattern image found...")` | `logger.info("No precomputed pattern image found...", extra={'event_type': 'pattern_image_generation'})` |
| pattern_agent.py | 62 | `print(f"Failed to generate pattern image: {e}")` | `logger.error(f"Failed to generate pattern image: {e}", exc_info=True, extra={'event_type': 'pattern_image_error'})` |
| pattern_agent.py | 114 | `print("Retrying without system message...")` | `logger.info("Retrying without system message for Anthropic compatibility...", extra={'event_type': 'pattern_agent_retry'})` |
| graph_util.py | 286 | `print("ValueError at graph_util.py\n")` | `logger.error("ValueError at graph_util.py", exc_info=True, extra={'event_type': 'graph_util_error'})` |
| static_util.py | 123 | `print("ValueError at graph_util.py\n")` | `logger.error("ValueError in static_util.py", exc_info=True, extra={'event_type': 'static_util_error'})` |

### 3.4 Each File Template

Add at top of each file:
```python
import logging
logger = logging.getLogger(__name__)
```

### 3.5 Verification

```bash
# Ensure no print statements remain
grep -r "print(" quantagent/trend_agent.py quantagent/pattern_agent.py \
     quantagent/graph_util.py quantagent/static_util.py

# Should return empty or only commented lines
```

---

## 4. Phase 3: Agent Logging Instrumentation (20 min)

### 4.1 Objectives
- Add agent_start event logging at function entry
- Add agent_end event logging at function exit
- Include symbol and summary metadata

### 4.2 Tasks

| Task ID | Description | File(s) | Effort | Acceptance |
|---------|-------------|---------|--------|------------|
| P3-T1 | Instrument indicator_agent.py | `quantagent/indicator_agent.py` | 5 min | AC-3.1 |
| P3-T2 | Instrument pattern_agent.py | `quantagent/pattern_agent.py` | 5 min | AC-3.2 |
| P3-T3 | Instrument trend_agent.py | `quantagent/trend_agent.py` | 5 min | AC-3.3 |
| P3-T4 | Instrument decision_agent.py | `quantagent/decision_agent.py` | 5 min | AC-3.4 |

### 4.3 Implementation Pattern

Each agent node should follow this pattern:

```python
import logging
logger = logging.getLogger(__name__)

def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    symbol = state.get('stock_name', 'unknown')

    logger.info(
        f"Starting {agent_name} for {symbol}",
        extra={
            'event_type': 'agent_start',
            'symbol': symbol,
            'thread_id': state.get('thread_id')
        }
    )

    try:
        # ... existing agent logic ...

        logger.info(
            f"{agent_name} completed",
            extra={
                'event_type': 'agent_end',
                'symbol': symbol,
                'metadata': {...summary_data...}
            }
        )
    except Exception as e:
        logger.error(
            f"{agent_name} failed: {e}",
            exc_info=True,
            extra={
                'event_type': 'agent_error',
                'symbol': symbol
            }
        )
        raise

    return {...}
```

### 4.4 Metadata Summary (per agent)

| Agent | Summary Metadata Fields |
|-------|------------------------|
| indicator_agent | `rsi`, `macd`, `trend_direction` |
| pattern_agent | `pattern` (primary pattern detected) |
| trend_agent | `trend` (direction) |
| decision_agent | `signal`, `confidence`, `reasoning` (truncated) |

### 4.5 Verification

Run a backtest and query:
```sql
SELECT module, event_type, symbol, timestamp
FROM logs
WHERE event_type IN ('agent_start', 'agent_end')
ORDER BY timestamp;
```

---

## 5. Phase 4: Infrastructure Logging (15 min)

### 5.1 Objectives
- Log TradingGraph initialization details
- Log risk manager order rejections
- Enhance backtest logging with environment tags

### 5.2 Tasks

| Task ID | Description | File(s) | Effort | Acceptance |
|---------|-------------|---------|--------|------------|
| P4-T1 | Add TradingGraph init logging | `quantagent/trading_graph.py` | 6 min | AC-4.1 |
| P4-T2 | Add risk manager rejection logging | `quantagent/trading/risk_manager.py` | 5 min | AC-4.2 |
| P4-T3 | Enhance backtest logging | `quantagent/backtesting/backtest.py` | 4 min | AC-4.3 |

### 5.3 Implementation Details

#### P4-T1: TradingGraph Initialization

In `__init__()` method:
```python
logger.info("Initializing TradingGraph", extra={'event_type': 'graph_init'})

# After LLM setup:
logger.info(
    f"LLM configured: agent={settings.AGENT_LLM_PROVIDER}/{settings.AGENT_LLM_MODEL}",
    extra={'event_type': 'llm_config', 'metadata': {...}}
)

# After checkpointer setup (if enabled):
if use_checkpointing:
    logger.info("PostgreSQL checkpointer enabled", extra={'event_type': 'checkpointer_enabled'})

# At end of init:
logger.info("TradingGraph initialized successfully", extra={'event_type': 'graph_ready'})
```

#### P4-T2: Risk Manager Rejection

In validation method:
```python
if not valid:
    logger.warning(
        f"Order rejected: {reason}",
        extra={
            'event_type': 'order_rejected',
            'symbol': symbol,
            'metadata': {'reason': reason, 'side': side, 'quantity': quantity}
        }
    )
```

#### P4-T3: Backtest Enhancement

Add `extra={'event_type': '...', 'environment': 'backtest'}` to existing log calls.

### 5.4 Verification

```sql
SELECT * FROM logs WHERE event_type IN ('graph_init', 'graph_ready', 'order_rejected');
```

---

## 6. Phase 5: Streamlit UI Integration (25 min)

### 6.1 Objectives
- Implement functional logs view
- Add filtering capabilities
- Display logs from database

### 6.2 Tasks

| Task ID | Description | File(s) | Effort | Acceptance |
|---------|-------------|---------|--------|------------|
| P5-T1 | Implement logs view with filters | `apps/streamlit/views/logs.py` | 20 min | AC-5.1 to AC-5.5 |
| P5-T2 | Add Log model import to services | `apps/streamlit/services/*.py` | 5 min | - |

### 6.3 Implementation Outline

Replace placeholder content in `apps/streamlit/views/logs.py`:

```python
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def render(db, environment: str) -> None:
    st.subheader("Logs - System Event Viewer")

    # Filters row
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

    # Query and display
    try:
        with db.SessionLocal() as s:
            from quantagent.models import Log
            q = s.query(Log)

            # Apply filters
            if level_filter:
                q = q.filter(Log.level.in_(level_filter))
            if symbol_filter:
                q = q.filter(Log.symbol.ilike(f"%{symbol_filter}%"))
            if event_type_filter:
                q = q.filter(Log.event_type.ilike(f"%{event_type_filter}%"))

            # Time window
            window_start = datetime.utcnow() - timedelta(hours=int(hours_back))
            q = q.filter(Log.timestamp >= window_start)

            # Execute
            logs = q.order_by(Log.timestamp.desc()).limit(500).all()

            if not logs:
                st.info("No logs found for selected filters.")
                return

            # Display dataframe
            rows = [{
                "timestamp": log.timestamp,
                "level": log.level,
                "module": log.module,
                "event_type": log.event_type or "",
                "symbol": log.symbol or "",
                "message": log.message[:100] + "..." if len(log.message) > 100 else log.message,
            } for log in logs]

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Expandable details
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

### 6.4 Verification

1. Start Streamlit app
2. Navigate to Logs tab
3. Verify filters work
4. Verify dataframe displays

---

## 7. Phase 6: Entry Point Initialization (10 min)

### 7.1 Objectives
- Initialize logging at application startup
- Configure appropriate settings per entry point

### 7.2 Tasks

| Task ID | Description | File(s) | Effort | Acceptance |
|---------|-------------|---------|--------|------------|
| P6-T1 | Initialize in run_backtest.py | `examples/run_backtest.py` | 3 min | AC-6.1 |
| P6-T2 | Initialize in Streamlit app.py | `apps/streamlit/app.py` | 4 min | AC-6.2 |
| P6-T3 | Initialize in Flask web_interface.py | `apps/flask/web_interface.py` | 3 min | AC-6.3 |

### 7.3 Configuration Per Entry Point

| Entry Point | Console | DB | Environment |
|-------------|---------|-----|-------------|
| run_backtest.py | True | True | "backtest" |
| Streamlit app.py | False | True | (from session state) |
| Flask web_interface.py | True | False | "paper" |

### 7.4 Implementation Pattern

```python
from quantagent.logging_config import setup_logging

# At module level or in main()
setup_logging(
    level="INFO",
    log_to_console=True,  # or False for Streamlit
    log_to_db=True,       # or False for Flask
    environment="backtest"  # or from runtime
)
```

---

## 8. Testing & Validation (15 min)

### 8.1 Tasks

| Task ID | Description | Effort | Acceptance |
|---------|-------------|--------|------------|
| P7-T1 | Run existing test suite | 5 min | AC-REG-1 |
| P7-T2 | Run backtest and verify logs | 5 min | AC-3.5, AC-PERF-1 |
| P7-T3 | Manual Streamlit verification | 5 min | AC-5.1 to AC-5.5 |

### 8.2 Test Commands

```bash
# Regression tests
pytest tests/ -v

# Verify logging works
python examples/run_backtest.py

# Check database
psql $DATABASE_URL -c "SELECT COUNT(*) FROM logs;"

# Performance check (compare times)
time python examples/run_backtest.py  # with logging
LOG_TO_DB=false LOG_TO_CONSOLE=false time python examples/run_backtest.py  # baseline
```

---

## 9. Risk Assessment

### 9.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Database connection failures during logging | Low | Medium | Graceful error handling in handler |
| Performance degradation >5% | Low | High | Async writes (future), minimal logging |
| Migration conflicts with existing schema | Low | Medium | Test migration on clean DB first |
| Circular import issues | Medium | Low | Import logging module lazily if needed |

### 9.2 Implementation Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Incomplete print() replacement | Low | Low | grep verification after changes |
| Missing agent instrumentation | Low | Medium | Acceptance test coverage |
| Streamlit session state conflicts | Low | Low | Test with fresh session |

---

## 10. Rollback Plan

If issues arise during implementation:

1. **Database**: Run `alembic downgrade -1` to remove logs table
2. **Code**: Revert to previous commit (no breaking changes expected)
3. **Configuration**: Remove LOG_* variables from .env

The implementation is designed to be additive with no breaking changes to existing functionality.

---

## 11. Files Summary

### New Files (1)
- `quantagent/logging_config.py`

### Modified Files (16)

| File | Phase | Changes |
|------|-------|---------|
| `quantagent/models.py` | 1 | Add Log model |
| `alembic/versions/*.py` | 1 | New migration file |
| `quantagent/settings.py` | 1 | Add LOG_* settings |
| `.env.example` | 1 | Add logging documentation |
| `quantagent/trend_agent.py` | 2, 3 | Replace print, add instrumentation |
| `quantagent/pattern_agent.py` | 2, 3 | Replace print, add instrumentation |
| `quantagent/graph_util.py` | 2 | Replace print |
| `quantagent/static_util.py` | 2 | Replace print |
| `quantagent/indicator_agent.py` | 3 | Add instrumentation |
| `quantagent/decision_agent.py` | 3 | Add instrumentation |
| `quantagent/trading_graph.py` | 4 | Add init logging |
| `quantagent/trading/risk_manager.py` | 4 | Add rejection logging |
| `quantagent/backtesting/backtest.py` | 4 | Enhance with event_type |
| `apps/streamlit/views/logs.py` | 5 | Full implementation |
| `apps/streamlit/app.py` | 6 | Initialize logging |
| `examples/run_backtest.py` | 6 | Initialize logging |
| `apps/flask/web_interface.py` | 6 | Initialize logging |

---

## 12. Definition of Done

- [ ] Phase 1: Infrastructure created and migration successful
- [ ] Phase 2: No print() statements in agent code
- [ ] Phase 3: All 4 agents log start/end events
- [ ] Phase 4: Infrastructure logging added
- [ ] Phase 5: Streamlit logs view functional
- [ ] Phase 6: All entry points initialize logging
- [ ] All acceptance criteria verified
- [ ] Existing tests pass
- [ ] Performance within bounds (<5% overhead)
- [ ] Documentation updated

---

## Related Documents

- Requirements: `docs/01_requirements/QuantAgent-yuk-RQ-structured-logging.md`
- Design: `docs/03_design/LOGGING_STRATEGY.md`
- Acceptance: `docs/05_acceptance_tests/QuantAgent-yuk-AC-structured-logging.md`
- Implementation: `docs/06_implementation/QuantAgent-yuk-IM-structured-logging.md`
