# Implementation Notes: Comprehensive Structured Logging System

**Issue ID:** QuantAgent-yuk
**Type:** Epic
**Priority:** P3 (Low - pending MVP validation)
**Status:** Open (Pre-implementation)
**Created:** 2026-01-04

---

## 1. Overview

This document provides implementation guidance and handoff notes for the structured logging system Epic. It includes proposed child issues, code templates, and implementation checklists.

---

## 2. Proposed Child Issues

The following child issues should be created in Beads to track granular implementation work. Each child can be executed independently after Phase 1 (Infrastructure) is complete.

### 2.1 Child Issue Summary

| Issue ID (Proposed) | Title | Phase | Effort | Dependencies |
|---------------------|-------|-------|--------|--------------|
| QuantAgent-yuk-01 | Logging Infrastructure Setup | 1 | 30 min | None |
| QuantAgent-yuk-02 | Replace print() Statements | 2 | 15 min | yuk-01 |
| QuantAgent-yuk-03 | Agent Logging Instrumentation | 3 | 20 min | yuk-01 |
| QuantAgent-yuk-04 | Infrastructure Logging | 4 | 15 min | yuk-01 |
| QuantAgent-yuk-05 | Streamlit Logs View | 5 | 25 min | yuk-01 through yuk-04 |
| QuantAgent-yuk-06 | Entry Point Initialization | 6 | 10 min | yuk-01 |

### 2.2 Child Issue Details

---

#### QuantAgent-yuk-01: Logging Infrastructure Setup

**Scope:**
- Add `Log` ORM model to `quantagent/models.py`
- Create Alembic migration for `logs` table
- Create `quantagent/logging_config.py` with:
  - `DatabaseLogHandler` class
  - `setup_logging()` function
- Add logging settings to `quantagent/settings.py`
- Update `.env.example` with documentation

**Files:**
- `quantagent/models.py` (modify)
- `quantagent/logging_config.py` (new)
- `quantagent/settings.py` (modify)
- `.env.example` (modify)
- `alembic/versions/*.py` (new migration)

**Acceptance Criteria:** AC-1.1 through AC-1.6

**Blocked By:** None

**Blocks:** All other child issues

---

#### QuantAgent-yuk-02: Replace print() Statements

**Scope:**
- Replace all `print()` calls in:
  - `quantagent/trend_agent.py` (3 occurrences)
  - `quantagent/pattern_agent.py` (3 occurrences)
  - `quantagent/graph_util.py` (1 occurrence)
  - `quantagent/static_util.py` (1 occurrence)
- Use appropriate log levels (INFO/ERROR)
- Add event_type metadata

**Files:**
- `quantagent/trend_agent.py` (modify)
- `quantagent/pattern_agent.py` (modify)
- `quantagent/graph_util.py` (modify)
- `quantagent/static_util.py` (modify)

**Acceptance Criteria:** AC-2.1 through AC-2.3

**Blocked By:** QuantAgent-yuk-01

**Blocks:** None (can run in parallel with yuk-03, yuk-04)

---

#### QuantAgent-yuk-03: Agent Logging Instrumentation

**Scope:**
- Add agent_start/agent_end logging to:
  - `quantagent/indicator_agent.py`
  - `quantagent/pattern_agent.py`
  - `quantagent/trend_agent.py`
  - `quantagent/decision_agent.py`
- Include symbol and summary metadata
- Use consistent event_type values

**Files:**
- `quantagent/indicator_agent.py` (modify)
- `quantagent/pattern_agent.py` (modify - may overlap with yuk-02)
- `quantagent/trend_agent.py` (modify - may overlap with yuk-02)
- `quantagent/decision_agent.py` (modify)

**Acceptance Criteria:** AC-3.1 through AC-3.5

**Blocked By:** QuantAgent-yuk-01

**Blocks:** QuantAgent-yuk-05 (logs view needs data)

**Note:** Consider merging with yuk-02 for files that overlap (trend_agent, pattern_agent).

---

#### QuantAgent-yuk-04: Infrastructure Logging

**Scope:**
- Add initialization logging to `quantagent/trading_graph.py`:
  - graph_init event
  - llm_config event
  - checkpointer_enabled event
  - graph_ready event
- Add rejection logging to `quantagent/trading/risk_manager.py`
- Enhance backtest logging with environment tags

**Files:**
- `quantagent/trading_graph.py` (modify)
- `quantagent/trading/risk_manager.py` (modify)
- `quantagent/backtesting/backtest.py` (modify)

**Acceptance Criteria:** AC-4.1 through AC-4.3

**Blocked By:** QuantAgent-yuk-01

**Blocks:** None

---

#### QuantAgent-yuk-05: Streamlit Logs View

**Scope:**
- Implement functional logs view in `apps/streamlit/views/logs.py`
- Add filters: level, symbol, event_type, time window
- Display logs in dataframe
- Add expandable detail view
- Handle DB connection gracefully

**Files:**
- `apps/streamlit/views/logs.py` (modify - full rewrite)

**Acceptance Criteria:** AC-5.1 through AC-5.6

**Blocked By:** QuantAgent-yuk-01 through yuk-04 (needs logs data to display)

**Blocks:** None

---

#### QuantAgent-yuk-06: Entry Point Initialization

**Scope:**
- Initialize logging in `examples/run_backtest.py`
- Initialize logging in `apps/streamlit/app.py`
- Initialize logging in `apps/flask/web_interface.py`

**Files:**
- `examples/run_backtest.py` (modify)
- `apps/streamlit/app.py` (modify)
- `apps/flask/web_interface.py` (modify)

**Acceptance Criteria:** AC-6.1 through AC-6.3

**Blocked By:** QuantAgent-yuk-01

**Blocks:** None

---

### 2.3 Dependency Graph

```
QuantAgent-yuk-01 (Infrastructure)
        │
        ├──► QuantAgent-yuk-02 (Print Replacement)
        │
        ├──► QuantAgent-yuk-03 (Agent Logging) ──► QuantAgent-yuk-05 (UI)
        │
        ├──► QuantAgent-yuk-04 (Infra Logging) ──► QuantAgent-yuk-05 (UI)
        │
        └──► QuantAgent-yuk-06 (Entry Points)
```

---

## 3. Code Templates

### 3.1 Log Model Template

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

### 3.2 DatabaseLogHandler Template

Create `quantagent/logging_config.py`:

```python
"""
Centralized logging configuration with database persistence.
"""

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
                symbol=self.symbol or getattr(record, 'symbol', None),
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
            if 'db' in locals():
                db.close()


def setup_logging(
    level: Optional[str] = None,
    log_to_console: bool = True,
    log_to_db: bool = True,
    environment: Optional[str] = None,
    symbol: Optional[str] = None
) -> logging.Logger:
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
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_to_console = log_to_console if log_to_console is not None else \
        os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
    log_to_db = log_to_db if log_to_db is not None else \
        os.getenv("LOG_TO_DB", "true").lower() == "true"

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Console handler (human-readable)
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # Database handler (structured)
    if log_to_db:
        db_handler = DatabaseLogHandler(environment=environment, symbol=symbol)
        root_logger.addHandler(db_handler)

    return root_logger
```

### 3.3 Agent Instrumentation Template

Pattern for each agent:

```python
import logging
logger = logging.getLogger(__name__)

def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    symbol = state.get('stock_name', 'unknown')

    logger.info(
        f"Starting <agent_name> for {symbol}",
        extra={
            'event_type': 'agent_start',
            'symbol': symbol,
            'thread_id': state.get('thread_id')
        }
    )

    try:
        # ... existing logic ...

        logger.info(
            f"<agent_name> completed",
            extra={
                'event_type': 'agent_end',
                'symbol': symbol,
                'metadata': {
                    # Add summary fields here
                }
            }
        )
    except Exception as e:
        logger.error(
            f"<agent_name> failed: {e}",
            exc_info=True,
            extra={
                'event_type': 'agent_error',
                'symbol': symbol
            }
        )
        # Handle or re-raise as appropriate

    return {...}
```

### 3.4 Settings Addition Template

Add to `quantagent/settings.py`:

```python
# Logging settings
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_CONSOLE: bool = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
LOG_TO_DB: bool = os.getenv("LOG_TO_DB", "true").lower() == "true"
```

### 3.5 .env.example Addition Template

```bash
# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_CONSOLE=true               # Show logs in terminal/stdout
LOG_TO_DB=true                    # Persist logs to PostgreSQL logs table
```

---

## 4. Implementation Checklist

### 4.1 Phase 1: Infrastructure Setup

- [ ] Add JSONB import to models.py
- [ ] Add Log model class to models.py
- [ ] Run: `python -m alembic revision --autogenerate -m "Add logs table"`
- [ ] Review generated migration file
- [ ] Run: `python -m alembic upgrade head`
- [ ] Verify table exists: `SELECT * FROM logs LIMIT 1;`
- [ ] Create quantagent/logging_config.py
- [ ] Add DatabaseLogHandler class
- [ ] Add setup_logging() function
- [ ] Add LOG_* settings to settings.py
- [ ] Update .env.example
- [ ] Test: `python -c "from quantagent.logging_config import setup_logging; setup_logging()"`

### 4.2 Phase 2: Print Replacement

- [ ] Add logger import to trend_agent.py
- [ ] Replace print() at line 36
- [ ] Replace print() at line 48
- [ ] Replace print() at line 104
- [ ] Add logger import to pattern_agent.py
- [ ] Replace print() at line 50
- [ ] Replace print() at line 62
- [ ] Replace print() at line 114
- [ ] Add logger import to graph_util.py
- [ ] Replace print() at line 286
- [ ] Add logger import to static_util.py
- [ ] Replace print() at line 123
- [ ] Verify: `grep -r "print(" quantagent/{trend,pattern}_agent.py quantagent/{graph,static}_util.py`

### 4.3 Phase 3: Agent Logging

- [ ] Add logger to indicator_agent.py
- [ ] Add agent_start log
- [ ] Add agent_end log with metadata
- [ ] Add logger to pattern_agent.py (if not done in Phase 2)
- [ ] Add agent_start log
- [ ] Add agent_end log with pattern metadata
- [ ] Add logger to trend_agent.py (if not done in Phase 2)
- [ ] Add agent_start log
- [ ] Add agent_end log with trend metadata
- [ ] Add logger to decision_agent.py
- [ ] Add agent_start log
- [ ] Add agent_end log with signal/confidence metadata

### 4.4 Phase 4: Infrastructure Logging

- [ ] Add logger to trading_graph.py
- [ ] Add graph_init log in __init__
- [ ] Add llm_config log after LLM setup
- [ ] Add checkpointer_enabled log if applicable
- [ ] Add graph_ready log at end of init
- [ ] Add logger to risk_manager.py
- [ ] Add order_rejected warning log
- [ ] Enhance backtest.py logs with event_type and environment

### 4.5 Phase 5: Streamlit UI

- [ ] Rewrite apps/streamlit/views/logs.py
- [ ] Add level multiselect filter
- [ ] Add symbol text input filter
- [ ] Add event_type text input filter
- [ ] Add hours_back number input
- [ ] Add query logic with filters
- [ ] Add dataframe display
- [ ] Add expandable details section
- [ ] Add error handling for DB disconnection
- [ ] Test with sample data

### 4.6 Phase 6: Entry Points

- [ ] Add setup_logging() to run_backtest.py
- [ ] Configure: console=True, db=True, environment="backtest"
- [ ] Add setup_logging() to streamlit/app.py
- [ ] Configure: console=False, db=True
- [ ] Add setup_logging() to flask/web_interface.py
- [ ] Configure: console=True, db=False

### 4.7 Validation

- [ ] Run: `pytest tests/ -v`
- [ ] Run backtest, verify console logs
- [ ] Query: `SELECT COUNT(*) FROM logs;`
- [ ] Query: `SELECT DISTINCT event_type FROM logs;`
- [ ] Open Streamlit, verify Logs tab works
- [ ] Measure performance overhead

---

## 5. Event Types Reference

| Event Type | Level | Module | Description |
|------------|-------|--------|-------------|
| `graph_init` | INFO | trading_graph | Graph initialization started |
| `llm_config` | INFO | trading_graph | LLM provider/model configured |
| `checkpointer_enabled` | INFO | trading_graph | Checkpointer initialized |
| `graph_ready` | INFO | trading_graph | Graph fully initialized |
| `agent_start` | INFO | *_agent | Agent execution started |
| `agent_end` | INFO | *_agent | Agent execution completed |
| `agent_error` | ERROR | *_agent | Agent execution failed |
| `agent_retry` | INFO | *_agent | Agent retrying after failure |
| `trend_image_generation` | INFO | trend_agent | Generating trend image |
| `trend_image_error` | ERROR | trend_agent | Failed to generate trend image |
| `pattern_image_generation` | INFO | pattern_agent | Generating pattern image |
| `pattern_image_error` | ERROR | pattern_agent | Failed to generate pattern image |
| `order_rejected` | WARNING | risk_manager | Order rejected by risk rules |
| `backtest_start` | INFO | backtest | Backtest run started |
| `backtest_end` | INFO | backtest | Backtest run completed |
| `graph_util_error` | ERROR | graph_util | Error in graph utilities |
| `static_util_error` | ERROR | static_util | Error in static utilities |

---

## 6. Known Considerations

### 6.1 Performance

- Database writes are synchronous in the initial implementation
- For high-volume logging, consider async buffered writes (future enhancement)
- The `Log.metadata` JSONB field may impact insert performance for large payloads

### 6.2 Thread Safety

- `SessionLocal()` creates new sessions per emit() call
- This is safe but creates connection overhead
- Connection pooling should handle this adequately

### 6.3 Migration Notes

- The migration adds 6 indexes to the `logs` table
- Initial migration on empty database should be fast
- Adding to production with existing data may require more careful planning

### 6.4 Circular Import Prevention

- `logging_config.py` imports from `database` and `models`
- Ensure no modules that import logging_config are imported by database/models
- If issues arise, use lazy imports inside `setup_logging()`

---

## 7. Testing Strategy

### 7.1 Unit Tests (New)

Create `tests/test_logging_config.py`:

```python
import pytest
import logging
from quantagent.logging_config import DatabaseLogHandler, setup_logging

def test_setup_logging_returns_logger():
    logger = setup_logging(log_to_console=True, log_to_db=False)
    assert isinstance(logger, logging.Logger)

def test_setup_logging_adds_console_handler():
    logger = setup_logging(log_to_console=True, log_to_db=False)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

def test_database_handler_graceful_failure():
    handler = DatabaseLogHandler()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="Test", args=(), exc_info=None
    )
    # Should not raise even if DB unavailable
    handler.emit(record)
```

### 7.2 Integration Tests

- Run backtest with logging enabled
- Query database for expected log entries
- Verify event_type values match documentation

### 7.3 Manual Tests

- Start Streamlit, navigate to Logs tab
- Apply filters, verify correct filtering
- Expand log entry, verify metadata display

---

## 8. Rollback Procedure

If issues arise during implementation:

1. **Revert code changes**: `git checkout -- quantagent/ apps/ examples/`
2. **Downgrade migration**: `python -m alembic downgrade -1`
3. **Remove new file**: `rm quantagent/logging_config.py`
4. **Remove env vars**: Delete LOG_* from .env

The implementation is designed to be non-breaking; existing functionality should work without logging configuration.

---

## 9. Post-Implementation Tasks

After initial implementation is complete:

1. **Monitor**: Track database growth rate
2. **Tune**: Adjust LOG_LEVEL based on verbosity needs
3. **Document**: Update LOGGING_STRATEGY.md with any deviations
4. **Future**: Plan async writes if performance issues arise

---

## Related Documents

- Requirements: `docs/01_requirements/QuantAgent-yuk-RQ-structured-logging.md`
- Design: `docs/03_design/LOGGING_STRATEGY.md`
- Acceptance: `docs/05_acceptance_tests/QuantAgent-yuk-AC-structured-logging.md`
- Planning: `docs/02_planning/QuantAgent-yuk-PL-structured-logging.md`
