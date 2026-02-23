# Acceptance Criteria: Comprehensive Structured Logging System

**Issue ID:** QuantAgent-yuk
**Type:** Epic
**Priority:** P3 (Low - pending MVP validation)
**Status:** Open
**Created:** 2026-01-04

---

## 1. Overview

This document defines the acceptance criteria and test oracles for the structured logging system implementation. Each criterion follows the Given/When/Then format and specifies measurable outcomes.

---

## 2. Phase 1: Infrastructure Setup

### AC-1.1: Log Model Created

```
Given the quantagent/models.py file exists
When the Log model is added
Then the model SHALL have the following columns:
  - id (Integer, primary key)
  - timestamp (DateTime, not null, indexed)
  - level (String(10), not null, indexed)
  - module (String(100), not null)
  - message (Text, not null)
  - environment (String(20), indexed)
  - symbol (String(20), indexed)
  - event_type (String(50), indexed)
  - metadata (JSONB)
  - thread_id (String(100), indexed)
  - checkpoint_id (String(100))
```

**Verification:**
```python
from quantagent.models import Log
assert hasattr(Log, 'id')
assert hasattr(Log, 'timestamp')
assert hasattr(Log, 'level')
assert hasattr(Log, 'module')
assert hasattr(Log, 'message')
assert hasattr(Log, 'environment')
assert hasattr(Log, 'symbol')
assert hasattr(Log, 'event_type')
assert hasattr(Log, 'metadata')
assert hasattr(Log, 'thread_id')
assert hasattr(Log, 'checkpoint_id')
```

### AC-1.2: Database Migration Runs Successfully

```
Given a clean database with existing schema
When running: python -m alembic upgrade head
Then the migration SHALL complete without errors
And a 'logs' table SHALL exist in the database
And the table SHALL have all required indexes
```

**Verification:**
```sql
SELECT table_name FROM information_schema.tables WHERE table_name = 'logs';
-- Should return 1 row

SELECT indexname FROM pg_indexes WHERE tablename = 'logs';
-- Should include: idx_logs_timestamp, idx_logs_level, idx_logs_environment,
--                 idx_logs_symbol, idx_logs_event_type, idx_logs_thread_id
```

### AC-1.3: Logging Config Module Created

```
Given the quantagent package exists
When the logging_config.py module is created
Then it SHALL export:
  - DatabaseLogHandler class
  - setup_logging() function
And setup_logging() SHALL accept parameters:
  - level (str, optional)
  - log_to_console (bool, default True)
  - log_to_db (bool, default True)
  - environment (str, optional)
  - symbol (str, optional)
```

**Verification:**
```python
from quantagent.logging_config import DatabaseLogHandler, setup_logging
import inspect

# Check setup_logging signature
sig = inspect.signature(setup_logging)
assert 'level' in sig.parameters
assert 'log_to_console' in sig.parameters
assert 'log_to_db' in sig.parameters
assert 'environment' in sig.parameters
assert 'symbol' in sig.parameters
```

### AC-1.4: Console Handler Outputs Human-Readable Format

```
Given logging is configured with log_to_console=True
When a log message is emitted at INFO level
Then console output SHALL match pattern:
  YYYY-MM-DD HH:MM:SS - module.name - LEVEL - message
```

**Verification:**
```python
import logging
from io import StringIO
from quantagent.logging_config import setup_logging

# Capture stdout
stream = StringIO()
handler = logging.StreamHandler(stream)
setup_logging(level="INFO", log_to_console=True, log_to_db=False)

logger = logging.getLogger("test.module")
logger.info("Test message")

output = stream.getvalue()
assert "test.module" in output
assert "INFO" in output
assert "Test message" in output
```

### AC-1.5: Database Handler Persists Logs

```
Given logging is configured with log_to_db=True
And a database connection is available
When a log message is emitted
Then a new row SHALL be inserted into the logs table
And the row SHALL contain the correct level, module, and message
```

**Verification:**
```python
from quantagent.logging_config import setup_logging
from quantagent.database import SessionLocal
from quantagent.models import Log
import logging

setup_logging(level="INFO", log_to_console=False, log_to_db=True, environment="test")

logger = logging.getLogger("test.db_handler")
logger.info("Database persistence test")

with SessionLocal() as session:
    log_entry = session.query(Log).filter(
        Log.module == "test.db_handler",
        Log.message == "Database persistence test"
    ).first()

    assert log_entry is not None
    assert log_entry.level == "INFO"
    assert log_entry.environment == "test"
```

### AC-1.6: Settings Module Updated

```
Given the quantagent/settings.py file exists
When logging settings are added
Then the following variables SHALL be exported:
  - LOG_LEVEL (str, default "INFO")
  - LOG_TO_CONSOLE (bool, default True)
  - LOG_TO_DB (bool, default True)
And values SHALL be read from environment variables
```

**Verification:**
```python
from quantagent import settings

assert hasattr(settings, 'LOG_LEVEL')
assert hasattr(settings, 'LOG_TO_CONSOLE')
assert hasattr(settings, 'LOG_TO_DB')
assert settings.LOG_LEVEL == "INFO"  # default
assert settings.LOG_TO_CONSOLE == True  # default
assert settings.LOG_TO_DB == True  # default
```

---

## 3. Phase 2: Print Statement Replacement

### AC-2.1: No Print Statements in Agent Code

```
Given the implementation is complete
When searching for print() in quantagent/
Then the following files SHALL NOT contain print() calls:
  - quantagent/trend_agent.py
  - quantagent/pattern_agent.py
  - quantagent/graph_util.py
  - quantagent/static_util.py
```

**Verification:**
```bash
grep -r "print(" quantagent/trend_agent.py quantagent/pattern_agent.py \
     quantagent/graph_util.py quantagent/static_util.py
# Should return empty (no matches)
```

### AC-2.2: Replacement Uses Appropriate Log Levels

```
Given print() statements are replaced with logger calls
When reviewing the replacements
Then status messages SHALL use logger.info()
And error messages SHALL use logger.error()
And warning messages SHALL use logger.warning()
```

**Verification (manual review):**
- `"No precomputed trend image found"` -> `logger.info()`
- `"Failed to generate trend image"` -> `logger.error()`
- `"Retrying without system message"` -> `logger.info()`
- `"ValueError at graph_util.py"` -> `logger.error()`

### AC-2.3: Event Types Added to Replacement Logs

```
Given print() statements are replaced
When the replacement logs are emitted
Then they SHALL include event_type in the extra dict
```

**Verification:**
```python
# In trend_agent.py, the replacement should look like:
logger.info(
    "No precomputed trend image found, generating with tool...",
    extra={'event_type': 'trend_image_generation'}
)
```

---

## 4. Phase 3: Agent Logging

### AC-3.1: Indicator Agent Logs Start/End

```
Given the indicator_agent.py is instrumented
When the indicator_agent_node function executes
Then it SHALL log an agent_start event at the beginning
And it SHALL log an agent_end event at the end
And both events SHALL include the symbol from state
```

**Verification:**
```python
from quantagent.database import SessionLocal
from quantagent.models import Log

# After running a backtest with symbol "BTC"
with SessionLocal() as session:
    start_log = session.query(Log).filter(
        Log.module == "quantagent.indicator_agent",
        Log.event_type == "agent_start"
    ).first()

    end_log = session.query(Log).filter(
        Log.module == "quantagent.indicator_agent",
        Log.event_type == "agent_end"
    ).first()

    assert start_log is not None
    assert end_log is not None
    assert start_log.symbol == "BTC"
    assert end_log.symbol == "BTC"
```

### AC-3.2: Pattern Agent Logs Start/End

```
Given the pattern_agent.py is instrumented
When the pattern_agent_node function executes
Then it SHALL log an agent_start event at the beginning
And it SHALL log an agent_end event at the end
And the agent_end event SHALL include pattern detection summary in metadata
```

**Verification:**
```python
with SessionLocal() as session:
    end_log = session.query(Log).filter(
        Log.module == "quantagent.pattern_agent",
        Log.event_type == "agent_end"
    ).first()

    assert end_log is not None
    assert end_log.metadata is not None
    assert 'pattern' in end_log.metadata
```

### AC-3.3: Trend Agent Logs Start/End

```
Given the trend_agent.py is instrumented
When the trend_agent_node function executes
Then it SHALL log an agent_start event at the beginning
And it SHALL log an agent_end event at the end
And the agent_end event SHALL include trend direction in metadata
```

**Verification:**
```python
with SessionLocal() as session:
    end_log = session.query(Log).filter(
        Log.module == "quantagent.trend_agent",
        Log.event_type == "agent_end"
    ).first()

    assert end_log is not None
    assert end_log.metadata is not None
    assert 'trend' in end_log.metadata
```

### AC-3.4: Decision Agent Logs Start/End

```
Given the decision_agent.py is instrumented
When the trade_decision_node function executes
Then it SHALL log an agent_start event at the beginning
And it SHALL log an agent_end event at the end
And the agent_end event SHALL include signal and confidence in metadata
```

**Verification:**
```python
with SessionLocal() as session:
    end_log = session.query(Log).filter(
        Log.module == "quantagent.decision_agent",
        Log.event_type == "agent_end"
    ).first()

    assert end_log is not None
    assert end_log.metadata is not None
    assert 'signal' in end_log.metadata
    assert 'confidence' in end_log.metadata
```

### AC-3.5: All Four Agents Log Per Analysis Cycle

```
Given a complete analysis cycle runs (all 4 agents)
When querying logs for a specific thread_id
Then there SHALL be exactly 8 log entries (4 starts + 4 ends)
And they SHALL be in chronological order
```

**Verification:**
```python
with SessionLocal() as session:
    cycle_logs = session.query(Log).filter(
        Log.thread_id == "test-thread-123",
        Log.event_type.in_(["agent_start", "agent_end"])
    ).order_by(Log.timestamp).all()

    assert len(cycle_logs) == 8

    # Verify order: indicator -> pattern -> trend -> decision (parallel may vary)
    modules = [log.module for log in cycle_logs]
    assert "quantagent.indicator_agent" in modules
    assert "quantagent.pattern_agent" in modules
    assert "quantagent.trend_agent" in modules
    assert "quantagent.decision_agent" in modules
```

---

## 5. Phase 4: Infrastructure Logging

### AC-4.1: TradingGraph Logs Initialization

```
Given logging is configured
When TradingGraph is instantiated
Then it SHALL log a graph_init event
And it SHALL log an llm_config event with provider/model details
And it SHALL log a graph_ready event when initialization completes
```

**Verification:**
```python
with SessionLocal() as session:
    init_logs = session.query(Log).filter(
        Log.module == "quantagent.trading_graph",
        Log.event_type.in_(["graph_init", "llm_config", "graph_ready"])
    ).all()

    event_types = [log.event_type for log in init_logs]
    assert "graph_init" in event_types
    assert "llm_config" in event_types
    assert "graph_ready" in event_types
```

### AC-4.2: Risk Manager Logs Rejections

```
Given an order violates risk rules
When the risk manager validates the order
Then it SHALL log an order_rejected event at WARNING level
And the log SHALL include the rejection reason in metadata
```

**Verification:**
```python
with SessionLocal() as session:
    rejection_log = session.query(Log).filter(
        Log.module == "quantagent.trading.risk_manager",
        Log.event_type == "order_rejected"
    ).first()

    assert rejection_log is not None
    assert rejection_log.level == "WARNING"
    assert rejection_log.metadata is not None
    assert 'reason' in rejection_log.metadata
```

### AC-4.3: Backtest Logs Include Environment Tag

```
Given a backtest run is executed
When the backtest logs events
Then all log entries SHALL have environment="backtest"
```

**Verification:**
```python
with SessionLocal() as session:
    backtest_logs = session.query(Log).filter(
        Log.event_type.in_(["backtest_start", "backtest_end"])
    ).all()

    for log in backtest_logs:
        assert log.environment == "backtest"
```

---

## 6. Phase 5: Streamlit UI Integration

### AC-5.1: Logs View Displays Data

```
Given logs exist in the database
When navigating to the Logs tab in Streamlit
Then logs SHALL be displayed in a dataframe
And the dataframe SHALL show: timestamp, level, module, event_type, symbol, message
```

**Verification (manual):**
1. Run backtest to generate logs
2. Open Streamlit app
3. Navigate to Logs tab
4. Verify dataframe displays with correct columns

### AC-5.2: Level Filter Works

```
Given logs exist at multiple levels (INFO, WARNING, ERROR)
When selecting only ERROR in the level filter
Then only ERROR level logs SHALL be displayed
```

**Verification (manual):**
1. Generate logs at multiple levels
2. Select "ERROR" in level multiselect
3. Verify only ERROR logs appear

### AC-5.3: Symbol Filter Works

```
Given logs exist for multiple symbols (BTC, SPX)
When entering "BTC" in the symbol filter
Then only logs with symbol containing "BTC" SHALL be displayed
```

**Verification (manual):**
1. Generate logs for multiple symbols
2. Enter "BTC" in symbol text input
3. Verify only BTC logs appear

### AC-5.4: Time Window Filter Works

```
Given logs exist from multiple time periods
When setting hours_back to 1
Then only logs from the last 1 hour SHALL be displayed
```

**Verification (manual):**
1. Generate logs at different times
2. Set hours_back to 1
3. Verify only recent logs appear

### AC-5.5: Expandable Details Work

```
Given logs with metadata exist
When clicking on the expander for a log entry
Then the full message SHALL be displayed
And the metadata SHALL be displayed as JSON
```

**Verification (manual):**
1. Find log with metadata in UI
2. Click expander
3. Verify metadata JSON is visible

### AC-5.6: Query Performance Acceptable

```
Given 10,000+ logs exist in the database
When loading the Logs view
Then the query SHALL complete within 2 seconds
```

**Verification:**
```python
import time
from quantagent.database import SessionLocal
from quantagent.models import Log
from datetime import datetime, timedelta

start = time.time()
with SessionLocal() as session:
    logs = session.query(Log).filter(
        Log.timestamp >= datetime.utcnow() - timedelta(hours=24)
    ).order_by(Log.timestamp.desc()).limit(500).all()
elapsed = time.time() - start

assert elapsed < 2.0, f"Query took {elapsed:.2f}s, expected <2s"
```

---

## 7. Phase 6: Entry Point Initialization

### AC-6.1: run_backtest.py Initializes Logging

```
Given examples/run_backtest.py is executed
When the script starts
Then setup_logging() SHALL be called with environment="backtest"
And console logging SHALL be enabled
And database logging SHALL be enabled
```

**Verification:**
```bash
python examples/run_backtest.py
# Verify console output shows formatted logs
# Verify logs appear in database with environment="backtest"
```

### AC-6.2: Streamlit app.py Initializes Logging

```
Given apps/streamlit/app.py is executed
When the app starts
Then setup_logging() SHALL be called
And console logging SHALL be disabled (to avoid Streamlit clutter)
And database logging SHALL be enabled
```

**Verification:**
```bash
streamlit run apps/streamlit/app.py
# Verify no log output in terminal
# Verify logs appear in database
```

### AC-6.3: Flask app Initializes Logging

```
Given apps/flask/web_interface.py is executed
When the app starts
Then setup_logging() SHALL be called with log_to_db=False
And console logging SHALL be enabled
```

**Verification:**
```bash
python apps/flask/web_interface.py
# Verify console output shows formatted logs
# Verify no logs in database (legacy app, DB disabled)
```

---

## 8. Performance Criteria

### AC-PERF-1: Latency Impact Within Bounds

```
Given a baseline backtest execution time (without logging)
When running the same backtest with logging enabled
Then the execution time SHALL NOT increase by more than 5%
```

**Verification:**
```python
import time

# Baseline (logging disabled)
os.environ["LOG_TO_DB"] = "false"
os.environ["LOG_TO_CONSOLE"] = "false"
start = time.time()
# Run backtest
baseline = time.time() - start

# With logging
os.environ["LOG_TO_DB"] = "true"
os.environ["LOG_TO_CONSOLE"] = "true"
start = time.time()
# Run same backtest
with_logging = time.time() - start

overhead = (with_logging - baseline) / baseline * 100
assert overhead < 5.0, f"Logging overhead {overhead:.2f}%, expected <5%"
```

### AC-PERF-2: No Blocking on DB Failure

```
Given the database is unavailable
When logging events are emitted
Then the application SHALL NOT crash
And console logging SHALL continue to work
```

**Verification:**
```python
# Disconnect database
# Run agent analysis
# Verify no exception raised
# Verify console still outputs logs
```

---

## 9. Error Handling Criteria

### AC-ERR-1: Graceful Handler Error Recovery

```
Given the DatabaseLogHandler encounters an error
When emit() is called
Then the error SHALL be handled by handleError()
And the application SHALL continue running
And no exception SHALL propagate to calling code
```

**Verification:**
```python
import logging
from quantagent.logging_config import DatabaseLogHandler

handler = DatabaseLogHandler()
handler.environment = None  # Force potential error

# This should not raise
record = logging.LogRecord(
    name="test", level=logging.INFO, pathname="", lineno=0,
    msg="Test", args=(), exc_info=None
)
handler.emit(record)  # Should not raise
```

---

## 10. Regression Criteria

### AC-REG-1: Existing Tests Pass

```
Given the logging implementation is complete
When running: pytest tests/
Then all existing tests SHALL pass
```

**Verification:**
```bash
pytest tests/ -v
# All tests should pass
```

### AC-REG-2: Backtest Produces Same Results

```
Given logging is enabled
When running a deterministic backtest
Then the trading decisions SHALL be identical to runs without logging
And the PnL calculations SHALL be identical
```

**Verification:**
```python
# Run backtest without logging, capture results
# Run backtest with logging, capture results
# Compare trading decisions and PnL
assert results_with_logging == results_without_logging
```

---

## 11. Documentation Criteria

### AC-DOC-1: .env.example Updated

```
Given the implementation is complete
When reviewing .env.example
Then it SHALL include:
  - LOG_LEVEL with description and default
  - LOG_TO_CONSOLE with description and default
  - LOG_TO_DB with description and default
```

**Verification:**
```bash
grep -E "LOG_LEVEL|LOG_TO_CONSOLE|LOG_TO_DB" .env.example
# Should show all three variables with comments
```

### AC-DOC-2: Event Types Documented

```
Given the implementation is complete
When reviewing LOGGING_STRATEGY.md
Then the Event Types Reference table SHALL include all implemented event types
```

**Verification (manual):**
- Review LOGGING_STRATEGY.md Event Types Reference table
- Verify all implemented event types are listed

---

## 12. Test Checklist Summary

| Phase | Test ID | Description | Type |
|-------|---------|-------------|------|
| 1 | AC-1.1 | Log model has required columns | Unit |
| 1 | AC-1.2 | Migration runs successfully | Integration |
| 1 | AC-1.3 | Logging config module exports correctly | Unit |
| 1 | AC-1.4 | Console handler format correct | Unit |
| 1 | AC-1.5 | Database handler persists logs | Integration |
| 1 | AC-1.6 | Settings module updated | Unit |
| 2 | AC-2.1 | No print statements remain | Static |
| 2 | AC-2.2 | Correct log levels used | Review |
| 2 | AC-2.3 | Event types added | Review |
| 3 | AC-3.1 | Indicator agent logs start/end | Integration |
| 3 | AC-3.2 | Pattern agent logs start/end | Integration |
| 3 | AC-3.3 | Trend agent logs start/end | Integration |
| 3 | AC-3.4 | Decision agent logs start/end | Integration |
| 3 | AC-3.5 | All agents log per cycle | Integration |
| 4 | AC-4.1 | TradingGraph logs initialization | Integration |
| 4 | AC-4.2 | Risk manager logs rejections | Integration |
| 4 | AC-4.3 | Backtest logs have environment tag | Integration |
| 5 | AC-5.1 | Logs view displays data | Manual |
| 5 | AC-5.2 | Level filter works | Manual |
| 5 | AC-5.3 | Symbol filter works | Manual |
| 5 | AC-5.4 | Time window filter works | Manual |
| 5 | AC-5.5 | Expandable details work | Manual |
| 5 | AC-5.6 | Query performance acceptable | Performance |
| 6 | AC-6.1 | run_backtest.py initializes logging | Integration |
| 6 | AC-6.2 | Streamlit app initializes logging | Integration |
| 6 | AC-6.3 | Flask app initializes logging | Integration |
| - | AC-PERF-1 | Latency impact <5% | Performance |
| - | AC-PERF-2 | No blocking on DB failure | Resilience |
| - | AC-ERR-1 | Graceful handler error recovery | Unit |
| - | AC-REG-1 | Existing tests pass | Regression |
| - | AC-REG-2 | Backtest results unchanged | Regression |
| - | AC-DOC-1 | .env.example updated | Review |
| - | AC-DOC-2 | Event types documented | Review |

---

## Related Documents

- Requirements: `docs/01_requirements/QuantAgent-yuk-RQ-structured-logging.md`
- Design: `docs/03_design/LOGGING_STRATEGY.md`
- Planning: `docs/02_planning/QuantAgent-yuk-PL-structured-logging.md`
- Implementation: `docs/06_implementation/QuantAgent-yuk-IM-structured-logging.md`
