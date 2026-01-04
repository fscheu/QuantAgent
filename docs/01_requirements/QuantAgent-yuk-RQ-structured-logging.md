# Requirements: Comprehensive Structured Logging System

**Issue ID:** QuantAgent-yuk
**Type:** Epic
**Priority:** P3 (Low - pending MVP validation)
**Status:** Open
**Created:** 2026-01-04

---

## 1. Overview

This document defines the functional requirements for implementing a comprehensive structured logging system in QuantAgent. The system addresses current visibility gaps, provides audit trail capabilities, and enables effective debugging through database-persisted, filterable logs.

### 1.1 Problem Statement

**Current State:**
- Agent nodes use `print()` statements instead of proper logging (4 files, ~10 occurrences)
- No centralized logging configuration
- No database persistence of logs
- Streamlit logs tab is a placeholder (not wired to any data source)
- Inconsistent log formats and levels across modules
- No environment-based filtering capability

**Impact:**
- Difficult to debug agent behavior in production/paper trading
- No audit trail for trading decisions
- Unable to correlate events across the agent pipeline
- Manual console inspection required for troubleshooting

### 1.2 Solution Summary

Implement a dual-handler logging system that:
1. Writes structured logs to PostgreSQL for persistence and querying
2. Outputs human-readable logs to console for development
3. Provides environment-based filtering (backtest/paper/prod)
4. Integrates with Streamlit UI for log viewing
5. Maintains performance within acceptable bounds (<5% latency impact)

---

## 2. Functional Requirements

### 2.1 Logging Infrastructure

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-LOG-001 | System SHALL provide a centralized logging configuration module | Must |
| FR-LOG-002 | System SHALL support dual-handler logging (console + database) | Must |
| FR-LOG-003 | System SHALL allow independent enable/disable of each handler via environment variables | Must |
| FR-LOG-004 | System SHALL support configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) | Must |
| FR-LOG-005 | Console handler SHALL output human-readable formatted text | Must |
| FR-LOG-006 | Database handler SHALL persist structured log entries to PostgreSQL | Must |

### 2.2 Database Schema

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-DB-001 | System SHALL create a `logs` table in PostgreSQL | Must |
| FR-DB-002 | Log entries SHALL include: timestamp, level, module, message | Must |
| FR-DB-003 | Log entries SHALL support optional metadata: environment, symbol, event_type | Must |
| FR-DB-004 | Log entries SHALL support optional tracing: thread_id, checkpoint_id | Should |
| FR-DB-005 | Log entries SHALL support JSONB metadata field for structured data | Should |
| FR-DB-006 | Database SHALL have indexes on: timestamp, level, environment, symbol, event_type, thread_id | Must |

### 2.3 Agent Logging

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-AGT-001 | Each agent node SHALL log an `agent_start` event when execution begins | Must |
| FR-AGT-002 | Each agent node SHALL log an `agent_end` event when execution completes | Must |
| FR-AGT-003 | Agent logs SHALL include the symbol being analyzed | Must |
| FR-AGT-004 | Agent logs SHALL include summary metadata (not full payloads) | Should |
| FR-AGT-005 | Agent retry events SHALL be logged at INFO level | Should |
| FR-AGT-006 | Agent errors SHALL be logged at ERROR level with exception info | Must |

### 2.4 Infrastructure Logging

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-INF-001 | TradingGraph initialization SHALL log configuration details | Should |
| FR-INF-002 | LLM provider/model configuration SHALL be logged at startup | Should |
| FR-INF-003 | Checkpointer initialization SHALL be logged if enabled | Should |
| FR-INF-004 | Risk manager order rejections SHALL be logged with rejection reason | Must |
| FR-INF-005 | Backtest start/end events SHALL include environment tag | Should |

### 2.5 Print Statement Replacement

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-PRT-001 | All `print()` statements in agent code SHALL be replaced with logger calls | Must |
| FR-PRT-002 | Replacement SHALL preserve the original message content | Must |
| FR-PRT-003 | Replacement SHALL use appropriate log levels (INFO for status, ERROR for failures) | Must |
| FR-PRT-004 | Replacement SHALL add event_type metadata where applicable | Should |

**Files requiring print() replacement:**
- `quantagent/trend_agent.py` (lines 36, 48, 104)
- `quantagent/pattern_agent.py` (lines 50, 62, 114)
- `quantagent/graph_util.py` (line 286)
- `quantagent/static_util.py` (line 123)

### 2.6 Streamlit UI Integration

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-UI-001 | Logs view SHALL display logs from database | Must |
| FR-UI-002 | Logs view SHALL support filtering by log level | Must |
| FR-UI-003 | Logs view SHALL support filtering by symbol | Should |
| FR-UI-004 | Logs view SHALL support filtering by event type | Should |
| FR-UI-005 | Logs view SHALL support time window selection | Must |
| FR-UI-006 | Logs view SHALL display logs in descending timestamp order | Must |
| FR-UI-007 | Logs view SHALL limit display to 500 entries for performance | Should |
| FR-UI-008 | Logs view SHALL provide expandable detail view for metadata | Should |

### 2.7 Configuration

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-CFG-001 | LOG_LEVEL environment variable SHALL control minimum log level | Must |
| FR-CFG-002 | LOG_TO_CONSOLE environment variable SHALL enable/disable console output | Must |
| FR-CFG-003 | LOG_TO_DB environment variable SHALL enable/disable database persistence | Must |
| FR-CFG-004 | Default configuration SHALL be: INFO level, console enabled, DB enabled | Must |
| FR-CFG-005 | Configuration SHALL be documented in .env.example | Must |

---

## 3. Non-Functional Requirements

### 3.1 Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-PERF-001 | Logging overhead SHALL NOT increase backtest latency by more than 5% | <5% |
| NFR-PERF-002 | Database writes SHALL NOT block agent execution | Async-capable |
| NFR-PERF-003 | Log query in Streamlit SHALL complete within 2 seconds | <2s |

### 3.2 Reliability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-REL-001 | Logging failures SHALL NOT crash the application | Graceful degradation |
| NFR-REL-002 | Database connection failures SHALL be handled silently | No exceptions |
| NFR-REL-003 | Console logging SHALL continue if DB logging fails | Independent handlers |

### 3.3 Maintainability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-MNT-001 | Logging configuration SHALL be centralized in one module | Single file |
| NFR-MNT-002 | Event types SHALL be documented for consistency | Reference table |
| NFR-MNT-003 | Log retention policy SHALL be documented (future enhancement) | Documented |

---

## 4. Scope

### 4.1 In Scope

- New `quantagent/logging_config.py` module
- New `Log` ORM model in `quantagent/models.py`
- Alembic migration for `logs` table
- Settings additions in `quantagent/settings.py`
- Print statement replacement in 4 files
- Agent instrumentation in 4 agent files
- Infrastructure logging in 3 files
- Streamlit logs view implementation
- Entry point initialization in 3 files
- Documentation updates

### 4.2 Out of Scope

- Async/buffered database writes (future optimization)
- Log rotation/retention policies (future enhancement)
- LLM prompt/response logging (Phase 3 enhancement)
- External alerting (Slack, email notifications)
- Log export functionality
- Log sampling for high-volume scenarios

---

## 5. Constraints

| Constraint | Description |
|------------|-------------|
| C-001 | Must use existing SQLAlchemy/Alembic infrastructure |
| C-002 | Must integrate with existing `quantagent.database` module |
| C-003 | Must not require additional dependencies beyond stdlib `logging` |
| C-004 | Must maintain backward compatibility (existing code works without logging) |
| C-005 | Must follow existing code patterns in the repository |

---

## 6. Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| PostgreSQL database | Infrastructure | Available |
| SQLAlchemy ORM | Library | In use |
| Alembic migrations | Library | In use |
| Streamlit | Library | In use |
| Python logging stdlib | Library | Available |

---

## 7. User Stories

### US-001: Developer Debugging
**As a** developer
**I want to** see structured logs from agent execution
**So that** I can debug issues without adding print statements

**Acceptance Criteria:**
- Logs appear in console during local development
- Logs include module name, timestamp, and message
- Log level filtering works via environment variable

### US-002: Backtest Analysis
**As a** quantitative analyst
**I want to** query historical logs from backtest runs
**So that** I can analyze agent behavior over time

**Acceptance Criteria:**
- Logs from backtests are persisted to database
- Logs can be filtered by symbol and time range
- Agent start/end events show which agents ran

### US-003: Streamlit Monitoring
**As a** system operator
**I want to** view recent logs in the Streamlit UI
**So that** I can monitor system health without database access

**Acceptance Criteria:**
- Logs view shows recent entries (last 24h default)
- Can filter by log level (INFO, WARNING, ERROR)
- Can expand entries to see metadata details

### US-004: Production Audit
**As a** compliance officer
**I want to** have an audit trail of trading decisions
**So that** I can review the reasoning behind executed trades

**Acceptance Criteria:**
- Decision agent logs include signal, confidence, and reasoning summary
- Logs are timestamped and immutable
- Logs include environment tag (paper/prod)

---

## 8. Definition of Done

- [ ] All `print()` statements in agent code replaced with logger calls
- [ ] `logs` table created and populated during backtest execution
- [ ] Streamlit logs view displays logs with filtering
- [ ] Environment variables documented in `.env.example`
- [ ] Performance validated (<5% latency increase)
- [ ] All existing tests pass
- [ ] New unit tests for DatabaseLogHandler
- [ ] Migration runs without errors on fresh database

---

## Related Documents

- Design: `docs/03_design/LOGGING_STRATEGY.md` (source strategy document)
- Acceptance: `docs/05_acceptance_tests/QuantAgent-yuk-AC-structured-logging.md`
- Planning: `docs/02_planning/QuantAgent-yuk-PL-structured-logging.md`
- Implementation: `docs/06_implementation/QuantAgent-yuk-IM-structured-logging.md`
