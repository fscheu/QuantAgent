# Acceptance Criteria: Extend Minimal Operational Observability

**Issue:** QuantAgent-s62  
**Related:** [RQ](../01_requirements/QuantAgent-s62-RQ-operational-observability.md) | [DS](../03_design/QuantAgent-s62-DS-operational-observability.md)

---

## AC1 — Dashboard: Real Scheduler Status

**Given** a DB connection is available and at least one `SchedulerHeartbeat` row exists for the selected environment  
**When** the user opens the Dashboard tab  
**Then** the "Scheduler Status" widget shows a non-placeholder status (green/yellow/red) with a real last-run time, not the text "unknown (MVP placeholder)".

**Given** a DB connection is available but no heartbeat rows exist  
**When** the user opens the Dashboard tab  
**Then** the scheduler status widget shows an explicit "No data" or "Stopped" indicator without crashing.

---

## AC2 — Paper Trading Tab: Positions, Orders & PnL

**Given** the Paper Trading tab is open with environment = `paper`  
**When** the DB has orders and trades for the `paper` environment  
**Then** the tab shows:
- An orders table with rows filtered to `environment=paper`.
- A trades/PnL summary showing daily realized PnL for today.
- An open positions table (all positions, with a note that environment scoping is not available).

**Given** no orders or trades exist for `paper`  
**When** the user opens the Paper Trading tab  
**Then** each section shows an explicit "No data" message, not an error or blank page.

---

## AC3 — Paper Trading Tab: LLM Telemetry Section

**Given** the DB has `Log` rows with `event_type='llm_call'` and `environment='paper'` within the last 24 hours  
**When** the user opens the Paper Trading tab  
**Then** the LLM Cost & Latency section displays:
- Total LLM call count (non-zero integer).
- Total tokens (input + output sum).
- Average call latency in milliseconds.
- Approximate cost or "-" if tokens are 0.

**Given** no LLM telemetry log rows exist for the environment  
**When** the user opens the Paper Trading tab  
**Then** the LLM section displays an explicit info message: "No LLM telemetry data found for this environment." without raising an exception.

---

## AC4 — Logs View: Environment Filter

**Given** the Logs tab is open  
**When** the user selects environment = `paper` from the environment filter  
**Then** only log entries with `environment = 'paper'` (or `NULL` if "all" is selected) are returned.

**Given** environment filter is set to `all`  
**When** the query runs  
**Then** no environment filter is applied and all log entries matching other filters are shown.

---

## AC5 — Graceful Degradation

**Given** the DB is unavailable (DATABASE_URL not set or connection fails)  
**When** the user opens Dashboard, Paper Trading, or Logs tabs  
**Then** all new sections show their respective "no data" or "DB not available" messages without raising uncaught exceptions.

**Given** a new section returns partial or malformed data (e.g., null token fields in extra_data)  
**When** the section renders  
**Then** null/missing values are shown as "-" or 0, and the section continues to render the other available fields.

---

## Testability Notes

- AC1–AC3 can be validated manually by seeding `SchedulerHeartbeat`, `Order`, `Trade`, and `Log` rows in a local DB and loading the Streamlit app.
- AC4 can be validated manually by checking that log counts change when environment filter is switched.
- AC5 can be validated by running the app without a DB (unset DATABASE_URL) and confirming no crash.
- Unit tests for `get_environment_metrics()` can seed mock `Log` rows and assert aggregate shape/values — particularly: correct filtering by `event_type` and `environment`, correct handling of null `extra_data`, and correct empty-result shape when no rows match.
