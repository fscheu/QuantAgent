# Requirements: Extend Minimal Operational Observability

**Issue:** QuantAgent-s62  
**Parent:** QuantAgent-kkj (M2 Milestone)  
**Related:** [PL](../02_planning/QuantAgent-s62-PL-operational-observability.md) | [DS](../03_design/QuantAgent-s62-DS-operational-observability.md) | [AC](../05_acceptance_tests/QuantAgent-s62-AC-operational-observability.md)  
**Priority:** P1  
**Type:** Feature

---

## Context

QuantAgent has partial observability pieces already in place:
- `QuantAgent-vje`: `SchedulerHeartbeat` model + `paper_trading.py` view showing scheduler status, last run, duration, recent runs.
- `QuantAgent-69d`: `llm_telemetry.py` with per-call LLM metrics persisted to `logs` table and aggregation functions (`get_session_metrics`, `get_backtest_metrics`).

For M2 paper trading operation, the operator still needs an integrated view without SSH or log grepping. Specifically:
1. The Dashboard scheduler status section is still a placeholder ("Status: unknown (MVP placeholder)").
2. LLM telemetry is persisted but never surfaced in any Streamlit view.
3. The Logs view has no environment filter, making paper troubleshooting require manual inspection.
4. The Paper Trading tab lacks an integrated view of positions, orders, PnL, and cost/latency metrics.

---

## Functional Requirements

### FR1 — Dashboard: Wire Scheduler Status

The Dashboard tab scheduler status widget must show real data from `SchedulerHeartbeat` for the current environment, replacing the existing placeholder text. Must show at minimum: status (green/yellow/red), last run time, and next-run estimate.

### FR2 — Paper Trading Tab: Integrated Operational View

The Paper Trading tab must include, for the selected `paper` environment:
- Scheduler status, last run, next run estimate.
- Open positions (current state).
- Recent orders (filtered by environment, last N).
- Recent trades with PnL summary (filtered by environment, last N).
- Portfolio/PnL summary: total portfolio value, daily PnL.

### FR3 — Paper Trading Tab: LLM Telemetry Section

The Paper Trading tab must expose a summary of LLM telemetry relevant to paper trading operations:
- Total calls, total tokens (input + output), estimated cost (if pricing constants available), and average latency (duration_ms avg).
- Filtered by `environment = 'paper'` from the `logs` table.
- Scoped to a configurable time window (e.g., last 24h by default).

### FR4 — Logs View: Environment Filter

The Logs view must include an environment dropdown filter (paper/backtest/all) so the operator can troubleshoot paper trading without inspecting all log entries.

### FR5 — Graceful Degradation

All new UI sections must degrade explicitly when:
- DB is unavailable.
- No data exists yet for the environment (first run, no heartbeats).
- Telemetry is missing or null fields are present.

No section may raise an unhandled exception or silently break the UI.

---

## Out of Scope

- Real-time WebSocket updates.
- Advanced analytics or data warehouse queries.
- Real broker integration.
- Duplicating or replacing existing `QuantAgent-vje` heartbeat logic.
- New DB tables or migrations.
