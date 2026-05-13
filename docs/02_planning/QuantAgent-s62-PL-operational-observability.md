# Planning: Extend Minimal Operational Observability

**Issue:** QuantAgent-s62  
**Related:** [RQ](../01_requirements/QuantAgent-s62-RQ-operational-observability.md) | [DS](../03_design/QuantAgent-s62-DS-operational-observability.md) | [AC](../05_acceptance_tests/QuantAgent-s62-AC-operational-observability.md)

---

## Summary

Four targeted changes to wire existing observability primitives into the Streamlit UI:
1. Wire Dashboard scheduler status placeholder with real heartbeat data.
2. Extend Paper Trading tab with integrated positions/orders/PnL view.
3. Add LLM telemetry section to Paper Trading tab.
4. Add environment filter to Logs view.

No new tables. No new models. No new migrations. Reuse `QuantAgent-vje` + `QuantAgent-69d` primitives exclusively.

---

## Task Breakdown

### Task 1 — `quantagent/llm_telemetry.py`: Add `get_environment_metrics()`

Add a new aggregation function `get_environment_metrics(db_session, environment, hours_back=24)` that:
- Queries `Log` rows with `event_type='llm_call'` and `environment=environment`.
- Optionally scoped to the last `hours_back` hours.
- Returns the same aggregate shape as existing `get_session_metrics()` / `get_backtest_metrics()`.

This is the only code change outside `apps/streamlit/`.

**Estimated effort:** small (10–20 lines).

### Task 2 — `apps/streamlit/services/db.py`: Add `get_paper_llm_metrics()`

Add `get_paper_llm_metrics(environment, hours_back=24)` method to `DbHandle` that:
- Opens a session and calls `get_environment_metrics()` from `llm_telemetry`.
- Returns the aggregate dict or an empty/default dict on failure.

**Estimated effort:** small (~15 lines).

### Task 3 — `apps/streamlit/views/dashboard.py`: Wire Scheduler Status

Replace the placeholder block:
```python
st.write("Status: unknown (MVP placeholder)")
st.write("Next run: -  | Last run: -  | Errors: -")
```

With a real call to `db.get_latest_heartbeat(environment)` that renders:
- Status emoji + text (reuse logic from `paper_trading.py:_calculate_status()`).
- Last run (relative time, reuse `_humanize_time()` from `paper_trading.py`).
- Link/pointer to Paper Trading tab for full detail.

**Estimated effort:** small (~20 lines).

### Task 4 — `apps/streamlit/views/paper_trading.py`: Add Positions/Orders/PnL + Telemetry

Extend `render()` to add two new sections below the existing scheduler status + recent runs:

**Section A — Positions, Orders & PnL** (after recent runs):
- Open positions table (all positions, since `Position` has no environment column).
- Recent orders table (filtered by environment, last 20).
- PnL summary: sum unrealized PnL from positions + daily realized PnL from trades.

**Section B — LLM Cost & Latency (last 24h)**:
- Call `db.get_paper_llm_metrics(environment)`.
- Show: total calls, total tokens, avg latency (ms), approx cost (if estimable).
- If no data: explicit info message.

**Estimated effort:** medium (~60–80 lines).

### Task 5 — `apps/streamlit/views/logs.py`: Add Environment Filter

Add environment selectbox filter (options: `all`, `paper`, `backtest`) before current filters. Apply to query when not `all`.

**Estimated effort:** small (~15 lines).

---

## Implementation Order

1. `llm_telemetry.py` — Task 1 (no UI dependencies)
2. `services/db.py` — Task 2 (depends on Task 1)
3. `views/dashboard.py` — Task 3 (depends on Task 2)
4. `views/paper_trading.py` — Task 4 (depends on Task 2)
5. `views/logs.py` — Task 5 (independent)

---

## Files to Touch

| File | Change |
|---|---|
| `quantagent/llm_telemetry.py` | Add `get_environment_metrics()` |
| `apps/streamlit/services/db.py` | Add `get_paper_llm_metrics()` |
| `apps/streamlit/views/dashboard.py` | Wire scheduler status, import helpers from paper_trading |
| `apps/streamlit/views/paper_trading.py` | Add positions/orders/PnL + telemetry sections |
| `apps/streamlit/views/logs.py` | Add environment filter |

**No other files.** No migrations, no new models, no changes to scheduler or core engine.

---

## Risks

| Risk | Mitigation |
|---|---|
| `Position` lacks environment column — can't filter by paper | Show all positions; add note in UI. Post-MVP: add environment to Position model (separate ticket). |
| LLM telemetry logs may be sparse or absent if paper scheduler hasn't run LLM calls yet | Explicit "no telemetry data" message; section doesn't crash. |
| `_calculate_status()` and `_humanize_time()` are private to `paper_trading.py` | Extract to `apps/streamlit/utils/scheduler_ui.py` or inline in dashboard. Prefer inline to avoid new files unless >3 usages. |
| DB-side `environment` column on `Log` uses `String(20)` not Enum — filter must use string literal | Query with `Log.environment == environment` (string); already consistent with existing Logs view. |
