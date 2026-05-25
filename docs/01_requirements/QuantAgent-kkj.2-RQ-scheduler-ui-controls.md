# QuantAgent-kkj.2 — Requirements: Scheduler UI Controls (Start/Stop)

**Issue:** QuantAgent-kkj.2  
**Parent:** QuantAgent-kkj (M2 Milestone Tracking)  
**Type:** feature  
**Priority:** P1  
**Created:** 2026-05-25  

---

## Context

`TradingScheduler` for paper trading was implemented in `QuantAgent-3o4` and hardened
in `QuantAgent-sft` (heartbeat running/stuck/error, auditable signal→order→trade→position
chain). The Paper Trading view in Streamlit (`apps/streamlit/views/paper_trading.py`)
already shows scheduler status and recent run history, but provides no lifecycle controls.

Today, starting or stopping the scheduler requires terminal access:

```bash
python apps/paper_trading.py --environment paper
```

This is a practical barrier for M3: any evaluator or operator must have shell access to
trigger a paper trading cycle. The QuantAgent-aki readiness report (M2 pilot) explicitly
recommended adding UI-based scheduler controls as a prerequisite for advancing to M3.

---

## Problem Statement

An operator using the Streamlit UI cannot:
- Start a paper trading scheduler cycle (with configurable parameters)
- Stop a running scheduler gracefully
- Observe state changes immediately after triggering start/stop

---

## Requirements

### RQ-01 — Start Scheduler from UI

The Paper Trading view must expose a **Start** control that:

1. Accepts the following parameters before launching:
   - **Strategies/Environment**: hardcoded `paper` (no user input needed — M3 scope is paper only)
   - **Asset universe**: text input, comma-separated symbols (default: `BTC,SPX`)
   - **Mode**: single-cycle (`run_once`) or continuous (interval-based via APScheduler)
   - **Interval** (only for continuous mode): interval in hours, minimum 0.25h
2. Launches the scheduler as a **subprocess** (survives Streamlit reruns)
3. Persists the subprocess PID so the UI can send a stop signal
4. Disables the Start button / shows a warning if the scheduler is already running
   (determined via heartbeat `status == "running"` in DB)

### RQ-02 — Stop Scheduler from UI

The Paper Trading view must expose a **Stop** control that:

1. Sends `SIGTERM` to the persisted subprocess PID
2. Waits briefly (≤ 2s) for graceful termination; sends `SIGKILL` on timeout
3. Clears the persisted PID after termination
4. Disables the Stop button when the scheduler is not running
   (no active PID or heartbeat `status` is `stopped`/`completed`/`error`)

### RQ-03 — State Refresh After Action

After Start or Stop is triggered:

1. The UI must refresh scheduler status within ≤ 5 seconds without requiring manual
   browser reload
2. The existing status indicators (running/stuck/error/stopped emojis, last run, etc.)
   must update to reflect the new state
3. A Streamlit `st.rerun()` or polling loop with short sleep is acceptable

### RQ-04 — Heartbeat Integrity

The QuantAgent-sft heartbeat semantics must not be altered:
- `status` values: `running`, `completed`, `error` (only set by TradingScheduler itself)
- The UI does not write to `scheduler_heartbeats` table
- Status determination logic in `_calculate_status()` remains unchanged

### RQ-05 — PID Persistence

The subprocess PID must be persisted in a way that survives brief Streamlit server
restarts and is not stored in ephemeral `session_state` alone. Acceptable options:

- **Option A (preferred):** Write PID to a temp file with known path
  (e.g., `/tmp/quantagent_scheduler.pid`)
- **Option B:** Store PID in the DB via a lightweight `SchedulerControl` table
  (adds migration complexity; not preferred for this scope)

Option A is preferred: simpler, no schema change, consistent with the subprocess approach.

### RQ-06 — User Manual Update

The section of the user manual that instructs operators to use CLI to start the scheduler
must be updated to describe the UI-based controls.

---

## Out of Scope

- Broker real connection
- New strategies
- Pause/resume scheduler (not supported by current APScheduler setup)
- Redesigning dashboard or other views
- Environment selection in Start form (always paper)

---

## Acceptance Criteria

- [ ] AC-01: From the Paper Trading UI, the operator can initiate the scheduler with
  asset universe and mode (single-cycle or continuous).
- [ ] AC-02: From the Paper Trading UI, the operator can stop the running scheduler
  gracefully.
- [ ] AC-03: After Start/Stop, the scheduler status in the UI updates within ≤ 5 seconds.
- [ ] AC-04: The Start button is disabled or shows a warning when the scheduler is
  already running.
- [ ] AC-05: The Stop button is disabled when the scheduler is stopped.
- [ ] AC-06: The QuantAgent-sft heartbeat states and semantics are unmodified.
- [ ] AC-07: The user manual reflects UI-based scheduler operation.

---

## Dependencies

| Issue | Description | Status |
|---|---|---|
| QuantAgent-sft | Paper runtime hardening (heartbeat) | ✅ merged |
| QuantAgent-vje | Scheduler monitoring dashboard | ✅ merged |
| QuantAgent-3o4 | TradingScheduler implementation | ✅ merged |

---

## Key Files

| File | Role |
|---|---|
| `apps/streamlit/views/paper_trading.py` | Target view to extend with controls |
| `apps/paper_trading.py` | CLI entry point — subprocess target |
| `quantagent/trading/scheduler.py` | TradingScheduler (do not modify internals) |
| `quantagent/models.py` | SchedulerHeartbeat model (read-only from UI) |
| `apps/streamlit/services/db.py` | DB service (read-only from UI) |
| `docs/user-manual/` | User manual to update |
