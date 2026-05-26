# QuantAgent-kkj.2 — Implementation: Scheduler UI Controls (Start/Stop)

**Issue:** QuantAgent-kkj.2  
**Phase:** implementer  
**Run-ID:** 20260526T082650Z-QuantAgent-kkj.2-implementer  
**Date:** 2026-05-26  

---

## Summary

Added Start/Stop lifecycle controls to the Paper Trading view in Streamlit and a
`--environment` CLI flag to `apps/paper_trading.py`. The scheduler now runs as a
detached subprocess whose PID is persisted in `/tmp/quantagent_scheduler.pid`.

---

## Files Changed

| File | Change |
|---|---|
| `apps/streamlit/views/paper_trading.py` | Added PID helpers, `_render_scheduler_controls()`, wired into `render()` |
| `apps/paper_trading.py` | Added `--environment` CLI argument |
| `docs/user-manual/paper-trading-automation.md` | Added UI-based start/stop section; corrected `--interval-hours` flag |
| `docs/06_implementation/QuantAgent-kkj.2-IM-scheduler-ui-controls.md` | This document |

---

## Design Decisions Followed

- **Subprocess + PID file** as per the planning doc (`QuantAgent-kkj.2-PL`).
- PID written to `/tmp/quantagent_scheduler.pid`; stale PID detected via `os.kill(pid, 0)`.
- `--enable` always passed to subprocess so the scheduler starts even when the env
  flag `TRADING_SCHEDULER_ENABLED` is not set.
- Start button disabled when `is_alive AND hb_running`; Stop button disabled when
  `not is_alive`.
- `_render_scheduler_controls()` placed before `db.get_latest_heartbeat()` so controls
  are always visible even when no heartbeat exists yet.
- Heartbeat semantics (status values, `_calculate_status()`) not modified.

---

## Quality Gates

| Gate | Result |
|---|---|
| `git status --short` | 3 files modified, repo otherwise clean |
| `python -m compileall -q .` | OK |
| `ruff check apps/paper_trading.py apps/streamlit/views/paper_trading.py` | All checks passed |
| `ruff check --fix .` | 14 auto-fixed (pre-existing), 2 remaining in test files (pre-existing, not introduced) |
| `pytest tests/trading/ -x -q --tb=short` | 35 passed |

---

## Acceptance Criteria Verification

| AC | Status |
|---|---|
| AC-01: Start from UI with asset universe and mode | ✅ `_render_scheduler_controls()` exposes assets/mode/interval inputs |
| AC-02: Stop from UI gracefully | ✅ SIGTERM → 2s wait → SIGKILL; PID cleared |
| AC-03: Status updates within ≤ 5s after action | ✅ `time.sleep(2)` + `st.rerun()` after Start; `time.sleep(1)` + `st.rerun()` after Stop |
| AC-04: Start disabled when already running | ✅ `disabled=scheduler_running` on Start button |
| AC-05: Stop disabled when stopped | ✅ `disabled=not is_alive` on Stop button |
| AC-06: QuantAgent-sft heartbeat states unmodified | ✅ `_calculate_status()` unchanged; UI only reads DB |
| AC-07: User manual updated | ✅ New "Starting and Stopping from the UI" section added |
