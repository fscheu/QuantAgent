# QuantAgent-kkj.2 — Planning: Scheduler UI Controls (Start/Stop)

**Issue:** QuantAgent-kkj.2  
**Phase:** planner  
**Run-ID:** 20260525T204955Z-QuantAgent-kkj.2-planner  
**Created:** 2026-05-25  

---

## Summary

Add Start/Stop lifecycle controls to the Paper Trading view in Streamlit. The scheduler
currently runs only from CLI; this change makes it operable from the UI as a subprocess,
persisting its PID to a temp file for stop signaling.

---

## Approach Decision: Subprocess + PID File

The TradingScheduler uses APScheduler's `BackgroundScheduler`, which owns its own thread
pool and cannot be safely shared across Streamlit's re-render lifecycle. The two options:

| Option | Pros | Cons |
|---|---|---|
| **Thread in Streamlit process** | No subprocess overhead | Streamlit reruns can clobber thread state; scheduler dies on reload |
| **Subprocess (chosen)** | Survives Streamlit reruns and minor restarts | Slightly more complex start/stop signaling |

**Decision:** subprocess, with PID written to `/tmp/quantagent_scheduler.pid`.

This matches the existing `apps/paper_trading.py` CLI contract; the UI invokes it as a
subprocess rather than calling the Python class directly.

---

## Architecture

```
Paper Trading View (Streamlit)
  ├── _render_scheduler_controls(db, environment)        [NEW]
  │     ├── Reads PID from /tmp/quantagent_scheduler.pid
  │     ├── Reads heartbeat status from DB (existing)
  │     ├── [Start] → subprocess.Popen(["python", "apps/paper_trading.py", ...])
  │     │             → writes PID to /tmp/quantagent_scheduler.pid
  │     │             → st.rerun() after 2s
  │     └── [Stop]  → os.kill(pid, SIGTERM) → 2s wait → SIGKILL if alive
  │                   → removes PID file
  │                   → st.rerun()
  │
  └── render(db, environment)                            [MODIFIED]
        → calls _render_scheduler_controls() before status card
```

### Subprocess Command

The implementer must invoke `apps/paper_trading.py` as a detached subprocess. The exact
invocation:

```python
import subprocess, sys, os

cmd = [
    sys.executable,
    "apps/paper_trading.py",
    "--environment", "paper",
    "--assets", assets_str,
]
if mode == "once":
    cmd.append("--run-once")
else:
    cmd += ["--interval-hours", str(interval_hours)]

proc = subprocess.Popen(
    cmd,
    cwd=repo_root,
    start_new_session=True,      # detach from Streamlit's process group
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
```

The `--run-once` flag must be verified in `apps/paper_trading.py`. Check if it exists;
add it if missing (this is a minimal addition to the CLI, not a scheduler core change).

### PID File Path

`/tmp/quantagent_scheduler.pid` — chosen because:
- Survives brief Streamlit server restarts
- Readable/writable by the app process without permissions issues
- Cleaned up on stop or process exit

A stale PID file (process no longer alive) must be detected by checking
`os.kill(pid, 0)` before showing the scheduler as running.

---

## File Changes

| File | Change Type | Description |
|---|---|---|
| `apps/streamlit/views/paper_trading.py` | modify | Add `_render_scheduler_controls()` and call it from `render()` |
| `apps/paper_trading.py` | modify (minimal) | Add `--run-once` flag if not present; verify `--environment` flag exists |
| `docs/user-manual/paper-trading-automation.md` | update | Add section on UI-based start/stop |
| `docs/01_requirements/QuantAgent-kkj.2-RQ-scheduler-ui-controls.md` | new | Requirements (planner artifact) |
| `docs/02_planning/QuantAgent-kkj.2-PL-scheduler-ui-controls.md` | new | This planning doc |
| `docs/05_acceptance_tests/QuantAgent-kkj.2-AC-scheduler-ui-controls.md` | new | Acceptance tests |
| `docs/06_implementation/QuantAgent-kkj.2-IM-scheduler-ui-controls.md` | implementer | Created by implementer |

---

## Implementation Steps (for implementer phase)

### Step 1 — Audit `apps/paper_trading.py` CLI

Check if `--run-once` and `--environment` flags exist. If `--run-once` is missing, add
it: when present, call `scheduler.run_once()` and exit instead of starting the APScheduler
loop. This is a safe addition: the CLI entry point already has this pattern.

### Step 2 — Add `_render_scheduler_controls()` to paper_trading view

New function placed before `_render_status_card()` call in `render()`. Structure:

```
def _render_scheduler_controls(db, environment):
    pid = _read_pid()
    is_alive = _pid_is_alive(pid)
    hb = db.get_latest_heartbeat(environment)  # already available
    hb_running = hb and hb.get("status") == "running"
    scheduler_running = is_alive and hb_running

    st.subheader("Scheduler Controls")
    
    with st.expander("Start Scheduler", expanded=not scheduler_running):
        assets_input = st.text_input("Assets", value="BTC,SPX")
        mode = st.radio("Mode", ["Single cycle", "Continuous"])
        if mode == "Continuous":
            interval = st.number_input("Interval (hours)", min_value=0.25, value=1.0)
        
        start_disabled = scheduler_running
        if st.button("▶ Start", disabled=start_disabled):
            _launch_subprocess(assets_input, mode, interval, environment)
            time.sleep(2)
            st.rerun()

    col1, _ = st.columns([1, 3])
    with col1:
        stop_disabled = not is_alive
        if st.button("■ Stop", disabled=stop_disabled):
            _stop_subprocess(pid)
            time.sleep(1)
            st.rerun()
```

### Step 3 — Helper functions

- `_read_pid() -> Optional[int]`: reads `/tmp/quantagent_scheduler.pid`; returns None
  if file absent or content invalid
- `_pid_is_alive(pid) -> bool`: uses `os.kill(pid, 0)` to probe; returns False if pid
  is None or OSError (process not found)
- `_write_pid(pid: int)`: writes PID to file
- `_clear_pid()`: removes PID file
- `_launch_subprocess(assets, mode, interval, environment)`: builds and starts subprocess,
  writes PID file
- `_stop_subprocess(pid)`: SIGTERM → 2s wait → SIGKILL; calls `_clear_pid()`

### Step 4 — Wire into `render()`

Insert `_render_scheduler_controls(db, environment)` call at the top of the `render()`
function body (after the `db.ok` check), before the existing `heartbeat = db.get_latest_heartbeat()`.

### Step 5 — Update user manual

In `docs/user-manual/paper-trading-automation.md`, add a new section **"Starting and
Stopping from the UI"** before the existing "Starting the Scheduler" (CLI) section.
The CLI section should remain as an alternative / advanced option.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Stale PID file if scheduler crashes | Medium | Always validate PID liveness via `os.kill(pid,0)` before treating as running |
| Multiple subprocess launches (double-start) | Low | Start button disabled when PID file present AND pid alive |
| `--run-once` flag missing in CLI | Low | Step 1 explicitly audits; add if absent |
| Streamlit permission error on `/tmp` write | Very Low | `/tmp` writable by any user; no concern in current deployment |
| subprocess inherits DB session from Streamlit | Not applicable | subprocess is a new process; gets its own DB connection |

---

## Quality Gates (for implementer)

- [ ] `git status --short` — repo clean before and after
- [ ] `python -m compileall -q apps/streamlit/views/paper_trading.py apps/paper_trading.py`
- [ ] `python -m pytest tests/ -x -q --tb=short` — all existing tests pass
- [ ] `--run-once` flag verified in `apps/paper_trading.py`
- [ ] PID file path is `/tmp/quantagent_scheduler.pid` (not hardcoded to a venv or home path)
- [ ] Start button disabled when scheduler is running
- [ ] Stop button disabled when scheduler is stopped

---

## Next Phase

**Implementer** can proceed immediately. No design review needed beyond this plan.
All decisions are resolved:
- subprocess approach (not thread)
- PID in `/tmp/quantagent_scheduler.pid`
- `--run-once` flag in CLI (to be added if absent)
- UI wired into existing `render()` function — no new view
