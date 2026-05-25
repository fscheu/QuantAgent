# QuantAgent-kkj.2 — Acceptance Tests: Scheduler UI Controls (Start/Stop)

**Issue:** QuantAgent-kkj.2  
**Created:** 2026-05-25  

---

## Automated Tests (pytest)

These must pass in the implementer phase. All tests use the existing test infrastructure
(shared venv, no real DB required for unit tests — mock DB handle as needed).

### AT-01 — PID file helpers

```
File: tests/streamlit/test_paper_trading_controls.py (new)

test_read_pid_returns_none_when_file_absent()
  → _read_pid() returns None when PID file does not exist

test_read_pid_returns_int_when_valid()
  → write "12345\n" to tmp file → _read_pid() returns 12345

test_read_pid_returns_none_on_invalid_content()
  → write "notanumber\n" → _read_pid() returns None

test_pid_is_alive_returns_false_for_none()
  → _pid_is_alive(None) is False

test_pid_is_alive_returns_false_for_dead_process()
  → use a PID known to not exist (e.g., 999999999) → returns False

test_pid_is_alive_returns_true_for_current_process()
  → _pid_is_alive(os.getpid()) is True

test_write_and_clear_pid(tmp_path)
  → _write_pid(12345, pid_path) → read back → assert 12345
  → _clear_pid(pid_path) → file absent

test_clear_pid_does_not_raise_when_file_absent()
  → _clear_pid(nonexistent_path) does not raise
```

### AT-02 — Subprocess launch (smoke)

```
test_launch_subprocess_creates_pid_file(tmp_path, monkeypatch)
  → monkeypatch subprocess.Popen to return fake proc with pid=99999
  → call _launch_subprocess("BTC,SPX", "once", 1.0, "paper", pid_path=tmp_path/pid)
  → assert PID file exists and contains 99999

test_launch_subprocess_uses_run_once_flag_for_single_mode(tmp_path, monkeypatch)
  → capture Popen args
  → assert "--run-once" in args when mode == "once"

test_launch_subprocess_uses_interval_for_continuous_mode(tmp_path, monkeypatch)
  → capture Popen args
  → assert "--interval-hours" in args when mode == "continuous"
```

### AT-03 — Stop subprocess

```
test_stop_subprocess_sends_sigterm_and_clears_pid(tmp_path, monkeypatch)
  → monkeypatch os.kill and os.waitpid (or psutil)
  → write PID file with fake pid → _stop_subprocess(pid, pid_path)
  → assert SIGTERM sent, PID file removed

test_stop_subprocess_no_error_when_process_already_dead(tmp_path, monkeypatch)
  → os.kill raises ProcessLookupError → function handles gracefully, clears PID file
```

### AT-04 — CLI `--run-once` flag

```
tests/test_paper_trading_cli.py (update or create)

test_run_once_flag_exits_after_single_cycle(monkeypatch)
  → monkeypatch TradingScheduler.run_once to return {}
  → invoke CLI with ["--run-once", "--environment", "paper", "--assets", "BTC"]
  → assert run_once called exactly once, scheduler.start() NOT called
```

### AT-05 — Heartbeat non-modification

```
test_paper_trading_view_does_not_write_heartbeat(monkeypatch, db_mock)
  → render paper_trading view with a mock db
  → assert db_mock.SessionLocal.execute was NOT called with INSERT/UPDATE on
    scheduler_heartbeats
  (ensures the view only reads, never writes heartbeats)
```

---

## Manual Verification Checklist

These must be verified manually against the running Streamlit app. Use the QA environment.

### MV-01 — Start (stopped → running)

1. Open Paper Trading tab; confirm scheduler shows as stopped
2. In "Scheduler Controls" expander, set assets = `BTC,SPX`, mode = Single cycle
3. Click **▶ Start**
4. Wait ≤ 5 seconds; confirm status updates (at minimum, heartbeat shows "running")
5. Confirm `/tmp/quantagent_scheduler.pid` exists with a valid PID

### MV-02 — Start button disabled when running

1. While scheduler is running (MV-01), confirm **▶ Start** button is disabled or shows
   a warning message

### MV-03 — Stop (running → stopped)

1. With scheduler running, click **■ Stop**
2. Wait ≤ 5 seconds; confirm status updates to stopped/completed
3. Confirm `/tmp/quantagent_scheduler.pid` no longer exists (or is cleared)

### MV-04 — Stop button disabled when stopped

1. While scheduler is stopped, confirm **■ Stop** button is disabled

### MV-05 — Continuous mode

1. Start scheduler with mode = Continuous, interval = 0.5h
2. Confirm process is running and PID file exists
3. Confirm scheduler does not exit immediately (stays in APScheduler loop)
4. Stop scheduler; confirm graceful termination

### MV-06 — Heartbeat states unchanged

1. Run a full single-cycle start → wait for completion
2. Inspect recent runs table; confirm status values are `running`/`completed`/`error`
   (unchanged from pre-QuantAgent-kkj.2 behavior)

### MV-07 — User manual reflects UI controls

1. Open User Manual tab in Streamlit
2. Navigate to "Paper Trading Automation" section
3. Confirm presence of "Starting and Stopping from the UI" subsection with accurate
   instructions

---

## Regression Scope

The implementer must confirm the following views are unaffected:

- Dashboard view: no regressions in load or metrics
- Orders & Positions view: no regressions
- Existing Paper Trading status card and recent runs table: unmodified behavior
- All passing tests in `tests/` remain passing (`pytest tests/ -x -q`)
