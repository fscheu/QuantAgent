# QuantAgent-uzq — Acceptance Tests: Fix TradingScheduler Heartbeat and Scheduler Unit-Test Regressions

**Issue ID:** QuantAgent-uzq  
**Blocks:** QuantAgent-82t (Re-enable unit tests in CI)

---

## Gate Command (authoritative)

```bash
DATABASE_URL=postgresql://test:test@localhost:5432/quantagent_test \
  /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python \
  -m pytest tests/test_vje_scheduler_heartbeat_backend.py tests/trading/test_scheduler.py \
  -v --tb=short --maxfail=10 -m "not integration and not slow"
```

All 8 previously-failing tests must pass. No new failures introduced.

---

## Heartbeat Backend Tests (`tests/test_vje_scheduler_heartbeat_backend.py`)

### AC-1: `test_upsert_heartbeat_start_updates_existing_row_for_environment` PASSES

**Given:** A fresh SQLite in-memory DB with heartbeat + active_positions tables  
**When:** `_upsert_heartbeat_start()` is called twice with different timestamps  
**Then:** Only 1 `SchedulerHeartbeat` row exists; both calls return the same `id`; the row's `timestamp` is the second call's value; `status == "running"`; `assets` matches scheduler config  

**Verification:**
```bash
pytest tests/test_vje_scheduler_heartbeat_backend.py::test_upsert_heartbeat_start_updates_existing_row_for_environment -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** `AttributeError: 'TradingScheduler' object has no attribute '_upsert_heartbeat_start'`

---

### AC-2: `test_upsert_heartbeat_complete_sets_last_trade_id` PASSES

**Given:** One `Trade` row pre-inserted; a running heartbeat row exists  
**When:** `_upsert_heartbeat_complete(heartbeat, stats)` is called  
**Then:** The heartbeat row has `status == "completed"`, `completed_at` is set, `last_trade_id` points to the pre-inserted trade, and `stats["processed"] == 1`  

**Verification:**
```bash
pytest tests/test_vje_scheduler_heartbeat_backend.py::test_upsert_heartbeat_complete_sets_last_trade_id -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** `AttributeError: 'TradingScheduler' object has no attribute '_upsert_heartbeat_complete'`

---

### AC-3: `test_analyze_and_trade_continues_when_heartbeat_start_fails` PASSES

**Given:** `_upsert_heartbeat_start` is mocked to return `None` (simulating a failure)  
**When:** `analyze_and_trade()` runs with a healthy data provider and strategy  
**Then:** `stats["processed"] == 1`, `stats["errors"] == 0`; `_upsert_heartbeat_complete` is called once with `None` as the first argument  

**Verification:**
```bash
pytest tests/test_vje_scheduler_heartbeat_backend.py::test_analyze_and_trade_continues_when_heartbeat_start_fails -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** `analyze_and_trade()` does not call the heartbeat methods at all

---

### AC-4: `test_analyze_and_trade_full_cycle_writes_completed_heartbeat` PASSES

**Given:** Real SQLite session; all assets produce valid signals  
**When:** `analyze_and_trade()` completes  
**Then:** A `SchedulerHeartbeat` row exists in the DB with `status == "completed"` and `completed_at` not null; `stats["processed"] == 1`  

**Verification:**
```bash
pytest tests/test_vje_scheduler_heartbeat_backend.py::test_analyze_and_trade_full_cycle_writes_completed_heartbeat -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** No `SchedulerHeartbeat` row written; `assert hb is not None` fails

---

### AC-5: `test_analyze_and_trade_per_asset_error_completes_heartbeat_with_error_count` PASSES

**Given:** Real SQLite session; data provider returns empty DataFrame (triggers DataFetchError)  
**When:** `analyze_and_trade()` completes  
**Then:** A `SchedulerHeartbeat` row exists with `status == "completed"`; `stats["errors"] == 1`, `stats["processed"] == 0`  

**Verification:**
```bash
pytest tests/test_vje_scheduler_heartbeat_backend.py::test_analyze_and_trade_per_asset_error_completes_heartbeat_with_error_count -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** No heartbeat written; `assert hb is not None` fails

---

## Scheduler Unit Tests (`tests/trading/test_scheduler.py`)

### AC-6: `test_trading_scheduler_long_signal_executes_order` PASSES

**Given:** `DummySession` with a properly configured query mock; strategy returns LONG signal  
**When:** `run_once()` executes  
**Then:** `stats["processed"] == 1`, `stats["errors"] == 0`; `order_manager.execute_decision` called once with `decision=TradeSignal.LONG`  

**Verification:**
```bash
pytest tests/trading/test_scheduler.py::test_trading_scheduler_long_signal_executes_order -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** `TypeError: unsupported operand type(s) for +=: 'Mock' and 'int'` from `update_candle_tracking`; `stats["processed"] == 0`

---

### AC-7: `test_trading_scheduler_short_signal_executes_order` PASSES

**Given:** `DummySession` with fixed query mock; strategy returns SHORT signal  
**When:** `run_once()` executes  
**Then:** `stats["processed"] == 1`, `stats["errors"] == 0`; `order_manager.execute_decision` called with `decision=TradeSignal.SHORT`  

**Verification:**
```bash
pytest tests/trading/test_scheduler.py::test_trading_scheduler_short_signal_executes_order -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** Same TypeError as AC-6

---

### AC-8: `test_trading_scheduler_hold_signal_skips_execution` PASSES

**Given:** `DummySession` with fixed query mock; strategy returns `None` (HOLD)  
**When:** `run_once()` executes  
**Then:** `stats["processed"] == 1`, `stats["errors"] == 0`; `order_manager.execute_decision` not called  

**Verification:**
```bash
pytest tests/trading/test_scheduler.py::test_trading_scheduler_hold_signal_skips_execution -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** Same TypeError as AC-6

---

## Non-Regression: Previously-Passing Tests Remain Green

The following tests must still pass after the fix:

```bash
pytest tests/test_vje_scheduler_heartbeat_backend.py \
       tests/trading/test_scheduler.py \
       -v --tb=short -m "not integration and not slow"
```

Previously-passing tests that must not regress:
- `test_db_handle_returns_latest_heartbeat_dict`
- `test_db_handle_recent_heartbeats_are_limited_and_sorted_desc`
- `test_db_handle_missing_heartbeat_table_fails_closed`
- `test_db_handle_returns_none_for_different_environment`
- `test_db_handle_ok_false_short_circuits_queries`
- `test_scheduler_settings_validation_interval_zero`
- `test_scheduler_settings_validation_negative_interval`
- `test_scheduler_settings_validation_empty_assets`
- `test_scheduler_start_happy_path`
- `test_scheduler_start_disabled_config`
- `test_scheduler_stop_graceful_shutdown`
- `test_scheduler_stop_idempotent`
- `test_scheduler_transient_error_continue_with_next_asset`
- `test_scheduler_analysis_failure_continues_processing`
- `test_scheduler_environment_tagging_in_execution`
- `test_scheduler_double_start_idempotent`
- `test_scheduler_processes_multiple_assets`
- `test_scheduler_tracks_last_run_stats`
