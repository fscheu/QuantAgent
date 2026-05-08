# QuantAgent-3uf — Acceptance Tests: Fix PositionMonitor Unit-Test Regressions

**Issue ID:** QuantAgent-3uf

---

## Gate Command (authoritative)

```bash
DATABASE_URL=postgresql://test:test@localhost:5432/quantagent_test \
  /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python \
  -m pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"
```

All tests previously failing must now pass. No new failures introduced.

---

## AC-1: `test_only_one_active_position_per_symbol` PASSES

**Given:** A fresh, empty database  
**When:** The test opens one position, closes it, opens another  
**Then:** Exactly 1 active position exists after each open; the second active is SELL  

**Verification:**
```bash
pytest tests/test_position_monitor.py::test_only_one_active_position_per_symbol -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** `assert 46 == 1` (stale data from previous runs)

---

## AC-2: `test_position_with_all_optional_fields` PASSES

**Given:** Actual Trade and Signal records exist in the DB  
**When:** `open_position()` is called with `trade_id=<real_id>` and `signal_id=<real_id>`  
**Then:** The position is committed without FK violation; all optional fields are set correctly  

**Verification:**
```bash
pytest tests/test_position_monitor_constraints.py::test_position_with_all_optional_fields -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** `sqlalchemy.exc.IntegrityError: ForeignKeyViolation (trade_id=123 not present)`

---

## AC-3: `test_get_active_position_returns_most_recent_if_multiple` PASSES

**Given:** A fresh DB with pos1 created via monitor, pos2 inserted directly  
**When:** `get_active_position("BTCUSDT")` is called  
**Then:** Returns pos1 (oldest, lowest id) due to deterministic `ORDER BY id ASC`  

**Verification:**
```bash
pytest tests/test_position_monitor_constraints.py::test_get_active_position_returns_most_recent_if_multiple -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** `assert 1 == 96` (returned oldest stale record instead of current test's pos1)

---

## AC-4: `test_closed_position_not_returned_by_get_active` PASSES

**Given:** A fresh DB  
**When:** A position is opened and immediately closed  
**Then:** `get_active_position()` returns `None` (no other active positions in DB)  

**Verification:**
```bash
pytest tests/test_position_monitor_constraints.py::test_closed_position_not_returned_by_get_active -v --tb=short
# Expected: PASSED
```

**Failure mode before fix:** Returned stale active BTCUSDT positions from prior runs

---

## AC-5: Full gate passes without new regressions

**Given:** All changes are applied  
**When:** The full CI gate command runs  
**Then:** All previously-passing tests still pass; the 4 fixed tests now pass  

**Verification:**
```bash
DATABASE_URL=postgresql://test:test@localhost:5432/quantagent_test \
  /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python \
  -m pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow" \
  | tail -5
# Expected: no FAILED lines; 4 previously-failing tests now show PASSED
```

---

## AC-6: Tests run with no DATABASE_URL (SQLite fallback)

**Given:** `DATABASE_URL` is not set in the environment  
**When:** The test suite runs  
**Then:** All 4 tests pass using the `conftest.py` SQLite in-memory fixture  

**Verification:**
```bash
unset DATABASE_URL
/mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python \
  -m pytest tests/test_position_monitor.py tests/test_position_monitor_constraints.py \
  -v --tb=short
# Expected: all tests PASSED (SQLite in-memory from conftest.py)
```

---

## Non-Regression Checks

The following existing passing tests must remain green after changes:

```bash
pytest tests/test_position_monitor.py -v --tb=short -m "not integration and not slow"
pytest tests/test_position_monitor_constraints.py -v --tb=short -m "not integration and not slow"
```

Specifically, tests that were already passing before this fix:
- `test_open_position`
- `test_get_active_position`
- `test_update_candle_tracking_*`
- `test_close_position_*`
- `test_active_position_has_required_fields`
- `test_default_values_are_correct`
- `test_candles_since_entry_never_decrements`
- `test_candles_direction_never_exceeds_prediction_horizon`
- `test_accuracy_is_between_zero_and_one`
- `test_closed_position_has_closed_at_timestamp`
- `test_close_already_closed_position_is_idempotent`
- `test_update_tracking_on_closed_position_still_increments`
- `test_zero_quantity_position`
- `test_candle_tracking_with_equal_prices`
- `test_accuracy_calculation_*`
- `test_all_exit_policies_can_be_set`
- `test_position_persists_after_session_close`
