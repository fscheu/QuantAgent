# Test Report: Position Reversal Fix (QuantAgent-g3c)

**Issue ID:** QuantAgent-g3c  
**Test Date:** 2026-01-04  
**Tester:** Tester Agent  
**Branch:** `feature/QuantAgent-g3c-position-reversal-fix`  
**Status:** Tests created, awaiting execution  

---

## Test Suite Overview

Created comprehensive test suite in `tests/test_order_manager_reversal.py` to validate position reversal fix.

**Test Strategy:**
- Structure validation: Verify two orders + two trades created
- Constraint validation: Close qty must match existing position exactly
- Error handling: Failed close prevents open, failed open leaves flat position
- State flow: Position transitions SHORT→FLAT→LONG or LONG→FLAT→SHORT
- Edge cases: Zero position, equal sizes, different sizes

**Total Tests:** 13

---

## Test Cases

### 1. Structure Validation

#### `test_reversal_short_to_long_structure`
**Purpose:** Verify SHORT→LONG reversal produces correct structure  
**Validates:**
- Two orders created (close + open)
- Two `portfolio.execute_trade()` calls
- Two DB persistence operations
- Returns final filled order

#### `test_reversal_long_to_short_structure`
**Purpose:** Verify LONG→SHORT reversal produces correct structure  
**Validates:** Same as above for opposite direction

---

### 2. Constraint Validation

#### `test_reversal_close_qty_matches_existing`
**Purpose:** Verify close order qty == abs(existing position qty)  
**Critical constraint:** Close order must use exact existing qty, NOT freshly calculated size  
**Validates:**
- Close order created with `qty = abs(existing_qty)`
- Close order side is opposite of existing position
- Precision maintained (no rounding errors)

#### `test_reversal_different_sizes`
**Purpose:** Verify close qty ≠ open qty (normal scenario)  
**Validates:**
- Close uses existing position qty
- Open uses newly calculated qty
- Both orders execute successfully even when sizes differ

---

### 3. Error Handling

#### `test_reversal_close_fails_prevents_open`
**Purpose:** If close order fails, new position must NOT open  
**Validates:**
- Broker failure on close → return None
- `portfolio.execute_trade` never called
- No inconsistent state

#### `test_reversal_open_fails_leaves_flat_position`
**Purpose:** If open fails after close succeeds, position left FLAT  
**Validates:**
- Close executes successfully
- Open failure → return None
- Position in consistent FLAT state (not SHORT/LONG)

#### `test_reversal_validation_fails_on_close`
**Purpose:** RiskManager rejects close order  
**Validates:**
- Circuit breaker active → reversal blocked
- No portfolio changes

#### `test_reversal_validation_fails_on_open`
**Purpose:** RiskManager rejects open order after close succeeds  
**Validates:**
- Close succeeds → portfolio FLAT
- Open rejected → return None
- Position left FLAT (acceptable outcome)

---

### 4. State Flow

#### `test_reversal_state_flow_short_to_long`
**Purpose:** Validate position state transitions during reversal  
**Validates:**
- Initial state: SHORT (qty < 0)
- After close: FLAT (qty = 0)
- After open: LONG (qty > 0)

---

### 5. Edge Cases

#### `test_reversal_edge_case_zero_position`
**Purpose:** FLAT position (qty=0) should NOT trigger reversal  
**Validates:**
- Single order executed (not two)
- Normal flow used (not reversal path)

#### `test_non_reversal_unaffected`
**Purpose:** New positions and additions to existing positions work unchanged  
**Validates:**
- No existing position → single order
- Same direction → single order (not reversal)

---

### 6. Real Bug Scenario

#### `test_bug_scenario_short_to_long_reversal`
**Purpose:** Reproduce exact conditions from bug report  
**Setup:**
- Portfolio: $106,909.42
- Existing SHORT: -0.0330943811250786 BTC @ $106,000
- New LONG signal: confidence 68%, price $106,045.33
- Calculated size: ~0.034277 BTC (differs from existing!)

**Before fix:** ValueError "Trying to buy 0.0342770443640196 shares but SHORT position is only 0.0330943811250786"

**After fix:** Should succeed with:
- Close order: BUY 0.0330943811250786 BTC (exact existing qty)
- Open order: BUY ~0.034277 BTC (newly calculated)

**Validates:**
- No ValueError
- Two orders executed
- Final order is BUY (LONG)

---

## Test Execution

### Commands

**Environment setup:**
```bash
cd /mnt/c/Users/BAISCF/repos_local/QuantAgent
source venv_wsl/bin/activate
python -V  # Verify Python 3.12.3
```

**Run reversal tests:**
```bash
pytest tests/test_order_manager_reversal.py -v
```

**Run with coverage:**
```bash
pytest tests/test_order_manager_reversal.py -v --cov=quantagent.trading.order_manager --cov-report=term-missing
```

**Run specific test:**
```bash
pytest tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_close_qty_matches_existing -v
```

**Run bug scenario test only:**
```bash
pytest tests/test_order_manager_reversal.py::TestPositionReversalRealBugScenario::test_bug_scenario_short_to_long_reversal -v
```

---

## Expected Results

### All Tests Pass ✓

```
tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_short_to_long_structure PASSED
tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_long_to_short_structure PASSED
tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_close_qty_matches_existing PASSED
tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_different_sizes PASSED
tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_close_fails_prevents_open PASSED
tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_open_fails_leaves_flat_position PASSED
tests/test_order_manager_reversal.py::TestPositionReversal::test_non_reversal_unaffected PASSED
tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_state_flow_short_to_long PASSED
tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_edge_case_zero_position PASSED
tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_validation_fails_on_close PASSED
tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_validation_fails_on_open PASSED
tests/test_order_manager_reversal.py::TestPositionReversalRealBugScenario::test_bug_scenario_short_to_long_reversal PASSED

=============================== 13 passed in X.XXs ===============================
```

---

## Known Limitations

### Cannot Execute in Current Environment

⚠️ **This agent session has no command execution capability** (no `bash` tool).

Tests have been created but NOT executed. The user must run them manually using the commands above.

---

## Coverage Analysis

**Target code:**
- `quantagent/trading/order_manager.py`
  - Lines ~110-132: Reversal detection logic
  - Lines ~263-440: `_execute_reversal()` method

**Expected coverage:**
- Reversal detection: ✓ Both directions (LONG→SHORT, SHORT→LONG)
- Two-order execution: ✓ Close + Open flow
- Error paths: ✓ Close fails, Open fails, Validation fails
- Edge cases: ✓ Zero position, equal sizes, different sizes
- State transitions: ✓ Position state flow validated

**Uncovered scenarios:**
- Database transaction failures (mock doesn't fully simulate SQLAlchemy behavior)
- Concurrent reversals (out of scope)
- Network/broker timeouts (handled by existing broker logic)

---

## Manual Validation Steps

After automated tests pass, perform manual integration test:

1. **Run backtest with reversal scenario:**
```bash
python examples/run_backtest.py --symbol BTC --start-date 2024-01-01 --end-date 2024-12-31
```

2. **Check logs for reversal execution:**
```
BTC: Position reversal detected - existing qty: -0.033, new side: OrderSide.BUY
BTC: Executing reversal - Step 1: Close BUY 0.033000
BTC: Close order filled - BUY 0.033000 @ $42,424.24
BTC: Portfolio updated - close BUY 0.033000 executed
BTC: Executing reversal - Step 2: Open BUY 0.034277
BTC: New order filled - BUY 0.034277 @ $42,424.24
BTC: Portfolio updated - new BUY 0.034277 executed
BTC: Position reversal completed successfully
```

3. **Verify database:**
```sql
SELECT * FROM trades WHERE symbol = 'BTC' ORDER BY created_at DESC LIMIT 2;
-- Should show two trades: close and open
```

4. **Check portfolio consistency:**
- Position qty should match open order qty
- No orphaned positions
- Cash balance correct

---

## Next Steps

1. **User executes tests:**
   ```bash
   source venv_wsl/bin/activate
   pytest tests/test_order_manager_reversal.py -v
   ```

2. **If all tests PASS:**
   - Update this document with actual test output
   - Run integration test (backtest)
   - Request code review
   - Merge to `main`

3. **If any test FAILS:**
   - Capture failure output
   - Create Fail Report (see template below)
   - Hand off to Implementer for fix

---

## Fail Report Template

**If tests fail, fill this out:**

### Failed Test
```
Test: test_reversal_close_qty_matches_existing
Command: pytest tests/test_order_manager_reversal.py::TestPositionReversal::test_reversal_close_qty_matches_existing -v
Exit code: 1
```

### Error Output
```
[Paste stacktrace here]
```

### Analysis
- **File:** quantagent/trading/order_manager.py
- **Line:** [line number from stacktrace]
- **Hypothesis:**
  - [ ] Bug in implementation (reversal logic incorrect)
  - [ ] Contract broken (portfolio.get_position behavior changed)
  - [ ] Test assumption wrong (mock setup incorrect)

### Recommendation
[Hand off to Implementer / Request clarification / Update test]

---

## Related Documentation

- [Implementation Doc](/docs/06_implementation/QuantAgent-g3c-IM-position-reversal-fix.md)
- [Planning Doc](/docs/02_planning/QuantAgent-g3c-PL-position-reversal-fix.md)
- [Test Strategy](/docs/03_design/TESTING_PATTERNS.md) (if exists)
