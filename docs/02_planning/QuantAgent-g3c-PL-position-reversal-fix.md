# Planning: Position Reversal Bug Fix

**Issue ID:** QuantAgent-g3c
**Date:** 2026-01-04
**Status:** Ready for Implementation
**Level:** MINIMAL

---

## Objective

Fix the bug where position reversals (SHORT to LONG or LONG to SHORT) fail due to size calculation mismatch between the new position size and the existing position that must be closed first.

---

## Problem Statement

### Error Log

```
BTC: Size calculated - TradeSignal.LONG 0.034277 @ $106045.33 (confidence=68.0%, portfolio=$106909.42)
BTC: Trade validation passed - proceeding to execution
BTC: Order filled - OrderSide.BUY 0.034277 @ $107105.78 (slippage: 1.00%)
BTC: Portfolio update failed - Trying to buy 0.0342770443640196 shares but SHORT position in BTC is only 0.0330943811250786
```

### Root Cause

The `OrderManager.execute_decision()` flow:

1. Calculates new position size: `qty = PositionSizer.calculate_size(...)` = 0.034277
2. Validates trade via `RiskManager.validate_trade()` - passes (allows reversals)
3. Creates Order with calculated qty (0.034277)
4. Places order with broker - succeeds
5. Calls `PortfolioManager.execute_trade()` - **FAILS**

The failure occurs because `PortfolioManager._execute_buy()` (lines 162-167) validates:

```python
if pos["qty"] < 0:  # SHORT position exists
    if abs(pos["qty"]) < qty:  # Trying to buy MORE than SHORT position
        raise ValueError(f"Trying to buy {qty} shares but SHORT position is only {abs(pos['qty'])}")
```

The system calculates a fresh position size (0.034277) but tries to use it to close a SHORT position of different size (0.033094).

### Expected Behavior

Position reversal should be handled as two logical operations:
1. **Close existing position:** Buy exactly `abs(SHORT_qty)` to close SHORT
2. **Open new position:** Buy calculated size to open LONG

---

## Scope

### In Scope

- Fix position reversal logic in `OrderManager.execute_decision()`
- Handle both reversal directions: SHORT->LONG and LONG->SHORT
- Ensure broker and portfolio updates are consistent

### Out of Scope

- Changing position sizing strategy
- Modifying risk management rules
- Adding pyramiding or other position management strategies
- Refactoring unrelated code

---

## Implementation Plan

### Option A: Two-Order Reversal (Recommended)

When a reversal is detected, execute two separate orders:
1. Close order: Opposite side with existing position qty
2. Open order: New side with calculated qty

**Pros:**
- Clear audit trail (two trades recorded)
- Each operation is validated independently
- Matches real-world broker behavior

**Cons:**
- More complex logic
- Two database entries per reversal

### Option B: Combined Order with Adjusted Size

Calculate total quantity needed: `close_qty + new_position_qty`

**Cons:**
- Portfolio manager would need significant changes
- Single trade record doesn't reflect two distinct operations
- Less clear audit trail

### Decision: Option A

Two-order reversal is cleaner, matches real-world trading, and requires fewer changes to `PortfolioManager`.

---

## Tasks

### Task 1: Detect Position Reversal in OrderManager (30 min)

**File:** `quantagent/trading/order_manager.py`

**Location:** `execute_decision()` method, after size calculation (around line 100)

**Logic:**
```
1. Get current position for symbol from portfolio
2. Determine if this is a reversal:
   - existing_qty > 0 (LONG) AND decision == SHORT -> reversal
   - existing_qty < 0 (SHORT) AND decision == LONG -> reversal
3. If reversal:
   - First, execute close order
   - Then, execute open order
4. If not reversal:
   - Continue with existing flow
```

### Task 2: Implement Close Order Logic (30 min)

**File:** `quantagent/trading/order_manager.py`

**Add helper method or inline logic to:**
1. Create close order with:
   - `side = SELL` if closing LONG, `BUY` if closing SHORT
   - `qty = abs(existing_position_qty)`
   - `symbol, price, order_type` from context
2. Execute close order through existing broker/portfolio flow
3. Verify position is closed (qty == 0)

### Task 3: Update Main Flow for Reversal (30 min)

**File:** `quantagent/trading/order_manager.py`

**Modify `execute_decision()`:**
1. If reversal detected and close order succeeded:
   - Proceed to create new position order with calculated size
   - Execute open order through existing flow
2. If close order failed:
   - Log error
   - Return None (do not open new position)

### Task 4: Add Unit Tests (45 min)

**File:** `tests/test_order_manager.py` (or new file)

**Test cases:**
1. SHORT to LONG reversal succeeds
2. LONG to SHORT reversal succeeds
3. Reversal with different sizes (close qty != open qty)
4. Failed close order prevents open order
5. Non-reversal trades unaffected

---

## Validation

### How to Test

1. **Unit Tests:**
   ```bash
   pytest tests/test_order_manager.py -v -k reversal
   ```

2. **Manual Test:**
   - Run backtest with data that triggers position reversals
   - Verify no "Portfolio update failed" errors
   - Verify positions are correctly closed and opened

### Success Criteria

1. Position reversals complete without error
2. Both close and open trades are recorded in database
3. Portfolio state is consistent after reversal
4. Existing non-reversal trades work unchanged

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Two orders increase latency | Low | Low | Both execute in same session, no real latency |
| Transaction rollback on second order failure | Medium | Medium | Wrap both orders in try/except, rollback if second fails |
| Existing tests break | Low | Medium | Run full test suite before PR |

---

## Dependencies

- None (self-contained fix)

---

## Files to Modify

| File | Change |
|------|--------|
| `quantagent/trading/order_manager.py` | Add reversal detection and two-order execution logic |
| `tests/test_order_manager.py` | Add reversal test cases |

---

## Estimated Time

| Task | Time |
|------|------|
| Task 1: Detect reversal | 30 min |
| Task 2: Close order logic | 30 min |
| Task 3: Update main flow | 30 min |
| Task 4: Unit tests | 45 min |
| **Total** | **~2.5 hours** |

---

## Related Documentation

- [SHORT_POSITIONS_IMPLEMENTATION.md](/docs/06_implementation/SHORT_POSITIONS_IMPLEMENTATION.md) - SHORT position implementation details
- [POSITION_MANAGEMENT_STRATEGIES.md](/docs/03_design/POSITION_MANAGEMENT_STRATEGIES.md) - Position management design
