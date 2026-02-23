# Implementation: Position Reversal Bug Fix

**Issue ID:** QuantAgent-g3c  
**Date:** 2026-01-04  
**Implementer:** Implementer Agent  
**Status:** Implemented  

---

## Summary

Fixed position reversal bug where SHORT→LONG or LONG→SHORT transitions failed with `ValueError` due to size mismatch between calculated new position size and existing position quantity.

---

## What Changed

### File: `quantagent/trading/order_manager.py`

#### 1. Added Reversal Detection in `execute_decision()`

**Location:** After size calculation (lines 107-119)

**Logic:**
```python
# Check for position reversal
current_position = self.portfolio.get_position(symbol)
existing_qty = current_position.get("qty", 0.0) if current_position else 0.0

is_reversal = (existing_qty > 0 and side == OrderSide.SELL) or (
    existing_qty < 0 and side == OrderSide.BUY
)

if is_reversal:
    return self._execute_reversal(...)
```

Detects reversal when:
- Existing LONG position (qty > 0) + new SELL order → SHORT reversal
- Existing SHORT position (qty < 0) + new BUY order → LONG reversal

#### 2. Added `_execute_reversal()` Method

**Location:** New method at end of class (lines 264-448)

**Flow:**

1. **Close existing position:**
   - Create order with opposite side and exact existing qty
   - Validate → Place → Execute → Persist trade

2. **Open new position:**
   - Create order with new side and calculated qty
   - Validate → Place → Execute → Persist trade

3. **Return:**
   - Returns filled order for new position
   - Returns None if either step fails

**Key Features:**
- Two separate orders = clear audit trail
- Each order validated independently
- Portfolio state consistent after each step
- Database commit after each trade
- Comprehensive logging at each step

---

## Why This Approach

**Two-Order Reversal** (vs single combined order):

✅ **Pros:**
- Clear audit trail (two Trade records)
- Each operation validated independently  
- Matches real-world broker behavior
- Portfolio manager unchanged (reuses existing logic)
- Transaction safety (can rollback on failure)

❌ **Cons:**
- Slightly more complex logic
- Two database entries per reversal

**Decision:** Two-order approach is cleaner and safer.

---

## Testing

### Manual Test

Run backtest with data that triggers reversals:

```bash
source venv_wsl/bin/activate
python examples/run_backtest.py --symbol BTC --start 2024-01-01 --end 2024-12-31
```

**Expected:**
- No "Portfolio update failed" errors
- Two trades logged per reversal (close + open)
- Position quantities match expected values
- Portfolio value consistent

### Unit Test (To be written by Tester)

See `docs/02_planning/QuantAgent-g3c-PL-position-reversal-fix.md` Task 4 for test cases:
1. SHORT to LONG reversal succeeds
2. LONG to SHORT reversal succeeds  
3. Reversal with different sizes (close qty ≠ open qty)
4. Failed close order prevents open order
5. Non-reversal trades unaffected

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Second order fails after first succeeds | First trade committed; portfolio in consistent flat state |
| Transaction rollback issues | Each trade committed separately with try/except |
| Existing non-reversal trades affected | Reversal logic only triggers when `is_reversal == True` |

---

## Files Modified

| File | Lines Changed | Type |
|------|---------------|------|
| `quantagent/trading/order_manager.py` | ~200 lines added | Feature |
| `docs/06_implementation/QuantAgent-g3c-IM-position-reversal-fix.md` | New file | Documentation |

---

## Verification Checklist

- [x] Code compiles without syntax errors
- [x] Reversal detection logic correct
- [x] Two-order execution implemented  
- [x] Error handling for each step
- [x] Logging comprehensive
- [ ] Unit tests pass (pending Tester)
- [ ] Integration test with backtest (pending Tester)
- [ ] Code review by human

---

## Next Steps

1. **Tester:** Write and execute unit tests per planning doc
2. **Tester:** Run integration test with backtest
3. **Human:** Code review
4. **Human:** Merge to main after all checks pass

---

## Related Documentation

- [Planning](/docs/02_planning/QuantAgent-g3c-PL-position-reversal-fix.md)
- [SHORT Positions Implementation](/docs/06_implementation/SHORT_POSITIONS_IMPLEMENTATION.md)
