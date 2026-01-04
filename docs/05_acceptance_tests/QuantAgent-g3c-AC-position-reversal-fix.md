# Acceptance Criteria: Position Reversal Bug Fix

**Issue ID:** QuantAgent-g3c
**Date:** 2026-01-04
**Status:** Ready for Validation

---

## Overview

This document defines acceptance criteria for the position reversal bug fix. The fix should allow the system to reverse positions (SHORT to LONG or LONG to SHORT) without errors.

---

## Acceptance Criteria

### AC-1: SHORT to LONG Reversal

```
Given a SHORT position exists for BTC with qty = -0.033094
And the system generates a LONG signal with calculated size = 0.034277
When OrderManager.execute_decision() is called with decision=LONG
Then:
  - The existing SHORT position is closed (BUY 0.033094)
  - A new LONG position is opened (BUY 0.034277)
  - No "Portfolio update failed" error occurs
  - Two Trade records are created in the database
  - Final position qty > 0 (LONG)
```

### AC-2: LONG to SHORT Reversal

```
Given a LONG position exists for BTC with qty = 0.050000
And the system generates a SHORT signal with calculated size = 0.040000
When OrderManager.execute_decision() is called with decision=SHORT
Then:
  - The existing LONG position is closed (SELL 0.050000)
  - A new SHORT position is opened (SELL 0.040000)
  - No errors occur
  - Two Trade records are created in the database
  - Final position qty < 0 (SHORT)
```

### AC-3: Non-Reversal Trades Unaffected

```
Given no position exists for ETH
And the system generates a LONG signal
When OrderManager.execute_decision() is called
Then:
  - A single LONG position is opened
  - One Trade record is created
  - Existing behavior is unchanged
```

### AC-4: Failed Close Prevents Open

```
Given a SHORT position exists for BTC
And the close order fails (e.g., insufficient capital for some reason)
When OrderManager.execute_decision() attempts reversal
Then:
  - The new LONG position is NOT opened
  - System remains in original SHORT state
  - Error is logged
  - Function returns None
```

### AC-5: Portfolio Consistency After Reversal

```
Given initial portfolio value = $100,000
And a SHORT position exists for BTC
When a LONG reversal is executed successfully
Then:
  - portfolio.get_total_value() reflects correct value
  - portfolio.cash is updated correctly (close adds cash, open subtracts cash)
  - portfolio.positions[BTC]["qty"] > 0
  - No orphaned or inconsistent state
```

---

## Test Cases

### Test Case 1: Basic SHORT to LONG Reversal

**Setup:**
```python
# Create portfolio with SHORT position
portfolio.positions["BTC"] = {
    "qty": -0.033094,
    "avg_cost": 105000.0,
    "current_price": 106000.0,
    "pnl": -33.09,
    "pnl_pct": -0.95
}
portfolio.cash = 103500.0  # Initial cash after opening SHORT
```

**Action:**
```python
result = order_manager.execute_decision(
    symbol="BTC",
    decision=TradeSignal.LONG,
    confidence=0.68,
    current_price=106045.33
)
```

**Expected:**
- `result` is not None (order executed)
- `portfolio.positions["BTC"]["qty"] > 0` (now LONG)
- Two trades in database for BTC

### Test Case 2: Basic LONG to SHORT Reversal

**Setup:**
```python
# Create portfolio with LONG position
portfolio.positions["ETH"] = {
    "qty": 2.5,
    "avg_cost": 3000.0,
    "current_price": 3100.0,
    "pnl": 250.0,
    "pnl_pct": 3.33
}
portfolio.cash = 92500.0
```

**Action:**
```python
result = order_manager.execute_decision(
    symbol="ETH",
    decision=TradeSignal.SHORT,
    confidence=0.75,
    current_price=3100.0
)
```

**Expected:**
- `result` is not None
- `portfolio.positions["ETH"]["qty"] < 0` (now SHORT)
- Two trades in database for ETH

### Test Case 3: Reversal with Different Sizes

**Setup:**
```python
# SHORT position smaller than new LONG size
portfolio.positions["BTC"] = {
    "qty": -0.01,  # Small SHORT
    "avg_cost": 100000.0,
    "current_price": 105000.0,
    "pnl": -50.0,
    "pnl_pct": -5.0
}
```

**Action:**
```python
result = order_manager.execute_decision(
    symbol="BTC",
    decision=TradeSignal.LONG,
    confidence=0.80,
    current_price=105000.0
)
# Assuming calculated size = 0.038 (larger than SHORT qty)
```

**Expected:**
- Close order: BUY 0.01 (exactly the SHORT qty)
- Open order: BUY 0.038 (calculated size for new LONG)
- Final position qty = 0.038

### Test Case 4: Non-Reversal Trade (No Existing Position)

**Setup:**
```python
# No position exists
portfolio.positions = {}
portfolio.cash = 100000.0
```

**Action:**
```python
result = order_manager.execute_decision(
    symbol="SOL",
    decision=TradeSignal.LONG,
    confidence=0.70,
    current_price=150.0
)
```

**Expected:**
- Single order executed (no reversal logic triggered)
- One trade in database
- `portfolio.positions["SOL"]["qty"] > 0`

### Test Case 5: Non-Reversal Trade (Same Direction as Existing)

**Setup:**
```python
# LONG position exists, LONG signal received
portfolio.positions["BTC"] = {
    "qty": 0.05,
    "avg_cost": 100000.0,
    "current_price": 105000.0,
    "pnl": 250.0,
    "pnl_pct": 5.0
}
```

**Action:**
```python
result = order_manager.execute_decision(
    symbol="BTC",
    decision=TradeSignal.LONG,
    confidence=0.65,
    current_price=105000.0
)
```

**Expected:**
- Trade rejected by RiskManager (position already exists, no adding)
- `result` is None
- Position unchanged

---

## Edge Cases

### Edge Case 1: Reversal at Zero Position

```
Given position qty = 0 (no position)
When LONG signal received
Then treat as normal open (not a reversal)
```

### Edge Case 2: Exact Same Size Reversal

```
Given SHORT qty = -0.034277
And calculated LONG size = 0.034277
When reversal executed
Then:
  - Close: BUY 0.034277
  - Open: BUY 0.034277
  - Final qty = 0.034277 (LONG)
```

### Edge Case 3: Very Small Position

```
Given SHORT qty = -0.000001 (dust)
When LONG signal with size = 0.05
Then reversal should still work correctly
```

---

## Negative Test Cases

### Negative Case 1: Insufficient Capital for Open After Close

```
Given SHORT position exists
And closing SHORT brings cash to $1000
And new LONG position requires $5000
When reversal attempted
Then:
  - Close order may succeed
  - Open order should fail validation
  - System should handle gracefully (log error, keep cash from close)
```

### Negative Case 2: Broker Failure on Close Order

```
Given broker.place_order() fails for close order
When reversal attempted
Then:
  - Open order is NOT executed
  - Original position remains
  - Error logged
```

---

## Validation Commands

```bash
# Run all position reversal tests
pytest tests/test_order_manager.py -v -k reversal

# Run specific test
pytest tests/test_order_manager.py::test_short_to_long_reversal -v

# Run with coverage
pytest tests/test_order_manager.py --cov=quantagent/trading/order_manager --cov-report=term-missing
```

---

## Definition of Done

1. All acceptance criteria pass
2. All test cases pass
3. No regressions in existing tests (`pytest tests/` passes)
4. Error log from bug report no longer occurs in similar scenarios
5. Code reviewed and merged
