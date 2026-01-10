# Requirements: Trade P&L Calculation

**Issue:** QuantAgent-r78
**Type:** Bug Fix
**Priority:** P1
**Status:** Ready for Implementation

---

## Problem Statement

Trade objects created when closing positions do not calculate `pnl` and `pnl_pct` fields. All trades show `$0.00` P&L regardless of actual performance, making backtest metrics unusable.

---

## Scope

### In Scope
- Calculate P&L when closing LONG positions
- Calculate P&L when closing SHORT positions
- Populate `Trade.pnl` (absolute value in dollars)
- Populate `Trade.pnl_pct` (percentage return)

### Out of Scope
- Commission handling (already tracked separately, marked TODO)
- Unrealized P&L calculation (works correctly via portfolio positions)
- Historical data migration (existing trades remain with NULL pnl)
- Partial position closes (current system closes full positions only)

---

## Functional Requirements

### FR-1: LONG Position P&L
When closing a LONG position, the system shall calculate:
- `pnl = (exit_price - entry_price) * quantity`
- `pnl_pct = ((exit_price - entry_price) / entry_price) * 100`

### FR-2: SHORT Position P&L
When closing a SHORT position, the system shall calculate:
- `pnl = (entry_price - exit_price) * quantity`
- `pnl_pct = ((entry_price - exit_price) / entry_price) * 100`

### FR-3: Opening Positions
When opening a new position (no prior position exists), `pnl` and `pnl_pct` shall remain `None`.

### FR-4: Increasing Positions
When increasing an existing position (same direction), `pnl` and `pnl_pct` shall remain `None`.

---

## Constraints

- P&L calculation must occur before Trade object is persisted to database
- Must use `Decimal` type for `pnl` to maintain precision consistency with model
- Must handle edge case where `entry_price` is zero or missing (log warning, set pnl to None)

---

## Definition of Done

1. Closed trades have non-null `pnl` and `pnl_pct` values
2. Backtest metrics (winning trades, losing trades, total P&L) reflect actual values
3. Existing consumers of `Trade.pnl` work without modification
