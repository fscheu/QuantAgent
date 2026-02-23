# Acceptance Criteria: Trade P&L Calculation

**Issue:** QuantAgent-r78
**Related:** [RQ](../01_requirements/QuantAgent-r78-RQ-trade-pnl-calculation.md) | [DS](../03_design/QuantAgent-r78-DS-trade-pnl-calculation.md)

---

## AC-1: LONG Position Profit

**Given** a LONG position opened at $60,000
**When** the position is closed at $65,000 with quantity 0.1
**Then** `trade.pnl` equals $500.00
**And** `trade.pnl_pct` equals 8.33% (approximately)

---

## AC-2: LONG Position Loss

**Given** a LONG position opened at $60,000
**When** the position is closed at $55,000 with quantity 0.1
**Then** `trade.pnl` equals -$500.00
**And** `trade.pnl_pct` equals -8.33% (approximately)

---

## AC-3: SHORT Position Profit

**Given** a SHORT position opened at $65,000
**When** the position is closed at $60,000 with quantity 0.1
**Then** `trade.pnl` equals $500.00
**And** `trade.pnl_pct` equals 7.69% (approximately)

---

## AC-4: SHORT Position Loss

**Given** a SHORT position opened at $60,000
**When** the position is closed at $65,000 with quantity 0.1
**Then** `trade.pnl` equals -$500.00
**And** `trade.pnl_pct` equals -8.33% (approximately)

---

## AC-5: Opening Position (No P&L)

**Given** no existing position for symbol
**When** a new position is opened
**Then** `trade.pnl` is None
**And** `trade.pnl_pct` is None

---

## AC-6: Backtest Metrics Non-Zero

**Given** a backtest with at least one closed trade
**When** backtest completes
**Then** `winning_trades + losing_trades > 0`
**And** `total_pnl != 0` (unless all trades net to exactly zero)

---

## Invariants

1. `trade.pnl` is Decimal or None (never float)
2. `trade.pnl_pct` is float or None
3. Closed trades (`closed_at IS NOT NULL`) always have pnl calculated
4. Opening trades (`closed_at IS NULL`) always have pnl = None

---

## Oracle: Manual Verification

```sql
-- After fix: All closed trades should have pnl
SELECT COUNT(*) as closed_without_pnl
FROM trades
WHERE closed_at IS NOT NULL AND pnl IS NULL;
-- Expected: 0 (for new trades after fix)
```

---

## Oracle: Backtest Regression

Run same backtest that showed $0.00 P&L:
```bash
python examples/run_backtest.py
```

**Before fix:**
```
Total P&L: $0.00
Winning Trades: 0
Losing Trades: 0
```

**After fix:**
```
Total P&L: != $0.00
Winning Trades: > 0 OR Losing Trades: > 0
```
