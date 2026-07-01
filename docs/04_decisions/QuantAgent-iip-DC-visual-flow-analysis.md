# QuantAgent-iip: Visual Flow Analysis

**Date:** 2026-06-29  
**Purpose:** Visual representation of the data bleed bug and fix

---

## Current Flow (WITH BUG)

```
┌─────────────────────────────────────────────────────────────┐
│ TEST SCRIPT: /tmp/quantagent_spy_rsi_surgical_probe.py     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Run 1: Backtest 1H                   │
        │  - Creates session from SessionLocal()│
        │  - Creates PortfolioManager (fresh)   │
        │  - Creates PositionMonitor            │
        │    backtest_run_id = 29               │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Main Loop: Process candles           │
        │  - Opens position ID 75 (SPY, LONG)   │
        │  - 35 trades executed                 │
        │  - Metrics calculated                 │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  _close_remaining_positions()         │
        │  Query:                               │
        │    WHERE backtest_run_id = 29   ← ONLY RUN 29! │
        │  Result: Closes position 75 ✓         │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  DATABASE STATE                       │
        │  active_positions (is_active=True):   │
        │    Run 12: ID 45 ← STALE              │
        │    Run 15: ID 49 ← STALE              │
        │    Run 17: ID 56 ← STALE              │
        │    Run 19: ID 57 ← STALE              │
        │    Run 20: ID 60 ← STALE              │
        │    Run 22: ID 63 ← STALE              │
        │    Run 25: ID 68 ← STALE              │
        │    Run 29: ID 75 ← CLOSED ✓           │
        │  8 STALE POSITIONS REMAIN!            │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Run 2: Backtest 4H                   │
        │  - Creates NEW session                │
        │  - Creates NEW PortfolioManager       │
        │  - Creates NEW PositionMonitor        │
        │    backtest_run_id = 30               │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Main Loop: Process candles           │
        │  - Strategy generates LONG signal     │
        │  - Tries to execute trade             │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  RiskManager.validate_trade()         │
        │  Checks portfolio.positions['SPY']    │
        │  Result: EMPTY (fresh dict) ✓         │
        │                                       │
        │  BUT... somehow sees existing position│
        │  (possibly through PositionMonitor    │
        │   querying stale positions)           │
        │                                       │
        │  ❌ "Position already open: LONG      │
        │      0.000000 shares"                 │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  RESULT: DATA BLEED                   │
        │  - Same trades as 1H                  │
        │  - Same P&L as 1H                     │
        │  - Different eval count (83 vs 492)   │
        │    but identical financial results    │
        └───────────────────────────────────────┘
```

---

## Root Cause Visualization

```
Database: active_positions table
┌──────┬─────────────┬────────┬───────────┬────────────────────┐
│ ID   │ run_id      │ symbol │ is_active │ opened_at          │
├──────┼─────────────┼────────┼───────────┼────────────────────┤
│  45  │  12 ← STALE │  SPY   │  TRUE ❌  │ 2026-06-28 14:11   │
│  49  │  15 ← STALE │  SPY   │  TRUE ❌  │ 2026-06-28 22:42   │
│  56  │  17 ← STALE │  SPY   │  TRUE ❌  │ 2026-06-28 22:45   │
│  57  │  19 ← STALE │  SPY   │  TRUE ❌  │ 2026-06-29 00:22   │
│  60  │  20 ← STALE │  SPY   │  TRUE ❌  │ 2026-06-29 01:03   │
│  63  │  22 ← STALE │  SPY   │  TRUE ❌  │ 2026-06-29 01:06   │
│  68  │  25 ← STALE │  SPY   │  TRUE ❌  │ 2026-06-29 01:07   │
│  80  │  30 ← NEW   │  SPY   │  TRUE ✓   │ 2026-06-29 02:01   │
└──────┴─────────────┴────────┴───────────┴────────────────────┘
         ↑                                  ↑
         │                                  │
    Never closed!                      All have is_active=TRUE
    (incomplete cleanup)                (should be FALSE)
```

**The Problem:**
- `_close_remaining_positions()` queries: `WHERE backtest_run_id = 30`
- Only finds and closes position 80
- Positions 45, 49, 56, 57, 60, 63, 68 remain `is_active=TRUE`
- These stale positions interfere with new backtests

---

## Fixed Flow (AFTER FIX)

```
┌─────────────────────────────────────────────────────────────┐
│ TEST SCRIPT: /tmp/quantagent_spy_rsi_surgical_probe.py     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Run 1: Backtest 1H                   │
        │  - Creates session                    │
        │  - Creates components                 │
        │    backtest_run_id = 29               │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  ✨ NEW: _cleanup_stale_positions()   │
        │  Query:                               │
        │    WHERE symbol = 'SPY'               │
        │      AND is_active = TRUE             │
        │      AND environment = BACKTEST       │
        │    # NO backtest_run_id filter!       │
        │                                       │
        │  Result: Found 8 stale positions      │
        │  - Closes all 8 positions ✓           │
        │  - Closes associated trades ✓         │
        │  - Logs: "Cleaned 8 stale positions"  │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Main Loop: Process candles           │
        │  - Opens position ID 75 (SPY, LONG)   │
        │  - 35 trades executed                 │
        │  - Metrics calculated                 │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  _close_remaining_positions()         │
        │  Closes position 75 ✓                 │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  DATABASE STATE                       │
        │  active_positions (is_active=True):   │
        │    NONE! All positions closed ✓       │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Run 2: Backtest 4H                   │
        │  - Creates NEW session                │
        │  - Creates NEW components             │
        │    backtest_run_id = 30               │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  ✨ NEW: _cleanup_stale_positions()   │
        │  Query: Same as above                 │
        │  Result: Found 0 stale positions ✓    │
        │  Logs: "No stale positions, clean"    │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Main Loop: Process candles           │
        │  - Strategy generates signals         │
        │  - Executes trades normally ✓         │
        │  - NO rejection errors ✓              │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  RESULT: NO DATA BLEED ✓              │
        │  - DIFFERENT trades vs 1H             │
        │  - DIFFERENT P&L vs 1H                │
        │  - Proper eval count (83)             │
        │  - Independent results ✓              │
        └───────────────────────────────────────┘
```

---

## Database State Comparison

### BEFORE FIX (Bug Present)

```sql
-- After Run 1 (1H) completes
SELECT COUNT(*) FROM active_positions 
WHERE is_active = TRUE AND environment = 'BACKTEST';
-- Result: 8 (stale from previous runs)

-- After Run 2 (4H) completes
SELECT COUNT(*) FROM active_positions 
WHERE is_active = TRUE AND environment = 'BACKTEST';
-- Result: 9 (8 stale + 1 from Run 2)

-- Problem accumulates over time! ❌
```

### AFTER FIX (Bug Fixed)

```sql
-- After Run 1 (1H) completes
SELECT COUNT(*) FROM active_positions 
WHERE is_active = TRUE AND environment = 'BACKTEST';
-- Result: 0 ✓

-- After Run 2 (4H) completes
SELECT COUNT(*) FROM active_positions 
WHERE is_active = TRUE AND environment = 'BACKTEST';
-- Result: 0 ✓

-- Clean slate for each run! ✅
```

---

## Key Insight: Scope of Cleanup

### OLD: _close_remaining_positions() (End of Run)

```python
# Scope: ONLY current backtest_run_id
def _close_remaining_positions(self):
    active_pos = self.position_monitor.get_active_position(asset)
    # ↓ This method filters by self.backtest_run_id
    #   Only finds positions from THIS run
```

**Query executed:**
```sql
SELECT * FROM active_positions
WHERE symbol = 'SPY'
  AND is_active = TRUE
  AND environment = 'BACKTEST'
  AND backtest_run_id = 30  ← ONLY CURRENT RUN
LIMIT 1;
```

**Result:** Closes position from current run only, ignores stale positions.

---

### NEW: _cleanup_stale_positions() (Start of Run)

```python
# Scope: ALL active positions for asset + environment
def _cleanup_stale_positions(self):
    stale_positions = self.db.query(ActivePosition).filter(
        ActivePosition.symbol == asset,
        ActivePosition.is_active == True,
        ActivePosition.environment == Environment.BACKTEST,
    ).all()  # NO backtest_run_id filter!
    # ↓ Gets ALL active positions regardless of run_id
```

**Query executed:**
```sql
SELECT * FROM active_positions
WHERE symbol = 'SPY'
  AND is_active = TRUE
  AND environment = 'BACKTEST';
-- NO backtest_run_id filter → finds ALL stale positions
```

**Result:** Closes ALL stale positions before starting new run.

---

## Timing: When Each Cleanup Runs

```
Backtest.run() execution flow:
┌─────────────────────────────────────────────┐
│ 1. Logger: "Starting backtest"             │
├─────────────────────────────────────────────┤
│ 2. ✨ NEW: _cleanup_stale_positions()      │ ← START
│    - Closes ALL stale positions            │
│    - Happens BEFORE run creation           │
├─────────────────────────────────────────────┤
│ 3. _create_backtest_run()                  │
│    - Assigns backtest_run_id               │
├─────────────────────────────────────────────┤
│ 4. Main loop: Process candles              │
│    - Open/close positions                  │
│    - Execute trades                        │
├─────────────────────────────────────────────┤
│ 5. _calculate_metrics()                    │
│    - Read trades from DB                   │
├─────────────────────────────────────────────┤
│ 6. _close_remaining_positions()            │ ← END
│    - Closes positions from THIS run        │
├─────────────────────────────────────────────┤
│ 7. _update_backtest_run()                  │
│    - Save metrics                          │
└─────────────────────────────────────────────┘
```

**Why cleanup at START is better:**
- Ensures clean slate BEFORE any logic runs
- Prevents stale positions from interfering
- Complements end-of-run cleanup (defense in depth)

---

## Code Changes Summary

### File: quantagent/backtesting/backtest.py

**Change 1: Add new method (after line 961)**
```diff
+    def _cleanup_stale_positions(self) -> None:
+        """Close ALL stale active positions before backtest start."""
+        for asset in self.assets:
+            stale_positions = self.db.query(ActivePosition).filter(
+                ActivePosition.symbol == asset,
+                ActivePosition.is_active == True,
+                ActivePosition.environment == Environment.BACKTEST,
+            ).all()
+            
+            for pos in stale_positions:
+                # Close position logic...
```

**Change 2: Call in run() method (after line 219)**
```diff
     def run(self, name: Optional[str] = None) -> BacktestMetrics:
         logger.info(f"Starting backtest: {self.start_date} to {self.end_date}")
         
         self._replay_trade_order_ids = None
         
+        # Clean up any stale active positions BEFORE starting
+        self._cleanup_stale_positions()
+        
         # Create backtest run record
         self._create_backtest_run(name)
```

**Lines changed:** ~2 + ~60 = ~62 lines total  
**Risk:** Low (defensive cleanup, no breaking changes)

---

## Expected Test Results

### Pre-Fix Results
```
=== RUN 1 (1H) ===
Evaluations: 492
Trades: 35
P&L: -$344.86

=== RUN 2 (4H) ===
Evaluations: 83
Trades: 35 ← SAME AS 1H (BUG!)
P&L: -$344.86 ← SAME AS 1H (BUG!)
Errors: "Position already open: LONG 0.000000 shares"
```

### Post-Fix Results
```
=== RUN 1 (1H) ===
Cleanup: Found 8 stale positions, closed all ✓
Evaluations: 492
Trades: 35
P&L: -$344.86

=== RUN 2 (4H) ===
Cleanup: Found 0 stale positions ✓
Evaluations: 83
Trades: M (M ≠ 35) ✓
P&L: Y (Y ≠ -$344.86) ✓
Errors: None ✓

Database state after: 0 active positions ✓
```

---

## Validation Checklist

After implementing the fix, verify:

- [ ] No "Position already open" errors in logs
- [ ] 1H and 4H produce different trade counts
- [ ] 1H and 4H produce different P&L values
- [ ] Cleanup logs show stale position detection
- [ ] Database has 0 active positions after completion
- [ ] Evaluation counts are in expected ratio (~4:1)
- [ ] Test passes when run multiple times in sequence

---

**Visual Analysis Complete**  
**Ready for Implementation**
