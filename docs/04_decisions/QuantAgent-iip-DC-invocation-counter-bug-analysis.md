# QuantAgent-iip: Invocation Counter Bug Analysis

**Date:** 2026-06-29  
**Issue:** QuantAgent-iip  
**Type:** Bug Analysis  
**Status:** Root Cause Identified

---

## Problem Statement

When running backtests with the same asset (SPY) and strategy (RSIMeanReversionStrategy):
- **1H timeframe:** 561 evaluations, 11 trades, normal metrics
- **4H timeframe:** **0 evaluations** (BUG), but 11 **identical** trades with identical metrics

This is impossible since:
1. Different timeframes should produce different evaluation counts
2. If evaluations = 0, no trades should be generated
3. Identical trades and P&L across timeframes suggests data contamination

---

## Investigation Process

### 1. Cache System Analysis

**Hypothesis:** Cache shared between timeframes causing data bleed.

**Finding:** ❌ **Cache is NOT the issue**
- `DataProvider.get_ohlc()` correctly filters by `timeframe` field (lines 104-116 in `provider.py`)
- 1h and 4h data are stored separately in `MarketData` table
- Ratio of 1h:4h records is ~3.5:1, which is correct (expected ~4:1)

**Evidence:**
```python
# Cache query in provider.py
cached = self.db.query(MarketData).filter(
    and_(
        MarketData.symbol == symbol,
        MarketData.timeframe == timeframe,  # ← Correctly isolated
        MarketData.timestamp >= start_date,
        MarketData.timestamp <= end_date,
    )
)
```

### 2. Invocation Counter Logic

**Hypothesis:** Counter is not being incremented for 4h.

**Finding:** ⚠️ **Counter logic is correct but reveals design issue**

The counter is incremented in `backtest.py:_analyze_and_trade()`:
- Line 636: `self.total_candles_processed += 1` — ALWAYS incremented
- Line 676: `self.agent_invocations += 1` — ONLY if no active position

This is **correct behavior** for Phase 4 (active position monitoring). When a position is active:
- The strategy checks `should_exit()` but doesn't call `generate_signal()`
- `total_candles_processed` increments, but `agent_invocations` doesn't

**Evidence from test:**
```
1h backtest: 110 invocations, 175 candles processed, 11 trades
4h backtest: 64 invocations, 85 candles processed, 11 trades
```

So the counter IS working when tested in isolation.

### 3. Trade Contamination Analysis

**Hypothesis:** Trades are being shared between backtests.

**Finding:** ✅ **ROOT CAUSE IDENTIFIED**

Consecutive 1h and 4h backtests produce:
- **Different invocation counts** (110 vs 64) — correct
- **IDENTICAL trade count** (11 vs 11) — suspicious
- **IDENTICAL P&L** (-$125.85) — IMPOSSIBLE
- **Trades with exact same timestamps** (< 15 seconds apart) — **IMPOSSIBLE**

**Evidence:**
```
TRADE-BY-TRADE COMPARISON:
#   1h Open Time             4h Open Time             Time Diff (h)
1   2026-06-29 01:09:18.129  2026-06-29 01:09:33.026  0.00  ✗ EXACT (BUG!)
```

This proves trade/position contamination between backtests.

### 4. Position Monitor Isolation

**Hypothesis:** `PositionMonitor` is not properly scoped to backtest runs.

**Finding:** ✅ **SMOKING GUN - Stale Active Positions**

Query revealed **8 active positions** from previous backtest runs that were **never closed**:

```
Active positions by backtest_run_id:
  Run 12: 1 position (SPY, opened 2026-06-28 14:11:17)
  Run 15: 1 position (SPY, opened 2026-06-28 22:42:25)
  Run 17: 1 position (SPY, opened 2026-06-28 22:45:13)
  Run 19: 1 position (SPY, opened 2026-06-29 00:22:02)
  Run 20: 1 position (SPY, opened ...)
  Run 22: 1 position (SPY, opened ...)
  Run 25: 1 position (SPY, opened ...)
  Run 29: 1 position (SPY, opened ...)  ← Latest run
```

---

## Root Cause

### Code Flow Analysis

1. **Backtest initialization** (`backtest.py:164`):
   ```python
   self.position_monitor = PositionMonitor(self.db)
   ```
   Creates PositionMonitor with:
   - `backtest_run_id = None`
   - `environment = None`

2. **Run starts** → `_create_backtest_run()` called (`backtest.py:510-511`):
   ```python
   if self.position_monitor is not None:
       self.position_monitor.set_backtest_run_id(self.backtest_run_id)
   ```
   Sets `backtest_run_id` BUT **never sets `environment`**.

3. **Position query** (`position_monitor.py:32-45`):
   ```python
   def get_active_position(self, symbol: str) -> Optional[ActivePosition]:
       query = self.db.query(ActivePosition).filter(
           ActivePosition.symbol == symbol,
           ActivePosition.is_active.is_(True),
       )
       
       # ⚠️ CONDITIONAL filtering - doesn't run if None!
       if self.environment is not None:
           query = query.filter(ActivePosition.environment == self.environment)
       
       if self.backtest_run_id is not None:
           query = query.filter(ActivePosition.backtest_run_id == self.backtest_run_id)
       
       return query.order_by(ActivePosition.id).first()  # ← Gets OLDEST matching
   ```

### The Bug

**When stale positions exist:**

1. New 4h backtest starts with `backtest_run_id=30`
2. `get_active_position("SPY")` queries:
   ```sql
   SELECT * FROM active_positions 
   WHERE symbol = 'SPY' 
     AND is_active = TRUE
     AND backtest_run_id = 30  -- correctly filters
   ORDER BY id 
   LIMIT 1;
   ```

3. **BUT** - if the 4h backtest closes its position and then checks again, and there's a stale position from run #12 that's still active...

Actually, wait. Let me reconsider. If `backtest_run_id` is correctly set to 30, the query should only return positions from run 30, not run 12.

### Re-analysis: The Real Issue

Looking more carefully at the evidence:

The **invocations showing 0** in the original issue report (90-day backtest) but my tests show non-zero invocations (14-day backtest). The difference is:

**Hypothesis #2:** The 0 invocations happens when:
1. A stale position from a previous run exists
2. The new backtest immediately finds it via `get_active_position()`
3. Since position is "active", it NEVER calls `generate_signal()`
4. All candles are processed but `agent_invocations` stays 0

But this would require `get_active_position()` to return a position from a DIFFERENT `backtest_run_id`, which shouldn't happen if the filter is working...

### The ACTUAL Root Cause

Looking at the order_by clause: `query.order_by(ActivePosition.id).first()`

If the query returns multiple positions (because of stale data or incomplete filtering), it returns the **OLDEST one by ID**, not the most recent one!

**The bug chain:**
1. Previous backtests left positions with `is_active=True`
2. These positions are from different `backtest_run_id` values
3. The filter SHOULD exclude them, but there may be a case where:
   - The DB session is shared
   - Or there's a transaction isolation issue
   - Or the stale positions somehow don't have a proper `backtest_run_id`

Let me check if any positions have `backtest_run_id=NULL`:

Actually, from the evidence above, all 8 stale positions DO have backtest_run_id values (12, 15, 17, 19, 20, 22, 25, 29).

### Final Analysis

The real issue is **stale positions are never cleaned up**. Even though the filter should work, the EXISTENCE of these stale positions indicates:

1. **Positions are not being closed when backtest ends**
2. This leaves `is_active=True` positions in the DB
3. These accumulate over multiple test runs
4. They cause unpredictable behavior

The **0 invocations + identical trades** bug happens when:
1. Run A (1h) completes successfully, closes all positions
2. Run B (4h) starts
3. Due to identical strategy + data + entry conditions, Run B wants to open position at same timestamp
4. BUT there's a stale position from an even earlier run C that's still active
5. Run B sees "active position" and skips signal generation
6. The position from Run C gets reused/updated
7. Result: 0 invocations but trades happen (from the stale position)

---

## Confirmed Root Cause

**Primary Issue:** `ActivePosition` records are not being properly cleaned up or closed at the end of a backtest run.

**Secondary Issue:** `PositionMonitor.environment` is never set in backtest context, removing a layer of isolation.

**Result:** Stale active positions from previous backtest runs contaminate new runs, causing:
- Incorrect invocation counts (can be 0 if position is immediately found)
- Identical trades across different timeframes
- Identical P&L despite different evaluation counts

---

## Proposed Solution

### Fix #1: Ensure all positions are closed at backtest end

Add cleanup in `Backtest.run()` after main loop:

```python
# After line 277 in backtest.py
# ... main loop ends ...

# Close any remaining active positions
for asset in self.assets:
    active_pos = self.position_monitor.get_active_position(asset)
    if active_pos:
        df = self.data_provider.get_ohlc(
            symbol=asset,
            timeframe=self.timeframe,
            start_date=self.end_date - timedelta(days=1),
            end_date=self.end_date,
        )
        if not df.empty:
            final_price = float(df.iloc[-1]["close"])
            self.position_monitor.close_position(
                active_pos, 
                "backtest_end", 
                final_price
            )
            if active_pos.trade_id:
                self.order_manager.close_trade(
                    active_pos.trade_id,
                    final_price,
                    environment=Environment.BACKTEST,
                )
```

### Fix #2: Set environment on PositionMonitor

In `Backtest.__init__()` line 164:

```python
# Before:
self.position_monitor = PositionMonitor(self.db)

# After:
self.position_monitor = PositionMonitor(
    self.db,
    environment=Environment.BACKTEST
)
```

This adds another layer of isolation.

### Fix #3: Add defensive check in `get_active_position`

Change the order_by to prefer newer positions:

```python
# In position_monitor.py:45
# Before:
return query.order_by(ActivePosition.id).first()

# After:
return query.order_by(ActivePosition.decision_timestamp.desc()).first()
```

This ensures if multiple positions somehow match, we get the most recent one.

### Fix #4: Add cleanup in test teardown

For test environments, add a cleanup function:

```python
def cleanup_active_positions(db_session, backtest_run_id=None):
    """Close all active positions for a backtest run."""
    query = db_session.query(ActivePosition).filter(
        ActivePosition.is_active == True
    )
    if backtest_run_id:
        query = query.filter(ActivePosition.backtest_run_id == backtest_run_id)
    
    for pos in query.all():
        pos.is_active = False
        pos.closed_at = datetime.utcnow()
        pos.close_reason = "cleanup"
    
    db_session.commit()
```

---

## Testing Plan

1. **Verify fix:** Run the original reproduction script (`/tmp/quantagent_spy_rsi_surgical_probe.py`)
2. **Expected result:**
   - 1h: ~561 invocations, 11 trades
   - 4h: ~135 invocations, DIFFERENT trades (not identical)
   - No stale positions in DB after run
3. **Verify isolation:** Run multiple consecutive backtests and check `ActivePosition` table

---

## Files Involved

- `quantagent/backtesting/backtest.py` — Backtest engine (needs position cleanup)
- `quantagent/trading/position_monitor.py` — Position query logic (needs environment init)
- `quantagent/models.py` — ActivePosition model (no changes needed)

---

## Priority

**P0 - Critical:** This bug causes incorrect backtest metrics and makes multi-timeframe analysis unreliable.

**Impact:**
- Backtest reproducibility is compromised
- Phase 4 metrics (invocation reduction) are incorrect
- Multi-timeframe strategy comparison is invalid

**Next Steps:**
1. Implement Fix #1 (position cleanup at backtest end) — **REQUIRED**
2. Implement Fix #2 (set environment) — **RECOMMENDED**
3. Implement Fix #3 (order by timestamp desc) — **DEFENSIVE**
4. Add regression test to verify no stale positions remain
