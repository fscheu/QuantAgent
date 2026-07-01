# QuantAgent-iip: Quick Implementation Reference

**For:** Developer implementing the stale position cleanup fix  
**Estimated Time:** 2-3 hours  
**Risk:** LOW  
**Files to Modify:** 1 file, 2 changes

---

## TL;DR

**Problem:** Stale `ActivePosition` records cause data bleed between backtests.

**Solution:** Add cleanup method, call at start of `run()`.

**Code:** ~62 lines total (60 new method + 2 line call)

---

## Quick Implementation

### Step 1: Add Method (backtest.py, after line 961)

<details>
<summary>Click to expand full code</summary>

```python
def _cleanup_stale_positions(self) -> None:
    """
    Close ALL stale active positions for assets in this backtest.
    
    This method is called at backtest START to prevent contamination from
    previous incomplete runs. It is more aggressive than _close_remaining_positions()
    which only closes positions for the CURRENT backtest_run_id.
    
    Why this is needed:
    - Previous backtests may have crashed or exited without proper cleanup
    - Stale ActivePosition records accumulate in the database
    - These stale positions interfere with new backtests:
      * Cause "Position already open" errors in RiskManager
      * Lead to incorrect trade rejection
      * Result in identical trades/P&L across different timeframes
    """
    if not self.assets:
        return
    
    logger.info(
        "Checking for stale active positions before backtest start",
        extra={"event_type": "stale_position_check"},
    )
    
    total_cleaned = 0
    
    for asset in self.assets:
        # Query ALL active positions for this asset + environment
        # regardless of backtest_run_id
        stale_positions = (
            self.db.query(ActivePosition)
            .filter(
                ActivePosition.symbol == asset,
                ActivePosition.is_active.is_(True),
                ActivePosition.environment == Environment.BACKTEST,
            )
            .all()
        )
        
        if not stale_positions:
            continue
        
        logger.warning(
            f"Found {len(stale_positions)} stale active positions for {asset}",
            extra={
                "event_type": "stale_positions_found",
                "symbol": asset,
                "count": len(stale_positions),
                "run_ids": [p.backtest_run_id for p in stale_positions],
            },
        )
        
        for pos in stale_positions:
            # Get price data for closing
            df = self.data_provider.get_ohlc(
                symbol=asset,
                timeframe=self.timeframe,
                start_date=self.start_date - timedelta(days=1),
                end_date=self.start_date,
            )
            
            # Fallback: try position's timestamp if no data at start_date
            if df.empty:
                df = self.data_provider.get_ohlc(
                    symbol=asset,
                    timeframe=self.timeframe,
                    start_date=pos.decision_timestamp - timedelta(days=1),
                    end_date=pos.decision_timestamp + timedelta(days=1),
                )
            
            if not df.empty:
                final_price = float(df.iloc[-1]["close"])
                
                # Close the position
                self.position_monitor.close_position(
                    pos,
                    reason="stale_cleanup",
                    exit_price=final_price,
                )
                
                # Close associated trade if exists
                if pos.trade_id:
                    try:
                        self.order_manager.close_trade(
                            pos.trade_id,
                            final_price,
                            environment=Environment.BACKTEST,
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to close trade {pos.trade_id} for stale position {pos.id}: {e}",
                            extra={
                                "event_type": "stale_cleanup_error",
                                "position_id": pos.id,
                                "trade_id": pos.trade_id,
                            },
                        )
                
                logger.info(
                    f"Closed stale position ID {pos.id} from run {pos.backtest_run_id} @ ${final_price:.2f}",
                    extra={
                        "event_type": "stale_position_closed",
                        "position_id": pos.id,
                        "backtest_run_id": pos.backtest_run_id,
                    },
                )
                total_cleaned += 1
            else:
                # No price data, force close
                logger.warning(
                    f"No price data for stale position {pos.id}, force closing",
                    extra={"event_type": "stale_position_force_close", "position_id": pos.id},
                )
                pos.is_active = False
                pos.closed_at = datetime.utcnow()
                pos.close_reason = "stale_cleanup_no_price"
                self.db.commit()
                total_cleaned += 1
    
    if total_cleaned > 0:
        logger.warning(
            f"Cleaned up {total_cleaned} stale positions before backtest start",
            extra={"event_type": "stale_cleanup_complete", "count": total_cleaned},
        )
```

</details>

### Step 2: Call Method (backtest.py, line ~220)

```python
def run(self, name: Optional[str] = None) -> BacktestMetrics:
    """Run backtest and return metrics."""
    logger.info(
        f"Starting backtest: {self.start_date} to {self.end_date}",
        extra={"event_type": "backtest_start", "environment": "backtest"},
    )
    # ... existing logging ...

    self._replay_trade_order_ids = None

    # NEW: Clean up any stale active positions BEFORE creating backtest run
    self._cleanup_stale_positions()  # ← ADD THIS LINE

    # Create backtest run record
    self._create_backtest_run(name)
```

---

## Testing

### 1. Pre-Check (verify bug exists)
```bash
cd /home/azureuser/repos/projects/QuantAgent

# Check stale positions
python3 << 'EOF'
from quantagent.database import SessionLocal
from quantagent.models import ActivePosition, Environment

session = SessionLocal()
count = session.query(ActivePosition).filter(
    ActivePosition.is_active == True,
    ActivePosition.environment == Environment.BACKTEST
).count()
print(f"Stale positions: {count}")
session.close()
EOF
```

### 2. Run Test
```bash
python3 /tmp/quantagent_spy_rsi_surgical_probe.py > /tmp/test_output.txt 2>&1
```

### 3. Verify Fix
```bash
# Should have NO "Position already open" errors
grep -i "position already open" /tmp/test_output.txt
# (empty output = good)

# Should show cleanup logs
grep -i "stale" /tmp/test_output.txt
# (shows cleanup messages = good)

# Should have 0 stale positions after
python3 << 'EOF'
from quantagent.database import SessionLocal
from quantagent.models import ActivePosition, Environment

session = SessionLocal()
count = session.query(ActivePosition).filter(
    ActivePosition.is_active == True,
    ActivePosition.environment == Environment.BACKTEST
).count()
print(f"Remaining stale positions: {count}")
assert count == 0, f"Expected 0, found {count}"
print("✅ SUCCESS")
session.close()
EOF
```

---

## Commit

```bash
git add quantagent/backtesting/backtest.py

git commit -m "fix(backtest): Clean up stale ActivePosition records at start

Problem:
- Sequential backtests (1H, 4H) produce identical trades/P&L
- 'Position already open' errors appear
- Stale ActivePosition records accumulate across runs

Root Cause:
- _close_remaining_positions() only closes positions for CURRENT
  backtest_run_id, leaving orphaned positions from previous runs
- Stale positions interfere with new backtests

Solution:
- Add _cleanup_stale_positions() called at backtest START
- Closes ALL active positions for same asset + environment,
  regardless of backtest_run_id
- Prevents contamination from previous incomplete runs

Testing:
- Run /tmp/quantagent_spy_rsi_surgical_probe.py
- Verify trades and P&L differ between 1H and 4H
- Verify no 'Position already open' errors
- Verify zero stale positions after completion

Related: QuantAgent-iip
Docs: 
  - docs/04_decisions/QuantAgent-iip-DC-data-bleed-root-cause-analysis.md
  - docs/06_implementation/QuantAgent-iip-IM-stale-position-cleanup.md
"
```

---

## Success Checklist

After implementation:

- [ ] No "Position already open" errors in logs
- [ ] 1H and 4H produce different trade counts
- [ ] 1H and 4H produce different P&L values
- [ ] Cleanup logs appear in output
- [ ] Zero stale positions in DB after run
- [ ] Test passes multiple times in sequence

---

## Rollback (if needed)

```bash
# Revert commit
git revert HEAD

# Manual cleanup of stale positions
python3 << 'EOF'
from quantagent.database import SessionLocal
from quantagent.models import ActivePosition, Environment
from datetime import datetime

session = SessionLocal()
positions = session.query(ActivePosition).filter(
    ActivePosition.is_active == True,
    ActivePosition.environment == Environment.BACKTEST
).all()

for pos in positions:
    pos.is_active = False
    pos.closed_at = datetime.utcnow()
    pos.close_reason = "emergency_cleanup"

session.commit()
print(f"Cleaned {len(positions)} positions")
session.close()
EOF
```

---

## Related Docs

📄 **Root Cause Analysis:**  
`docs/04_decisions/QuantAgent-iip-DC-data-bleed-root-cause-analysis.md`

📄 **Implementation Plan:**  
`docs/06_implementation/QuantAgent-iip-IM-stale-position-cleanup.md`

📄 **Visual Flow:**  
`docs/04_decisions/QuantAgent-iip-DC-visual-flow-analysis.md`

---

## Questions?

Check the full implementation plan for:
- Detailed explanation of the bug
- Database query examples
- Test utilities
- Alternative solutions

**Ready to implement!** 🚀
