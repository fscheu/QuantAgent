# QuantAgent-iip: Implementation Plan - Stale Position Cleanup

**Date:** 2026-06-29  
**Issue:** QuantAgent-iip  
**Type:** Implementation Guide  
**Related:** [Data Bleed Root Cause Analysis](./QuantAgent-iip-DC-data-bleed-root-cause-analysis.md)

---

## Overview

This document provides the **step-by-step implementation** of the stale position cleanup fix for the data bleed bug in sequential backtests.

**Goal:** Prevent `ActivePosition` contamination between backtest runs by cleaning up stale positions at backtest start.

---

## Implementation Steps

### Step 1: Add `_cleanup_stale_positions()` Method

**File:** `quantagent/backtesting/backtest.py`  
**Location:** Add after `_close_remaining_positions()` (around line 961)

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
    
    This method queries for ANY active position matching:
    - Same symbol(s) as this backtest
    - Same environment (BACKTEST)
    - ANY backtest_run_id (including previous runs)
    
    And closes them with reason "stale_cleanup".
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
                "position_ids": [p.id for p in stale_positions],
            },
        )
        
        for pos in stale_positions:
            # Get price data for closing (use start_date as reference)
            # Try to get recent price near backtest start
            df = self.data_provider.get_ohlc(
                symbol=asset,
                timeframe=self.timeframe,
                start_date=self.start_date - timedelta(days=1),
                end_date=self.start_date,
            )
            
            # Fallback: if no data at start_date, try position's timestamp
            if df.empty:
                df = self.data_provider.get_ohlc(
                    symbol=asset,
                    timeframe=self.timeframe,
                    start_date=pos.decision_timestamp - timedelta(days=1),
                    end_date=pos.decision_timestamp + timedelta(days=1),
                )
            
            if not df.empty:
                # Use the last available price
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
                        "symbol": asset,
                        "price": final_price,
                    },
                )
                total_cleaned += 1
            else:
                # No price data available, force close with NULL price
                logger.warning(
                    f"No price data available for stale position {pos.id}, force closing",
                    extra={
                        "event_type": "stale_position_force_close",
                        "position_id": pos.id,
                    },
                )
                pos.is_active = False
                pos.closed_at = datetime.utcnow()
                pos.close_reason = "stale_cleanup_no_price"
                self.db.commit()
                total_cleaned += 1
    
    if total_cleaned > 0:
        logger.warning(
            f"Cleaned up {total_cleaned} stale positions before backtest start",
            extra={
                "event_type": "stale_cleanup_complete",
                "count": total_cleaned,
            },
        )
    else:
        logger.info(
            "No stale positions found, proceeding with clean slate",
            extra={"event_type": "stale_cleanup_complete"},
        )
```

---

### Step 2: Call Cleanup at Backtest Start

**File:** `quantagent/backtesting/backtest.py`  
**Location:** In `run()` method, around line 222

**Current code:**
```python
def run(self, name: Optional[str] = None) -> BacktestMetrics:
    """Run backtest and return metrics."""
    logger.info(
        f"Starting backtest: {self.start_date} to {self.end_date}",
        extra={"event_type": "backtest_start", "environment": "backtest"},
    )
    # ... more logging ...

    self._replay_trade_order_ids = None

    # Create backtest run record
    self._create_backtest_run(name)
```

**Updated code:**
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
    # This prevents contamination from previous incomplete runs
    self._cleanup_stale_positions()

    # Create backtest run record
    self._create_backtest_run(name)
```

---

### Step 3: Add Test Utility (Optional but Recommended)

**File:** `tests/utils/__init__.py` (create if doesn't exist)

```python
"""Test utilities for QuantAgent."""

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from quantagent.models import ActivePosition, Environment


def cleanup_active_positions(
    db_session: Session,
    environment: Environment = Environment.BACKTEST,
    symbol: Optional[str] = None,
) -> int:
    """
    Close ALL active positions for an environment.
    
    Use this in test setup/teardown to ensure clean slate between tests.
    
    Args:
        db_session: SQLAlchemy session
        environment: Environment to clean (default: BACKTEST)
        symbol: Optional symbol filter (cleans all if None)
    
    Returns:
        Number of positions closed
    
    Example:
        >>> from quantagent.database import SessionLocal
        >>> from tests.utils import cleanup_active_positions
        >>> 
        >>> session = SessionLocal()
        >>> count = cleanup_active_positions(session, Environment.BACKTEST)
        >>> print(f"Cleaned {count} positions")
        >>> session.close()
    """
    query = db_session.query(ActivePosition).filter(
        ActivePosition.is_active == True,  # noqa: E712
        ActivePosition.environment == environment,
    )
    
    if symbol:
        query = query.filter(ActivePosition.symbol == symbol)
    
    positions = query.all()
    count = len(positions)
    
    if count == 0:
        return 0
    
    for pos in positions:
        pos.is_active = False
        pos.closed_at = datetime.utcnow()
        pos.close_reason = "test_cleanup"
    
    db_session.commit()
    
    return count
```

**File:** `tests/utils/conftest.py` (pytest fixtures)

```python
"""Pytest fixtures for database cleanup."""

import pytest

from quantagent.database import SessionLocal
from quantagent.models import Environment

from .cleanup import cleanup_active_positions


@pytest.fixture(scope="function")
def clean_backtest_positions():
    """
    Pytest fixture to clean up backtest positions before and after each test.
    
    Usage:
        def test_my_backtest(clean_backtest_positions):
            # Test runs with clean slate
            backtest = Backtest(...)
            metrics = backtest.run()
            assert ...
    """
    session = SessionLocal()
    
    # Clean up before test
    count_before = cleanup_active_positions(session, Environment.BACKTEST)
    if count_before > 0:
        print(f"\n[SETUP] Cleaned {count_before} stale positions before test")
    
    yield
    
    # Clean up after test
    count_after = cleanup_active_positions(session, Environment.BACKTEST)
    if count_after > 0:
        print(f"\n[TEARDOWN] Cleaned {count_after} positions after test")
    
    session.close()
```

---

### Step 4: Update Test Scripts

For standalone test scripts like `/tmp/quantagent_spy_rsi_surgical_probe.py`, add cleanup between runs:

```python
# At the top of the file
import sys
from pathlib import Path

REPO_ROOT = Path("/home/azureuser/repos/projects/QuantAgent")
sys.path.insert(0, str(REPO_ROOT))

from quantagent.backtesting.backtest import Backtest
from quantagent.database import SessionLocal
from quantagent.models import ActivePosition, Environment
from datetime import datetime

# ... existing imports ...


def cleanup_positions():
    """Clean up all active positions before tests."""
    session = SessionLocal()
    positions = session.query(ActivePosition).filter(
        ActivePosition.is_active == True,
        ActivePosition.environment == Environment.BACKTEST,
    ).all()
    
    for pos in positions:
        pos.is_active = False
        pos.closed_at = datetime.utcnow()
        pos.close_reason = "test_cleanup"
    
    session.commit()
    count = len(positions)
    session.close()
    
    if count > 0:
        print(f"\n[CLEANUP] Removed {count} stale positions before test")
    
    return count


# Main test execution
if __name__ == "__main__":
    print("=" * 70)
    print("SURGICAL TEST: SPY + RSIMeanReversionStrategy")
    print("=" * 70)
    
    # Clean up before starting
    cleanup_positions()
    
    # Run 1h test
    result_1h = run_single_test("1h")
    
    # Clean up between runs (optional, should not be needed with the fix)
    # cleanup_positions()
    
    # Run 4h test
    result_4h = run_single_test("4h")
    
    # ... rest of comparison logic ...
```

---

## Testing & Verification

### Pre-Implementation Check

Verify the current state of stale positions:

```bash
cd /home/azureuser/repos/projects/QuantAgent

python3 << 'EOF'
from quantagent.database import SessionLocal
from quantagent.models import ActivePosition, Environment

session = SessionLocal()
positions = session.query(ActivePosition).filter(
    ActivePosition.is_active == True,
    ActivePosition.environment == Environment.BACKTEST
).order_by(ActivePosition.backtest_run_id).all()

print(f"Current stale positions: {len(positions)}")
for pos in positions:
    print(f"  Run {pos.backtest_run_id}: ID {pos.id} | {pos.symbol} | {pos.opened_at}")

session.close()
EOF
```

Expected output (before fix):
```
Current stale positions: 9
  Run 12: ID 45 | SPY | 2026-06-28 14:11:17
  Run 15: ID 49 | SPY | 2026-06-28 22:42:25
  ...
```

---

### Post-Implementation Test

After implementing the fix:

```bash
# Run the surgical test
python3 /tmp/quantagent_spy_rsi_surgical_probe.py > /tmp/test_output.txt 2>&1

# Check for errors
grep -i "position already open" /tmp/test_output.txt
# Should return: (no output - no errors)

# Check cleanup logs
grep "stale" /tmp/test_output.txt
# Should show: stale position cleanup messages

# Verify final state
python3 << 'EOF'
from quantagent.database import SessionLocal
from quantagent.models import ActivePosition, Environment

session = SessionLocal()
count = session.query(ActivePosition).filter(
    ActivePosition.is_active == True,
    ActivePosition.environment == Environment.BACKTEST
).count()

print(f"Remaining stale positions: {count}")
assert count == 0, f"Expected 0 stale positions, found {count}"
print("✅ All stale positions cleaned up")

session.close()
EOF
```

---

### Integration Test

Create a new test file: `tests/backtesting/test_stale_position_cleanup.py`

```python
"""Test stale position cleanup in sequential backtests."""

from datetime import datetime, timedelta

import pytest

from quantagent.backtesting.backtest import Backtest
from quantagent.database import SessionLocal
from quantagent.models import ActivePosition, Environment
from quantagent.strategy.registry import build_strategy


def test_sequential_backtests_no_contamination():
    """
    Test that running two backtests sequentially does NOT contaminate results.
    
    This reproduces the bug scenario: 1H backtest followed by 4H backtest.
    With the fix, they should produce DIFFERENT trades and P&L.
    """
    asset = "SPY"
    strategy_name = "RSIMeanReversionStrategy"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    config = {
        "base_position_pct": 0.05,
        "max_daily_loss_pct": 0.05,
        "max_position_pct": 0.10,
        "slippage_pct": 0.01,
    }
    
    # Run 1H backtest
    strategy_1h = build_strategy(strategy_name)
    backtest_1h = Backtest(
        start_date=start_date,
        end_date=end_date,
        assets=[asset],
        timeframe="1h",
        initial_capital=100000.0,
        config=config,
        strategy=strategy_1h,
    )
    metrics_1h = backtest_1h.run(name="Test 1H")
    
    # Verify no stale positions after 1H
    session = SessionLocal()
    stale_after_1h = session.query(ActivePosition).filter(
        ActivePosition.is_active == True,
        ActivePosition.environment == Environment.BACKTEST,
    ).count()
    session.close()
    
    assert stale_after_1h == 0, f"Expected 0 stale positions after 1H, found {stale_after_1h}"
    
    # Run 4H backtest
    strategy_4h = build_strategy(strategy_name)
    backtest_4h = Backtest(
        start_date=start_date,
        end_date=end_date,
        assets=[asset],
        timeframe="4h",
        initial_capital=100000.0,
        config=config,
        strategy=strategy_4h,
    )
    metrics_4h = backtest_4h.run(name="Test 4H")
    
    # Verify no stale positions after 4H
    session = SessionLocal()
    stale_after_4h = session.query(ActivePosition).filter(
        ActivePosition.is_active == True,
        ActivePosition.environment == Environment.BACKTEST,
    ).count()
    session.close()
    
    assert stale_after_4h == 0, f"Expected 0 stale positions after 4H, found {stale_after_4h}"
    
    # Verify metrics are DIFFERENT
    # (Different timeframes should produce different results)
    assert metrics_1h.total_trades != metrics_4h.total_trades, \
        f"Trades should differ: 1H={metrics_1h.total_trades}, 4H={metrics_4h.total_trades}"
    
    assert metrics_1h.total_pnl != metrics_4h.total_pnl, \
        f"P&L should differ: 1H={metrics_1h.total_pnl}, 4H={metrics_4h.total_pnl}"
    
    # Verify evaluations are in expected ratio (~4:1)
    ratio = metrics_1h.agent_invocations / max(metrics_4h.agent_invocations, 1)
    assert 2.0 < ratio < 6.0, \
        f"Evaluation ratio 1H:4H should be ~4:1, got {ratio:.2f}"
    
    print(f"✅ Test passed:")
    print(f"  1H: {metrics_1h.total_trades} trades, {metrics_1h.agent_invocations} evals")
    print(f"  4H: {metrics_4h.total_trades} trades, {metrics_4h.agent_invocations} evals")
    print(f"  Ratio: {ratio:.2f}:1")
```

Run the test:
```bash
cd /home/azureuser/repos/projects/QuantAgent
pytest tests/backtesting/test_stale_position_cleanup.py -v -s
```

---

## Rollout Plan

### Phase 1: Immediate Fix (P0)
1. ✅ Implement `_cleanup_stale_positions()` method
2. ✅ Add call in `run()` method
3. ✅ Test with surgical probe script
4. ✅ Commit and push

### Phase 2: Test Infrastructure (P1)
1. Add test utility functions
2. Add pytest fixtures
3. Add integration test
4. Update existing tests to use fixtures

### Phase 3: Documentation (P2)
1. Update TESTING_PATTERNS.md
2. Update architecture.md
3. Add troubleshooting guide

---

## Commit Strategy

### Commit 1: Core Fix
```bash
git add quantagent/backtesting/backtest.py
git commit -m "fix(backtest): Clean up stale ActivePosition records at start

- Add _cleanup_stale_positions() to close ALL active positions
  for backtest assets regardless of backtest_run_id
- Call cleanup at start of run() to prevent contamination
- Fixes data bleed between sequential 1H and 4H backtests
- Prevents 'Position already open' errors

Related: QuantAgent-iip
"
```

### Commit 2: Test Utilities
```bash
git add tests/utils/cleanup.py tests/utils/conftest.py
git commit -m "test(utils): Add fixtures for active position cleanup

- Add cleanup_active_positions() utility function
- Add clean_backtest_positions pytest fixture
- Enables clean slate testing for backtests

Related: QuantAgent-iip
"
```

### Commit 3: Integration Test
```bash
git add tests/backtesting/test_stale_position_cleanup.py
git commit -m "test(backtest): Add regression test for stale position cleanup

- Verifies sequential backtests don't contaminate each other
- Tests 1H followed by 4H produces different results
- Checks no stale positions remain after completion

Related: QuantAgent-iip
"
```

---

## Success Criteria

### ✅ Must Have (Before Merge)
- [ ] `_cleanup_stale_positions()` implemented and called
- [ ] Surgical test passes without "Position already open" errors
- [ ] 1H and 4H backtests produce DIFFERENT metrics
- [ ] Zero stale positions remain after completion

### ✅ Should Have (Before Release)
- [ ] Test utilities added
- [ ] Integration test passes
- [ ] Manual verification on production-like data

### ✅ Nice to Have (Future)
- [ ] Database constraint to prevent multiple active positions
- [ ] Monitoring/alerting for stale positions
- [ ] Periodic cleanup job

---

## Rollback Plan

If the fix causes issues:

1. **Immediate:** Revert commit
   ```bash
   git revert <commit-hash>
   ```

2. **Manual cleanup:** Clear stale positions
   ```python
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
   session.close()
   ```

3. **Investigate:** Check logs for cleanup errors
   ```bash
   grep "stale_cleanup_error" logs/quantagent.log
   ```

---

## Related Documents

- [Root Cause Analysis](./QuantAgent-iip-DC-data-bleed-root-cause-analysis.md)
- [Original Bug Analysis](./QuantAgent-iip-DC-invocation-counter-bug-analysis.md)
- Issue: QuantAgent-iip in Beads

---

**Document Status:** READY FOR IMPLEMENTATION  
**Estimated Effort:** 2-3 hours  
**Risk Level:** Low (cleanup is defensive, won't break existing functionality)
