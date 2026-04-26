# QuantAgent-0b5 — Planning: Integrate PositionMonitor into TradingScheduler

**Issue ID:** QuantAgent-0b5  
**Title:** Integrate PositionMonitor into TradingScheduler  
**Type:** Task  
**Priority:** 2

---

## Objective

Integrate PositionMonitor into TradingScheduler to track positions, check exit conditions before LLM analysis, and record exit reasons in paper trading.

---

## Tasks

### Task 1: Add PositionMonitor to TradingScheduler.__init__
**Estimate:** 0.5h (30 minutes)

**What:**
- Import PositionMonitor in scheduler.py
- Add `self.position_monitor` attribute in `__init__`
- Initialize with `db_session=db_session, backtest_run_id=None`

**Code changes:**
```python
# At top of scheduler.py
from quantagent.trading.position_monitor import PositionMonitor

# In TradingScheduler.__init__
def __init__(self, ...) -> None:
    # ... existing init code ...
    
    # Add PositionMonitor
    self.position_monitor = PositionMonitor(
        db_session=db_session,
        backtest_run_id=None  # Paper trading
    )
```

**How to validate:**
```python
# Unit test
def test_scheduler_initializes_position_monitor():
    scheduler = TradingScheduler(...)
    assert isinstance(scheduler.position_monitor, PositionMonitor)
    assert scheduler.position_monitor.db == scheduler.db
    assert scheduler.position_monitor.backtest_run_id is None
```

**Dependencies:** None

---

### Task 2: Implement _check_exit_conditions() Helper
**Estimate:** 1h (60 minutes)

**What:**
- Add private method `_check_exit_conditions(position, current_price)`
- Check stop loss, take profit, max hold candles
- Return tuple: (should_exit: bool, exit_reason: Optional[str])

**Code:**
```python
def _check_exit_conditions(
    self, position: ActivePosition, current_price: float
) -> tuple[bool, Optional[str]]:
    """Check if position should exit."""
    from quantagent.models import OrderSide
    
    # Stop loss
    if position.side == OrderSide.BUY:
        if current_price <= float(position.stop_loss):
            return (True, "stop_loss")
    else:  # SELL
        if current_price >= float(position.stop_loss):
            return (True, "stop_loss")
    
    # Take profit
    if position.side == OrderSide.BUY:
        if current_price >= float(position.take_profit):
            return (True, "take_profit")
    else:
        if current_price <= float(position.take_profit):
            return (True, "take_profit")
    
    # Max hold
    if position.max_hold_candles:
        if position.candles_since_entry >= position.max_hold_candles:
            return (True, "max_hold")
    
    return (False, None)
```

**How to validate:**
```python
def test_check_exit_conditions_stop_loss_long():
    position = mock_position(side=BUY, stop_loss=98.0)
    should_exit, reason = scheduler._check_exit_conditions(position, 97.0)
    assert should_exit is True
    assert reason == "stop_loss"

def test_check_exit_conditions_no_exit():
    position = mock_position(side=BUY, stop_loss=98.0, take_profit=105.0)
    should_exit, reason = scheduler._check_exit_conditions(position, 101.0)
    assert should_exit is False
    assert reason is None
```

**Dependencies:** Task 1

---

### Task 3: Implement _execute_position_exit() Helper
**Estimate:** 1.5h (90 minutes)

**What:**
- Add private method `_execute_position_exit(position, reason, price, symbol)`
- Close position via PositionMonitor
- Execute exit order via OrderManager
- Update Trade record with exit_signal
- Add structured logging

**Code:**
```python
def _execute_position_exit(
    self,
    position: ActivePosition,
    exit_reason: str,
    exit_price: float,
    symbol: str,
) -> None:
    """Execute position exit."""
    from quantagent.models import OrderSide, TradeSignal, Trade
    
    # Determine exit signal (opposite)
    exit_signal = (
        TradeSignal.SHORT if position.side == OrderSide.BUY
        else TradeSignal.LONG
    )
    
    # Close position
    self.position_monitor.close_position(
        position, reason=exit_reason, exit_price=exit_price
    )
    
    # Execute exit order
    try:
        order = self.order_manager.execute_decision(
            symbol=symbol,
            decision=exit_signal,
            confidence=1.0,
            current_price=exit_price,
            environment=self.environment,
            trigger_signal_id=position.signal_id,
        )
    except Exception as exc:
        self.db.rollback()
        logger.error(
            "Exit order failed for %s: %s", symbol, exc,
            extra={"event_type": "scheduler.exit_order_error", "symbol": symbol}
        )
        raise ExecutionError(f"Exit failed: {exc}") from exc
    
    # Update Trade record
    if position.trade_id:
        trade = self.db.query(Trade).filter(Trade.id == position.trade_id).first()
        if trade:
            trade.exit_signal = exit_reason
            trade.closed_at = position.closed_at
            self.db.commit()
```

**How to validate:**
```python
def test_execute_position_exit():
    position = create_test_position()
    scheduler._execute_position_exit(position, "stop_loss", 97.0, "BTC-USD")
    
    # Position closed
    db.refresh(position)
    assert position.is_active is False
    assert position.close_reason == "stop_loss"
    
    # Exit order executed
    orders = db.query(Order).filter(Order.symbol == "BTC-USD").all()
    assert orders[-1].side == OrderSide.SELL
```

**Dependencies:** Task 2

---

### Task 4: Add Position Monitoring to _process_asset()
**Estimate:** 2h (120 minutes)

**What:**
- After `_fetch_market_data()`, add position check logic
- Get active position for symbol
- If position exists:
  - Calculate prev_close
  - Update candle tracking
  - Check exit conditions
  - If exit needed: execute exit and return early
- Insert **before** `strategy.generate_signal()` call

**Code location:** In `_process_asset()` method

**Code changes:**
```python
def _process_asset(self, symbol: str) -> None:
    df = self._fetch_market_data(symbol)
    kline_data = format_ohlcv_for_agents(df)
    current_price = float(df["close"].iloc[-1])
    
    # NEW: Position monitoring and exit check
    position = self.position_monitor.get_active_position(symbol)
    if position:
        # Calculate prev_close (fallback to current if not enough data)
        prev_close = (
            float(df["close"].iloc[-2]) if len(df) >= 2
            else current_price
        )
        
        # Update tracking
        self.position_monitor.update_candle_tracking(
            position, current_price, prev_close
        )
        
        logger.debug(
            "Position tracked: %s (candles=%d)",
            symbol, position.candles_since_entry,
            extra={"event_type": "scheduler.position_tracked", "symbol": symbol}
        )
        
        # Check exit conditions
        should_exit, exit_reason = self._check_exit_conditions(
            position, current_price
        )
        
        if should_exit:
            self._execute_position_exit(
                position, exit_reason, current_price, symbol
            )
            logger.info(
                "Position exit: %s reason=%s price=%.2f",
                symbol, exit_reason, current_price,
                extra={
                    "event_type": "scheduler.position_exit",
                    "symbol": symbol,
                    "environment": self.environment.value,
                    "extra_data": {
                        "reason": exit_reason,
                        "exit_price": current_price
                    }
                }
            )
            return  # Skip LLM analysis
    
    # Existing code continues: LLM analysis
    thread_id = self._make_thread_id(symbol)
    try:
        signal = self.strategy.generate_signal(...)
    # ... rest unchanged ...
```

**How to validate:**
- Integration test with active position → verify tracking updated
- Integration test with stop loss → verify early return, no LLM call
- Integration test without position → verify normal flow

**Dependencies:** Tasks 1-3

---

### Task 5: Add Position Creation After New Orders
**Estimate:** 1h (60 minutes)

**What:**
- After successful order execution (not exit), create ActivePosition
- Extract stop loss, take profit from signal
- Call `position_monitor.open_position()`

**Code location:** In `_process_asset()`, after `order_manager.execute_decision()` succeeds

**Code changes:**
```python
# After: order = self.order_manager.execute_decision(...)
if order:
    # NEW: Open ActivePosition for tracking
    # Extract levels from signal (with fallbacks)
    stop_loss = getattr(signal, 'stop_loss', None) or (
        current_price * 0.98 if trade_signal == TradeSignal.LONG
        else current_price * 1.02
    )
    take_profit = getattr(signal, 'take_profit', None) or (
        current_price * 1.04 if trade_signal == TradeSignal.LONG
        else current_price * 0.96
    )
    
    self.position_monitor.open_position(
        symbol=symbol,
        side=order.side,
        entry_price=current_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        quantity=order.quantity,
        exit_policy="sl_tp_only",
        signal_id=db_signal.id if db_signal else None,
        backtest_run_id=None,
    )
    
    logger.info(
        "Position opened: %s %s [entry=%.2f, sl=%.2f, tp=%.2f]",
        symbol, order.side.value, current_price, stop_loss, take_profit,
        extra={"event_type": "scheduler.position_opened", "symbol": symbol}
    )
    
    # ... existing logging continues ...
```

**How to validate:**
```python
def test_scheduler_opens_position_after_signal():
    mock_strategy.generate_signal.return_value = signal_with_levels()
    scheduler.analyze_and_trade()
    
    position = position_monitor.get_active_position("BTC-USD")
    assert position is not None
    assert position.is_active is True
```

**Dependencies:** Task 4

---

### Task 6: Add Unit Tests for Exit Logic
**Estimate:** 1.5h (90 minutes)

**What:**
- Create `tests/trading/test_scheduler_position_monitor.py`
- Add unit tests for new helper methods:
  - `test_check_exit_conditions_stop_loss_long()`
  - `test_check_exit_conditions_stop_loss_short()`
  - `test_check_exit_conditions_take_profit_long()`
  - `test_check_exit_conditions_take_profit_short()`
  - `test_check_exit_conditions_max_hold()`
  - `test_check_exit_conditions_no_exit()`
  - `test_execute_position_exit_closes_position()`
  - `test_execute_position_exit_creates_order()`
  - `test_execute_position_exit_updates_trade()`

**How to validate:**
```bash
pytest tests/trading/test_scheduler_position_monitor.py -v
# Expected: All 9+ tests PASS
```

**Dependencies:** Tasks 2-3

---

### Task 7: Add Integration Tests
**Estimate:** 2h (120 minutes)

**What:**
- Create `tests/integration/test_scheduler_position_integration.py`
- Add end-to-end tests:
  - `test_scheduler_tracks_position_on_tick()` — Full cycle: position → tracking → no exit → LLM analysis
  - `test_scheduler_exits_on_stop_loss()` — Full cycle: position → stop loss hit → exit order → LLM skipped
  - `test_scheduler_exits_on_take_profit()` — Full cycle: position → take profit hit → exit
  - `test_scheduler_opens_position_after_signal()` — No position → LLM signal → order → position created
  - `test_scheduler_handles_multiple_assets()` — Two assets, one exits, one continues

**Test structure:**
```python
def test_scheduler_exits_on_stop_loss(
    db_session, mock_data_provider, mock_strategy, scheduler
):
    # Setup: create position with stop loss at 98
    position = position_monitor.open_position(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=105.0,
        quantity=Decimal("1.0"),
        exit_policy="sl_tp_only",
    )
    
    # Mock market data: price drops to 97
    mock_data_provider.set_ohlc("BTC-USD", current_close=97.0, prev_close=99.0)
    
    # Execute scheduler
    stats = scheduler.analyze_and_trade()
    
    # Verify position closed
    db_session.refresh(position)
    assert position.is_active is False
    assert position.close_reason == "stop_loss"
    
    # Verify exit order created
    orders = db_session.query(Order).filter(Order.symbol == "BTC-USD").all()
    exit_order = orders[-1]
    assert exit_order.side == OrderSide.SELL
    
    # Verify LLM not called
    mock_strategy.generate_signal.assert_not_called()
    
    # Verify stats
    assert stats["processed"] == 1
    assert stats["errors"] == 0
```

**How to validate:**
```bash
pytest tests/integration/test_scheduler_position_integration.py -v
# Expected: All 5+ tests PASS
```

**Dependencies:** Tasks 1-5

---

### Task 8: Verify Existing Tests Pass
**Estimate:** 0.5h (30 minutes)

**What:**
- Run existing scheduler tests
- Fix any regressions caused by new code
- Ensure backward compatibility (tests without positions work as before)

**Commands:**
```bash
pytest tests/trading/test_scheduler.py -v
pytest tests/test_backtest.py -v  # Verify backtest path unaffected
```

**How to validate:**
- All pre-existing tests PASS
- No new failures introduced

**Dependencies:** Tasks 1-5

---

### Task 9: Update Documentation
**Estimate:** 0.5h (30 minutes)

**What:**
- Update `docs/03_technical/TESTING_PATTERNS.md` with position monitoring examples
- Add docstrings to new methods
- Update scheduler.py module docstring

**Files to update:**
- `quantagent/trading/scheduler.py` — Method docstrings
- `docs/03_technical/TESTING_PATTERNS.md` — Integration test examples

**How to validate:**
- Documentation is clear and accurate
- Docstrings follow project conventions

**Dependencies:** Tasks 1-7

---

## Total Estimate

**Total: 10.5 hours** (630 minutes)

**Breakdown:**
- Core implementation: 5.0h (Tasks 1-5)
- Testing: 4.0h (Tasks 6-8)
- Documentation: 0.5h (Task 9)
- Buffer: 1.0h (for unexpected issues)

**Note:** Original estimate was 8h (480 min). Planning identified more test coverage needed, increasing to 10.5h.

---

## Execution Order

### Phase 1: Core Implementation (Day 1)
1. **Task 1** — Init PositionMonitor (0.5h)
2. **Task 2** — Exit conditions helper (1h)
3. **Task 3** — Exit execution helper (1.5h)
4. **Task 4** — Integrate into _process_asset (2h)
5. **Task 5** — Position creation (1h)

**Total Phase 1:** 6 hours

---

### Phase 2: Testing (Day 2)
6. **Task 6** — Unit tests (1.5h)
7. **Task 7** — Integration tests (2h)
8. **Task 8** — Verify existing tests (0.5h)

**Total Phase 2:** 4 hours

---

### Phase 3: Polish (Optional)
9. **Task 9** — Documentation (0.5h)

---

## Risks & Mitigations

### Risk 1: OrderManager Assumptions
**Description:** OrderManager may not support "exit" orders directly

**Mitigation:**
- Use same `execute_decision()` method with opposite signal
- If issues arise: add `is_exit=True` parameter (small change)

**Probability:** Low  
**Impact:** Low (easy fix)

---

### Risk 2: Race Conditions
**Description:** Multiple scheduler instances may clash on same position

**Mitigation:**
- PositionMonitor uses DB transactions
- `get_active_position()` filters by `is_active=True`
- First instance to close wins (second gets no active position)

**Probability:** Low (paper trading is single-instance)  
**Impact:** Medium

---

### Risk 3: Test Flakiness
**Description:** Integration tests may be flaky with mocks

**Mitigation:**
- Use fixtures for clean DB state
- Mock only external dependencies (data provider, LLM)
- Keep PositionMonitor and OrderManager real (not mocked)

**Probability:** Medium  
**Impact:** Low

---

### Risk 4: Stop Loss Precision
**Description:** Float comparison may miss exact price levels

**Mitigation:**
- Use `<=` and `>=` (not `<` and `>`)
- Accept triggering at or past level (conservative)

**Probability:** Very Low  
**Impact:** Very Low

---

## Testing Strategy Summary

### Unit Tests (Task 6)
- Test exit condition logic in isolation
- Mock Position objects
- Fast, no DB needed

### Integration Tests (Task 7)
- Test full scheduler → PositionMonitor → OrderManager flow
- Real DB (test fixture)
- Mock only DataProvider and LLM
- Cover happy paths and edge cases

### Regression Tests (Task 8)
- Existing tests ensure backward compatibility
- No changes to tests needed (only run and verify)

---

## Rollback Plan

If critical issues arise post-merge:

### Option 1: Feature Flag
```python
# Add to settings.py
ENABLE_POSITION_MONITOR_IN_SCHEDULER = os.getenv("ENABLE_POSITION_MONITOR", "true").lower() == "true"

# In scheduler.py
if settings.ENABLE_POSITION_MONITOR_IN_SCHEDULER:
    position = self.position_monitor.get_active_position(symbol)
    # ... position monitoring logic ...
```

Set env var to disable:
```bash
export ENABLE_POSITION_MONITOR=false
```

---

### Option 2: Revert Commit
```bash
git revert <commit-hash>
git push origin main
```

---

## Success Criteria

- [ ] PositionMonitor integrated into TradingScheduler
- [ ] Exit checks performed before LLM analysis
- [ ] Exit orders executed when conditions met
- [ ] Exit reasons recorded in Trade model
- [ ] 9+ unit tests added and passing
- [ ] 5+ integration tests added and passing
- [ ] All existing tests pass
- [ ] Documentation updated

---

## Dependencies

**External:**
- QuantAgent-sfc (CLOSED) — SQLite/JSONB fix (required for tests)

**Internal:**
- PositionMonitor already implemented (QuantAgent-boi, QuantAgent-on4)
- OrderManager already supports execute_decision()
- Trade model already has exit_signal field

---

## Post-Completion Tasks

1. Monitor paper trading logs for exit events
2. Verify exit reasons appearing in database
3. Check LLM call reduction (should skip analysis on exits)
4. Update team documentation with new behavior
5. Consider adding metrics dashboard for exit tracking

---

## Final Checklist

**Before starting implementation:**
- [ ] Read PositionMonitor implementation
- [ ] Review OrderManager interface
- [ ] Understand current _process_asset() flow
- [ ] Set up test fixtures

**During implementation:**
- [ ] Task 1: Init PositionMonitor ✓
- [ ] Task 2: Exit conditions helper ✓
- [ ] Task 3: Exit execution helper ✓
- [ ] Task 4: Integrate monitoring ✓
- [ ] Task 5: Position creation ✓
- [ ] Task 6: Unit tests ✓
- [ ] Task 7: Integration tests ✓
- [ ] Task 8: Regression tests ✓
- [ ] Task 9: Documentation ✓

**After implementation:**
- [ ] All tests pass
- [ ] Code review complete
- [ ] Merge to main
- [ ] Monitor production logs
- [ ] Close Beads issue
