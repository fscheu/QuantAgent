# QuantAgent-0b5 — Design: Integrate PositionMonitor into TradingScheduler

**Issue ID:** QuantAgent-0b5  
**Title:** Integrate PositionMonitor into TradingScheduler  
**Type:** Task

---

## Design Overview

Add PositionMonitor integration to TradingScheduler's `_process_asset()` method to track positions, check exit conditions before LLM analysis, and record exit reasons.

---

## Architecture Changes

### Current Flow (Before)
```
TradingScheduler.analyze_and_trade():
  for each asset:
    1. _process_asset(symbol):
       a. Fetch market data
       b. Run LLM strategy.generate_signal()
       c. If signal != HOLD:
          - Execute order via OrderManager
```

### New Flow (After)
```
TradingScheduler.analyze_and_trade():
  for each asset:
    1. _process_asset(symbol):
       a. Fetch market data
       b. Check for active position
       c. If position exists:
          i.   Update tracking (candles, price)
          ii.  Check exit conditions
          iii. If exit needed:
               - Close position
               - Execute exit order
               - Record exit reason
               - SKIP LLM analysis
               - RETURN early
       d. Run LLM strategy.generate_signal()
       e. If signal != HOLD:
          - Execute order via OrderManager
          - Open new ActivePosition via PositionMonitor
```

**Key change:** Exit check happens **before** expensive LLM call.

---

## Implementation Details

### 1. Add PositionMonitor to __init__

```python
class TradingScheduler:
    def __init__(
        self,
        *,
        trading_graph: TradingGraph,
        order_manager: OrderManager,
        data_provider: DataProvider,
        db_session: Session,
        scheduler_settings: Optional[settings.SchedulerSettings] = None,
        scheduler_factory: Optional[Callable[[], BackgroundScheduler]] = None,
        strategy: Optional[LLMAgentStrategy] = None,
    ) -> None:
        # ... existing init code ...
        
        # NEW: Add PositionMonitor
        from quantagent.trading.position_monitor import PositionMonitor
        self.position_monitor = PositionMonitor(
            db_session=db_session,
            backtest_run_id=None  # Paper trading doesn't use backtest_run_id
        )
```

---

### 2. Add Position Monitoring to _process_asset

Insert **after** `_fetch_market_data()` and **before** `strategy.generate_signal()`:

```python
def _process_asset(self, symbol: str) -> None:
    df = self._fetch_market_data(symbol)
    kline_data = format_ohlcv_for_agents(df)
    current_price = float(df["close"].iloc[-1])
    
    # NEW: Check for active position and handle exits
    position = self.position_monitor.get_active_position(symbol)
    if position:
        prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else current_price
        
        # Update tracking
        self.position_monitor.update_candle_tracking(
            position, current_price, prev_close
        )
        
        # Check exit conditions
        should_exit, exit_reason = self._check_exit_conditions(
            position, current_price
        )
        
        if should_exit:
            self._execute_position_exit(
                position, exit_reason, current_price, symbol
            )
            # Early return - skip LLM analysis
            logger.info(
                "Position exit executed for %s: %s (price=%.2f)",
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
            return
    
    # Existing code continues: LLM analysis, order execution, etc.
    thread_id = self._make_thread_id(symbol)
    try:
        signal = self.strategy.generate_signal(
            kline_data, symbol, self.config.timeframe,
            current_price, thread_id=thread_id
        )
    # ... rest of existing code ...
```

---

### 3. Implement Exit Condition Check

```python
def _check_exit_conditions(
    self, position: ActivePosition, current_price: float
) -> tuple[bool, Optional[str]]:
    """
    Check if position should exit based on price levels.
    
    Returns:
        (should_exit, exit_reason) tuple
    """
    from quantagent.models import OrderSide
    
    # Stop loss check
    if position.side == OrderSide.BUY:
        if current_price <= float(position.stop_loss):
            return (True, "stop_loss")
    elif position.side == OrderSide.SELL:
        if current_price >= float(position.stop_loss):
            return (True, "stop_loss")
    
    # Take profit check
    if position.side == OrderSide.BUY:
        if current_price >= float(position.take_profit):
            return (True, "take_profit")
    elif position.side == OrderSide.SELL:
        if current_price <= float(position.take_profit):
            return (True, "take_profit")
    
    # Max hold candles check
    if position.max_hold_candles is not None:
        if position.candles_since_entry >= position.max_hold_candles:
            return (True, "max_hold")
    
    # Trailing stop check (if implemented)
    if position.trailing_stop_pct is not None and position.highest_price_seen:
        # Implementation depends on trailing stop logic
        pass
    
    return (False, None)
```

---

### 4. Implement Position Exit Execution

```python
def _execute_position_exit(
    self,
    position: ActivePosition,
    exit_reason: str,
    exit_price: float,
    symbol: str,
) -> None:
    """Execute exit order and close position."""
    from quantagent.models import OrderSide, TradeSignal, Trade
    
    # Determine opposite signal for exit
    if position.side == OrderSide.BUY:
        exit_signal = TradeSignal.SHORT  # Sell to close LONG
    else:
        exit_signal = TradeSignal.LONG   # Buy to close SHORT
    
    # Close position in PositionMonitor
    self.position_monitor.close_position(
        position, reason=exit_reason, exit_price=exit_price
    )
    
    # Execute exit order
    try:
        order = self.order_manager.execute_decision(
            symbol=symbol,
            decision=exit_signal,
            confidence=1.0,  # Exit is certain, not probabilistic
            current_price=exit_price,
            environment=self.environment,
            trigger_signal_id=position.signal_id,
        )
    except Exception as exc:
        self.db.rollback()
        logger.error(
            "Failed to execute exit order for %s: %s",
            symbol, exc,
            extra={
                "event_type": "scheduler.exit_order_error",
                "symbol": symbol,
                "environment": self.environment.value,
            }
        )
        raise ExecutionError(f"Exit order failed: {exc}") from exc
    
    # Update Trade record with exit reason
    if position.trade_id:
        trade = self.db.query(Trade).filter(Trade.id == position.trade_id).first()
        if trade:
            trade.exit_signal = exit_reason
            trade.closed_at = position.closed_at
            self.db.commit()
```

---

### 5. Open Position After New Order

When a **new** order is executed (not an exit), create ActivePosition:

```python
def _process_asset(self, symbol: str) -> None:
    # ... existing code ...
    
    # After successful order execution:
    try:
        order = self.order_manager.execute_decision(...)
    except Exception as exc:
        self.db.rollback()
        raise ExecutionError(str(exc)) from exc
    
    if order:
        # NEW: Open ActivePosition for tracking
        from quantagent.models import ExitPolicy
        
        # Extract stop loss, take profit from signal state_snapshot
        stop_loss = signal.stop_loss or (current_price * 0.98)  # Fallback
        take_profit = signal.take_profit or (current_price * 1.04)
        
        self.position_monitor.open_position(
            symbol=symbol,
            side=order.side,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=order.quantity,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            trade_id=None,  # Trade created by OrderManager separately
            signal_id=db_signal.id if db_signal else None,
            backtest_run_id=None,  # Paper trading
        )
        
        logger.info(...)
```

---

## Design Decisions

### Decision 1: Exit Check Before LLM Analysis

**Chosen:** Check exits **before** calling `strategy.generate_signal()`

**Rationale:**
- LLM calls are expensive (time + cost)
- If position should exit, LLM analysis is irrelevant
- Improves responsiveness (exit triggers faster)

**Trade-off:**
- Slightly more complex flow (early return)
- Benefit: Significant performance improvement

---

### Decision 2: Separate Exit Order Execution

**Chosen:** Use existing `OrderManager.execute_decision()` for exits

**Rationale:**
- Reuses existing order execution logic
- Maintains consistency (all orders go through same path)
- Handles fills, trades, commissions automatically

**Alternative considered:** Direct order creation
- **Rejected:** Would bypass OrderManager safeguards and logging

---

### Decision 3: Exit Reason in Trade.exit_signal

**Chosen:** Store exit reason in `Trade.exit_signal` field (string)

**Rationale:**
- Trade model already has `exit_signal` field (currently unused in paper trading)
- Semantic fit: "why did we exit?"
- No schema changes needed

**Field values:**
- `"stop_loss"`, `"take_profit"`, `"max_hold"`, `"trailing_stop"`

---

### Decision 4: PositionMonitor Instantiation

**Chosen:** Create single PositionMonitor instance in `__init__`

**Rationale:**
- Persistent across ticks
- Shares db_session with scheduler
- No need for multiple instances

**Alternative considered:** Create per-asset
- **Rejected:** Unnecessary overhead, same db_session needed

---

## Error Handling Strategy

### Exit Check Failures
```python
try:
    should_exit, reason = self._check_exit_conditions(position, current_price)
except Exception:
    logger.exception("Exit check failed for %s", symbol)
    # Continue with LLM analysis (conservative: don't exit on error)
    should_exit = False
```

### Exit Order Failures
```python
try:
    self._execute_position_exit(...)
except ExecutionError:
    # Position already closed in PositionMonitor
    # Log error, continue (position state is updated, order failed)
    logger.error("Exit order failed but position closed")
```

### Position Tracking Failures
```python
try:
    self.position_monitor.update_candle_tracking(...)
except Exception:
    logger.exception("Position tracking failed")
    # Continue (tracking is nice-to-have, not critical)
```

---

## Testing Strategy

### Unit Tests (New)
- `test_check_exit_conditions_stop_loss()`
- `test_check_exit_conditions_take_profit()`
- `test_check_exit_conditions_no_exit()`
- `test_execute_position_exit()`

### Integration Tests (New)
- `test_scheduler_tracks_position_on_tick()` — Verify candle tracking
- `test_scheduler_exits_on_stop_loss()` — Full flow: position → exit → order
- `test_scheduler_exits_on_take_profit()` — Full flow
- `test_scheduler_opens_position_after_signal()` — New position creation

### Existing Tests
- All existing scheduler tests must pass (backward compatibility)
- Tests without active positions should behave identically

---

## Rollback Plan

If issues arise:

1. **Revert changes to scheduler.py:**
   ```bash
   git checkout origin/main -- quantagent/trading/scheduler.py
   ```

2. **Disable position monitoring:**
   - Add feature flag: `ENABLE_POSITION_MONITOR=false`
   - Skip position checks if flag is false

---

## Open Questions

None — design is straightforward and builds on existing, tested PositionMonitor implementation.
