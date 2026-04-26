# QuantAgent-0b5 — Requirements: Integrate PositionMonitor into TradingScheduler

**Issue ID:** QuantAgent-0b5  
**Title:** Integrate PositionMonitor into TradingScheduler  
**Type:** Task  
**Priority:** 2  
**Labels:** enhancement, trading  
**Estimated:** 480 minutes (8 hours)

---

## Objective

Integrate the existing PositionMonitor component into TradingScheduler to track active positions, check exit conditions before re-analyzing, and record exit reasons in paper trading.

---

## Background

### Current State (Problem)

PositionMonitor was implemented and tested in previous issues (QuantAgent-boi, QuantAgent-on4) and successfully integrated into the Backtest path. However, **TradingScheduler (paper trading) does not use PositionMonitor at all**.

**Gaps in paper trading:**
1. **No position tracking**: `analyze_and_trade()` runs the full LLM graph on every tick without checking if positions should exit first
2. **No candle tracking**: Active positions don't get updated with price movements via `PositionMonitor.update_candle_tracking()`
3. **No exit reason recording**: When positions close, the exit reason (stop loss, take profit, directional accuracy) is never recorded

**Current flow (broken):**
```
Every tick:
  1. Fetch market data
  2. Run full LLM graph (expensive)
  3. Execute order if signal != HOLD
  
Missing:
  - Check if open position should exit first
  - Update position tracking with new candle
  - Record exit reason in Trade model
```

---

## Scope

### In Scope
1. **Add PositionMonitor to TradingScheduler**
   - Initialize PositionMonitor in `__init__`
   - Pass `db_session` and `backtest_run_id=None` (paper trading)

2. **Update tracking on each tick**
   - Before analysis: call `PositionMonitor.update_candle_tracking()` for open positions
   - Pass current price and previous close

3. **Check for exits before re-analysis**
   - After updating tracking: check if position should exit
   - If exit needed: execute exit order, close position with reason
   - Skip LLM analysis if exit was executed

4. **Record exit reason in Trade model**
   - When closing position: update Trade record with exit_reason
   - Map position close_reason to Trade fields

5. **Add integration tests**
   - Test: position tracked on each tick
   - Test: stop loss exit bypasses LLM
   - Test: take profit exit bypasses LLM
   - Test: exit reasons persisted to DB

### Out of Scope
- Modifying PositionMonitor logic (already implemented and tested)
- Changing LLM graph behavior
- Backtesting integration (already done in previous issues)
- Order execution logic changes (OrderManager handles this)
- Paper broker changes

---

## Requirements

### FR-1: PositionMonitor Initialization
**Description:** TradingScheduler instantiates PositionMonitor

**Requirements:**
- Add `position_monitor` attribute to TradingScheduler
- Initialize in `__init__` with `db_session` and `backtest_run_id=None`
- Ensure PositionMonitor persists across scheduler ticks

---

### FR-2: Position Tracking on Each Tick
**Description:** Update position tracking before analyzing

**Requirements:**
- In `_process_asset()`, before calling `strategy.generate_signal()`:
  - Get active position for symbol: `position_monitor.get_active_position(symbol)`
  - If position exists:
    - Extract current_price from market data
    - Extract prev_close (previous candle close)
    - Call `position_monitor.update_candle_tracking(position, current_price, prev_close)`

**Exit conditions to check:**
- Stop loss hit: `current_price <= position.stop_loss` (LONG) or `current_price >= position.stop_loss` (SHORT)
- Take profit hit: `current_price >= position.take_profit` (LONG) or `current_price <= position.take_profit` (SHORT)
- Max hold candles reached: `position.candles_since_entry >= position.max_hold_candles` (if configured)

---

### FR-3: Exit Before Re-Analysis
**Description:** Check if position should exit and execute exit order

**Requirements:**
- After updating tracking, evaluate exit conditions
- If exit needed:
  1. Determine exit reason: "stop_loss", "take_profit", "max_hold", "trailing_stop"
  2. Call `position_monitor.close_position(position, reason, exit_price=current_price)`
  3. Execute exit order via OrderManager
  4. **Skip LLM analysis** for this tick (position was closed)
  5. Record exit in logs

**Exit order execution:**
- Use `order_manager.execute_decision()` with opposite signal:
  - If LONG position: execute SELL/SHORT signal
  - If SHORT position: execute BUY/LONG signal
- Pass `environment=self.environment`

---

### FR-4: Record Exit Reason in Trade
**Description:** Persist exit reason to Trade model

**Requirements:**
- When closing position via PositionMonitor:
  - Query associated Trade record: `db.query(Trade).filter(Trade.id == position.trade_id).first()`
  - If Trade exists:
    - Set `Trade.exit_signal = position.close_reason`
    - Set `Trade.closed_at = position.closed_at`
    - Commit to DB

**Field mapping:**
- `position.close_reason` → `Trade.exit_signal`
- `position.closed_at` → `Trade.closed_at`
- `position.accuracy` → stored in ActivePosition, not duplicated in Trade

---

### FR-5: Logging and Observability
**Description:** Add structured logging for position monitoring

**Requirements:**
- Log when position tracking is updated
- Log when exit check is performed
- Log when exit is executed (reason, price)
- Use `event_type` for categorization:
  - `"scheduler.position_tracked"`
  - `"scheduler.exit_check"`
  - `"scheduler.position_exit"`

---

## Acceptance Criteria

### AC-1: PositionMonitor Instantiated
**Given** TradingScheduler is initialized  
**When** checking `scheduler.position_monitor` attribute  
**Then** it is an instance of PositionMonitor with `db_session` and `backtest_run_id=None`

---

### AC-2: Position Tracking on Each Tick
**Given** an active position exists for a symbol  
**When** `analyze_and_trade()` is called  
**Then**:
- `PositionMonitor.update_candle_tracking()` is called with current price and prev close
- Position's `candles_since_entry` increments
- Position's `candles_direction` is updated (if < prediction_horizon)

---

### AC-3: Stop Loss Exit
**Given** an active LONG position with entry_price=100, stop_loss=98  
**And** current_price=97 (below stop loss)  
**When** `analyze_and_trade()` runs  
**Then**:
- Position is closed with `close_reason="stop_loss"`
- Exit order is executed (SELL)
- LLM analysis is **skipped** for this tick
- Trade record updated with `exit_signal="stop_loss"`

---

### AC-4: Take Profit Exit
**Given** an active LONG position with entry_price=100, take_profit=105  
**And** current_price=106 (above take profit)  
**When** `analyze_and_trade()` runs  
**Then**:
- Position is closed with `close_reason="take_profit"`
- Exit order is executed (SELL)
- LLM analysis is **skipped**
- Trade record updated with `exit_signal="take_profit"`

---

### AC-5: No Exit, Continue Analysis
**Given** an active position that doesn't meet exit conditions  
**When** `analyze_and_trade()` runs  
**Then**:
- Position tracking is updated
- Exit check performed, no exit needed
- LLM analysis proceeds normally
- No exit order executed

---

### AC-6: Integration Tests Pass
**Given** new integration tests are added  
**When** running test suite  
**Then** at least 3 new tests pass:
1. `test_scheduler_tracks_position_on_tick()`
2. `test_scheduler_exits_on_stop_loss()`
3. `test_scheduler_exits_on_take_profit()`

---

### AC-7: Existing Tests Pass
**Given** existing scheduler unit tests  
**When** running test suite  
**Then** all existing tests still pass (backward compatibility)

---

## Constraints

- **No PositionMonitor changes**: Use existing implementation as-is
- **Minimal TradingScheduler changes**: Only add position monitoring logic
- **Backward compatible**: Existing behavior preserved (no position = continue as before)
- **Paper trading only**: `backtest_run_id=None` for TradingScheduler

---

## Non-Functional Requirements

### NFR-1: Performance
- Position monitoring should add < 50ms per tick
- Exit checks are O(1) (simple price comparisons)

### NFR-2: Robustness
- Graceful handling if PositionMonitor fails (log error, continue)
- Transaction boundaries clear (commit after position updates)

### NFR-3: Observability
- All position events logged with structured fields
- Exit reasons traceable in logs and database

---

## Definition of Done

- [ ] PositionMonitor integrated into TradingScheduler
- [ ] Position tracking called on each tick
- [ ] Exit checks performed before LLM analysis
- [ ] Exit orders executed when needed
- [ ] Exit reasons recorded in Trade model
- [ ] 3+ new integration tests added and passing
- [ ] Existing scheduler tests pass
- [ ] Structured logging added
- [ ] Documentation updated (this file)
