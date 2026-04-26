# QuantAgent-0b5 — Acceptance Criteria: Integrate PositionMonitor into TradingScheduler

**Issue ID:** QuantAgent-0b5  
**Title:** Integrate PositionMonitor into TradingScheduler  
**Type:** Task

---

## Acceptance Criteria

### AC-1: PositionMonitor Initialization

**Given** TradingScheduler is instantiated  
**When** checking the `scheduler.position_monitor` attribute  
**Then**:
- It exists and is an instance of PositionMonitor
- It has `db_session` set to the same session as scheduler
- It has `backtest_run_id=None` (paper trading context)

**Verification:**
```python
def test_scheduler_initializes_position_monitor():
    scheduler = TradingScheduler(...)
    assert isinstance(scheduler.position_monitor, PositionMonitor)
    assert scheduler.position_monitor.db == scheduler.db
    assert scheduler.position_monitor.backtest_run_id is None
```

---

### AC-2: Position Tracking on Each Tick

**Given** an active LONG position exists for symbol "BTC-USD"  
**And** position has `candles_since_entry=2`, `candles_direction=["up", "down"]`  
**When** `analyze_and_trade()` is called  
**And** current price is 101, previous close is 100  
**Then**:
- `PositionMonitor.update_candle_tracking()` is called with:
  - `position` = the active position
  - `current_price` = 101
  - `prev_close` = 100
- After update:
  - `position.candles_since_entry` = 3
  - `position.candles_direction` = ["up", "down", "up"] (101 > 100)

**Verification:**
```python
def test_scheduler_updates_position_tracking():
    # Setup: create active position
    position = position_monitor.open_position(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=105.0,
        quantity=Decimal("1.0"),
        exit_policy="sl_tp_only",
    )
    
    # Mock market data with current=101, prev=100
    mock_data_provider.set_data(...)
    
    # Run scheduler
    scheduler.analyze_and_trade()
    
    # Verify tracking updated
    db.refresh(position)
    assert position.candles_since_entry == 1
    assert position.candles_direction == ["up"]
```

---

### AC-3: Stop Loss Exit (LONG Position)

**Given** an active LONG position:
- Symbol: "BTC-USD"
- Entry price: 100
- Stop loss: 98
- Take profit: 105
**And** current market price drops to 97  
**When** `analyze_and_trade()` is called  
**Then**:
1. Position is closed via `PositionMonitor.close_position()`
   - `close_reason` = "stop_loss"
   - `exit_price` = 97
   - `is_active` = False
2. Exit order is executed:
   - Side: SELL (opposite of BUY)
   - Signal: SHORT
   - Price: 97
3. LLM analysis is **skipped** (no `strategy.generate_signal()` call)
4. Trade record updated:
   - `Trade.exit_signal` = "stop_loss"
   - `Trade.closed_at` = position.closed_at
5. Log entry created with `event_type="scheduler.position_exit"`

**Verification:**
```python
def test_scheduler_exits_long_on_stop_loss():
    # Setup
    position = create_long_position(stop_loss=98.0)
    mock_data(current_price=97.0)
    
    # Execute
    scheduler.analyze_and_trade()
    
    # Verify
    db.refresh(position)
    assert position.is_active is False
    assert position.close_reason == "stop_loss"
    
    # Verify exit order executed
    orders = db.query(Order).filter(Order.symbol == "BTC-USD").all()
    exit_order = orders[-1]
    assert exit_order.side == OrderSide.SELL
    
    # Verify LLM not called (mock)
    mock_strategy.generate_signal.assert_not_called()
```

---

### AC-4: Stop Loss Exit (SHORT Position)

**Given** an active SHORT position:
- Symbol: "ETH-USD"
- Entry price: 100
- Stop loss: 102
- Take profit: 95
**And** current market price rises to 103  
**When** `analyze_and_trade()` is called  
**Then**:
1. Position closed with `close_reason="stop_loss"`
2. Exit order executed: BUY (opposite of SELL)
3. LLM analysis skipped
4. Trade record updated with exit_signal="stop_loss"

**Verification:**
```python
def test_scheduler_exits_short_on_stop_loss():
    position = create_short_position(stop_loss=102.0)
    mock_data(current_price=103.0)
    
    scheduler.analyze_and_trade()
    
    db.refresh(position)
    assert position.close_reason == "stop_loss"
    
    exit_order = get_last_order("ETH-USD")
    assert exit_order.side == OrderSide.BUY
```

---

### AC-5: Take Profit Exit (LONG Position)

**Given** an active LONG position:
- Entry price: 100
- Stop loss: 98
- Take profit: 105
**And** current price rises to 106  
**When** `analyze_and_trade()` is called  
**Then**:
1. Position closed with `close_reason="take_profit"`
2. Exit order executed: SELL
3. LLM analysis skipped
4. Trade record updated with exit_signal="take_profit"

**Verification:**
```python
def test_scheduler_exits_long_on_take_profit():
    position = create_long_position(take_profit=105.0)
    mock_data(current_price=106.0)
    
    scheduler.analyze_and_trade()
    
    db.refresh(position)
    assert position.close_reason == "take_profit"
```

---

### AC-6: Take Profit Exit (SHORT Position)

**Given** an active SHORT position:
- Entry price: 100
- Stop loss: 102
- Take profit: 95
**And** current price drops to 94  
**When** `analyze_and_trade()` is called  
**Then**:
1. Position closed with `close_reason="take_profit"`
2. Exit order executed: BUY
3. LLM analysis skipped
4. Trade record updated with exit_signal="take_profit"

**Verification:**
```python
def test_scheduler_exits_short_on_take_profit():
    position = create_short_position(take_profit=95.0)
    mock_data(current_price=94.0)
    
    scheduler.analyze_and_trade()
    
    db.refresh(position)
    assert position.close_reason == "take_profit"
```

---

### AC-7: No Exit, Continue Analysis

**Given** an active position that doesn't meet exit conditions:
- LONG position
- Entry: 100, Stop: 98, Target: 105
- Current price: 101 (between stop and target)
**When** `analyze_and_trade()` is called  
**Then**:
1. Position tracking is updated (candles_since_entry incremented)
2. Exit check performed, returns (False, None)
3. LLM analysis proceeds normally (`strategy.generate_signal()` called)
4. No exit order executed
5. Position remains active

**Verification:**
```python
def test_scheduler_continues_analysis_without_exit():
    position = create_long_position(stop_loss=98.0, take_profit=105.0)
    mock_data(current_price=101.0)
    
    scheduler.analyze_and_trade()
    
    # Position still active
    db.refresh(position)
    assert position.is_active is True
    assert position.candles_since_entry == 1
    
    # LLM was called
    mock_strategy.generate_signal.assert_called_once()
```

---

### AC-8: Max Hold Candles Exit

**Given** an active position with `max_hold_candles=5`  
**And** `candles_since_entry=5` (limit reached)  
**When** `analyze_and_trade()` is called  
**Then**:
1. Position closed with `close_reason="max_hold"`
2. Exit order executed
3. Trade record updated with exit_signal="max_hold"

**Verification:**
```python
def test_scheduler_exits_on_max_hold_candles():
    position = create_long_position(max_hold_candles=5)
    # Simulate 5 candles passing
    for _ in range(5):
        position_monitor.update_candle_tracking(position, 100.0, 99.0)
    
    scheduler.analyze_and_trade()
    
    db.refresh(position)
    assert position.close_reason == "max_hold"
```

---

### AC-9: New Position Opened After Signal

**Given** no active position exists for "BTC-USD"  
**When** scheduler runs and LLM returns LONG signal  
**And** order is executed successfully  
**Then**:
1. New ActivePosition created via `PositionMonitor.open_position()`
2. Position has:
   - `symbol="BTC-USD"`
   - `side=OrderSide.BUY`
   - `entry_price` = current price
   - `stop_loss` from signal
   - `take_profit` from signal
   - `is_active=True`
   - `backtest_run_id=None`

**Verification:**
```python
def test_scheduler_opens_position_after_signal():
    mock_strategy.generate_signal.return_value = TradingSignal(
        action="LONG",
        confidence=0.85,
        stop_loss=98.0,
        take_profit=105.0,
    )
    mock_data(current_price=100.0)
    
    scheduler.analyze_and_trade()
    
    position = position_monitor.get_active_position("BTC-USD")
    assert position is not None
    assert position.side == OrderSide.BUY
    assert position.entry_price == 100.0
    assert position.is_active is True
```

---

### AC-10: Multiple Assets Handled Independently

**Given** positions exist for "BTC-USD" and "ETH-USD"  
**And** BTC position should exit (stop loss hit)  
**And** ETH position should continue (no exit condition)  
**When** `analyze_and_trade()` runs for both assets  
**Then**:
- BTC position closed, exit order executed, LLM skipped
- ETH position tracked, no exit, LLM analysis runs

**Verification:**
```python
def test_scheduler_handles_multiple_assets():
    btc_pos = create_long_position("BTC-USD", stop_loss=98.0)
    eth_pos = create_long_position("ETH-USD", stop_loss=1800.0)
    
    mock_data_multi({
        "BTC-USD": 97.0,  # Below stop loss
        "ETH-USD": 1850.0  # Above stop loss
    })
    
    scheduler.analyze_and_trade()
    
    # BTC exited
    db.refresh(btc_pos)
    assert btc_pos.is_active is False
    
    # ETH tracked but active
    db.refresh(eth_pos)
    assert eth_pos.is_active is True
```

---

### AC-11: Existing Scheduler Tests Pass

**Given** existing scheduler unit tests  
**When** running full test suite  
**Then** all pre-existing tests pass (backward compatibility)

**Verification:**
```bash
pytest tests/trading/test_scheduler.py -v
# Expected: All existing tests PASS
```

---

### AC-12: Integration Tests Pass

**Given** new integration tests added  
**When** running test suite  
**Then** at least 3 integration tests pass:
1. `test_scheduler_tracks_position_on_tick()`
2. `test_scheduler_exits_on_stop_loss()`
3. `test_scheduler_exits_on_take_profit()`

**Verification:**
```bash
pytest tests/integration/test_scheduler_position_monitor.py -v
# Expected: All new tests PASS
```

---

## Edge Cases

### Edge Case 1: Position Without Trade ID
**Given** position exists but `trade_id` is None  
**When** exit is executed  
**Then** no Trade update attempted, no error raised

---

### Edge Case 2: Exit Order Fails
**Given** position should exit  
**And** OrderManager.execute_decision() raises exception  
**When** exit is attempted  
**Then**:
- Position is already closed in DB
- Exception is logged
- ExecutionError is raised
- Transaction is rolled back

---

### Edge Case 3: Missing Market Data
**Given** data provider returns DataFrame with < 2 rows  
**When** calculating prev_close  
**Then** fallback: prev_close = current_price (no error)

---

### Edge Case 4: Concurrent Position Updates
**Given** multiple scheduler instances running (multi-worker)  
**When** both try to close same position  
**Then** only first one succeeds (DB constraint or locking)

---

## Performance Criteria

### Perf-1: Exit Check Latency
- Exit condition check adds < 5ms per asset
- Uses simple float comparisons (O(1))

### Perf-2: Position Tracking Latency
- `update_candle_tracking()` adds < 10ms per asset
- Single DB update (no complex queries)

### Perf-3: LLM Analysis Skip
- When exit triggers, saves 2-5 seconds (LLM call avoided)

---

## Boundary Conditions

### Boundary 1: Price Exactly at Stop Loss
**Given** current_price == stop_loss (exact equality)  
**Then** exit triggered (use `<=` for LONG, `>=` for SHORT)

### Boundary 2: Price Exactly at Take Profit
**Given** current_price == take_profit  
**Then** exit triggered

### Boundary 3: Zero Candles Since Entry
**Given** position just opened (candles_since_entry=0)  
**When** tracking updated  
**Then** candles_since_entry becomes 1

---

## Manual Test Procedure

1. **Setup:**
   - Start paper trading scheduler
   - Create test position manually in DB

2. **Stop Loss Test:**
   - Set stop_loss = current_price + 1
   - Wait for next tick
   - Verify position closed, order executed

3. **Take Profit Test:**
   - Set take_profit = current_price - 1
   - Wait for next tick
   - Verify position closed

4. **Tracking Test:**
   - Create position with prediction_horizon=3
   - Wait 3 ticks
   - Verify candles_direction has 3 entries

5. **Database Verification:**
   ```sql
   SELECT * FROM active_positions WHERE symbol='BTC-USD';
   SELECT * FROM trades WHERE symbol='BTC-USD' ORDER BY id DESC LIMIT 1;
   ```
   - Verify close_reason matches
   - Verify exit_signal in Trade record

---

## Definition of Done Checklist

- [ ] AC-1: PositionMonitor initialized ✓
- [ ] AC-2: Position tracking on each tick ✓
- [ ] AC-3: Stop loss exit (LONG) ✓
- [ ] AC-4: Stop loss exit (SHORT) ✓
- [ ] AC-5: Take profit exit (LONG) ✓
- [ ] AC-6: Take profit exit (SHORT) ✓
- [ ] AC-7: No exit, continue analysis ✓
- [ ] AC-8: Max hold candles exit ✓
- [ ] AC-9: New position opened ✓
- [ ] AC-10: Multiple assets handled ✓
- [ ] AC-11: Existing tests pass ✓
- [ ] AC-12: Integration tests pass ✓
- [ ] All edge cases handled
- [ ] Performance criteria met
- [ ] Manual test procedure executed
