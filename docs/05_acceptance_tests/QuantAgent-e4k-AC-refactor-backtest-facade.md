# QuantAgent-e4k — Acceptance Tests: Refactor Backtest to depend only on OrderManager

**Issue:** QuantAgent-e4k  
**Phase:** Acceptance / Verification

---

## AC-1: Backtest has no direct reference to sub-components

**Given** the `Backtest` class after the refactor  
**When** an instance is created with valid arguments  
**Then** the instance has no `position_sizer`, `risk_manager`, or `broker` attributes

**Verification:**
```python
bt = Backtest(start_date=..., end_date=..., assets=["BTC"])
assert not hasattr(bt, "position_sizer")
assert not hasattr(bt, "risk_manager")
assert not hasattr(bt, "broker")
assert hasattr(bt, "order_manager")
```

---

## AC-2: OrderManager exposes `reset_daily_tracker`

**Given** an `OrderManager` instance with a mocked `RiskManager`  
**When** `order_manager.reset_daily_tracker()` is called  
**Then** `risk_manager.reset_daily_tracker()` is called exactly once

**Verification (unit test):**
```python
risk_manager = Mock()
om = OrderManager(position_sizer=..., risk_manager=risk_manager, ...)
om.reset_daily_tracker()
risk_manager.reset_daily_tracker.assert_called_once()
```

---

## AC-3: OrderManager exposes `close_trade` — success path

**Given** an `OrderManager` with mocked DB, broker, portfolio, and risk_manager  
**And** the DB returns a `Trade` with `id=1, symbol="BTC", side=BUY, quantity=0.5`  
**When** `order_manager.close_trade(1, 50000.0, environment=Environment.BACKTEST)` is called  
**Then** a filled `Order` is returned  
**And** `broker.place_order` was called with a SELL order for 0.5 BTC  
**And** `portfolio.execute_trade` was called  
**And** `risk_manager.on_trade_executed` was called

---

## AC-4: `close_trade` returns None when trade not found

**Given** DB query returns None for the requested `trade_id`  
**When** `order_manager.close_trade(999, 50000.0)` is called  
**Then** the return value is `None`  
**And** no order is placed

---

## AC-5: `close_trade` returns None when risk manager rejects

**Given** a valid trade exists  
**And** `risk_manager.validate_trade` returns `(False, "circuit breaker")`  
**When** `order_manager.close_trade(1, 50000.0)` is called  
**Then** the return value is `None`  
**And** `broker.place_order` is NOT called

---

## AC-6: No regression in existing test suite

**Given** the refactored code  
**When** the full test suite is run (`pytest tests/ -x`)  
**Then** all previously passing tests continue to pass

**Key tests to verify:**
- `tests/test_trading_components.py` — RiskManager, PositionSizer, PortfolioManager
- `tests/test_order_manager_reversal.py` — existing OrderManager tests
- Any backtest integration tests

---

## AC-7: `run()` and `run_replay()` still call daily reset via OrderManager

**Given** a Backtest run with mocked OrderManager  
**When** the backtest processes a new day boundary  
**Then** `order_manager.reset_daily_tracker()` is called (not `risk_manager.reset_daily_tracker()`)
