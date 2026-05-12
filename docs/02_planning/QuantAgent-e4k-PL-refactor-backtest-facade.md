# QuantAgent-e4k — Planning: Refactor Backtest to depend only on OrderManager

**Issue:** QuantAgent-e4k  
**Phase:** Planner  
**Run ID:** 20260512T024029Z-QuantAgent-e4k-planner

---

## Scope Summary

Two files change:

| File | Change type |
|---|---|
| `quantagent/backtesting/backtest.py` | Remove 3 attribute assignments; replace 2 method calls |
| `quantagent/trading/order_manager.py` | Add 2 new methods |

One test file should be added or extended:

| File | Change type |
|---|---|
| `tests/test_order_manager_facade.py` (new) | Unit tests for the two new methods |

---

## Precise Changeset

### 1. `quantagent/backtesting/backtest.py`

**Remove** from `__init__` (lines 167–169):
```python
self.position_sizer = components.position_sizer   # DELETE
self.risk_manager = components.risk_manager        # DELETE
self.broker = components.broker                    # DELETE
```

**Replace** in `run()` (line 256) and `run_replay()` (line 373):
```python
# Before:
self.risk_manager.reset_daily_tracker()
# After:
self.order_manager.reset_daily_tracker()
```

No other changes to `backtest.py`. The two `self.order_manager.close_trade(...)` calls already exist (lines 423 and 653) and will work once the method is added to `OrderManager`.

### 2. `quantagent/trading/order_manager.py`

**Add** `reset_daily_tracker()` method:
```python
def reset_daily_tracker(self) -> None:
    """Delegate daily P&L reset to the internal risk manager."""
    self.risk_manager.reset_daily_tracker()
```

**Add** `close_trade()` method:
```python
def close_trade(
    self,
    trade_id: int,
    current_price: float,
    environment=None,
) -> Optional[Order]:
    """Close an open trade by executing an opposing market order.

    Looks up the Trade by ID, creates a closing order (opposite side,
    same quantity), fills it through the broker, and updates portfolio
    state and risk tracker.

    Returns the filled Order, or None if trade not found or rejected.
    """
    trade = self.db.query(Trade).filter(Trade.id == trade_id).first()
    if trade is None:
        logger.warning(f"close_trade: Trade {trade_id} not found")
        return None

    symbol = trade.symbol
    qty = float(trade.quantity)
    close_side = OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY

    is_valid, reason = self.risk_manager.validate_trade(symbol, close_side, qty, current_price)
    if not is_valid:
        logger.warning(f"{symbol}: Close trade {trade_id} rejected — {reason}")
        return None

    close_order = Order(
        symbol=symbol,
        side=close_side,
        quantity=qty,
        price=current_price,
        order_type=OrderType.MARKET,
        environment=environment,
    )

    try:
        self.db.add(close_order)
        self.db.flush()
    except Exception as e:
        logger.error(f"{symbol}: Failed to persist close order — {e}")
        self.db.rollback()
        return None

    try:
        filled_order = self.broker.place_order(close_order)
    except Exception as e:
        logger.error(f"{symbol}: Broker failed on close_trade — {e}")
        return None

    try:
        close_trade_record = self.portfolio.execute_trade(
            filled_order, filled_order.average_fill_price
        )
    except Exception as e:
        logger.error(f"{symbol}: Portfolio update failed on close_trade — {e}")
        return None

    self.risk_manager.on_trade_executed(close_trade_record)

    try:
        self.db.commit()
    except Exception as e:
        logger.error(f"{symbol}: DB commit failed on close_trade — {e}")
        self.db.rollback()
        return None

    logger.info(
        f"{symbol}: Closed trade {trade_id} @ ${current_price:.2f} "
        f"({close_side.name} {qty:.6f})"
    )
    return filled_order
```

The `Trade` import is already available in `order_manager.py`'s model imports (add it if not present).

---

## Import Check

In `order_manager.py`, the current imports are:
```python
from quantagent.models import Order, OrderSide, OrderType, Signal, TradeSignal
```
`Trade` needs to be added: `from quantagent.models import Order, OrderSide, OrderType, Signal, Trade, TradeSignal`

---

## Tests

New file `tests/test_order_manager_facade.py` (or extend an existing order_manager test):

1. `test_reset_daily_tracker_delegates` — mock `risk_manager`; call `order_manager.reset_daily_tracker()`; assert `risk_manager.reset_daily_tracker` was called once.
2. `test_close_trade_success` — mock DB (trade found), mock broker (fills), mock portfolio, mock risk_manager; assert filled order returned and `portfolio.execute_trade` called.
3. `test_close_trade_not_found` — mock DB returns None; assert method returns None.
4. `test_close_trade_rejected_by_risk` — mock risk_manager returns `(False, "reason")`; assert method returns None and broker not called.

---

## Risk / Notes

- `close_trade` already has a local variable named `close_trade` inside `_execute_reversal` in the existing file — the new method is a method on `self`, so there is no conflict.
- The `Trade` query uses `self.db`; this is the same session pattern used in all other `OrderManager` methods.
- No DB migration needed.

---

## Acceptance Criteria Reference

See `docs/05_acceptance_tests/QuantAgent-e4k-AC-refactor-backtest-facade.md`.
