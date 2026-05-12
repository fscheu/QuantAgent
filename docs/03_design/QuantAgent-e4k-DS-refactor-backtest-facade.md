# QuantAgent-e4k — Design: Refactor Backtest to depend only on OrderManager

**Issue:** QuantAgent-e4k  
**Pattern:** Facade / Law of Demeter  
**Scope:** `backtesting/`, `trading/`

---

## Motivation

`Backtest` currently violates the [Law of Demeter](https://en.wikipedia.org/wiki/Law_of_Demeter) by reaching through `OrderManager` to call sub-components directly. `OrderManager` already acts as a façade that encapsulates `PositionSizer`, `RiskManager`, and `PaperBroker`. The `Backtest` class should not need to know about those sub-components at all.

Additionally, `Backtest` calls `self.order_manager.close_trade(...)` which is a non-existent method, creating a latent `AttributeError` on every trade close.

---

## Before / After Dependency Graph

```
BEFORE:
                    ┌─────────────┐
                    │   Backtest  │
                    └──┬────┬──┬──┘
             ┌─────────┘    │  └──────────┐
             ▼             ▼              ▼
      PositionSizer   RiskManager      PaperBroker
             ▲             ▲              ▲
             └─────────────┼──────────────┘
                           │
                    ┌──────┴──────┐
                    │ OrderManager│  (also held by Backtest)
                    └─────────────┘

AFTER:
                    ┌─────────────┐
                    │   Backtest  │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │ OrderManager│
                    └──┬────┬──┬──┘
             ┌─────────┘    │  └──────────┐
             ▼             ▼              ▼
      PositionSizer   RiskManager      PaperBroker
```

---

## Design Decisions

### DD-1: Thin delegation methods on OrderManager

`reset_daily_tracker()` and `close_trade()` are thin wrappers. They do not duplicate logic — they delegate to the already-existing internal components. This keeps `OrderManager` as a true facade.

### DD-2: `close_trade` reuses the existing execution pipeline

Rather than implementing a new code path, `close_trade` follows the exact same flow as a normal opposing order:
1. Validate via `RiskManager`
2. Create `Order`
3. Fill via `PaperBroker`
4. Update via `PortfolioManager`
5. Update risk tracker

This ensures consistent P&L calculation, commission handling, and DB persistence.

### DD-3: `Backtest` still receives `components` from StrategyAssembler

`StrategyAssembler.build_components()` returns `TradingComponents` which still exposes all sub-components. `Backtest.__init__` simply stops storing the ones it doesn't need. This avoids changing the assembler or the dataclass.

---

## `OrderManager.close_trade` — Detailed Design

**Signature:**
```python
def close_trade(
    self,
    trade_id: int,
    current_price: float,
    environment=None,
) -> Optional[Order]:
```

**Flow:**
1. `db.query(Trade).filter(Trade.id == trade_id).first()` → if None, log warning, return None
2. Derive `close_side`: SELL if `trade.side == BUY`, else BUY
3. `risk_manager.validate_trade(symbol, close_side, qty, current_price)` → if invalid, log warning, return None
4. Create `Order(symbol, close_side, qty, current_price, MARKET, environment)`
5. `db.add(order); db.flush()` → if fails, rollback, return None
6. `broker.place_order(order)` → if fails, log error, return None
7. `portfolio.execute_trade(filled_order, filled_order.average_fill_price)` → if fails, log error, return None
8. `risk_manager.on_trade_executed(close_trade_record)`
9. `db.commit()` → if fails, rollback, return None
10. Return `filled_order`

**Import addition needed:**
```python
# In order_manager.py imports:
from quantagent.models import Order, OrderSide, OrderType, Signal, Trade, TradeSignal
```

---

## Backtest Changes — Design

Only three lines removed from `__init__`:
```python
# These three lines are deleted:
self.position_sizer = components.position_sizer
self.risk_manager = components.risk_manager
self.broker = components.broker
```

`self.order_manager` and `self.portfolio` remain (portfolio is used for equity tracking, not trading execution).

`reset_daily_tracker` call sites:
- `run()` line 256: `self.risk_manager.reset_daily_tracker()` → `self.order_manager.reset_daily_tracker()`
- `run_replay()` line 373: `self.risk_manager.reset_daily_tracker()` → `self.order_manager.reset_daily_tracker()`
