# SHORT Positions Implementation

**Date:** 2026-01-02
**Component:** `quantagent/portfolio/manager.py` (PortfolioManager)
**Status:** Implemented

## Overview

This document describes the implementation of SHORT position support in the PortfolioManager class. Previously, the system only supported LONG positions (buy to open, sell to close). Now it supports both LONG and SHORT positions.

## Conceptual Model

### LONG Positions (Original)
- **BUY** to open position
- **SELL** to close position
- **P&L**: Profit when price increases
- **Formula**: `pnl = qty × (current_price - entry_price)`

### SHORT Positions (New)
- **SELL** to open position (sell without owning)
- **BUY** to close position (buy to cover)
- **P&L**: Profit when price decreases
- **Formula**: `pnl = qty × (entry_price - current_price)`

## Position Representation

Positions use **signed quantities** to distinguish direction:
- **Positive qty**: LONG position (e.g., `qty: 100.0`)
- **Negative qty**: SHORT position (e.g., `qty: -100.0`)

```python
self.positions[symbol] = {
    "qty": 100.0,           # LONG: +100 shares
    "avg_cost": 50.25,      # Average entry price
    "current_price": 52.00, # Current market price
    "pnl": 175.0,           # Unrealized P&L
    "pnl_pct": 3.48         # P&L percentage
}

self.positions[symbol] = {
    "qty": -100.0,          # SHORT: -100 shares
    "avg_cost": 50.25,      # Average entry price (sold at)
    "current_price": 48.00, # Current market price
    "pnl": 225.0,           # Unrealized P&L (profit, price dropped)
    "pnl_pct": -4.48        # P&L percentage (inverted)
}
```

## Problem Solved

### Original Code (Line 77-81)
```python
if order.side == OrderSide.SELL:
    if symbol in self.positions:
        entry_price_for_sell = self.positions[symbol]["avg_cost"]
    else:
        raise ValueError(f"No position in {symbol} to sell")  # ← Blocked SHORT
```

This assumed SELL always closes an existing LONG position, preventing SHORT position opening.

## Implementation Changes

### 1. `execute_trade()` - Multi-scenario Handling

**Lines 75-97** - Determines if order opens/closes LONG/SHORT:

```python
# Determine entry price and position action
entry_price_for_sell = None
if order.side == OrderSide.SELL:
    if symbol in self.positions and self.positions[symbol]["qty"] > 0:
        # Closing LONG position
        entry_price_for_sell = self.positions[symbol]["avg_cost"]
    elif symbol not in self.positions or self.positions[symbol]["qty"] == 0:
        # Opening new SHORT position
        entry_price_for_sell = None
    elif self.positions[symbol]["qty"] < 0:
        # Increasing existing SHORT position
        entry_price_for_sell = self.positions[symbol]["avg_cost"]
```

**Lines 102-118** - Cash management for all scenarios:

```python
# Update cash based on action
if order.side == OrderSide.BUY:
    if symbol in self.positions and self.positions[symbol]["qty"] < 0:
        # Closing SHORT: pay for buy-back
        self.cash -= trade_value
    else:
        # Opening/increasing LONG: pay for shares
        self.cash -= trade_value
else:  # SELL
    if symbol in self.positions and self.positions[symbol]["qty"] > 0:
        # Closing LONG: receive cash
        self.cash += trade_value
    else:
        # Opening/increasing SHORT: receive cash from sale
        self.cash += trade_value
```

### 2. `_execute_sell()` - Open SHORT or Close LONG

**Lines 157-189** - Handles both scenarios:

```python
def _execute_sell(self, symbol: str, qty: float, price: float) -> None:
    """Update position for SELL order (close LONG or open SHORT)."""
    if symbol not in self.positions:
        # Open new SHORT position (negative qty)
        self.positions[symbol] = {
            "qty": -qty,
            "avg_cost": price,
            "current_price": price,
            "pnl": 0.0,
            "pnl_pct": 0.0,
        }
    else:
        pos = self.positions[symbol]
        if pos["qty"] > 0:
            # Close LONG position
            if pos["qty"] < qty:
                raise ValueError(f"Insufficient qty in {symbol}...")
            pos["qty"] -= qty
        else:
            # Increase SHORT position (more negative)
            total_qty = pos["qty"] - qty
            pos["avg_cost"] = (
                abs(pos["qty"]) * pos["avg_cost"] + qty * price
            ) / abs(total_qty)
            pos["qty"] = total_qty

        pos["current_price"] = price
        if pos["qty"] == 0:
            pos["avg_cost"] = 0.0

    self._update_position_pnl(symbol)
```

### 3. `_execute_buy()` - Open LONG or Close SHORT

**Lines 135-155** - Handles both scenarios:

```python
def _execute_buy(self, symbol: str, qty: float, price: float) -> None:
    """Update position for BUY order (open LONG or close SHORT)."""
    if symbol not in self.positions:
        # Open new LONG position
        self.positions[symbol] = {
            "qty": qty,
            "avg_cost": price,
            "current_price": price,
            "pnl": 0.0,
            "pnl_pct": 0.0,
        }
    else:
        pos = self.positions[symbol]
        if pos["qty"] < 0:
            # Close SHORT position
            if abs(pos["qty"]) < qty:
                raise ValueError(f"Trying to buy {qty}...")
            pos["qty"] += qty  # Reduce negative
        else:
            # Increase LONG position
            total_qty = pos["qty"] + qty
            pos["avg_cost"] = (pos["qty"] * pos["avg_cost"] + qty * price) / total_qty
            pos["qty"] = total_qty

        pos["current_price"] = price
        if pos["qty"] == 0:
            pos["avg_cost"] = 0.0

    self._update_position_pnl(symbol)
```

### 4. `_update_position_pnl()` - Inverse P&L for SHORT

**Lines 191-212** - Calculates P&L based on position direction:

```python
def _update_position_pnl(self, symbol: str) -> None:
    """Calculate unrealized P&L (works for LONG and SHORT)."""
    if symbol not in self.positions:
        return

    pos = self.positions[symbol]
    if pos["qty"] == 0:
        pos["pnl"] = 0.0
        pos["pnl_pct"] = 0.0
    else:
        if pos["qty"] > 0:
            # LONG: profit when price increases
            pos["pnl"] = pos["qty"] * (pos["current_price"] - pos["avg_cost"])
            pos["pnl_pct"] = ((pos["current_price"] - pos["avg_cost"]) / pos["avg_cost"]) * 100
        else:
            # SHORT: profit when price decreases (inverse P&L)
            pos["pnl"] = abs(pos["qty"]) * (pos["avg_cost"] - pos["current_price"])
            pos["pnl_pct"] = ((pos["avg_cost"] - pos["current_price"]) / pos["avg_cost"]) * 100
```

### 5. `get_total_value()` - Portfolio Value with SHORT

**Lines 214-230** - Calculates total value considering SHORT positions:

```python
def get_total_value(self) -> float:
    """Calculate total portfolio value (cash + positions).

    For LONG: value = qty × current_price
    For SHORT: value = qty × (2 × avg_cost - current_price)
              = initial_sale_proceeds - current_liability
    """
    position_value = 0.0
    for pos in self.positions.values():
        if pos["qty"] > 0:
            # LONG: value = shares × price
            position_value += pos["qty"] * pos["current_price"]
        else:
            # SHORT: value = initial proceeds - current buyback cost
            position_value += abs(pos["qty"]) * (2 * pos["avg_cost"] - pos["current_price"])

    return self.cash + position_value
```

### 6. `_persist_positions()` - Save Correct Side

**Line 349** - Determines side from quantity sign:

```python
side=OrderSide.BUY if pos_data["qty"] >= 0 else OrderSide.SELL,
```

## Examples

### Opening SHORT Position

```python
# SELL 100 shares at $50 (open SHORT)
order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=100)
trade = portfolio.execute_trade(order, fill_price=50.00)

# Position state:
# positions["AAPL"] = {
#     "qty": -100.0,
#     "avg_cost": 50.00,
#     "current_price": 50.00,
#     "pnl": 0.0
# }
# cash += 5000.00 (received from sale)
```

### Closing SHORT Position (Profit)

```python
# Price drops to $45, BUY 100 to close SHORT
order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
trade = portfolio.execute_trade(order, fill_price=45.00)

# Position closed:
# positions["AAPL"]["qty"] = 0.0
# cash -= 4500.00 (paid to buy back)
# Realized P&L = $500 profit (sold at $50, bought at $45)
```

### Closing SHORT Position (Loss)

```python
# Price rises to $55, BUY 100 to close SHORT
order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=100)
trade = portfolio.execute_trade(order, fill_price=55.00)

# Position closed:
# positions["AAPL"]["qty"] = 0.0
# cash -= 5500.00 (paid to buy back)
# Realized P&L = -$500 loss (sold at $50, bought at $55)
```

## Testing Considerations

### Unit Tests Required
1. **Open SHORT** - SELL without existing position
2. **Close SHORT** - BUY to cover SHORT position
3. **Increase SHORT** - SELL more on existing SHORT
4. **Partial Close SHORT** - BUY less than SHORT qty
5. **P&L Calculation** - Verify inverse P&L for SHORT
6. **Portfolio Value** - Verify total value with mixed LONG/SHORT
7. **Cash Flow** - Verify cash changes for all scenarios
8. **Database Persistence** - Verify correct `side` saved

### Edge Cases
- Opening SHORT when LONG position exists (should error)
- Closing SHORT with qty > position (should error)
- Mixed LONG/SHORT positions in same portfolio
- Transitioning from LONG to SHORT (close LONG, then open SHORT)

### 7. Trade Record Creation - Entry/Exit Price Logic

**Lines 104-140** - Correctly assigns entry/exit prices based on action type:

```python
# Determine action type from position state BEFORE trade
is_opening = position_qty_before == 0
is_closing_long = position_qty_before > 0 and order.side == OrderSide.SELL
is_closing_short = position_qty_before < 0 and order.side == OrderSide.BUY

if is_opening:
    # Opening new position (LONG or SHORT)
    entry_price = fill_price
    exit_price = None
    opened_at = now
    closed_at = None
elif is_closing_long or is_closing_short:
    # Closing existing position
    entry_price = avg_cost (from position)
    exit_price = fill_price
    closed_at = now
else:
    # Increasing existing position
    entry_price = fill_price
    exit_price = None
    opened_at = now
```

This fixes the original bug where opening a SHORT position (SELL with no position) would set `entry_price=None`, violating the NOT NULL constraint.

## Related Files

- **Implementation**: `quantagent/portfolio/manager.py`
- **Models**: `quantagent/models.py` (Position, OrderSide, Trade)
- **Risk Management**: `quantagent/risk/manager.py` (may need margin validation)
- **Tests**: `tests/test_portfolio_manager.py` (to be created/updated)

## Migration Notes

No database migration required. The `Position.side` field already exists and uses the `OrderSide` enum. The implementation now correctly sets `side=SELL` for SHORT positions.

## Future Enhancements

1. **Margin Requirements**: Implement margin validation for SHORT positions in RiskManager
2. **Borrowing Costs**: Track borrowing fees for SHORT positions
3. **Short Interest**: Monitor short interest and availability
4. **Stop Loss**: Enhanced stop-loss logic for SHORT positions
5. **Reporting**: Separate reporting for LONG vs SHORT performance

## References

- Original issue: Line 81 in `manager.py` raised `ValueError` for SELL without position
- TODO comment: Line 308 "Handle short positions" - now implemented
- Related: `quantagent/models.py` Position model with `side` field
