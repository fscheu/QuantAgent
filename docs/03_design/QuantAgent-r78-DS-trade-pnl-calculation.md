# Design: Trade P&L Calculation

**Issue:** QuantAgent-r78
**Related:** [RQ](../01_requirements/QuantAgent-r78-RQ-trade-pnl-calculation.md)

---

## Change Location

**File:** `quantagent/portfolio/manager.py`
**Method:** `execute_trade()`
**Lines:** ~116-140 (after position type determination, before Trade instantiation)

---

## Current Flow

```
1. Determine position_qty_before
2. Execute buy/sell (_execute_buy/_execute_sell)
3. Determine is_opening, is_closing_long, is_closing_short
4. Set entry_price, exit_price based on action type
5. Create Trade object (WITHOUT pnl/pnl_pct)  <-- BUG
6. Persist to database
```

---

## Modified Flow

```
1-4. [unchanged]
5. IF is_closing_long OR is_closing_short:
      Calculate pnl and pnl_pct
6. Create Trade object (WITH pnl/pnl_pct for closing trades)
7. Persist to database
```

---

## Implementation Contract

### Insertion Point
After line 127 (the `else` branch for increasing positions), before Trade instantiation (line 129).

### Signature (pseudo)
```python
pnl: Decimal | None = None
pnl_pct: float | None = None

if is_closing_long or is_closing_short:
    # Calculate here using entry_price, exit_price, fill_qty
    # Set pnl (Decimal) and pnl_pct (float)
```

### Trade Constructor Change
Pass `pnl=pnl` and `pnl_pct=pnl_pct` to Trade constructor.

---

## Data Types

| Field | Type | Notes |
|-------|------|-------|
| `entry_price` | `Decimal` | Already available as local var |
| `exit_price` | `Decimal` | Already available as local var |
| `fill_qty` | `float` | From parameter |
| `pnl` | `Decimal` | Model expects Numeric(18,8) |
| `pnl_pct` | `float` | Model expects Float |

---

## Edge Cases

| Case | Handling |
|------|----------|
| `entry_price` is 0 or None | Set pnl=None, log warning |
| `exit_price` is None | Not possible for closing trades (already validated) |
| Opening/increasing position | pnl remains None (no change) |

---

## No Changes Required

- `Trade` model (already has nullable `pnl` and `pnl_pct` fields)
- `backtest.py` `_calculate_metrics()` (already handles `t.pnl` correctly)
- Database schema (no migration needed)
