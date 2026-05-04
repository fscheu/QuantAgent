# Requirements: Support commissions in P&L calculation

**Issue:** QuantAgent-les  
**Type:** Feature  
**Priority:** P3  
**Status:** Open  
**Blocked by:** QuantAgent-r78

## Objective
Calculate **net** realized P&L for **closing trades** by deducting commission costs sourced from broker/order execution.

## Context
- `Trade.commission` exists (Numeric(18,8)) but is currently hardcoded to `0` in `quantagent/portfolio/manager.py`.
- Base gross P&L and `pnl_pct` calculations were introduced in **QuantAgent-r78** (commission explicitly out of scope there).

## Scope
### In scope
1. **Commission sourcing**
   - Commission must be provided by the execution path (broker/order execution), not hardcoded in portfolio.
   - Commission must be stored on the created `Trade` as `trade.commission`.

2. **Net P&L for closing trades**
   - For closing trades, compute net realized P&L:
     - `net_pnl = gross_pnl - commission`
   - Persist `trade.pnl` as **net** P&L (replace current gross value for closing trades).

3. **Net pnl_pct for closing trades**
   - For closing trades, compute `pnl_pct` using the net return after costs.

4. **Commission model support**
   - Support at least:
     - **Per-trade fixed fee** (absolute amount per execution)
     - **Percentage fee** (rate applied to notional)

### Out of scope
- Broker integrations beyond the existing paper/backtest broker path (no real exchange API integration).
- Persisting or reporting **separate** gross vs net P&L fields (single `Trade.pnl` field remains).
- Allocating both entry+exit commissions into a single “round-trip trade” object (system currently records trades per execution).

## Functional Requirements
### FR-1: Capture commission from execution
When an order is filled,
- **Given** execution returns a commission value for that fill/execution
- **When** the `Trade` record is created
- **Then** `trade.commission` equals the execution commission (Decimal, currency units)

### FR-2: Net P&L on close
When a trade closes an existing position,
- **Then** `trade.pnl` must reflect **net** P&L after commission cost.

### FR-3: Net pnl_pct on close
When a trade closes an existing position,
- **Then** `trade.pnl_pct` must reflect net return after commission cost.

### FR-4: Non-closing trades unchanged
When a trade opens a new position or increases an existing position,
- **Then** `trade.pnl` remains `None`
- **And** `trade.pnl_pct` remains `None`
- **And** `trade.commission` is still recorded (may be 0 depending on model)

## Constraints
- Use `Decimal` for commission and net P&L math to match DB model precision.
- Commission must never be negative.
- Default commission behavior must preserve existing results when no commission config is provided (i.e., default commission = 0).

## Definition of Done
- Commission is no longer hardcoded to 0 in `PortfolioManager.execute_trade()`.
- Closing trades store net `pnl` and net `pnl_pct` (after costs).
- System supports fixed-fee and percentage commission models in the paper/backtest execution path.
