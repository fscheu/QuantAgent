# Acceptance Criteria: Support commissions in P&L calculation

**Issue:** QuantAgent-les  
**Related:** [RQ](../01_requirements/QuantAgent-les-RQ-commissions-pnl.md) | [DS](../03_design/QuantAgent-les-DS-commissions-pnl.md)

## AC-1: Commission is persisted on trades
**Given** a filled order execution that reports a non-zero commission
**When** a `Trade` record is created from that execution
**Then** `trade.commission` equals that commission value

## AC-2: Net P&L is reduced by commission (LONG close)
**Given** a LONG position opened at $60,000
**And** the position is closed at $65,000 with quantity 0.1
**And** execution commission is $10.00
**When** the close trade is recorded
**Then** `trade.pnl` equals $490.00 (net)

## AC-3: Net P&L is reduced by commission (SHORT close)
**Given** a SHORT position opened at $65,000
**And** the position is closed at $60,000 with quantity 0.1
**And** execution commission is $10.00
**When** the close trade is recorded
**Then** `trade.pnl` equals $490.00 (net)

## AC-4: Net pnl_pct reflects costs
**Given** a closing trade with `entry_price`, `exit_price`, and filled `quantity`
**And** a non-zero `trade.commission`
**When** `trade.pnl_pct` is computed
**Then** `trade.pnl_pct` equals `(trade.pnl / (entry_price * quantity)) * 100` within reasonable rounding

## AC-5: Commission model = fixed fee
**Given** commission model is configured as fixed fee (e.g., $2.50 per fill)
**When** an order is filled
**Then** the resulting fill/trade commission equals $2.50

## AC-6: Commission model = percentage of notional
**Given** commission model is configured as percentage (e.g., 0.10% = 0.001)
**When** an order is filled with notional $10,000
**Then** the resulting fill/trade commission equals $10.00

## AC-7: Defaults preserve prior behavior
**Given** no commission model/config is provided (default)
**When** trades are executed
**Then** commission is 0
**And** net P&L equals prior gross P&L behavior
