# Design: Support commissions in P&L calculation

**Issue:** QuantAgent-les  
**Related:** [RQ](../01_requirements/QuantAgent-les-RQ-commissions-pnl.md) | QuantAgent-r78

## Level of detail
**STANDARD** (new behavior + config surface + data flow changes).

## Affected components
- `quantagent/trading/paper_broker.py` (commission computation for simulated fills)
- `quantagent/trading/order_manager.py` (ensure commission/fill is persisted with the order)
- `quantagent/portfolio/manager.py` (use execution commission; compute net pnl/pnl_pct)
- `quantagent/strategy/assembler.py` (wire config for commission model into broker)

## Data model (no migrations)
Use existing tables/fields:
- `fills.commission` (per execution)
- `trades.commission` (commission attributed to the `Trade` record)

No new columns required.

## Execution → commission sourcing contract
### Contract
- The broker execution step must produce a commission value per fill.
- The order execution path must persist that commission in `Fill.commission` and/or make it available to portfolio trade creation.

### Minimal implementation approach
- In `PaperBroker.place_order(order)` create a `Fill` object and attach it to `order.fills`:
  - `quantity`, `price` (actual fill price), `commission` (computed per selected commission model)
  - Rationale: `Fill` already exists and is the natural execution-level carrier of commissions.

## Commission models
### Supported models
1. **Fixed per-trade**
   - `commission = fixed_fee`
2. **Percentage of notional**
   - `commission = notional * rate`
   - `notional = abs(fill_qty) * fill_price`

### Configuration shape (proposed)
Keep config minimal and consistent with existing `slippage_pct` wiring.

- Add a `commission` section in portfolio profile / overrides, passed into `PaperBroker`:
  - `commission_model: "none" | "fixed" | "pct"`
  - `commission_fixed: float` (currency)
  - `commission_pct: float` (e.g., `0.001` for 0.1%)

Defaults: `commission_model="none"` (all commissions = 0).

## Net P&L / pnl_pct computation
### Definitions
- `gross_pnl` is the pre-cost P&L currently computed in `PortfolioManager.execute_trade()` (QuantAgent-r78).
- `commission` is the execution commission for the **current filled order**.

### Closing trades
For `is_closing_long` or `is_closing_short`:
- `net_pnl = gross_pnl - commission`
- Persist:
  - `Trade.commission = commission`
  - `Trade.pnl = net_pnl` (net)

### pnl_pct base
Compute net percent return using the same capital base used in QuantAgent-r78 (entry notional):
- `entry_notional = entry_price * qty`
- `pnl_pct = (net_pnl / entry_notional) * 100`

Notes:
- `qty` is the filled quantity used for P&L.
- If `entry_notional <= 0`, keep `pnl` and `pnl_pct` as `None` (consistent with existing edge guard).

## Trade ↔ fill commission attribution
- For now, attribute the commission from the current fill/execution to the `Trade` created for that execution.
- This does **not** attempt to “roll up” entry+exit commissions into a single round-trip trade (out of scope).

## Risks / gotchas
- Ensure commission uses `Decimal` end-to-end (avoid float drift).
- Ensure notional uses `abs(qty)` for percentage commission.
- Confirm the order/SQLAlchemy session lifecycle: attaching `Fill` to `order.fills` must result in persistence with the order/trade transaction.

## Open questions
1. Should net realized P&L at close include **both** entry and exit commissions (round-trip), or only the current closing execution commission? (Current proposal: only current execution.)
2. Should commissions also adjust `PortfolioManager.cash` (cash flows) or only realized P&L reporting? (Current proposal: P&L reporting only; cash behavior remains unchanged unless explicitly required.)
