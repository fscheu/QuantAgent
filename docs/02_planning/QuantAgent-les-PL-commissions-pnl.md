# Planning: Support commissions in P&L calculation

**Issue:** QuantAgent-les  
**Related:** [RQ](../01_requirements/QuantAgent-les-RQ-commissions-pnl.md) | [DS](../03_design/QuantAgent-les-DS-commissions-pnl.md) | [AC](../05_acceptance_tests/QuantAgent-les-AC-commissions-pnl.md)
**Complexity:** STANDARD

## Tasks

### Task 1: Add commission config wiring (~1h)
- Update `StrategyAssembler` resolved config to include commission settings (defaults to none/0).
- Pass resolved commission settings into `PaperBroker`.

### Task 2: Produce commission at execution (~1h)
- Update `PaperBroker.place_order()` to compute commission per selected model.
- Create a `Fill` record (or equivalent carrier) with computed commission and associate it to the order.

### Task 3: Use commission in trade persistence (~1h)
- Update `PortfolioManager.execute_trade()`:
  - source commission from order execution (e.g., latest fill commission)
  - set `Trade.commission`
  - compute `Trade.pnl` and `Trade.pnl_pct` as **net** values for closing trades

### Task 4: Regression check / validation (~0.5h)
- Run a representative backtest and confirm metrics change when commission is enabled and remain unchanged when disabled.

## Dependencies
- Depends on QuantAgent-r78 being present (gross P&L calculation on close).

## Risks
- SQLAlchemy session/cascade behavior when attaching `Fill` objects to an already-flushed `Order`.
- Decimal vs float precision drift.

## Rollout
- Default commission config is none (0) to avoid changing existing backtests.
- Enable commission in a dedicated profile/override to validate behavior.

## Validation commands
```bash
pytest -q
python examples/run_backtest.py
```
