# Planning: Backtest depends only on OrderManager (facade)

**Issue ID:** QuantAgent-e4k
**Status:** Ready for Implementation
**Level:** STANDARD

---

## Objective

Refactor `Backtest` to interact with trading execution components exclusively through `OrderManager`, removing direct references to `PositionSizer`, `RiskManager`, and `PaperBroker`.

## Task Breakdown

### Task 1: Confirm current Backtest execution touchpoints (0.5h)
- Identify all current usages in `quantagent/backtesting/backtest.py` of:
  - `RiskManager` (e.g., daily reset)
  - `PortfolioManager` (equity curve)
  - `PaperBroker` / `PositionSizer` (should not be used directly)
- List the exact methods/fields that need facade coverage.

### Task 2: Make OrderManager facade complete for Backtest needs (0.5–1h)
- If missing, add minimal methods to `quantagent/trading/order_manager.py`:
  - `reset_daily_tracker()`
  - `get_portfolio_snapshot()` (or equivalent accessors)
- Keep methods thin delegations (no logic changes).

### Task 3: Refactor Backtest to depend only on OrderManager (1–2h)
- Update `Backtest.__init__`:
  - stop storing `position_sizer`, `risk_manager`, `broker` (and preferably `portfolio`)
  - keep `order_manager` as the only execution facade reference
- Update run-loop daily reset callsite to use `order_manager`
- Update equity curve recording to use `order_manager` facade method(s)
- Ensure imports reflect the new dependency graph.

### Task 4: Quick regression validation (0.5–1h)
- Run focused tests:
  - `pytest tests/test_backtest.py -v`
  - `pytest tests/test_backtest_integration.py -v`
- Optional smoke run:
  - `python examples/run_backtest.py` (or the smallest existing backtest example)

## Risks
- Backtest may be relying on additional execution internals not obvious at first glance; mitigate by `rg` search for `risk_manager|broker|position_sizer|portfolio` usages.

## Rollout
- No migration/rollout steps (refactor-only)
- Merge when tests pass and acceptance criteria are met

## Related Docs
- `docs/03_design/backtesting_engine.md`
- `docs/03_design/strategy_assembler_architecture.md`
- `docs/03_design/POSITION_MANAGEMENT_STRATEGIES.md`
