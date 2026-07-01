# 2026-07-01 — Implementation note: eliminate SHORT 0.000000 residue in backtests

## Change summary

Implemented a narrow fix for false short-open rejections caused by float residue after closing short positions in backtests.

## Files changed

- `quantagent/portfolio/manager.py`
- `quantagent/trading/risk_manager.py`

## Implementation details

### `quantagent/portfolio/manager.py`
Added near-zero quantity normalization to portfolio state.

Behavior:
- maintain `_qty_epsilon = 1e-8`
- after `_execute_buy()` and `_execute_sell()` update a position:
  - if `abs(pos["qty"]) < _qty_epsilon`
    - set `pos["qty"] = 0.0`
    - set `pos["avg_cost"] = 0.0`

Why here:
- this is the canonical in-memory portfolio state used by trading/risk paths
- fixing state at the source is better than compensating downstream only

### `quantagent/trading/risk_manager.py`
Added defensive tolerance before position concentration checks.

Behavior:
- maintain `_position_epsilon = 1e-8`
- when reading existing portfolio qty:
  - if `abs(existing_qty) < _position_epsilon`
    - treat as `0.0`

Why here:
- protects against future tiny residues or alternate paths that might bypass direct normalization
- keeps risk classification aligned with intended flat position semantics

## Why epsilon is `1e-8` and not `1e-9`

Observed traced residual:
- `-1.865295473635342e-09`

With `1e-9`, that residual survived.
With `1e-8`, it collapses to flat as intended.

This was determined from traced RSI minimal reproduction, not guessed.

## Validation performed

### Automated tests
Executed focused regression subset successfully:

```bash
pytest -q \
  tests/test_order_manager_reversal.py \
  tests/test_backtest_end_to_end_timeframe_isolation.py \
  tests/test_backtest_run_isolation.py \
  tests/test_backtest_metrics_run_scoping.py \
  tests/test_stale_position_cleanup.py
```

Result:
- `19 passed`

### Runtime reproductions
Validated through:
- focused short close/reopen repro
- focused RSI traced repro
- final rerun of `scripts/run_three_strategy_backtests.py`

Confirmation criterion:
- no `SHORT 0.000000` rejections in final three-strategy run

## Remaining follow-ups outside this fix

Not addressed here:
- `SAWarning` in backtest metrics query (`Trade.id.in_(subquery)` coercion)
- Triple Screen generating zero trades in current synthetic harness
- 52-Week High processing zero candles in current synthetic harness
- broader strategy-level performance or tuning

## Keep / don’t keep

Keep:
- the code fix in portfolio/risk modules
- this implementation note

Do not keep as product conclusions:
- synthetic-run PnL as business signal
- temporary repro scripts/logs used only during investigation
