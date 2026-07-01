# 2026-07-01 — SHORT 0.000000 residue bug in backtest executions

## Context

While validating post-deploy QA and running `scripts/run_three_strategy_backtests.py`, QuantAgent emitted repeated rejections like:

- `Position already open: SHORT 0.000000 shares. Cannot add to existing SHORT position (prevents over-concentration)`

At first glance this looked like a stale-position contamination issue across strategy runs. It was not.

The bug reproduced cleanly in the RSI backtest path and did not depend on QA database state.

## Symptom

During the three-strategy backtest run:
- RSI Mean Reversion would execute shorts, close them, and later start rejecting new shorts.
- The rejection message displayed `SHORT 0.000000 shares`.
- Triple Screen and 52-Week High were red herrings for the original bug; Triple Screen isolated produced zero trades and did not reproduce the symptom.

## Root cause

The root cause was float residue left in the in-memory portfolio position after closing a short.

Concrete observed case from traced RSI reproduction:
- trade quantity: `43.67551354`
- in-memory short position before close: `-43.67551354186529`
- after applying the close, portfolio quantity became:
  - `-1.865295473635342e-09`

That residual negative epsilon was:
- small enough to render as `0.000000` in logs,
- but still negative enough for `RiskManager` to interpret it as an open short position.

So the system was effectively saying:
- visually: zero shares,
- logically: still short.

Classic float dust. Unforced error, but there it was.

## Why the first fix was insufficient

An initial mitigation used epsilon `1e-9` for:
- portfolio normalization,
- defensive risk validation.

That was conceptually correct but numerically too strict.

The traced residual `-1.865295473635342e-09` is larger in magnitude than `1e-9`, so the normalization did not fire and the bug persisted in the three-strategy execution.

## Final fix applied

Two tolerances were kept, but widened to `1e-8`:

### 1. Portfolio normalization
File:
- `quantagent/portfolio/manager.py`

Behavior:
- after position updates in `_execute_buy()` and `_execute_sell()`, if `abs(qty) < 1e-8`:
  - set `qty = 0.0`
  - set `avg_cost = 0.0`

This removes zombie positions at the source of truth for trading state.

### 2. Defensive risk normalization
File:
- `quantagent/trading/risk_manager.py`

Behavior:
- before applying concentration checks, if `abs(existing_qty) < 1e-8`, treat it as `0.0`

This prevents a near-zero leftover from being classified as an active short/long even if some other path reintroduces minor float dust.

## Evidence used to confirm the fix

### Focused RSI traced reproduction
A dedicated traced reproduction showed:
- before close: non-zero negative short qty
- after close: `portfolio_qty = 0.0`
- repeated across subsequent RSI close cycles after widening epsilon

### Final three-strategy confirmation
Running `scripts/run_three_strategy_backtests.py` after the epsilon adjustment produced:
- no `SHORT 0.000000` log entries
- RSI still generated risk rejections later, but now for a legitimate reason:
  - `Daily loss limit exceeded`

That distinction matters:
- original bug = false rejection due to zombie short residue
- current rejection = intended risk control behavior

## Validation summary

Validated with:
- focused short close/reopen reproduction
- focused RSI minimal reproduction
- final three-strategy end-to-end rerun
- regression-oriented test subset:
  - `tests/test_order_manager_reversal.py`
  - `tests/test_backtest_end_to_end_timeframe_isolation.py`
  - `tests/test_backtest_run_isolation.py`
  - `tests/test_backtest_metrics_run_scoping.py`
  - `tests/test_stale_position_cleanup.py`

Result:
- original `SHORT 0.000000` bug resolved
- three-strategy execution no longer reproduces that symptom

## Non-goals / what this did not solve

This fix does NOT imply:
- RSI strategy is profitable
- Triple Screen is generating expected trades
- 52-Week High is fully validated in the current synthetic harness
- all backtest warnings are gone (`SAWarning` on subquery coercion still exists)

It solved one specific bug:
- false short-open rejections caused by float residue after short closes.

## Operational takeaway

When a trading engine stores portfolio quantities in floats, closing logic needs explicit normalization. Otherwise logs and business rules diverge:
- logs round to zero,
- rules still see a signed non-zero.

That is how you get a portfolio that is both flat and not flat at the same time. Schrödinger would be proud; operations, not so much.
