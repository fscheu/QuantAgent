# QuantAgent-l8r — Requirements: Trade P&L CI regression

**Issue ID:** QuantAgent-l8r  
**Related:** `QuantAgent-82t`, `QuantAgent-r78`

## Objective
Restore the CI gate by fixing the regression in `tests/test_r78_trade_pnl_calculation.py` without changing the production P&L formulas.

## Scope in
- Make `tests/test_r78_trade_pnl_calculation.py` deterministic under the CI PostgreSQL database.
- Assert against the trade created by the current test case instead of unrelated historical rows.
- Preserve the existing P&L contract from `QuantAgent-r78` for the scenarios exercised by this file.

## Scope out
- Changes to `quantagent/portfolio/manager.py` unless the tests prove production logic is actually wrong.
- Workflow/CI YAML changes.
- Unrelated `QuantAgent-82t` blockers.

## Constraints
- Keep the diff minimal and test-focused.
- Reuse existing repository test fixtures/patterns when possible.
- The resulting test file must pass both standalone and inside the CI gate command.

## Done
- `tests/test_r78_trade_pnl_calculation.py` passes in the shared CI venv.
- The exact `QuantAgent-82t` gate no longer fails because of this module.
- The fix is explained in issue-specific docs/artifacts.