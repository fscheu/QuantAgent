# QuantAgent-l8r — Plan: Trade P&L CI regression

**Complexity:** Minimal

## Task 1 — Confirm root cause
- Re-run `tests/test_r78_trade_pnl_calculation.py` under the shared CI venv.
- Verify whether failures come from production formulas or from test isolation/query selection.

## Task 2 — Apply minimal test fix
- Update the test module to use isolated database state and/or row selection tied to the orders created by each test.
- Keep production code untouched unless root-cause verification proves otherwise.

## Task 3 — Revalidate gate impact
- Run the targeted file.
- Run the exact `QuantAgent-82t` gate command and confirm the file is no longer a blocker.
- If other blockers remain, leave them for follow-up tickets instead of widening scope.