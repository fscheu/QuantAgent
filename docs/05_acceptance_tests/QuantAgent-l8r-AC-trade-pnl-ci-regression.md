# QuantAgent-l8r — Acceptance Criteria: Trade P&L CI regression

**Related:** `docs/01_requirements/QuantAgent-l8r-RQ-trade-pnl-ci-regression.md`

## AC-1 — Current closing trade is selected
**Given** a test opens and closes a position inside one test case  
**When** it queries the resulting closed trade  
**Then** it validates the trade created by that test case, not an older row left in the shared CI database.

## AC-2 — Opening trade remains unclosed
**Given** a test only opens or increases a position  
**When** it queries the resulting trade rows  
**Then** it only inspects the rows created by that test case  
**And** those rows keep `pnl=None` and `exit_price=None` when appropriate.

## AC-3 — CI blocker removed
**Given** the exact `QuantAgent-82t` pytest gate command  
**When** it runs after this change  
**Then** `tests/test_r78_trade_pnl_calculation.py` contributes zero failures.

## Oracle
- `python -m pytest tests/test_r78_trade_pnl_calculation.py -v`
- `python -m pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"` no longer stops on this file.