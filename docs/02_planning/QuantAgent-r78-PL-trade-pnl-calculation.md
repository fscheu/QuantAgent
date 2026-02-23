# Planning: Trade P&L Calculation

**Issue:** QuantAgent-r78
**Related:** [RQ](../01_requirements/QuantAgent-r78-RQ-trade-pnl-calculation.md) | [DS](../03_design/QuantAgent-r78-DS-trade-pnl-calculation.md) | [AC](../05_acceptance_tests/QuantAgent-r78-AC-trade-pnl-calculation.md)
**Complexity:** MINIMAL

---

## Tasks

### Task 1: Implement P&L Calculation (~30 min)
**File:** `quantagent/portfolio/manager.py`
**Location:** `execute_trade()` method, lines 116-140

**Steps:**
1. Add P&L calculation logic after line 127 (after position type determination)
2. Initialize `pnl = None`, `pnl_pct = None`
3. If `is_closing_long`: calculate `pnl = (exit - entry) * qty`
4. If `is_closing_short`: calculate `pnl = (entry - exit) * qty`
5. Calculate `pnl_pct` for closing trades
6. Pass `pnl` and `pnl_pct` to Trade constructor
7. Add edge case guard for invalid entry_price

### Task 2: Write Unit Tests (~30 min)
**File:** `tests/portfolio/test_manager_pnl.py` (new)

**Test cases:**
- `test_pnl_calculation_long_profit`
- `test_pnl_calculation_long_loss`
- `test_pnl_calculation_short_profit`
- `test_pnl_calculation_short_loss`
- `test_pnl_none_for_opening_trade`

### Task 3: Verify Backtest (~15 min)
**Command:** `python examples/run_backtest.py`

**Verify:**
- Total P&L != $0.00
- Winning/Losing trades classified correctly
- Win rate and profit factor calculated

---

## Dependencies

- None (self-contained fix)

---

## Risks

| Risk | Mitigation |
|------|------------|
| Precision loss in float conversion | Use Decimal arithmetic, convert to float only for pnl_pct |
| Division by zero | Guard against entry_price <= 0 |

---

## Validation Commands

```bash
# Run unit tests
pytest tests/portfolio/test_manager_pnl.py -v

# Run existing portfolio tests (regression)
pytest tests/portfolio/ -v

# Run backtest verification
python examples/run_backtest.py
```

---

## Rollout

1. Implement fix on feature branch `feature/QuantAgent-r78`
2. Run all tests: `pytest tests/ -v`
3. Run backtest: `python examples/run_backtest.py`
4. PR review
5. Merge to main

---

## Estimated Effort

| Task | Time |
|------|------|
| Implementation | 30 min |
| Unit tests | 30 min |
| Verification | 15 min |
| **Total** | **~1.25 hours** |
