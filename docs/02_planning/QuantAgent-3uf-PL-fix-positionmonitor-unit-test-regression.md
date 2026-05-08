# QuantAgent-3uf — Plan: Fix PositionMonitor Unit-Test Regressions

**Issue ID:** QuantAgent-3uf
**Feature branch:** `feature/QuantAgent-3uf-fix-positionmonitor-unit-test-regression`

---

## References

- Requirements: `docs/01_requirements/QuantAgent-3uf-RQ-fix-positionmonitor-unit-test-regression.md`
- Design: `docs/03_design/QuantAgent-3uf-DS-fix-positionmonitor-unit-test-regression.md`
- Acceptance tests: `docs/05_acceptance_tests/QuantAgent-3uf-AC-fix-positionmonitor-unit-test-regression.md`

---

## Task List

### Task 1: Remove local `db_session` from `test_position_monitor.py`
**Estimate:** 5 min  
**File:** `tests/test_position_monitor.py`

Remove the local `db_session` fixture and its unused imports (`os`, `create_engine`,
`sessionmaker` if not used elsewhere). The `position_monitor` fixture stays.

Also remove the module-level docstring note about PostgreSQL requirement (the tests
will use SQLite via conftest.py).

**Validate:**
```bash
pytest tests/test_position_monitor.py -v --tb=short -m "not integration and not slow"
# test_only_one_active_position_per_symbol should now PASS
```

---

### Task 2: Remove local `db_session` from `test_position_monitor_constraints.py`
**Estimate:** 5 min  
**File:** `tests/test_position_monitor_constraints.py`

Same removal as Task 1 for the constraints file.

**Validate:**
```bash
pytest tests/test_position_monitor_constraints.py::test_closed_position_not_returned_by_get_active -v
# Should PASS
```

---

### Task 3: Fix FK in `test_position_with_all_optional_fields`
**Estimate:** 10 min  
**File:** `tests/test_position_monitor_constraints.py`

Replace hardcoded `trade_id=123, signal_id=456` with actual DB records created in
the test. Add imports for `Trade`, `Signal`, `TradeSignal`. Use `db_session.flush()`
to populate IDs before calling `open_position`.

See design doc for the exact code pattern.

**Validate:**
```bash
pytest tests/test_position_monitor_constraints.py::test_position_with_all_optional_fields -v
# Should PASS (no FK violation)
```

---

### Task 4: Add `ORDER BY` to `get_active_position`
**Estimate:** 2 min  
**File:** `quantagent/trading/position_monitor.py`

In `get_active_position`, change:
```python
return query.first()
```
to:
```python
return query.order_by(ActivePosition.id).first()
```

**Validate:**
```bash
pytest tests/test_position_monitor_constraints.py::test_get_active_position_returns_most_recent_if_multiple -v
# Should PASS
```

---

### Task 5: Run full gate command
**Estimate:** 2 min  

```bash
DATABASE_URL=postgresql://test:test@localhost:5432/quantagent_test \
  /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python \
  -m pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"
```

All 4 previously failing tests must be PASSED. No new failures.

---

## Execution Order

Tasks 1, 2, and 4 are independent and can be executed in any order.
Task 3 depends on Task 2 (must be in same file edit).
Task 5 must be last.

Recommended: 1 → 2+3 (same file) → 4 → 5

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| SQLite incompatibility with JSON columns | Low | Medium | conftest.py already tested with JSON; QuantAgent-4ch resolved this |
| Other tests importing `db_session` from these files | Very Low | Low | pytest discovers fixtures from conftest.py; local removals don't affect other files |
| `flush()` behavior differs between SQLite and PostgreSQL | Very Low | Low | `flush()` is standard SQLAlchemy; both DBs support it |
| `ORDER BY id` changes production behavior | Low | Low | Only adds determinism; normal case (one active position) is unaffected |

---

## Commit Plan

Single commit on feature branch:

```
fix(tests): fix PositionMonitor unit-test regressions (QuantAgent-3uf)

- Remove local db_session fixtures from both position_monitor test files
  so tests use conftest.py SQLite isolation instead of shared PostgreSQL
- Fix FK violation in test_position_with_all_optional_fields by creating
  actual Trade/Signal records instead of using hardcoded non-existent IDs
- Add ORDER BY to get_active_position for deterministic .first() results

Fixes: test_only_one_active_position_per_symbol (46 stale rows in shared DB)
       test_position_with_all_optional_fields (ForeignKeyViolation on trade_id=123)
       test_get_active_position_returns_most_recent_if_multiple (non-deterministic ordering)
       test_closed_position_not_returned_by_get_active (stale active positions)
```

---

## Success Criteria

- [ ] 4 failing tests now pass
- [ ] No previously-passing tests broken
- [ ] Tests run cleanly with and without `DATABASE_URL`
- [ ] Changes limited to the 3 files listed above
