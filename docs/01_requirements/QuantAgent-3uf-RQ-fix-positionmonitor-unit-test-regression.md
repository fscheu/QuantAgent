# QuantAgent-3uf — Requirements: Fix PositionMonitor Unit-Test Regressions

**Issue ID:** QuantAgent-3uf
**Title:** Fix PositionMonitor unit-test regressions exposed by CI gate
**Type:** Bug
**Priority:** 1

---

## Objective

Restore 4 failing PositionMonitor tests so the CI gate introduced by `QuantAgent-82t`
(re-enable unit tests) passes cleanly without PositionMonitor failures.

---

## Background

Re-evaluation of `QuantAgent-82t` (re-enable CI unit tests) exposed 4 failing tests:

- `tests/test_position_monitor.py::test_only_one_active_position_per_symbol`
- `tests/test_position_monitor_constraints.py::test_position_with_all_optional_fields`
- `tests/test_position_monitor_constraints.py::test_get_active_position_returns_most_recent_if_multiple`
- `tests/test_position_monitor_constraints.py::test_closed_position_not_returned_by_get_active`

Failures are caused by **test infrastructure issues**, not bugs in the business logic of
`PositionMonitor` itself.

---

## Root Cause Analysis

### RC-1: Local `db_session` fixtures override central isolation

Both `test_position_monitor.py` and `test_position_monitor_constraints.py` define a
local `db_session` fixture that reads `DATABASE_URL` from the environment. When
`DATABASE_URL` is set (as it is in CI with PostgreSQL), this local fixture overrides
the centralized SQLite in-memory fixture from `tests/conftest.py`.

Consequences:
- All tests share a single persistent PostgreSQL database.
- Data is committed (not rolled back) between tests.
- Active positions from earlier runs accumulate in the `active_positions` table.
- By the time `test_only_one_active_position_per_symbol` runs, 46 stale active
  BTCUSDT positions exist, causing `assert len == 1` to fail.
- `test_closed_position_not_returned_by_get_active` returns a stale position from a
  previous run instead of `None`.

### RC-2: FK violation with hardcoded `trade_id` / `signal_id`

`test_position_with_all_optional_fields` calls `position_monitor.open_position()`
with `trade_id=123, signal_id=456`. These are FKs to `trades.id` and `signals.id`.
Since no parent records with those IDs exist, PostgreSQL raises
`ForeignKeyViolation`, crashing the commit.

### RC-3: Non-deterministic ordering in `get_active_position`

`PositionMonitor.get_active_position()` uses `.first()` with no `ORDER BY`.
When the test manually inserts a second active position for the same symbol
(`test_get_active_position_returns_most_recent_if_multiple`), the query returns
whichever record PostgreSQL happens to surface first (id=1, from a previous run),
not the one the test just created (e.g., id=96). This makes the assertion
`active.id == pos1.id` non-deterministic in a shared DB.

---

## Requirements

### R1: Test isolation via centralized SQLite fixture

Both test files must use the `db_session` fixture from `tests/conftest.py`
(SQLite in-memory, function-scoped, with FK enforcement). The local `db_session`
definitions in both test files must be removed.

**Acceptance:** Each test starts with an empty database and cannot be affected by
data from other tests.

### R2: Valid FK references in optional-fields test

`test_position_with_all_optional_fields` must create actual `Trade` and `Signal`
records in the test DB before referencing their IDs as `trade_id` / `signal_id`.
The commit must succeed without FK violations.

**Acceptance:** The test passes with FK constraints enforced (both PostgreSQL and
SQLite + `PRAGMA foreign_keys=ON`).

### R3: Deterministic query ordering in `get_active_position`

`PositionMonitor.get_active_position()` must add `ORDER BY ActivePosition.id`
(ascending) to its query so that when multiple active positions exist for the same
symbol, `first()` always returns the earliest-inserted record consistently.

**Acceptance:** `test_get_active_position_returns_most_recent_if_multiple` passes
deterministically because the query order is explicit, not dependent on DB internals.

---

## Out of Scope

- Changing CI workflow (no modification to `.github/workflows/`)
- Resolving benchmark fixture (`benchmark_data_dir`) or PnL calculation issues
- Refactoring `PositionMonitor` beyond the `ORDER BY` addition
- Adding new tests or changing test assertions beyond what's needed for the fixes

---

## Files Affected

| File | Change |
|---|---|
| `tests/test_position_monitor.py` | Remove local `db_session` fixture |
| `tests/test_position_monitor_constraints.py` | Remove local `db_session`; fix FK in `test_position_with_all_optional_fields` |
| `quantagent/trading/position_monitor.py` | Add `ORDER BY ActivePosition.id` to `get_active_position` |

---

## Gate Command

```bash
DATABASE_URL=postgresql://test:test@localhost:5432/quantagent_test \
  /mnt/actions-runner/autodev-runtime/venvs/QuantAgent/.venv/bin/python \
  -m pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"
```

All 4 previously failing tests must appear as `PASSED`. No new failures.
