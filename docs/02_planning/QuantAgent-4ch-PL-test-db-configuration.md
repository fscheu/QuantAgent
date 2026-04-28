# QuantAgent-4ch — Planning: Test Database Configuration

**Issue ID:** QuantAgent-4ch  
**Title:** Fix remaining unit test failures (stale paths and missing data)  
**Type:** Bug  
**Priority:** 1

---

## Objective

Fix unit test failures by replacing PostgreSQL database dependency with SQLite for tests.

---

## Tasks

### Task 1: Update conftest.py with db_session Fixture
**Estimate:** 0.5h

**What:**
- Add `db_session` fixture to `tests/conftest.py`
- Use SQLite in-memory engine (`sqlite:///:memory:`)
- Create all tables via `Base.metadata.create_all()`
- Yield session for test use
- Cleanup: close session, drop tables, dispose engine

**Why:**
- Provides isolated, fast test database
- No external dependencies (PostgreSQL)

**How to validate:**
```bash
pytest tests/test_backtest.py::TestBacktest::test_backtest_initialization -v
# Should PASS (no psycopg2 errors)
```

**Dependencies:** None

---

### Task 2: Enable SQLite Foreign Keys
**Estimate:** 0.25h

**What:**
- In `db_session` fixture, after creating engine, execute:
  ```python
  from sqlalchemy import event
  @event.listens_for(test_engine, "connect")
  def set_sqlite_pragma(dbapi_conn, connection_record):
      cursor = dbapi_conn.cursor()
      cursor.execute("PRAGMA foreign_keys=ON")
      cursor.close()
  ```

**Why:**
- SQLite disables foreign key constraints by default
- Tests need FK enforcement (same as PostgreSQL)

**How to validate:**
```python
# Test that FK violation raises error
def test_fk_enforcement(db_session):
    from quantagent.models import Trade
    with pytest.raises(Exception):
        trade = Trade(order_id=99999)  # Non-existent order
        db_session.add(trade)
        db_session.commit()
```

**Dependencies:** Task 1

---

### Task 3: Remove or Update Existing db_session Fixtures
**Estimate:** 0.25h

**What:**
- Search for existing `db_session` fixtures in test files:
  ```bash
  grep -r "def db_session" tests/
  ```
- If found in `test_backtest.py` or other files, remove them (use centralized fixture from `conftest.py`)
- Update any test-specific cleanup logic if needed

**Why:**
- Avoid fixture conflicts
- Single source of truth for test database setup

**How to validate:**
```bash
pytest tests/ -v
# All tests using db_session pass
```

**Dependencies:** Task 1

---

### Task 4: Run Full Test Suite
**Estimate:** 0.25h

**What:**
- Execute full test suite:
  ```bash
  pytest tests/ -v
  ```
- Verify all unit tests pass
- Check for any tests failing due to SQLite incompatibilities
- Fix any issues (e.g., PostgreSQL-specific SQL)

**Why:**
- Comprehensive validation
- Catch edge cases

**How to validate:**
```bash
pytest tests/ -v | grep -E "(PASSED|FAILED|ERROR)" | sort | uniq -c
# Should show all PASSED, no FAILED/ERROR related to database
```

**Dependencies:** Tasks 1-3

---

### Task 5: Document Changes (Optional)
**Estimate:** 0.25h

**What:**
- Update `docs/03_design/TESTING_PATTERNS.md` with database testing guidance:
  - How to use `db_session` fixture
  - SQLite vs PostgreSQL considerations
  - When to use integration tests for PostgreSQL-specific features

**Why:**
- Help future developers understand test database setup
- Prevent regressions

**How to validate:**
- Read doc and verify clarity
- Ensure examples are accurate

**Dependencies:** Task 4

---

## Total Estimate

**1.5 hours** (3 core tasks + 2 optional/validation tasks)

---

## Execution Order

1. **Task 1** (core fix)
2. **Task 2** (FK enforcement)
3. **Task 3** (cleanup)
4. **Task 4** (validation)
5. **Task 5** (documentation) — optional, can be deferred

---

## Risks & Mitigations

### Risk 1: SQLite Incompatibility
**Description:** Some tests may use PostgreSQL-specific features (ENUM, ARRAY, etc.)

**Mitigation:**
- QuantAgent models use standard SQLAlchemy types (no PostgreSQL-specific types identified)
- If issues arise, wrap PostgreSQL-specific tests with `@pytest.mark.postgres` and skip in unit tests

**Probability:** Low  
**Impact:** Low

---

### Risk 2: Test Cleanup Incomplete
**Description:** Fixture cleanup may leave state (if using file-based SQLite)

**Mitigation:**
- Use in-memory SQLite (`:memory:`) — automatically cleaned up on session close
- Explicitly drop tables and dispose engine in fixture cleanup

**Probability:** Very Low  
**Impact:** Low

---

### Risk 3: Foreign Key Constraint Differences
**Description:** SQLite FK enforcement may differ from PostgreSQL

**Mitigation:**
- Explicitly enable FK constraints in SQLite (`PRAGMA foreign_keys=ON`)
- Task 2 addresses this

**Probability:** Low (with mitigation)  
**Impact:** Medium

---

## Testing Strategy

### Unit Tests (SQLite)
- All tests using `db_session` fixture
- Fast, isolated, no external dependencies
- Run on every commit

### Integration Tests (PostgreSQL) — Future
- If PostgreSQL-specific features are tested, mark with `@pytest.mark.integration`
- Skip by default (`pytest -m "not integration"`)
- Run manually or in CI with PostgreSQL service

---

## Rollout Plan

### Step 1: Local Development
- Developer implements Tasks 1-3
- Runs `pytest tests/` locally
- Verifies all tests pass

### Step 2: Code Review
- Submit PR with:
  - Updated `tests/conftest.py`
  - This planning doc + RQ/DS/AC docs
- Reviewer runs tests locally

### Step 3: CI Validation
- CI pipeline runs `pytest tests/`
- Verifies tests pass in clean environment (no PostgreSQL)

### Step 4: Merge to Main
- Once approved and CI passes, merge
- All future tests benefit from SQLite fixture

---

## Success Criteria

- [ ] All unit tests pass with `pytest tests/`
- [ ] No `psycopg2.OperationalError` errors
- [ ] Tests run in < 60s (target: < 30s)
- [ ] No PostgreSQL required for unit tests
- [ ] Documentation updated (optional)

---

## Next Steps After This Issue

1. **Implementer Agent**: Execute Tasks 1-4
2. **Tester Agent**: Run test suite, verify all ACs
3. **Merge**: Squash commit to main branch
4. **Close Issue**: Update Beads status to `closed`
