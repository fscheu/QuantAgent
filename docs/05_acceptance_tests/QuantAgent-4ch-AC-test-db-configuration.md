# QuantAgent-4ch — Acceptance Criteria: Test Database Configuration

**Issue ID:** QuantAgent-4ch  
**Title:** Fix remaining unit test failures (stale paths and missing data)  
**Type:** Bug

---

## AC-1: Test Database Isolation

**Given** a test requiring database access (e.g., `test_backtest.py`)  
**When** the test runs via `pytest`  
**Then**:
- Test uses SQLite in-memory database
- Test does NOT attempt to connect to PostgreSQL at `localhost:5432`
- No `psycopg2.OperationalError` is raised

**Verification:**
```bash
pytest tests/test_backtest.py::TestBacktest::test_backtest_initialization -v
# PASSED (no database connection errors)
```

---

## AC-2: Clean State Per Test

**Given** two tests that both use `db_session` fixture  
**When** tests run sequentially  
**Then**:
- Second test does not see data from first test
- Each test starts with empty tables
- No foreign key constraint violations from leftover data

**Verification:**
```python
# Test 1 creates BacktestRun record
def test_creates_run(db_session):
    from quantagent.models import BacktestRun
    run = BacktestRun(run_id="test-123", ...)
    db_session.add(run)
    db_session.commit()
    assert db_session.query(BacktestRun).count() == 1

# Test 2 starts with clean database
def test_empty_db(db_session):
    from quantagent.models import BacktestRun
    assert db_session.query(BacktestRun).count() == 0  # ✓ Clean state
```

Run both:
```bash
pytest tests/test_backtest.py::TestBacktest::test_creates_run tests/test_backtest.py::TestBacktest::test_empty_db -v
# Both PASSED
```

---

## AC-3: No External Dependencies

**Given** a CI environment without PostgreSQL installed  
**When** `pytest tests/` runs  
**Then**:
- All unit tests pass
- No errors about missing database services
- No Docker/PostgreSQL required

**Verification:**
```bash
# Simulate CI environment (no PostgreSQL)
docker run --rm -it python:3.12 bash
pip install -e .[test]
pytest tests/ -v
# All unit tests PASSED
```

---

## AC-4: Integration Tests Marked (If Applicable)

**Given** tests that explicitly require PostgreSQL features  
**When** reviewing test suite  
**Then**:
- Such tests are marked `@pytest.mark.integration` or `@pytest.mark.postgres`
- Standard `pytest` run skips them (use `pytest -m "not integration"`)
- Documentation explains how to run integration tests

**Current state:** No PostgreSQL-specific tests identified. If added in future, must be marked.

**Verification:**
```bash
pytest -m "not integration" tests/
# All unit tests run (integration tests skipped)
```

---

## AC-5: Backward Compatibility

**Given** existing test files (`test_backtest.py`, `test_agents.py`, etc.)  
**When** `db_session` fixture is updated  
**Then**:
- No changes required to test function signatures
- No changes required to test assertions
- Tests continue to work as before (only underlying database changes)

**Verification:**
```bash
# Before fix: test fails at fixture setup
# After fix: test runs with new fixture, same test code
git diff tests/  # Should show no changes to test files (only conftest.py)
```

---

## Negative Test Cases

### NT-1: Invalid Schema
**Given** a test that creates a model with invalid fields  
**When** saving to database  
**Then** SQLAlchemy validation raises appropriate error (just like with PostgreSQL)

**Example:**
```python
def test_invalid_model(db_session):
    from quantagent.models import BacktestRun
    with pytest.raises(Exception):  # IntegrityError, ValidationError, etc.
        run = BacktestRun(run_id=None)  # run_id is NOT NULL
        db_session.add(run)
        db_session.commit()
```

### NT-2: Foreign Key Constraints
**Given** a test that violates foreign key constraint  
**When** saving to database  
**Then** SQLite enforces constraint and raises error

**Example:**
```python
def test_fk_violation(db_session):
    from quantagent.models import Trade
    with pytest.raises(Exception):
        trade = Trade(order_id=99999)  # Order doesn't exist
        db_session.add(trade)
        db_session.commit()
```

---

## Performance Criteria

### P-1: Fast Execution
**Given** the full unit test suite  
**When** running `pytest tests/`  
**Then**:
- Total execution time < 60 seconds (target: < 30s)
- Individual test overhead < 50ms per test

**Rationale:** In-memory SQLite is fast; no network/disk I/O.

---

## Boundary Conditions

### B-1: Empty Database
**Given** a test with `db_session` fixture  
**When** test queries database before inserting data  
**Then** returns empty results (not error)

```python
def test_empty_query(db_session):
    from quantagent.models import BacktestRun
    assert db_session.query(BacktestRun).all() == []  # ✓
```

### B-2: Large Data
**Given** a test inserting 1000+ records  
**When** test runs  
**Then** completes successfully without memory/performance issues

```python
def test_bulk_insert(db_session):
    from quantagent.models import Signal
    signals = [Signal(...) for _ in range(1000)]
    db_session.bulk_save_objects(signals)
    db_session.commit()
    assert db_session.query(Signal).count() == 1000  # ✓
```

---

## Oracles (How to Verify Success)

### Oracle 1: Test Exit Code
```bash
pytest tests/test_backtest.py
echo $?
# 0 = success
```

### Oracle 2: Verbose Output
```bash
pytest tests/ -v | grep -E "(PASSED|FAILED|ERROR)"
# All PASSED, no FAILED or ERROR
```

### Oracle 3: Database File Not Created
```bash
pytest tests/
ls -la *.db
# No .db files created (in-memory database used)
```

### Oracle 4: No PostgreSQL Connection Attempts
```bash
# Monitor network connections during test run
lsof -i :5432  # PostgreSQL port
# (no connections to port 5432 during pytest)
```

---

## Manual Test Procedure

1. **Setup:**
   ```bash
   cd /path/to/QuantAgent
   git checkout feature/QuantAgent-4ch-test-db-fix
   pip install -e .[test]
   ```

2. **Run Failing Test (Before Fix):**
   ```bash
   git checkout main  # Before fix
   pytest tests/test_backtest.py::TestBacktest::test_backtest_initialization
   # Should fail with psycopg2.OperationalError
   ```

3. **Run Fixed Test (After Fix):**
   ```bash
   git checkout feature/QuantAgent-4ch-test-db-fix
   pytest tests/test_backtest.py::TestBacktest::test_backtest_initialization
   # Should PASS
   ```

4. **Run Full Suite:**
   ```bash
   pytest tests/ -v
   # All unit tests PASS
   ```

5. **Verify No External Dependencies:**
   ```bash
   # Stop PostgreSQL if running
   docker stop quantagent-postgres || true
   pytest tests/
   # Still PASS (no PostgreSQL required)
   ```

6. **Check Performance:**
   ```bash
   time pytest tests/
   # Should complete in < 60s (target: < 30s)
   ```

---

## Definition of Done (Testing Checklist)

- [ ] AC-1: Test uses SQLite (verified with passing test)
- [ ] AC-2: Clean state per test (verified with sequential tests)
- [ ] AC-3: No PostgreSQL required (verified in clean environment)
- [ ] AC-4: Integration tests marked (N/A if none exist)
- [ ] AC-5: Backward compatible (no test code changes)
- [ ] NT-1: Schema validation works
- [ ] NT-2: Foreign key constraints enforced
- [ ] P-1: Fast execution (< 60s for full suite)
- [ ] B-1: Empty database queries work
- [ ] B-2: Large data inserts work
- [ ] All oracles verified
- [ ] Manual test procedure completed successfully
