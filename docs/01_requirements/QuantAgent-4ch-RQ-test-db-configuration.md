# QuantAgent-4ch — Requirements: Fix Test Database Configuration

**Issue ID:** QuantAgent-4ch  
**Title:** Fix remaining unit test failures (stale paths and missing data)  
**Type:** Bug  
**Priority:** 1  
**Labels:** openclaw:design_approved testing

---

## Objective

Fix unit test failures caused by incorrect database configuration. Tests attempt to connect to PostgreSQL database `quantagent` which doesn't exist in test/CI environments, causing test suite to fail at setup.

---

## Scope

### In Scope
- Configure test fixtures to use SQLite (in-memory or file-based) for unit tests
- Update test database fixtures in `tests/conftest.py`
- Ensure test isolation (each test gets clean database state)
- Validate that tests using `db_session` fixture work correctly

### Out of Scope
- Integration tests that require real PostgreSQL (keep as-is, mark appropriately)
- Performance testing of database operations
- Changes to production database configuration
- Changes to test logic or assertions (only fixture setup)

---

## Current Behavior (Broken)

1. Test runner executes `pytest`
2. Tests requiring database (e.g., `test_backtest.py`) use `db_session` fixture
3. Fixture calls `SessionLocal()` which reads `DATABASE_URL` from environment
4. `DATABASE_URL` points to PostgreSQL: `postgresql://user:password@localhost:5432/quantagent`
5. PostgreSQL database `quantagent` doesn't exist → `psycopg2.OperationalError`
6. **Test fails at setup before any test logic runs**

Example error:
```
E   psycopg2.OperationalError: connection to server at "localhost" (127.0.0.1), 
    port 5432 failed: FATAL:  database "quantagent" does not exist
```

---

## Expected Behavior (Fixed)

1. Test runner executes `pytest`
2. Tests requiring database use `db_session` fixture
3. Fixture creates **test-specific SQLite database** (in-memory or temporary file)
4. Each test gets isolated, clean database state
5. Test runs successfully
6. Database is cleaned up after test completes

---

## Acceptance Criteria

### AC-1: Test Database Isolation
**Given** a test requiring database access  
**When** the test runs  
**Then** it uses a separate SQLite test database, not production PostgreSQL

### AC-2: Clean State Per Test
**Given** multiple tests using `db_session` fixture  
**When** tests run sequentially  
**Then** each test starts with empty database tables (no leftover data from previous tests)

### AC-3: No External Dependencies
**Given** a CI environment without PostgreSQL installed  
**When** pytest runs  
**Then** all unit tests pass without requiring external database services

### AC-4: Integration Tests Marked
**Given** tests that explicitly require PostgreSQL (if any)  
**When** reviewing test suite  
**Then** they are marked with `@pytest.mark.integration` or similar, and skipped in standard unit test runs

### AC-5: Backward Compatibility
**Given** existing test files (`test_backtest.py`, etc.)  
**When** fixture is updated  
**Then** tests continue to work without modifications to test code (only fixture changes)

---

## Constraints

- **No test code changes**: Fix must be in `conftest.py` fixtures only
- **SQLAlchemy compatibility**: Solution must work with existing SQLAlchemy models
- **Fast execution**: In-memory SQLite preferred for speed
- **CI/CD compatibility**: Must work in GitHub Actions / automation environments

---

## Definition of Done

- [ ] `db_session` fixture updated to use SQLite for tests
- [ ] All unit tests pass locally with `pytest`
- [ ] All unit tests pass in CI environment (if configured)
- [ ] Test execution time remains reasonable (no significant slowdown)
- [ ] Documentation updated (this file + implementation notes)
