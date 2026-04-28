# QuantAgent-4ch — Design: Test Database Configuration

**Issue ID:** QuantAgent-4ch  
**Title:** Fix remaining unit test failures (stale paths and missing data)  
**Type:** Bug

---

## Design Overview

Update `tests/conftest.py` to provide a test-specific database configuration that uses SQLite instead of PostgreSQL for unit tests. This ensures tests can run in any environment without external database dependencies.

---

## Affected Components

### Modified
- `tests/conftest.py` — Add `db_session` fixture with SQLite configuration

### Not Modified
- `quantagent/database.py` — Production database engine (no changes)
- `quantagent/models.py` — ORM models (no changes)
- Individual test files — Test logic unchanged (only fixture dependency)

---

## Technical Approach

### 1. Test Database Fixture

Create `db_session` fixture in `tests/conftest.py`:

```python
@pytest.fixture
def db_session():
    """Provide isolated SQLite database session for testing."""
    # Create in-memory SQLite engine
    from sqlalchemy import create_engine
    from quantagent.models import Base
    
    test_engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(test_engine)
    
    # Create session
    from sqlalchemy.orm import sessionmaker
    TestSession = sessionmaker(bind=test_engine)
    session = TestSession()
    
    yield session
    
    # Cleanup
    session.close()
    Base.metadata.drop_all(test_engine)
    test_engine.dispose()
```

### 2. Key Design Decisions

#### Why SQLite In-Memory?
- **Fast**: No disk I/O, ideal for unit tests
- **Isolated**: Each test gets fresh database
- **No setup**: No external services required
- **Compatible**: SQLAlchemy abstracts differences

#### Why Not Mock Database?
- Need real SQL constraints, foreign keys
- Need real transaction handling
- Need real ORM behavior (relationships, cascades)
- Mocks would create tautological tests

#### Fixture Scope
- **Scope**: `function` (default)
- **Reason**: Each test needs clean state
- **Alternative**: `class` scope could batch tests, but risks state leakage

### 3. SQLite Compatibility

Most SQLAlchemy code is database-agnostic, but watch for:

**Supported in SQLite:**
- Foreign keys (must enable: `PRAGMA foreign_keys=ON`)
- Transactions
- Basic constraints (NOT NULL, UNIQUE, CHECK)

**PostgreSQL-specific features NOT in SQLite:**
- ENUM types (use VARCHAR/TEXT instead)
- ARRAY columns (use JSON or separate table)
- `RETURNING` in some contexts (SQLAlchemy handles)

**Current QuantAgent models**: All should be compatible (uses standard SQLAlchemy types).

---

## Alternative Approaches Considered

### ❌ Use PostgreSQL in Docker for Tests
**Why rejected:**
- Slow startup (2-5s per test run)
- External dependency (Docker + image)
- Overkill for unit tests
- CI complexity

**When to use:** Integration tests explicitly testing PostgreSQL-specific behavior.

### ❌ Monkeypatch DATABASE_URL
**Why rejected:**
- Still requires mocking/managing global state
- Risk of contaminating production config
- Doesn't provide database isolation

**Better:** Explicit test fixture with test engine.

### ❌ Single Persistent Test Database
**Why rejected:**
- State leakage between tests
- Requires cleanup logic per test
- Hard to parallelize tests
- Risk of non-deterministic failures

**Better:** Fresh in-memory database per test.

---

## Migration Path

### Step 1: Add Fixture
Add `db_session` fixture to `tests/conftest.py`.

### Step 2: Verify Tests
Run `pytest tests/test_backtest.py -v` to confirm tests pass.

### Step 3: Expand Coverage
Verify all tests using `db_session` work (grep for `db_session` in tests/).

### Step 4: Document
Update `docs/03_design/TESTING_PATTERNS.md` with database testing guidance.

---

## Validation

### Before Fix
```bash
$ pytest tests/test_backtest.py::TestBacktest::test_backtest_initialization
# ERROR: psycopg2.OperationalError: database "quantagent" does not exist
```

### After Fix
```bash
$ pytest tests/test_backtest.py::TestBacktest::test_backtest_initialization
# PASSED
```

### Full Suite
```bash
$ pytest tests/ -v
# All unit tests pass
```

---

## Open Questions

None — design is straightforward.
