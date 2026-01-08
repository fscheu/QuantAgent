# Test Failure Analysis Report v2 — QuantAgent-fg0

**Issue ID**: QuantAgent-fg0  
**Branch**: feature/QuantAgent-fg0-test-revision  
**Date**: 2026-01-08  
**Reporter**: Tester Agent  
**Status**: 🟡 **21 FAILED**, 279 PASSED, 20 SKIPPED

---

## Executive Summary

After initial fixes documented in `QuantAgent-fg0-IM-tests.md`, test suite now shows **significant improvement**:
- ✅ **279 tests passing** (up from ~244)
- ❌ **21 tests failing** (down from 56)
- ⏭️ **20 tests skipped**
- ⚠️ **10,266 warnings** (non-blocking)

Remaining failures fall into **6 categories**:

1. **Retry logic cap mismatch** (1 failure) — test expectation
2. **Backtest config snapshot field missing** (1 failure) — schema evolution
3. **Backtest integration data unavailable** (2 failures) — external dependency
4. **Checkpointing setup AttributeError** (1 failure) — implementation bug
5. **Database unique constraint violation** (1 failure) — test isolation
6. **Mock interface type errors** (14 failures) — mock return_value shape
7. **Missing test data file** (1 failure) — repository structure

---

## Category 1: Retry Logic — Max Wait Cap Mismatch

**Count**: 1 failure  
**Severity**: 🟢 Low  
**Type**: Test expectation needs adjustment

### Failure

**File**: `tests/test_agent_utils_retry.py:548`

```python
def test_very_large_max_wait_applied():
    wait = _calculate_exponential_backoff_wait(
        attempt=10, base_wait=2.0, max_wait=3600.0
    )
    assert wait == 3600.0  # Should be capped
    ^^^^^^^^^^^^^^^^^^^^^
E   assert 2048.0 == 3600.0
```

### Analysis

- **What's tested**: Exponential backoff should cap at `max_wait=3600.0`
- **Actual behavior**: Returns `2048.0` (2^11 seconds, attempt 10)
- **Expected behavior**: Test assumes cap is enforced
- **Root cause**: Implementation applies cap at different threshold OR cap logic is incorrect

### Hypothesis

1. **Implementation correct, test wrong**: Backoff formula is `base_wait * 2^attempt`, so attempt 10 → 2048s < 3600s (cap not hit yet)
2. **Implementation wrong**: Cap should be enforced but isn't being applied
3. **Boundary confusion**: Test needs attempt=11 or higher to trigger cap

### Recommended Action

**→ Review** `quantagent/agent_utils.py:_calculate_exponential_backoff_wait` to verify cap logic, then:
- If cap logic is correct: adjust test to use attempt=12 (4096 > 3600)
- If cap logic is missing: add enforcement in implementation

---

## Category 2: Backtest Config Snapshot — Missing Field

**Count**: 1 failure  
**Severity**: 🟡 Medium  
**Type**: Schema evolution / backward compatibility

### Failure

**File**: `tests/test_backtest.py:142`

```python
def test_backtest_config_snapshot_includes_all_params(self, db_session):
    # ...
    assert field in snapshot
E   AssertionError: assert 'model_provider' in {...}
```

### Analysis

- **What's tested**: Backtest config snapshot includes all expected parameters
- **Missing field**: `'model_provider'`
- **Present fields**: `'agent_llm_provider'`, `'agent_llm_model'`, etc.
- **Root cause**: Field name changed or test expectation outdated

### Hypothesis

1. **Rename**: `model_provider` → `agent_llm_provider` (schema evolution)
2. **Deprecation**: Field no longer captured in snapshot
3. **Test bug**: Test references wrong field name

### Recommended Action

**→ Update test** to use correct field name (`agent_llm_provider`) OR remove assertion if field is deprecated

---

## Category 3: Backtest Integration — External Data Unavailable

**Count**: 2 failures  
**Severity**: 🔴 High (blocks CI)  
**Type**: External dependency / data provider limitation

### Failures

**File**: `tests/test_backtest_integration.py`

#### 3.1 `test_backtest_with_trades_execution` (line 222)

```python
assert len(signals) > 0
E   assert 0 > 0
```

**Captured logs**:
```
ERROR yfinance: $BTC-USD: possibly delisted; no price data found (1h 2023-12-02 -> 2024-01-01)
Yahoo error = "1h data not available for startTime=1701475200 and endTime=1704067200. 
The requested range must be within the last 730 days."
```

#### 3.2 `test_backtest_date_range_iteration` (line 491)

```python
assert len(invoke_calls) == 10
E   assert 0 == 10
```

**Same root cause**: No data fetched → no iterations → no graph invocations

### Analysis

- **What's tested**: Backtest execution with real market data
- **Data source**: yfinance API (Yahoo Finance)
- **Date range**: 2023-12-02 to 2024-01-01 (1-hour intervals)
- **Limitation**: Yahoo Finance 1h data only available for **last 730 days**
- **Current date**: 2026-01-08 → test data is ~2 years old (>730 days)

### Hypothesis

1. **Hard-coded dates**: Test uses static 2023/2024 dates that expired
2. **No mock fallback**: Tests require real API calls (no fixtures)
3. **Design issue**: Integration tests should use relative date ranges or fixtures

### Recommended Action

**→ Choice A**: Update test to use **relative date range** (e.g., `end=now`, `start=now - 30 days`)  
**→ Choice B**: Mock `yfinance` data provider with fixture data  
**→ Choice C**: Mark tests as `@pytest.mark.integration` and document external dependency

---

## Category 4: Checkpointing — AttributeError on Setup

**Count**: 1 failure  
**Severity**: 🔴 Critical  
**Type**: Implementation bug

### Failure

**File**: `tests/test_checkpointing_resume.py:395`

```python
tg = TradingGraph(use_checkpointing=True)
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
quantagent/trading_graph.py:150: _setup_checkpointer
    checkpointer.setup()
    ^^^^^^^^^^^^^^^^^^
E   AttributeError: 'ExitStack' object has no attribute 'setup'
```

**Exception chain**:
```python
# trading_graph.py:153
raise ValueError(f"Failed to connect to PostgreSQL at {db_url}: {str(e)}")
E   ValueError: Failed to connect to PostgreSQL at postgresql://test-db: 
    'ExitStack' object has no attribute 'setup'
```

### Analysis

- **What's tested**: LangGraph with different thread IDs (checkpointing enabled)
- **Error location**: `quantagent/trading_graph.py:150`
- **Root cause**: `checkpointer` is `ExitStack` object instead of actual checkpointer
- **Expected type**: Should be `PostgresSaver` or similar with `.setup()` method

### Hypothesis

1. **Factory bug**: `_setup_checkpointer()` returns wrong object type
2. **Import error**: PostgresSaver not imported/instantiated correctly
3. **Conditional bug**: Checkpointer path returns ExitStack in error case

### Code Location

```python
# quantagent/trading_graph.py:57
self.checkpointer, self._checkpointer_context = self._setup_checkpointer()

# line 150
checkpointer.setup()  # ← ExitStack has no .setup()
```

### Recommended Action

**→ Fix implementation** in `trading_graph.py:_setup_checkpointer()`:
- Return correct checkpointer type (PostgresSaver)
- Ensure ExitStack is only used as context manager, not returned as checkpointer

---

## Category 5: Database Migrations — Unique Constraint Violation

**Count**: 1 failure  
**Severity**: 🟡 Medium  
**Type**: Test isolation / fixture cleanup

### Failure

**File**: `tests/test_migrations.py:196`

```python
def test_insert_position(db):
    # ...
    db.commit()
    
E   psycopg2.errors.UniqueViolation: duplicate key value violates unique constraint "ix_positions_symbol"
E   DETAIL: Key (symbol)=(BTC) already exists.
```

### Analysis

- **What's tested**: Insert position record into database
- **Constraint**: `ix_positions_symbol` (unique index on `symbol` column)
- **Root cause**: Previous test left `BTC` record in `positions` table
- **Impact**: Test isolation broken

### Hypothesis

1. **Fixture cleanup incomplete**: `db` fixture doesn't truncate `positions` table
2. **Transaction rollback missing**: Test doesn't wrap in transaction
3. **Parallel execution**: Multiple tests writing same symbol

### Recommended Action

**→ Update fixture** to ensure cleanup:
```python
@pytest.fixture
def db():
    # setup
    yield session
    session.query(Position).delete()  # ← Add cleanup
    session.commit()
```

OR **→ Update test** to handle existing records:
```python
def test_insert_position(db):
    db.query(Position).filter_by(symbol="BTC").delete()  # ← Clean before insert
    # ... rest of test
```

---

## Category 6: Mock Interface Type Errors

**Count**: 14 failures  
**Severity**: 🔴 Critical  
**Type**: Mock configuration bug (tautological mocks)

### Failure Pattern

**All in**: `tests/test_trading_components.py::TestFullEndToEndIntegration`

```python
quantagent/trading/order_manager.py:114: execute_decision
    is_reversal = (existing_qty > 0 and side == OrderSide.SELL) or (
                   ^^^^^^^^^^^^^^^^
E   TypeError: '>' not supported between instances of 'Mock' and 'int'
```

### Affected Tests (14)

1. `test_full_flow_long_valid_trade_executes_all_steps`
2. `test_full_flow_short_valid_trade_executes_all_steps`
3. `test_full_flow_invalid_trade_rejected_before_broker`
4. `test_full_flow_position_too_large_rejected`
5. `test_full_flow_circuit_breaker_active`
6. `test_daily_pnl_tracking_across_trades`
7. `test_short_to_long_reversal`
8. `test_long_to_short_reversal`
9. `test_reversal_with_different_sizes`
10. `test_reversal_close_order_fails`
11. `test_non_reversal_unchanged`
12. `test_reversal_order_objects_created`
13. `test_reversal_broker_receives_correct_sequence`
14. `test_reversal_using_tradesiganl_enum`

### Analysis

- **What's tested**: Full trading flow with OrderManager
- **Error location**: `order_manager.py:114`
- **Code context**:
  ```python
  existing_qty = self.portfolio.get_position_quantity(symbol)
  is_reversal = (existing_qty > 0 and side == OrderSide.SELL) or ...
  ```
- **Root cause**: `self.portfolio.get_position_quantity()` returns `Mock` object instead of numeric value
- **Expected**: Should return `Decimal`, `float`, or `int`

### Hypothesis

1. **Mock not configured**: Test setup doesn't specify `return_value` for `get_position_quantity`
2. **Mock shape wrong**: Mock exists but `return_value` is another Mock (default behavior)
3. **Test regression**: Recent change to OrderManager added `.get_position_quantity()` call, tests not updated

### Code Location

**Test file**: `tests/test_trading_components.py` (class `TestFullEndToEndIntegration`)

**Example setup** (needs investigation):
```python
# Likely missing:
self.portfolio.get_position_quantity.return_value = Decimal("0")
```

### Recommended Action

**→ Fix test setup** in `TestFullEndToEndIntegration`:

```python
@pytest.fixture
def setup(self):
    # ... existing mocks ...
    self.portfolio.get_position_quantity = Mock(return_value=Decimal("0"))
    # OR
    self.portfolio.get_position_quantity.return_value = Decimal("0")
```

**Note**: This is a **test bug**, not production code bug. These tests use tautological mocks that don't validate real behavior.

---

## Category 7: Missing Test Data File

**Count**: 1 failure  
**Severity**: 🟡 Medium  
**Type**: Repository structure / test data management

### Failure

**File**: `tests/test_parallel_execution.py:21`

```python
df = pd.read_csv("benchmark/btc/BTC_4h_1.csv")
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   FileNotFoundError: [Errno 2] No such file or directory: 'benchmark/btc/BTC_4h_1.csv'
```

### Analysis

- **What's tested**: Parallel agent execution
- **Missing file**: `benchmark/btc/BTC_4h_1.csv`
- **Expected location**: Repository root (relative path)
- **Root cause**: Test data not committed OR test expects data generation

### Hypothesis

1. **Gitignore**: `benchmark/` directory excluded from repo
2. **Test setup incomplete**: Test expects user to generate data first
3. **CI/CD missing step**: Data generation script not run in pipeline

### Recommended Action

**→ Choice A**: Commit `benchmark/btc/BTC_4h_1.csv` to repository  
**→ Choice B**: Generate fixture data in test setup:
```python
@pytest.fixture
def btc_data():
    # Generate or fetch sample data
    return pd.DataFrame(...)
```
**→ Choice C**: Skip test if data not available:
```python
@pytest.mark.skipif(not os.path.exists("benchmark/btc/BTC_4h_1.csv"), 
                    reason="Test data not available")
```

---

## Summary Table

| Category | Count | Severity | Type | Fix Owner |
|----------|-------|----------|------|-----------|
| 1. Retry cap | 1 | 🟢 Low | Test expectation | Tester |
| 2. Config field | 1 | 🟡 Medium | Schema evolution | Tester |
| 3. Data unavailable | 2 | 🔴 High | External dependency | Implementer |
| 4. Checkpointing | 1 | 🔴 Critical | Implementation bug | Implementer |
| 5. Unique constraint | 1 | 🟡 Medium | Test isolation | Tester |
| 6. Mock type errors | 14 | 🔴 Critical | Mock config | Tester |
| 7. Missing data file | 1 | 🟡 Medium | Repo structure | Implementer |
| **TOTAL** | **21** | — | — | — |

---

## Tester Action Items (Can Fix Now)

### 1. Update retry cap test
**File**: `tests/test_agent_utils_retry.py:548`
**Fix**: Use `attempt=12` to trigger cap OR verify implementation logic

### 2. Update backtest config test
**File**: `tests/test_backtest.py:142`
**Fix**: Replace `'model_provider'` with `'agent_llm_provider'`

### 3. Fix unique constraint test
**File**: `tests/test_migrations.py:196`
**Fix**: Add cleanup before insert OR update fixture

### 4. Fix mock return values
**File**: `tests/test_trading_components.py` (TestFullEndToEndIntegration)
**Fix**: Configure `portfolio.get_position_quantity.return_value = Decimal("0")`

---

## Implementer Action Items (Production Code)

### 1. Fix checkpointing AttributeError
**File**: `quantagent/trading_graph.py:150`
**Issue**: `_setup_checkpointer()` returns ExitStack instead of PostgresSaver

### 2. Fix backtest integration data dates
**File**: `tests/test_backtest_integration.py`
**Issue**: Use relative date ranges OR mock data provider

### 3. Add missing test data file
**File**: `benchmark/btc/BTC_4h_1.csv`
**Issue**: Commit data OR generate in fixture OR skip test

---

## Test Execution Summary

```
Command: pytest tests/ -v
Duration: 754.91s (12:34)
Results:
  ✅ 279 passed
  ❌ 21 failed
  ⏭️ 20 skipped
  ⚠️ 10,266 warnings
```

**Overall Status**: 🟡 **Partially Passing**  
**Blocker Issues**: 3 (checkpointing, backtest integration, mock errors)  
**Quick Wins**: 4 (can be fixed in test code only)

---

## Next Steps

1. **Tester**: Fix 4 test-code issues (categories 1, 2, 5, 6)
2. **Implementer**: Fix 3 production issues (categories 3, 4, 7)
3. **Re-run**: Full test suite after fixes
4. **Target**: 295+ passing tests (21 current failures resolved)

---

**END OF REPORT**
