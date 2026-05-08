# Requirements: Fix Missing Benchmark Fixture for Parallel Execution Test

**Issue ID:** QuantAgent-40j  
**Type:** Bug  
**Priority:** P1  
**Blocker for:** QuantAgent-82t (Re-enable unit tests in CI)

---

## Context

The test `tests/test_parallel_execution.py::test_parallel_execution` is currently failing in the CI gate because it:
1. Hardcodes a non-existent file path: `benchmark/btc/BTC_4h_1.csv`
2. Is collected by the CI gate command: `pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"`
3. Blocks the CI pipeline at `maxfail=10`

The test was discovered when validating QuantAgent-82t with the exact CI gate command on a clean integration branch.

---

## Objective

Prevent `test_parallel_execution` from blocking the fast unit test CI gate while preserving the test's value for performance validation.

---

## Scope

### In Scope
- Mark the test with appropriate pytest marker to exclude from CI gate
- Document why this test is excluded from the fast unit test suite

### Out of Scope
- Adding benchmark data to the repository
- Creating complex fixtures or mocking for the test
- Modifying the CI workflow configuration
- Resolving other test failures (PositionMonitor, PnL) discovered in the same run

---

## Analysis

### Test Characteristics
The test `test_parallel_execution`:
- **Performance-focused**: Measures execution time to validate parallel agent execution (4-5s parallel vs 6-9s sequential)
- **Integration-level**: Instantiates real `TradingGraph`, executes full graph with all agents
- **LLM-dependent**: Makes actual LLM API calls (no mocking)
- **Data-dependent**: Requires real benchmark OHLCV data for meaningful results
- **Timing-based**: Success criteria based on execution time thresholds

### Why This Test Should Be Excluded from Fast CI Gate

1. **Not a unit test**: Tests full system integration with real LLM calls
2. **Requires external data**: Depends on benchmark files not in version control
3. **Slow execution**: Takes 4-9 seconds minimum (potentially longer with API latency)
4. **Timing-sensitive**: May have flaky results depending on API response times
5. **High cost**: Makes multiple real LLM API calls per run

### Existing Pytest Markers

From `pytest.ini`:
- `slow`: marks tests as slow (deselect with `-m "not slow"`)
- `integration`: marks tests as integration tests
- `api`: marks tests requiring API calls
- `vision`: marks tests using vision-capable LLMs

---

## Recommended Solution

Mark the test with **`@pytest.mark.integration`** because:
- The test validates system-level integration behavior (parallel graph execution)
- It requires multiple components working together (TradingGraph, all agents, LLM providers)
- It makes real API calls
- The CI gate already excludes `integration` tests: `-m "not integration and not slow"`

**Alternative:** `@pytest.mark.slow` would also work, but `integration` more accurately describes the test's nature.

**Not recommended:** Adding `@pytest.mark.api` alone wouldn't exclude it from the current CI gate filter.

---

## Definition of Done

- [ ] `tests/test_parallel_execution.py::test_parallel_execution` is marked with `@pytest.mark.integration`
- [ ] The test is excluded from collection when running the CI gate command
- [ ] The CI gate command `pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"` no longer fails due to this test
- [ ] Test remains available for manual execution and integration test suites

---

## Testing the Fix

```bash
# Verify test is excluded from CI gate
cd ~/repos/projects/QuantAgent
pytest tests/test_parallel_execution.py -v -m "not integration and not slow"
# Expected: 0 tests collected

# Verify test can still be run explicitly
pytest tests/test_parallel_execution.py -v
# Expected: test runs (may fail due to missing data, but that's acceptable for integration tests)

# Verify test is included in integration suite
pytest tests/test_parallel_execution.py -v -m integration
# Expected: test is collected
```

---

## Related Files

- `tests/test_parallel_execution.py` - Test file requiring marker
- `pytest.ini` - Marker definitions (no changes needed)
- `.github/workflows/main-ci-deploy.yml` - CI workflow using the gate command (no changes needed)

---

## Notes

- This is a minimal fix that addresses the immediate blocker
- The test remains valuable for performance regression testing in integration suites
- If benchmark data is added to the repo in the future, the marker should remain (test is inherently slow/integration-level)
- The test's timing-based assertions make it unsuitable for fast CI gates regardless of data availability
