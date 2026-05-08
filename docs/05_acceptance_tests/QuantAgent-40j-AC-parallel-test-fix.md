# Acceptance Criteria: Fix Missing Benchmark Fixture for Parallel Execution Test

**Issue ID:** QuantAgent-40j  
**Related:** [QuantAgent-40j-RQ-parallel-test-fix.md](../01_requirements/QuantAgent-40j-RQ-parallel-test-fix.md)

---

## Success Criteria

### 1. Test Marker Applied

**Given:** The file `tests/test_parallel_execution.py` exists  
**When:** The test function `test_parallel_execution` is inspected  
**Then:**
- The function has the decorator `@pytest.mark.integration` applied
- The decorator is placed before the function definition
- No other markers are required

### 2. CI Gate Exclusion

**Given:** The repository is in a clean state with the marker applied  
**When:** Running the CI gate command:
```bash
pytest tests/ -v --tb=short --maxfail=10 -m "not integration and not slow"
```
**Then:**
- `test_parallel_execution` is **not collected** (excluded from run)
- The command completes without errors from this specific test
- Exit code is 0 (or reflects failures from other legitimate unit tests)

### 3. Explicit Execution Works

**Given:** The marker is applied  
**When:** Running the test explicitly:
```bash
pytest tests/test_parallel_execution.py::test_parallel_execution -v
```
**Then:**
- The test is collected and attempts to run
- Whether it passes or fails is irrelevant (it may fail due to missing benchmark data)
- The test is not skipped or marked as deselected

### 4. Integration Suite Inclusion

**Given:** The marker is applied  
**When:** Running integration tests:
```bash
pytest tests/ -v -m integration
```
**Then:**
- `test_parallel_execution` is collected and included
- The test appears in the test collection list

### 5. Test Discovery

**Given:** The marker is applied  
**When:** Running `pytest --collect-only tests/test_parallel_execution.py`  
**Then:**
- The test is listed with marker `<IntegrationMarker>`
- Output shows: `<Function test_parallel_execution>`

---

## Negative Cases

### 1. Marker Not Applied
**When:** The marker is removed or commented out  
**Then:** The CI gate command collects the test and fails with `FileNotFoundError`

### 2. Wrong Marker Applied
**When:** Only `@pytest.mark.api` is applied (without `integration` or `slow`)  
**Then:** The CI gate command still collects the test (since the filter is `not integration and not slow`)

### 3. Typo in Marker
**When:** The marker is misspelled (e.g., `@pytest.mark.integraton`)  
**Then:** pytest fails with "unknown marker" error (due to `--strict-markers` in pytest.ini)

---

## Verification Commands

Run these commands to verify acceptance criteria:

```bash
# Change to repo directory
cd ~/repos/projects/QuantAgent

# AC1: Verify marker is present
grep -n "@pytest.mark.integration" tests/test_parallel_execution.py

# AC2: Verify CI gate excludes test
pytest tests/test_parallel_execution.py -v -m "not integration and not slow"
# Expected output: "collected 0 items" or similar

# AC3: Verify explicit execution works
pytest tests/test_parallel_execution.py::test_parallel_execution -v --collect-only
# Expected: test is collected

# AC4: Verify integration suite inclusion
pytest tests/test_parallel_execution.py -v -m integration --collect-only
# Expected: test is collected

# AC5: Verify marker is recognized
pytest tests/test_parallel_execution.py --collect-only
# Expected output includes: markers: integration
```

---

## Boundary Conditions

1. **Multiple markers**: If other markers are added later (e.g., `@pytest.mark.api`), the test should still be excluded from CI gate
2. **Pytest version**: Fix should work with pytest >= 7.0 (current repo version)
3. **Future CI changes**: If CI gate filter changes to include integration tests, this test may fail (acceptable behavior)

---

## Rollback Criteria

If the fix causes issues:
- The marker can be removed to restore original behavior
- No data migrations or complex rollback needed
- Single-line change, easily revertible

---

## Non-Functional Acceptance

- **Performance**: No impact (test is excluded from fast CI gate)
- **Maintainability**: Standard pytest marker, well-documented pattern
- **Documentation**: No additional docs needed beyond this AC file
