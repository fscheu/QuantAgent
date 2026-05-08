# Planning: Fix Missing Benchmark Fixture for Parallel Execution Test

**Issue ID:** QuantAgent-40j  
**Related:**
- [Requirements](../01_requirements/QuantAgent-40j-RQ-parallel-test-fix.md)
- [Acceptance Criteria](../05_acceptance_tests/QuantAgent-40j-AC-parallel-test-fix.md)

---

## Overview

This is a minimal fix to prevent `test_parallel_execution` from blocking the CI gate by marking it as an integration test. The fix requires a single-line decorator addition.

**Estimated effort:** 5 minutes  
**Risk level:** Very Low  
**Complexity:** Trivial

---

## Tasks

### Task 1: Add Integration Marker to Test
**Effort:** 2 minutes  
**Assignee:** Implementer agent

**Actions:**
1. Open `tests/test_parallel_execution.py`
2. Add `@pytest.mark.integration` decorator before `def test_parallel_execution():`
3. Ensure `import pytest` is present at the top of the file (already exists)

**File to modify:**
- `tests/test_parallel_execution.py` (line ~14, before function definition)

**Change:**
```python
# Before (line 14):
def test_parallel_execution():

# After:
@pytest.mark.integration
def test_parallel_execution():
```

**Validation:**
```bash
grep -A1 "@pytest.mark.integration" tests/test_parallel_execution.py
```

---

### Task 2: Verify CI Gate Exclusion
**Effort:** 2 minutes  
**Assignee:** Tester agent

**Actions:**
1. Run the exact CI gate command
2. Verify test is not collected
3. Verify no FileNotFoundError occurs

**Commands:**
```bash
cd ~/repos/projects/QuantAgent
pytest tests/test_parallel_execution.py -v -m "not integration and not slow"
```

**Expected output:**
```
collected 0 items
```

---

### Task 3: Verify Integration Suite Inclusion
**Effort:** 1 minute  
**Assignee:** Tester agent

**Actions:**
1. Run integration marker filter
2. Verify test is collected

**Commands:**
```bash
pytest tests/test_parallel_execution.py -v -m integration --collect-only
```

**Expected output:**
```
<Function test_parallel_execution>
1 test collected
```

---

## Dependencies

**None.** This is a self-contained change with no external dependencies.

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Test permanently excluded from all CI | Low | Very Low | Test can still be run manually and in integration suites |
| Typo in marker name | Low | Very Low | `--strict-markers` in pytest.ini will catch unknown markers |
| Wrong marker applied | Low | Very Low | Acceptance tests verify CI gate exclusion |

---

## Testing Strategy

### Pre-commit Testing
1. Verify marker syntax is correct
2. Run CI gate command locally
3. Verify test is excluded

### Post-merge Testing
- CI pipeline will validate the fix automatically
- Integration test suite can be run separately if needed

---

## Rollout Plan

1. **Implement:** Add the marker (Task 1)
2. **Verify:** Run local tests (Tasks 2-3)
3. **Commit:** Single commit with clear message
4. **Validate:** CI gate runs successfully

**No staged rollout needed** - this is a test configuration change with no production impact.

---

## Commit Strategy

**Single commit:**
```
fix(tests): Mark test_parallel_execution as integration test

- Add @pytest.mark.integration to test_parallel_execution
- Prevents CI gate failure due to missing benchmark data
- Test remains available for integration test suites

Resolves: QuantAgent-40j
Unblocks: QuantAgent-82t
```

**Files changed:**
- `tests/test_parallel_execution.py` (+1 line)

---

## Validation Checklist

Before marking complete:
- [ ] Marker added to test function
- [ ] CI gate command excludes test (0 items collected)
- [ ] Integration filter includes test (1 item collected)
- [ ] Test can be run explicitly
- [ ] No pytest warnings about unknown markers
- [ ] Acceptance criteria met (see AC document)

---

## Next Steps

After this fix is merged:
1. QuantAgent-82t can proceed to re-enable unit tests in CI
2. Consider adding benchmark data fixtures in the future (separate ticket)
3. Consider performance regression tracking for parallel execution (separate ticket)

---

## Notes

- This fix does not address the root cause (missing benchmark data)
- The test remains valuable for performance validation
- Future work may add proper fixtures or sample data
- The timing-based nature of the test makes it inherently unsuitable for fast CI gates
