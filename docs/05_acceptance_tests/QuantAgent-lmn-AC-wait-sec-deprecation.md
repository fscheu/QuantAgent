# Acceptance Criteria: Remove wait_sec Deprecation

**Issue ID:** QuantAgent-lmn  
**Related:** [QuantAgent-lmn-RQ-wait-sec-deprecation.md](../01_requirements/QuantAgent-lmn-RQ-wait-sec-deprecation.md)

---

## AC1: Function Signature Updated

**Given** the `invoke_with_retry` function in `quantagent/agent_utils.py`  
**When** inspecting the function signature  
**Then** the `wait_sec` parameter should not be present

---

## AC2: Deprecation Logic Removed

**Given** the `invoke_with_retry` function implementation  
**When** reading the function body  
**Then** there should be no code handling `wait_sec` parameter  
**And** no deprecation warning should be emitted

---

## AC3: Tests Updated

**Given** the test file `tests/test_agent_utils_retry.py`  
**When** searching for `wait_sec` usage  
**Then** no test should use `wait_sec` parameter  
**And** all relevant tests should use `base_wait` instead

---

## AC4: Deprecation Test Removed

**Given** the test file `tests/test_agent_utils_retry.py`  
**When** looking for deprecation warning tests  
**Then** `test_deprecation_warning_for_wait_sec` should not exist

---

## AC5: Test Suite Passes

**Given** all changes have been applied  
**When** running the test suite  
**Then** all tests should pass without errors  
**And** no deprecation warnings should appear in test output

---

## AC6: No Regression

**Given** existing functionality using `base_wait`  
**When** executing retry logic  
**Then** behavior should remain unchanged from before

---

## Validation Commands

```bash
# Search for wait_sec in codebase
grep -r "wait_sec" quantagent/ tests/ --include="*.py"
# Should return no results

# Run tests
pytest tests/test_agent_utils_retry.py -v

# Run full test suite
pytest
```
