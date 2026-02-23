# Planning: API Retry Logic Enhancement

**Issue ID:** QuantAgent-aao
**Type:** PL (Planning)
**Created:** 2026-01-03
**Related:**
- [Requirements](../01_requirements/QuantAgent-aao-RQ-api-retry-enhancement.md)
- [Design](../03_design/QuantAgent-aao-DS-api-retry-enhancement.md)
- [Acceptance Tests](../05_acceptance_tests/QuantAgent-aao-AC-api-retry-enhancement.md)

---

## Overview

This document defines the implementation plan for enhancing `invoke_with_retry` with exponential backoff, multi-provider error handling, and configurable parameters.

**Estimated Total Effort:** 6-8 hours
**Risk Level:** Low (isolated change, backwards compatible)

---

## Task Breakdown

### Phase 1: Configuration Setup (1h)

#### Task 1.1: Add RETRY_CONFIG to default_config.py
**Effort:** 30 min
**File:** `quantagent/default_config.py`

**Actions:**
1. Add `RETRY_CONFIG` dictionary with default values
2. Add inline comments explaining each parameter

**Expected Output:**
```python
RETRY_CONFIG = {
    "max_retries": 5,
    "base_wait": 4.0,
    "max_wait": 60.0,
    "exponential_base": 2,
    "jitter": True,
    "jitter_factor": 0.5,
}
```

**Validation:**
- Import succeeds: `from quantagent.default_config import RETRY_CONFIG`
- Values are accessible: `RETRY_CONFIG["max_retries"] == 5`

**Dependencies:** None

---

#### Task 1.2: Add logging import to agent_utils.py
**Effort:** 15 min
**File:** `quantagent/agent_utils.py`

**Actions:**
1. Add `import logging` and `import random`
2. Create module-level logger: `logger = logging.getLogger(__name__)`
3. Add import for `RETRY_CONFIG`

**Expected Output:**
```python
import logging
import random
import time
from typing import Any, Callable, TypeVar

from quantagent.default_config import RETRY_CONFIG

logger = logging.getLogger(__name__)
```

**Validation:**
- File parses without errors
- Existing tests still pass

**Dependencies:** Task 1.1

---

### Phase 2: Core Implementation (2.5h)

#### Task 2.1: Implement _is_retryable_error helper
**Effort:** 45 min
**File:** `quantagent/agent_utils.py`

**Actions:**
1. Define `RETRYABLE_ERROR_PATTERNS` tuple
2. Implement `_is_retryable_error(error: Exception) -> bool`
3. Handle: error type name matching, HTTP status codes

**Expected Output:**
```python
RETRYABLE_ERROR_PATTERNS = (
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "Timeout",
    "ConnectionError",
    "ConnectError",
)

def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error should trigger a retry."""
    error_name = type(error).__name__

    # Check error name patterns
    if any(pattern in error_name for pattern in RETRYABLE_ERROR_PATTERNS):
        return True

    # Check HTTP status code if available
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        return status_code == 429 or status_code >= 500

    return False
```

**Validation:**
- Unit test for each retryable error type
- Unit test for non-retryable errors (Auth, BadRequest)

**Dependencies:** Task 1.2

---

#### Task 2.2: Implement _calculate_wait_time helper
**Effort:** 30 min
**File:** `quantagent/agent_utils.py`

**Actions:**
1. Implement exponential backoff calculation
2. Apply max_wait cap
3. Add jitter with configurable factor
4. Apply floor to prevent zero/negative waits

**Expected Output:**
```python
def _calculate_wait_time(
    attempt: int,
    base_wait: float,
    max_wait: float,
    exponential_base: int,
    jitter: bool,
    jitter_factor: float,
) -> float:
    """Calculate wait time with exponential backoff and optional jitter."""
    wait = min(base_wait * (exponential_base ** attempt), max_wait)

    if jitter:
        jitter_range = wait * jitter_factor
        wait = wait + random.uniform(-jitter_range / 2, jitter_range / 2)
        wait = max(0.1, wait)  # Floor

    return wait
```

**Validation:**
- Test exponential progression without jitter
- Test max_wait cap applied
- Test jitter within expected bounds

**Dependencies:** Task 1.2

---

#### Task 2.3: Refactor invoke_with_retry main function
**Effort:** 1h 15min
**File:** `quantagent/agent_utils.py`

**Actions:**
1. Update function signature with new parameters
2. Add `wait_sec` deprecation handling
3. Resolve config values (local override vs global default)
4. Update retry loop to use `_is_retryable_error`
5. Update retry loop to use `_calculate_wait_time`
6. Replace `print()` with `logger.warning()` and `logger.error()`
7. Update docstring with examples

**Expected Output:** See Design Document section "Module Structure"

**Validation:**
- AC1.1, AC1.2, AC1.3 pass
- Backwards compatibility: existing calls work

**Dependencies:** Tasks 2.1, 2.2

---

### Phase 3: Testing (2h)

#### Task 3.1: Create test file structure
**Effort:** 15 min
**File:** `tests/test_agent_utils.py` (new or extend)

**Actions:**
1. Create/update test file
2. Add pytest imports and fixtures
3. Add mock helpers for error classes

**Expected Output:**
```python
import pytest
import random
from unittest.mock import Mock, patch

from quantagent.agent_utils import (
    invoke_with_retry,
    _is_retryable_error,
    _calculate_wait_time,
)

@pytest.fixture
def mock_rate_limit_error():
    """Create a mock RateLimitError."""
    error = Exception("rate limited")
    error.__class__.__name__ = "RateLimitError"
    return error
```

**Dependencies:** Phase 2 complete

---

#### Task 3.2: Implement core retry tests
**Effort:** 45 min
**File:** `tests/test_agent_utils.py`

**Actions:**
1. Test success on first try (AC1.1)
2. Test success after retries (AC1.2)
3. Test max retries exceeded (AC1.3)

**Test Cases:**
```python
def test_invoke_with_retry_success_first_try():
    ...

def test_invoke_with_retry_success_after_retries():
    ...

def test_invoke_with_retry_max_retries_exceeded():
    ...
```

**Validation:** All P0 core tests pass

**Dependencies:** Task 3.1

---

#### Task 3.3: Implement backoff tests
**Effort:** 30 min
**File:** `tests/test_agent_utils.py`

**Actions:**
1. Test exponential wait calculation (AC2.1)
2. Test max_wait cap (AC2.2)
3. Test jitter bounds (AC2.3)

**Dependencies:** Task 3.1

---

#### Task 3.4: Implement multi-provider tests
**Effort:** 30 min
**File:** `tests/test_agent_utils.py`

**Actions:**
1. Test OpenAI errors retry (AC3.1)
2. Test non-retryable errors fail immediately (AC3.5)
3. Test HTTP status code handling (AC3.7, AC3.8)

**Dependencies:** Task 3.1

---

### Phase 4: Validation and Cleanup (1h)

#### Task 4.1: Run existing test suite
**Effort:** 15 min

**Actions:**
1. Run `pytest tests/`
2. Verify no regressions
3. Fix any failing tests

**Validation:** All existing tests pass

**Dependencies:** Phase 3 complete

---

#### Task 4.2: Verify backwards compatibility
**Effort:** 30 min

**Actions:**
1. Check each agent file imports work
2. Manually verify call patterns in agents still work
3. Run any integration tests if available

**Files to verify:**
- `quantagent/decision_agent.py`
- `quantagent/trend_agent.py`
- `quantagent/pattern_agent.py`
- `quantagent/indicator_agent.py`

**Dependencies:** Phase 3 complete

---

#### Task 4.3: Update docstrings and inline docs
**Effort:** 15 min
**File:** `quantagent/agent_utils.py`

**Actions:**
1. Ensure docstring has usage examples
2. Add inline comments for complex logic
3. Verify type hints are complete

**Dependencies:** Task 4.2

---

## Execution Order (Critical Path)

```
Task 1.1 (config)
    |
    v
Task 1.2 (imports)
    |
    +---> Task 2.1 (_is_retryable_error)
    |           |
    |           v
    +---> Task 2.2 (_calculate_wait_time)
                |
                v
          Task 2.3 (main function)
                |
                v
          Task 3.1 (test setup)
                |
    +-----------+-----------+
    |           |           |
    v           v           v
Task 3.2    Task 3.3    Task 3.4
(core)      (backoff)   (providers)
    |           |           |
    +-----------+-----------+
                |
                v
          Task 4.1 (run tests)
                |
                v
          Task 4.2 (verify compat)
                |
                v
          Task 4.3 (docs)
```

---

## Dependencies

### Internal Dependencies
- `quantagent/default_config.py` - Must be modified first
- `quantagent/agent_utils.py` - Main file to modify
- Existing test infrastructure in `tests/`

### External Dependencies
- None (all stdlib)

### Blocking Dependencies
- None - this change is self-contained

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Existing tests break | Low | High | Run test suite after each phase |
| Agent behavior changes subtly | Low | Medium | Verify each agent call pattern |
| Jitter causes test flakiness | Medium | Low | Seed random in tests |
| Import errors in agents | Low | High | Test imports before/after |

---

## Validation Checkpoints

### Checkpoint 1: After Phase 1
- [ ] `RETRY_CONFIG` importable
- [ ] No syntax errors in modified files
- [ ] Existing tests still pass

### Checkpoint 2: After Phase 2
- [ ] `invoke_with_retry` handles new parameters
- [ ] Helper functions work in isolation
- [ ] Manual test: function works end-to-end

### Checkpoint 3: After Phase 3
- [ ] All P0 acceptance tests pass
- [ ] All P1 acceptance tests pass
- [ ] Code coverage >80% on new code

### Checkpoint 4: After Phase 4
- [ ] Full test suite passes
- [ ] Each agent file verified
- [ ] Docstrings complete

---

## Rollout Strategy

1. **Development Branch**
   - Create branch: `feature/QuantAgent-aao-api-retry`
   - Implement all phases
   - Run full test suite

2. **Review**
   - Code review by human
   - Verify backwards compatibility
   - Check logging output format

3. **Merge**
   - Merge to main after approval
   - Monitor for any runtime issues

4. **Post-Merge**
   - Verify in staging/dev environment
   - Check agent logs for new format
   - Confirm no unexpected retry behavior

---

## Test Commands

```bash
# Run all tests
pytest tests/ -v

# Run only agent_utils tests
pytest tests/test_agent_utils.py -v

# Run with coverage
pytest tests/test_agent_utils.py --cov=quantagent/agent_utils --cov-report=term-missing

# Run specific test
pytest tests/test_agent_utils.py::test_invoke_with_retry_success_first_try -v
```

---

## Files Modified Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `quantagent/default_config.py` | Modify | Add RETRY_CONFIG |
| `quantagent/agent_utils.py` | Modify | Refactor invoke_with_retry |
| `tests/test_agent_utils.py` | Create/Modify | Add unit tests |

---

## Definition of Done

- [ ] All tasks completed
- [ ] All checkpoints validated
- [ ] All P0/P1 acceptance tests pass
- [ ] Existing test suite passes
- [ ] Code coverage >80%
- [ ] Docstrings updated
- [ ] Commit prepared with descriptive message
- [ ] Issue QuantAgent-aao ready to close

---

## References

- [Requirements](../01_requirements/QuantAgent-aao-RQ-api-retry-enhancement.md)
- [Design](../03_design/QuantAgent-aao-DS-api-retry-enhancement.md)
- [Acceptance Tests](../05_acceptance_tests/QuantAgent-aao-AC-api-retry-enhancement.md)
