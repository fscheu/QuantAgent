# Acceptance Tests: API Retry Logic Enhancement

**Issue ID:** QuantAgent-aao
**Type:** AC (Acceptance Tests)
**Created:** 2026-01-03
**Related:**
- [Requirements](../01_requirements/QuantAgent-aao-RQ-api-retry-enhancement.md)
- [Design](../03_design/QuantAgent-aao-DS-api-retry-enhancement.md)

---

## Overview

This document defines acceptance criteria and test cases for validating the enhanced `invoke_with_retry` function. All tests should be executable without real API calls (mocked).

---

## Test Categories

1. **Core Retry Functionality** - Basic retry behavior
2. **Exponential Backoff** - Wait time calculations
3. **Multi-Provider Support** - Error handling across providers
4. **Configuration** - Global and per-call config
5. **Backwards Compatibility** - Existing code continues to work
6. **Edge Cases** - Boundary conditions

---

## AC1: Core Retry Functionality

### AC1.1: Success on First Attempt

```
Given: A function that succeeds on first call
When: invoke_with_retry is called
Then:
  - Function is called exactly once
  - Result is returned immediately
  - No sleep is executed
  - No warning logs are emitted
```

**Test Data:**
```python
Input: mock_fn returns "success"
Expected: "success"
Call count: 1
Sleep calls: 0
```

### AC1.2: Success After Retries

```
Given: A function that fails twice then succeeds
When: invoke_with_retry is called with retries=5
Then:
  - Function is called 3 times total
  - Result from third call is returned
  - Warning logged for each retry attempt
  - Sleep called twice with appropriate wait times
```

**Test Data:**
```python
Input: mock_fn raises RateLimitError, RateLimitError, then returns "success"
Expected: "success"
Call count: 3
Warning logs: 2 (attempts 1 and 2)
```

### AC1.3: Max Retries Exceeded

```
Given: A function that always fails with retryable error
When: invoke_with_retry is called with retries=3
Then:
  - Function is called exactly 3 times
  - RuntimeError is raised with message containing "Max retries (3) exceeded"
  - Error log emitted on final failure
  - Sleep called 2 times (between attempts, not after last)
```

**Test Data:**
```python
Input: mock_fn always raises RateLimitError("rate limited")
Expected Exception: RuntimeError("Max retries (3) exceeded: rate limited")
Call count: 3
Sleep calls: 2
```

---

## AC2: Exponential Backoff

### AC2.1: Exponential Wait Time Calculation

```
Given: invoke_with_retry configured with base_wait=4, exponential_base=2, jitter=False
When: Function fails multiple times before succeeding
Then: Wait times follow exponential pattern: 4s, 8s, 16s, 32s, 60s (capped)
```

**Test Data:**
| Attempt | Expected Wait (jitter=False) |
|---------|------------------------------|
| 1       | 4.0s                         |
| 2       | 8.0s                         |
| 3       | 16.0s                        |
| 4       | 32.0s                        |
| 5       | 60.0s (capped)               |

**Verification:**
```python
# Assert sleep calls match expected values
mock_sleep.assert_has_calls([
    call(4.0),
    call(8.0),
    call(16.0),
])
```

### AC2.2: Max Wait Cap Applied

```
Given: invoke_with_retry with base_wait=2, max_wait=10, jitter=False
When: Function fails 5 times
Then: Wait times are capped at max_wait: 2s, 4s, 8s, 10s, 10s
```

**Test Data:**
```python
Input: retries=6, base_wait=2, max_wait=10, jitter=False
Expected waits: [2.0, 4.0, 8.0, 10.0, 10.0]
```

### AC2.3: Jitter Within Bounds

```
Given: invoke_with_retry with base_wait=4, jitter=True, jitter_factor=0.5
When: Wait time is calculated for attempt 1
Then: Wait time is between 3.0s and 5.0s (4 +/- 1)
```

**Test Data:**
```python
# base_wait=4, jitter_factor=0.5
# jitter_range = 4 * 0.5 = 2
# wait = 4 + uniform(-1, 1) = [3.0, 5.0]

# Run 100 iterations, all should be in range
for _ in range(100):
    wait = _calculate_wait_time(attempt=0, base_wait=4, ...)
    assert 3.0 <= wait <= 5.0
```

### AC2.4: Jitter Reproducibility in Tests

```
Given: Random seed is set before test
When: invoke_with_retry with jitter=True is called
Then: Wait times are deterministic and can be asserted
```

**Test Data:**
```python
random.seed(42)
# Now jitter calculations are reproducible
```

---

## AC3: Multi-Provider Support

### AC3.1: OpenAI RateLimitError Triggers Retry

```
Given: Function raises openai.RateLimitError
When: invoke_with_retry is called
Then: Error is detected as retryable and retry occurs
```

**Test Data:**
```python
from openai import RateLimitError
mock_fn.side_effect = [RateLimitError("rate limited"), "success"]
result = invoke_with_retry(mock_fn)
assert result == "success"
assert mock_fn.call_count == 2
```

### AC3.2: Anthropic RateLimitError Triggers Retry

```
Given: Function raises anthropic.RateLimitError (mocked)
When: invoke_with_retry is called
Then: Error is detected as retryable and retry occurs
```

**Test Data:**
```python
# Mock Anthropic error
class MockAnthropicRateLimitError(Exception):
    pass
MockAnthropicRateLimitError.__module__ = "anthropic"
MockAnthropicRateLimitError.__name__ = "RateLimitError"

mock_fn.side_effect = [MockAnthropicRateLimitError(), "success"]
result = invoke_with_retry(mock_fn)
assert result == "success"
```

### AC3.3: Timeout Errors Trigger Retry

```
Given: Function raises timeout-related error
When: invoke_with_retry is called
Then: Error is detected as retryable and retry occurs
```

**Test Data (multiple error types):**
```python
# Test each timeout type
timeout_errors = [
    requests.exceptions.Timeout("timeout"),
    httpx.TimeoutException("timeout"),
    openai.APITimeoutError("timeout"),
]
for error in timeout_errors:
    mock_fn.side_effect = [error, "success"]
    result = invoke_with_retry(mock_fn)
    assert result == "success"
```

### AC3.4: Connection Errors Trigger Retry

```
Given: Function raises connection-related error
When: invoke_with_retry is called
Then: Error is detected as retryable and retry occurs
```

**Test Data:**
```python
connection_errors = [
    requests.exceptions.ConnectionError("connection failed"),
    httpx.ConnectError("connection failed"),
    openai.APIConnectionError("connection failed"),
]
```

### AC3.5: Authentication Error Does NOT Retry

```
Given: Function raises AuthenticationError (4xx)
When: invoke_with_retry is called
Then: Error is re-raised immediately without retry
```

**Test Data:**
```python
from openai import AuthenticationError
mock_fn.side_effect = AuthenticationError("Invalid API key")

with pytest.raises(AuthenticationError):
    invoke_with_retry(mock_fn, retries=5)

assert mock_fn.call_count == 1  # No retries
```

### AC3.6: BadRequestError Does NOT Retry

```
Given: Function raises BadRequestError (400)
When: invoke_with_retry is called
Then: Error is re-raised immediately without retry
```

**Test Data:**
```python
from openai import BadRequestError
mock_fn.side_effect = BadRequestError("Invalid request")

with pytest.raises(BadRequestError):
    invoke_with_retry(mock_fn, retries=5)

assert mock_fn.call_count == 1
```

### AC3.7: HTTP 429 Status Code Triggers Retry

```
Given: Function raises error with status_code=429
When: invoke_with_retry is called
Then: Error is detected as retryable
```

**Test Data:**
```python
class HTTPError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code

mock_fn.side_effect = [HTTPError(429), "success"]
result = invoke_with_retry(mock_fn)
assert result == "success"
```

### AC3.8: HTTP 5xx Status Codes Trigger Retry

```
Given: Function raises error with status_code in [500, 502, 503, 504]
When: invoke_with_retry is called
Then: Error is detected as retryable
```

**Test Data:**
```python
for status in [500, 502, 503, 504]:
    mock_fn.side_effect = [HTTPError(status), "success"]
    result = invoke_with_retry(mock_fn)
    assert result == "success"
```

---

## AC4: Configuration

### AC4.1: Global Config Applied by Default

```
Given: RETRY_CONFIG defines max_retries=5, base_wait=4
When: invoke_with_retry called without explicit params
Then: Uses values from RETRY_CONFIG
```

**Test Data:**
```python
# With RETRY_CONFIG["max_retries"] = 5
mock_fn.side_effect = [RateLimitError()] * 5 + ["success"]

# Should succeed because max_retries=5 allows 5 attempts
result = invoke_with_retry(mock_fn)
assert result == "success"
assert mock_fn.call_count == 6
```

### AC4.2: Per-Call Config Overrides Global

```
Given: RETRY_CONFIG defines max_retries=5
When: invoke_with_retry called with retries=2
Then: Uses retries=2 (explicit param wins)
```

**Test Data:**
```python
mock_fn.side_effect = [RateLimitError()] * 5

with pytest.raises(RuntimeError, match="Max retries \\(2\\) exceeded"):
    invoke_with_retry(mock_fn, retries=2)

assert mock_fn.call_count == 2  # Not 5
```

### AC4.3: Partial Override (Mix Global and Local)

```
Given: RETRY_CONFIG defines max_retries=5, base_wait=4
When: invoke_with_retry called with only base_wait=10
Then: Uses retries=5 (global), base_wait=10 (local)
```

**Test Data:**
```python
mock_fn.side_effect = [RateLimitError(), "success"]

with patch("time.sleep") as mock_sleep:
    result = invoke_with_retry(mock_fn, base_wait=10, jitter=False)

mock_sleep.assert_called_once_with(10.0)  # Local base_wait
assert mock_fn.call_count == 2
```

---

## AC5: Backwards Compatibility

### AC5.1: Legacy Signature Works

```
Given: Existing code calls invoke_with_retry(fn, arg, retries=3, wait_sec=4)
When: Code is executed with new implementation
Then: Function works, uses retries=3, base_wait=4.0
```

**Test Data:**
```python
# Legacy call pattern
mock_fn.side_effect = [RateLimitError(), "success"]

with patch("time.sleep") as mock_sleep:
    result = invoke_with_retry(mock_fn, retries=3, wait_sec=4)

assert result == "success"
mock_sleep.assert_called()  # Should have waited
```

### AC5.2: Deprecation Warning for wait_sec

```
Given: Code uses wait_sec parameter
When: invoke_with_retry is called
Then: Deprecation warning is logged
```

**Test Data:**
```python
with pytest.warns(DeprecationWarning, match="wait_sec is deprecated"):
    invoke_with_retry(mock_fn, wait_sec=4)
```

### AC5.3: Existing Agent Calls Work

```
Given: Calls in trend_agent.py, pattern_agent.py use current signature
When: Agents are executed
Then: No changes required, agents work correctly
```

**Verification:**
- Run existing test suite
- No test failures related to invoke_with_retry

---

## AC6: Edge Cases

### AC6.1: Zero Retries

```
Given: invoke_with_retry called with retries=1
When: First call fails
Then: RuntimeError raised immediately (no actual retries)
```

**Test Data:**
```python
mock_fn.side_effect = RateLimitError()

with pytest.raises(RuntimeError):
    invoke_with_retry(mock_fn, retries=1)

assert mock_fn.call_count == 1
```

### AC6.2: Very Large Max Wait

```
Given: max_wait=3600 (1 hour)
When: Exponential backoff exceeds max_wait
Then: Wait is capped at 3600s
```

### AC6.3: Jitter Does Not Produce Negative Wait

```
Given: base_wait=0.1, jitter=True, jitter_factor=1.0
When: Wait time calculated
Then: Wait time is always >= 0.1s (floor applied)
```

**Test Data:**
```python
for _ in range(1000):
    wait = _calculate_wait_time(attempt=0, base_wait=0.1, jitter=True, jitter_factor=1.0)
    assert wait >= 0.1
```

### AC6.4: Function Returns None Successfully

```
Given: Function returns None (valid result)
When: invoke_with_retry is called
Then: None is returned, not treated as failure
```

**Test Data:**
```python
mock_fn.return_value = None
result = invoke_with_retry(mock_fn)
assert result is None
assert mock_fn.call_count == 1
```

### AC6.5: Function With Args and Kwargs

```
Given: Function requires positional and keyword arguments
When: invoke_with_retry(fn, arg1, arg2, kwarg1=val1) is called
Then: Arguments are passed correctly to fn
```

**Test Data:**
```python
mock_fn.return_value = "success"
invoke_with_retry(mock_fn, "pos1", "pos2", key1="val1", key2="val2")

mock_fn.assert_called_once_with("pos1", "pos2", key1="val1", key2="val2")
```

---

## AC7: Logging

### AC7.1: Warning on Each Retry

```
Given: Function fails twice then succeeds
When: invoke_with_retry is called
Then: Two WARNING logs emitted with retry details
```

**Expected Log Format:**
```
WARNING - quantagent.agent_utils - Retry 1/5 after RateLimitError: waiting 2.00s
WARNING - quantagent.agent_utils - Retry 2/5 after RateLimitError: waiting 4.00s
```

**Verification:**
```python
with caplog.at_level(logging.WARNING):
    invoke_with_retry(mock_fn)

assert "Retry 1/5" in caplog.text
assert "Retry 2/5" in caplog.text
```

### AC7.2: Error on Final Failure

```
Given: All retries exhausted
When: RuntimeError is raised
Then: ERROR log emitted before exception
```

**Expected Log Format:**
```
ERROR - quantagent.agent_utils - Max retries (5) exceeded: rate limited
```

### AC7.3: No Info/Debug Logs on Success

```
Given: Function succeeds on first try
When: invoke_with_retry is called
Then: No logs emitted at WARNING or higher
```

---

## Test Matrix Summary

| Category | Test ID | Priority | Automated |
|----------|---------|----------|-----------|
| Core | AC1.1 | P0 | Yes |
| Core | AC1.2 | P0 | Yes |
| Core | AC1.3 | P0 | Yes |
| Backoff | AC2.1 | P0 | Yes |
| Backoff | AC2.2 | P0 | Yes |
| Backoff | AC2.3 | P1 | Yes |
| Backoff | AC2.4 | P1 | Yes |
| Multi-Provider | AC3.1 | P0 | Yes |
| Multi-Provider | AC3.2 | P0 | Yes |
| Multi-Provider | AC3.3 | P1 | Yes |
| Multi-Provider | AC3.4 | P1 | Yes |
| Multi-Provider | AC3.5 | P0 | Yes |
| Multi-Provider | AC3.6 | P1 | Yes |
| Multi-Provider | AC3.7 | P1 | Yes |
| Multi-Provider | AC3.8 | P1 | Yes |
| Config | AC4.1 | P0 | Yes |
| Config | AC4.2 | P0 | Yes |
| Config | AC4.3 | P1 | Yes |
| Compat | AC5.1 | P0 | Yes |
| Compat | AC5.2 | P2 | Yes |
| Compat | AC5.3 | P0 | Yes |
| Edge | AC6.1-AC6.5 | P1 | Yes |
| Logging | AC7.1-AC7.3 | P2 | Yes |

**Priority Legend:**
- P0: Must pass before merge
- P1: Should pass before merge
- P2: Nice to have, can be deferred

---

## Definition of Done Checklist

- [ ] All P0 tests pass
- [ ] All P1 tests pass
- [ ] Existing test suite passes (no regressions)
- [ ] Code coverage >80% on new code
- [ ] Manual verification: existing agent calls work unchanged
- [ ] Logging output matches expected format

---

## References

- [Requirements](../01_requirements/QuantAgent-aao-RQ-api-retry-enhancement.md)
- [Design](../03_design/QuantAgent-aao-DS-api-retry-enhancement.md)
- pytest documentation for mocking and fixtures
