# Implementation: API Retry Logic Enhancement

**Issue ID:** QuantAgent-aao  
**Type:** IM (Implementation)  
**Created:** 2026-01-03  
**Related:**
- [Requirements](../01_requirements/QuantAgent-aao-RQ-api-retry-enhancement.md)
- [Design](../03_design/QuantAgent-aao-DS-api-retry-enhancement.md)
- [Acceptance Tests](../05_acceptance_tests/QuantAgent-aao-AC-api-retry-enhancement.md)

---

## Summary

Enhanced `invoke_with_retry` in `quantagent/agent_utils.py` to support:
- True exponential backoff with jitter
- Multi-provider error handling (OpenAI, Anthropic, generic HTTP)
- Configurable retry parameters via `default_config.py`
- Structured logging via Python's `logging` module
- Backwards compatibility with existing code

---

## Changes Made

### 1. `quantagent/default_config.py`

Added `RETRY_CONFIG` dictionary with default retry parameters:

```python
RETRY_CONFIG = {
    "max_retries": 5,          # Maximum retry attempts
    "base_wait": 2.0,          # Base wait time in seconds
    "max_wait": 60.0,          # Maximum wait time cap
    "exponential_base": 2,     # Multiplier for exponential backoff
    "jitter": True,            # Add randomness to prevent thundering herd
    "jitter_factor": 0.5,      # Jitter range: wait * (1 +/- jitter_factor/2)
}
```

**Rationale:** Centralized configuration allows easy tuning without modifying code.

---

### 2. `quantagent/agent_utils.py`

#### Enhanced Imports
```python
import logging
import random
from quantagent.default_config import RETRY_CONFIG
```

#### New Helper Functions

##### `_is_retryable_error(error: Exception) -> bool`
Determines if an error should trigger a retry based on:
- Error type pattern matching (e.g., "RateLimitError", "Timeout")
- HTTP status code (429, 5xx are retryable; 4xx are not)
- Generic network/connection error names

##### `_calculate_wait_time(...) -> float`
Calculates wait time with exponential backoff:
```
wait = base_wait * (exponential_base ^ attempt)
wait = min(wait, max_wait)

if jitter:
    wait += random.uniform(-jitter_range/2, +jitter_range/2)
    wait = max(0.1, wait)  # Floor to prevent negative waits
```

#### Enhanced `invoke_with_retry` Function

**Signature changes:**
- Added parameters: `base_wait`, `max_wait`, `jitter`
- Kept `wait_sec` for backwards compatibility (logs deprecation warning)
- All new params default to `None` to enable global config fallback

**Behavior changes:**
- Uses exponential backoff instead of fixed wait time
- Checks if errors are retryable before retrying
- Non-retryable errors fail immediately without consuming retries
- Uses structured logging (`logger.warning`, `logger.error`) instead of `print()`
- Includes extra context in log entries for structured logging systems

**Algorithm:**
1. Resolve config (explicit params override global config)
2. For each attempt (up to `max_retries`):
   - Try calling `call_fn(*args, **kwargs)`
   - On success: return result
   - On error:
     - Check if retryable via `_is_retryable_error()`
     - If not retryable: log error and re-raise immediately
     - If retries exhausted: log error and raise `RuntimeError`
     - Otherwise: calculate wait time, log warning, sleep, retry

---

### 3. `tests/test_agent_utils_retry.py`

Created comprehensive test suite with 35+ test cases covering:

**Core Retry Functionality (AC1)**
- Success on first attempt (no retries)
- Success after retries
- Max retries exceeded

**Exponential Backoff (AC2)**
- Exponential calculation correctness
- Max wait cap applied
- Jitter within expected bounds
- Jitter floor prevents negative waits

**Multi-Provider Support (AC3)**
- OpenAI errors trigger retry
- Anthropic errors trigger retry
- Timeout/connection errors trigger retry
- Authentication errors do NOT retry
- BadRequest errors do NOT retry
- HTTP 429 and 5xx trigger retry

**Configuration (AC4)**
- Global config used by default
- Per-call config overrides global
- Partial overrides work correctly

**Backwards Compatibility (AC5)**
- Legacy `wait_sec` parameter works
- Deprecation warning logged
- Existing agent calls work unchanged

**Edge Cases (AC6)**
- Zero retries (retries=1)
- Functions returning None
- Functions with args/kwargs

**Logging (AC7)**
- Warning on each retry
- Error on final failure
- No logs on immediate success

---

## Testing

### How to Test

#### Run New Tests Only
```bash
pytest tests/test_agent_utils_retry.py -v
```

#### Run All Tests (Regression Check)
```bash
pytest -v
```

#### With Coverage
```bash
pytest tests/test_agent_utils_retry.py --cov=quantagent.agent_utils --cov-report=term-missing
```

### Expected Results

All tests in `test_agent_utils_retry.py` should pass:
- 35+ test cases covering P0 and P1 acceptance criteria
- Coverage >80% on modified code

Existing tests should continue passing (backwards compatibility).

---

## Backwards Compatibility

### What Still Works

✅ Existing code using `invoke_with_retry` continues to work without changes:
```python
# Old code in trend_agent.py
invoke_with_retry(graph_llm.invoke, messages, retries=3, wait_sec=4)
```

✅ `wait_sec` parameter mapped to `base_wait` internally  
✅ Same exception types raised (`RuntimeError` on exhaustion)  
✅ Same function signature structure

### What Changed (Improvements)

⚠️ Wait timing is now exponential instead of fixed  
⚠️ `wait_sec` parameter logs deprecation warning  
⚠️ Uses structured logging instead of `print()`  
⚠️ Default retries increased from 3 to 5 (can override)

**Migration Path:**
- No immediate changes required
- Optionally update to new parameter names:
  ```python
  # Old style (still works)
  invoke_with_retry(fn, retries=3, wait_sec=4)
  
  # New style (recommended)
  invoke_with_retry(fn, retries=3, base_wait=4.0)
  ```

---

## Deployment Notes

### Pre-Deployment Checklist

- [x] All new tests pass
- [x] Existing tests pass (regression)
- [x] Code formatted with black/isort
- [x] No linting errors (flake8)
- [x] Type hints validated (mypy)
- [x] Documentation updated

### Post-Deployment Monitoring

Monitor logs for:
- Retry frequency (too many retries may indicate API issues)
- Deprecation warnings (agents using `wait_sec`)
- Non-retryable errors (may indicate config issues)

### Rollback Plan

If issues arise:
1. Git revert to previous commit
2. Redeploy
3. Existing code continues working (no breaking changes)

---

## Known Limitations

1. **No circuit breaker**: Repeated failures won't temporarily disable the endpoint
2. **No per-error-type config**: All retryable errors use same backoff strategy
3. **No metrics**: Retry counts/timing not tracked (can add later with OpenTelemetry)

---

## Future Enhancements

Potential improvements for future iterations:
1. Circuit breaker pattern for repeated failures
2. Provider fallback (auto-switch to backup LLM on exhaustion)
3. Metrics/telemetry via OpenTelemetry
4. Per-error-type retry configuration
5. Adaptive backoff based on API response headers

---

## References

- Python logging: https://docs.python.org/3/library/logging.html
- Exponential backoff with jitter: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- OpenAI error types: https://platform.openai.com/docs/guides/error-codes
- Anthropic error types: https://docs.anthropic.com/en/api/errors
