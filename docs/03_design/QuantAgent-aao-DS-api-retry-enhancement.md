# Design: API Retry Logic Enhancement

**Issue ID:** QuantAgent-aao
**Type:** DS (Design)
**Created:** 2026-01-03
**Related:** [QuantAgent-aao-RQ-api-retry-enhancement.md](../01_requirements/QuantAgent-aao-RQ-api-retry-enhancement.md)

---

## Overview

This document specifies the technical design for enhancing `invoke_with_retry` in `quantagent/agent_utils.py` to support exponential backoff with jitter, multi-provider error handling, and configurable parameters.

---

## Current Implementation Analysis

### File: `quantagent/agent_utils.py`

```python
def invoke_with_retry(
    call_fn: Callable[..., T], *args, retries: int = 3, wait_sec: int = 4, **kwargs
) -> T:
```

**Current behavior:**
- Fixed wait time (`wait_sec=4`) between all retries
- Only catches `openai.RateLimitError` explicitly
- Generic `Exception` catch-all with same retry behavior
- Uses `print()` for logging
- Returns first successful result or raises `RuntimeError`

**Usage locations (9 calls):**
- `decision_agent.py`: 1 call
- `trend_agent.py`: 3 calls
- `pattern_agent.py`: 3 calls
- `indicator_agent.py`: 1 call

---

## Proposed Design

### 1. Enhanced Function Signature

```python
def invoke_with_retry(
    call_fn: Callable[..., T],
    *args,
    retries: int | None = None,
    base_wait: float | None = None,
    max_wait: float | None = None,
    jitter: bool | None = None,
    **kwargs
) -> T:
```

**Design decisions:**
- Parameters default to `None` to enable fallback to global config
- `wait_sec` renamed to `base_wait` for clarity (breaking change avoided via deprecation)
- Add `max_wait` cap for exponential growth
- Add `jitter` toggle for randomization

### 2. Configuration in `default_config.py`

```python
# Retry configuration for LLM API calls
RETRY_CONFIG = {
    "max_retries": 5,          # Maximum retry attempts
    "base_wait": 4.0,          # Base wait time in seconds
    "max_wait": 60.0,          # Maximum wait time cap
    "exponential_base": 2,     # Multiplier for exponential backoff
    "jitter": True,            # Add randomness to prevent thundering herd
    "jitter_factor": 0.5,      # Jitter range: wait * (1 +/- jitter_factor/2)
}
```

### 3. Exponential Backoff Algorithm

```
wait_time = min(base_wait * (exponential_base ^ attempt), max_wait)

if jitter:
    jitter_range = wait_time * jitter_factor
    wait_time = wait_time + random.uniform(-jitter_range/2, jitter_range/2)
    wait_time = max(0.1, wait_time)  # Floor to prevent negative/zero waits
```

**Example progression (base=4, exp=2, max=60):**

| Attempt | Base Wait | With Jitter (0.5 factor) |
|---------|-----------|--------------------------|
| 1       | 4s        | 3s - 5s                 |
| 2       | 8s        | 6s - 10s                |
| 3       | 16s       | 12s - 20s               |
| 4       | 32s       | 24s - 40s               |
| 5       | 60s (cap) | 45s - 75s (capped)      |
| 6+      | 60s (cap) | 45s - 75s (capped)      |

### 4. Multi-Provider Error Handling

**Retryable errors (will trigger retry):**
```python
RETRYABLE_EXCEPTIONS = (
    # OpenAI
    "openai.RateLimitError",
    "openai.APITimeoutError",
    "openai.APIConnectionError",

    # Anthropic
    "anthropic.RateLimitError",
    "anthropic.APITimeoutError",
    "anthropic.APIConnectionError",

    # Generic HTTP
    "requests.exceptions.Timeout",
    "requests.exceptions.ConnectionError",
    "httpx.TimeoutException",
    "httpx.ConnectError",
)
```

**Non-retryable errors (fail immediately):**
- `openai.AuthenticationError` (4xx)
- `anthropic.AuthenticationError` (4xx)
- `openai.BadRequestError` (400)
- `anthropic.BadRequestError` (400)
- Any error with HTTP status 4xx except 429

**Implementation approach:**
```python
def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error should trigger a retry."""
    error_type = f"{type(error).__module__}.{type(error).__name__}"

    # Check explicit retryable types
    if any(retryable in error_type for retryable in RETRYABLE_ERROR_TYPES):
        return True

    # Check HTTP status if available
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        # 429 (rate limit) and 5xx are retryable
        return status_code == 429 or status_code >= 500

    # Generic network/timeout errors are retryable
    if "timeout" in error_type.lower() or "connection" in error_type.lower():
        return True

    return False
```

### 5. Logging Strategy

**Replace `print()` with structured logging:**

```python
import logging

logger = logging.getLogger(__name__)

# On retry:
logger.warning(
    "Retry %d/%d after %s: waiting %.2fs",
    attempt + 1,
    max_retries,
    type(error).__name__,
    wait_time,
    extra={
        "retry_attempt": attempt + 1,
        "max_retries": max_retries,
        "error_type": type(error).__name__,
        "wait_seconds": wait_time,
    }
)

# On final failure:
logger.error(
    "Max retries (%d) exceeded: %s",
    max_retries,
    str(error),
    extra={
        "max_retries": max_retries,
        "final_error": str(error),
        "error_type": type(error).__name__,
    }
)
```

---

## Flow Diagram

```
invoke_with_retry(call_fn, *args, **kwargs)
                    |
                    v
    +---------------------------+
    | Load config (global/local)|
    | Resolve: retries, base,   |
    | max_wait, jitter          |
    +---------------------------+
                    |
                    v
    +---------------------------+
    |   attempt = 0             |
    +---------------------------+
                    |
        +----------+----------+
        |                     |
        v                     |
    +---------------------------+
    | Try: call_fn(*args)       |
    +---------------------------+
        |           |
     success      error
        |           |
        v           v
    +-------+   +---------------------------+
    | return|   | Is error retryable?       |
    | result|   +---------------------------+
    +-------+       |           |
                   yes          no
                    |           |
                    v           v
    +---------------------------+   +------------------+
    | attempt < max_retries?    |   | raise immediately|
    +---------------------------+   +------------------+
        |           |
       yes          no
        |           |
        v           v
    +---------------------------+   +------------------+
    | Calculate wait_time:      |   | raise RuntimeError|
    | base * 2^attempt          |   | "Max retries     |
    | Apply max_wait cap        |   |  exceeded"       |
    | Apply jitter if enabled   |   +------------------+
    +---------------------------+
                    |
                    v
    +---------------------------+
    | log.warning(retry info)   |
    | time.sleep(wait_time)     |
    | attempt += 1              |
    +---------------------------+
                    |
                    +-----> back to "Try: call_fn"
```

---

## Module Structure

### Modified Files

**`quantagent/agent_utils.py`**
```python
"""
Shared utilities for agent implementations.
Provides centralized retry logic with exponential backoff.
"""

import logging
import random
import time
from typing import Any, Callable, TypeVar

from quantagent.default_config import RETRY_CONFIG

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Error types that trigger retry
RETRYABLE_ERROR_PATTERNS = (
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "Timeout",
    "ConnectionError",
    "ConnectError",
)

def _is_retryable_error(error: Exception) -> bool:
    """Determine if error should trigger retry."""
    ...

def _calculate_wait_time(
    attempt: int,
    base_wait: float,
    max_wait: float,
    exponential_base: int,
    jitter: bool,
    jitter_factor: float,
) -> float:
    """Calculate wait time with exponential backoff and optional jitter."""
    ...

def invoke_with_retry(
    call_fn: Callable[..., T],
    *args,
    retries: int | None = None,
    base_wait: float | None = None,
    max_wait: float | None = None,
    jitter: bool | None = None,
    **kwargs
) -> T:
    """
    Centralized retry wrapper with exponential backoff.

    Handles rate limiting and transient errors across multiple LLM providers
    (OpenAI, Anthropic, Qwen). Uses exponential backoff with jitter.

    Args:
        call_fn: Function to call (typically llm.invoke)
        *args: Positional arguments for call_fn
        retries: Max retry attempts (default: from RETRY_CONFIG)
        base_wait: Base wait time in seconds (default: from RETRY_CONFIG)
        max_wait: Maximum wait time cap (default: from RETRY_CONFIG)
        jitter: Add randomness to wait time (default: from RETRY_CONFIG)
        **kwargs: Keyword arguments for call_fn

    Returns:
        Result from call_fn if successful

    Raises:
        RuntimeError: If all retries exhausted
        Exception: Re-raised if error is not retryable

    Example:
        >>> result = invoke_with_retry(llm.invoke, messages)
        >>> result = invoke_with_retry(llm.invoke, messages, retries=7, base_wait=4)
    """
    ...
```

**`quantagent/default_config.py`**
```python
# Add to existing file:

# Retry configuration for LLM API calls
RETRY_CONFIG = {
    "max_retries": 5,
    "base_wait": 2.0,
    "max_wait": 60.0,
    "exponential_base": 2,
    "jitter": True,
    "jitter_factor": 0.5,
}
```

---

## Backwards Compatibility

### Preserved Behavior

1. **Signature compatibility**: `retries` parameter still works
2. **Default behavior**: Without explicit params, uses new defaults (slightly different timing)
3. **Return type**: Same `T` type variable
4. **Exception type**: Still raises `RuntimeError` on exhaustion

### Breaking Changes (Minor)

1. **`wait_sec` parameter**: Deprecated in favor of `base_wait`
   - Mitigation: Accept both, log deprecation warning if `wait_sec` used

2. **Wait timing**: Now exponential instead of fixed
   - Mitigation: Document in release notes; generally an improvement

### Deprecation Handling

```python
def invoke_with_retry(
    call_fn: Callable[..., T],
    *args,
    retries: int | None = None,
    wait_sec: int | None = None,  # DEPRECATED
    base_wait: float | None = None,
    **kwargs
) -> T:
    if wait_sec is not None:
        logger.warning(
            "wait_sec is deprecated, use base_wait instead"
        )
        if base_wait is None:
            base_wait = float(wait_sec)
    ...
```

---

## Testing Strategy

### Unit Test Structure

```
tests/
  test_agent_utils.py  (new or extend existing)
    - test_invoke_with_retry_success_first_try
    - test_invoke_with_retry_success_after_retries
    - test_invoke_with_retry_max_retries_exceeded
    - test_exponential_backoff_calculation
    - test_jitter_within_bounds
    - test_max_wait_cap_applied
    - test_retryable_error_detection
    - test_non_retryable_error_immediate_fail
    - test_custom_config_override
    - test_backwards_compatibility_wait_sec
```

### Mock Strategy

```python
from unittest.mock import Mock, patch

def test_exponential_backoff():
    mock_fn = Mock(side_effect=[
        RateLimitError("rate limited"),
        RateLimitError("rate limited"),
        "success"
    ])

    with patch("time.sleep") as mock_sleep:
        result = invoke_with_retry(mock_fn, retries=3, base_wait=2, jitter=False)

    assert result == "success"
    assert mock_fn.call_count == 3
    # Verify exponential backoff: 2, 4 seconds
    mock_sleep.assert_any_call(2.0)
    mock_sleep.assert_any_call(4.0)
```

---

## Error Messages

### Retry Warning
```
WARNING - quantagent.agent_utils - Retry 2/5 after RateLimitError: waiting 4.23s
```

### Final Failure
```
ERROR - quantagent.agent_utils - Max retries (5) exceeded: Rate limit exceeded
```

### Non-Retryable Error
```
ERROR - quantagent.agent_utils - Non-retryable error (AuthenticationError): Invalid API key
```

---

## Dependencies

### No New External Dependencies

All functionality uses stdlib modules:
- `time` - sleep
- `logging` - structured logging
- `random` - jitter calculation
- `typing` - type hints

### Import Changes in `agent_utils.py`

```python
# Remove:
from openai import RateLimitError

# Add:
import logging
import random
from quantagent.default_config import RETRY_CONFIG
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Longer total wait on repeated failures | Medium | max_wait cap prevents excessive delays |
| Jitter introduces non-determinism | Low | Seed random in tests for reproducibility |
| Import errors if provider not installed | Low | Error pattern matching (strings) vs direct imports |
| Log spam on frequent retries | Low | WARNING level appropriate; can be filtered |

---

## Out of Scope (Documented for Future)

1. **Circuit breaker**: Not implemented; could add if retry storms become an issue
2. **Provider fallback**: Not implemented; would require significant changes to agent architecture
3. **Retry metrics/telemetry**: Not implemented; could add OpenTelemetry spans later
4. **Per-error-type retry config**: All retryable errors use same backoff

---

## References

- [Requirements Document](../01_requirements/QuantAgent-aao-RQ-api-retry-enhancement.md)
- [AWS Exponential Backoff](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
- Current code: `quantagent/agent_utils.py`
