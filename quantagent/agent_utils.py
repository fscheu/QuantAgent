"""
Shared utilities for agent implementations.

Provides centralized retry logic with exponential backoff and structured output helpers.
"""

import logging
import random
import time
from typing import Callable, TypeVar

from quantagent.default_config import RETRY_CONFIG

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Error patterns that trigger retry
RETRYABLE_ERROR_PATTERNS = (
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "Timeout",
    "ConnectionError",
    "ConnectError",
)


def _is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry.

    Args:
        error: The exception to check

    Returns:
        True if error should be retried, False otherwise
    """
    error_type = f"{type(error).__module__}.{type(error).__name__}"

    # Check explicit retryable patterns
    if any(pattern in error_type for pattern in RETRYABLE_ERROR_PATTERNS):
        return True

    # Check HTTP status if available
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        # 429 (rate limit) and 5xx are retryable
        return status_code == 429 or status_code >= 500

    # Generic network/timeout errors by name
    error_name_lower = type(error).__name__.lower()
    if "timeout" in error_name_lower or "connection" in error_name_lower:
        return True

    return False


def _calculate_wait_time(
    attempt: int,
    base_wait: float,
    max_wait: float,
    exponential_base: float,
    jitter: bool,
    jitter_factor: float,
) -> float:
    """
    Calculate wait time with exponential backoff and optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_wait: Base wait time in seconds
        max_wait: Maximum wait time cap
        exponential_base: Multiplier for exponential backoff
        jitter: Whether to add randomness
        jitter_factor: Jitter range as fraction of wait time

    Returns:
        Calculated wait time in seconds
    """
    # Exponential backoff: base * (exponential_base ^ attempt)
    wait = base_wait * (exponential_base**attempt)

    # Cap at max_wait
    wait = min(wait, max_wait)

    # Add jitter if enabled
    if jitter:
        jitter_range = wait * jitter_factor
        wait = wait + random.uniform(-jitter_range / 2, jitter_range / 2)
        # Floor to prevent negative/zero waits
        wait = max(0.1, wait)

    return wait


def invoke_with_retry(
    call_fn: Callable[..., T],
    *args,
    retries: int | None = None,
    wait_sec: float | None = None,
    base_wait: float | None = None,
    max_wait: float | None = None,
    jitter: bool | None = None,
    **kwargs,
) -> T:
    """
    Centralized retry wrapper with exponential backoff.

    Handles rate limiting and transient errors across multiple LLM providers
    (OpenAI, Anthropic, Qwen). Uses exponential backoff with jitter.

    Args:
        call_fn: Function to call (typically llm.invoke)
        *args: Positional arguments for call_fn
        retries: Max retry attempts (default: from RETRY_CONFIG)
        wait_sec: DEPRECATED - use base_wait instead
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
    # Handle deprecated wait_sec parameter
    if wait_sec is not None:
        logger.warning(
            "wait_sec parameter is deprecated, use base_wait instead",
            extra={"deprecated_param": "wait_sec"},
        )
        if base_wait is None:
            base_wait = float(wait_sec)

    # Resolve configuration (local override > global config)
    max_retries: int = retries if retries is not None else int(RETRY_CONFIG["max_retries"])
    resolved_base_wait: float = base_wait if base_wait is not None else float(RETRY_CONFIG["base_wait"])
    resolved_max_wait: float = max_wait if max_wait is not None else float(RETRY_CONFIG["max_wait"])
    resolved_jitter: bool = jitter if jitter is not None else bool(RETRY_CONFIG["jitter"])
    exponential_base: float = float(RETRY_CONFIG["exponential_base"])
    jitter_factor: float = float(RETRY_CONFIG["jitter_factor"])

    for attempt in range(max_retries):
        try:
            return call_fn(*args, **kwargs)
        except Exception as e:
            # Check if error is retryable
            if not _is_retryable_error(e):
                logger.error(
                    "Non-retryable error (%s): %s",
                    type(e).__name__,
                    str(e),
                    extra={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
                raise

            # Check if we have retries left
            if attempt >= max_retries - 1:
                logger.error(
                    "Max retries (%d) exceeded: %s",
                    max_retries,
                    str(e),
                    extra={
                        "max_retries": max_retries,
                        "final_error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                raise RuntimeError(f"Max retries ({max_retries}) exceeded: {str(e)}")

            # Calculate wait time and retry
            wait_time = _calculate_wait_time(
                attempt, resolved_base_wait, resolved_max_wait, exponential_base, resolved_jitter, jitter_factor
            )

            logger.warning(
                "Retry %d/%d after %s: waiting %.2fs",
                attempt + 1,
                max_retries,
                type(e).__name__,
                wait_time,
                extra={
                    "retry_attempt": attempt + 1,
                    "max_retries": max_retries,
                    "error_type": type(e).__name__,
                    "wait_seconds": wait_time,
                },
            )

            time.sleep(wait_time)

    # This should never be reached due to logic above, but satisfies mypy
    raise RuntimeError(f"Max retries ({max_retries}) exceeded")
