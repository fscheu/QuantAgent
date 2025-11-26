"""
Shared utilities for agent implementations.

Provides centralized retry logic and structured output helpers.
"""

import time
from typing import Any, Callable, TypeVar

from openai import RateLimitError

T = TypeVar("T")


def invoke_with_retry(
    call_fn: Callable[..., T],
    *args,
    retries: int = 3,
    wait_sec: int = 4,
    **kwargs
) -> T:
    """
    Centralized retry wrapper with exponential backoff.

    Handles RateLimitError and generic exceptions gracefully.
    Used across pattern_agent, trend_agent, and any LLM/tool calls.

    Args:
        call_fn: Function to call (typically llm.invoke or tool.invoke)
        *args: Positional arguments for call_fn
        retries: Number of retry attempts (default: 3)
        wait_sec: Seconds to wait between retries (default: 4)
        **kwargs: Keyword arguments for call_fn

    Returns:
        Result from call_fn if successful

    Raises:
        RuntimeError: If all retries are exhausted
    """
    for attempt in range(retries):
        try:
            return call_fn(*args, **kwargs)
        except RateLimitError:
            if attempt < retries - 1:
                print(
                    f"Rate limit hit, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                )
                time.sleep(wait_sec)
            else:
                raise RuntimeError(
                    f"Max retries ({retries}) exceeded due to rate limiting"
                )
        except Exception as e:
            if attempt < retries - 1:
                print(
                    f"Error: {type(e).__name__}, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
                )
                time.sleep(wait_sec)
            else:
                raise RuntimeError(f"Max retries ({retries}) exceeded: {str(e)}")
