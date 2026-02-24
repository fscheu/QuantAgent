"""
Unit tests for agent_utils retry logic enhancement (QuantAgent-aao).

Tests exponential backoff, multi-provider error handling, configuration,
and backwards compatibility.
"""

import logging
import random
from unittest.mock import Mock, patch

import pytest

from quantagent.agent_utils import (
    _calculate_wait_time,
    _is_retryable_error,
    invoke_with_retry,
)

# ============================================================================
# AC1: Core Retry Functionality
# ============================================================================


def test_invoke_with_retry_success_first_attempt():
    """AC1.1: Success on first attempt - no retries."""
    mock_fn = Mock(return_value="success")

    with patch("time.sleep") as mock_sleep:
        result = invoke_with_retry(mock_fn)

    assert result == "success"
    assert mock_fn.call_count == 1
    assert mock_sleep.call_count == 0


def test_invoke_with_retry_success_after_retries():
    """AC1.2: Success after retries - returns result from successful attempt."""

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(
        side_effect=[
            MockRateLimitError("rate limited"),
            MockRateLimitError("rate limited"),
            "success",
        ]
    )

    with patch("time.sleep"):
        result = invoke_with_retry(mock_fn, retries=5)

    assert result == "success"
    assert mock_fn.call_count == 3


def test_invoke_with_retry_max_retries_exceeded():
    """AC1.3: Max retries exceeded - raises RuntimeError."""

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(side_effect=MockRateLimitError("rate limited"))

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="Max retries \\(3\\) exceeded"):
            invoke_with_retry(mock_fn, retries=3)

    assert mock_fn.call_count == 3
    assert mock_sleep.call_count == 2  # No sleep after last attempt


# ============================================================================
# AC2: Exponential Backoff
# ============================================================================


def test_exponential_backoff_calculation():
    """AC2.1: Exponential wait time calculation without jitter."""
    # Test wait time progression: 2, 4, 8, 16, 32, 60 (capped)
    test_cases = [
        (0, 2.0),  # attempt 0: 2 * 2^0 = 2
        (1, 4.0),  # attempt 1: 2 * 2^1 = 4
        (2, 8.0),  # attempt 2: 2 * 2^2 = 8
        (3, 16.0),  # attempt 3: 2 * 2^3 = 16
        (4, 32.0),  # attempt 4: 2 * 2^4 = 32
        (5, 60.0),  # attempt 5: 2 * 2^5 = 64, capped at 60
    ]

    for attempt, expected_wait in test_cases:
        wait = _calculate_wait_time(
            attempt=attempt,
            base_wait=2.0,
            max_wait=60.0,
            exponential_base=2,
            jitter=False,
            jitter_factor=0.5,
        )
        assert wait == expected_wait, f"Attempt {attempt} expected {expected_wait}, got {wait}"


def test_max_wait_cap_applied():
    """AC2.2: Max wait cap prevents excessive delays."""
    # With base=2, max=10, check that waits are capped
    expected_waits = [2.0, 4.0, 8.0, 10.0, 10.0]

    for attempt, expected in enumerate(expected_waits):
        wait = _calculate_wait_time(
            attempt=attempt,
            base_wait=2.0,
            max_wait=10.0,
            exponential_base=2,
            jitter=False,
            jitter_factor=0.5,
        )
        assert wait == expected


def test_jitter_within_bounds():
    """AC2.3: Jitter produces values within expected range."""
    random.seed(42)  # For reproducibility

    # base_wait=4, jitter_factor=0.5
    # jitter_range = 4 * 0.5 = 2
    # wait = 4 + uniform(-1, 1) = [3.0, 5.0]

    for _ in range(100):
        wait = _calculate_wait_time(
            attempt=0,
            base_wait=4.0,
            max_wait=60.0,
            exponential_base=2,
            jitter=True,
            jitter_factor=0.5,
        )
        assert 3.0 <= wait <= 5.0, f"Wait {wait} outside expected range [3.0, 5.0]"


def test_jitter_does_not_produce_negative_wait():
    """AC6.3: Jitter floor prevents negative wait times."""
    random.seed(42)

    for _ in range(1000):
        wait = _calculate_wait_time(
            attempt=0,
            base_wait=0.1,
            max_wait=60.0,
            exponential_base=2,
            jitter=True,
            jitter_factor=1.0,
        )
        assert wait >= 0.1, f"Wait {wait} is below floor"


# ============================================================================
# AC3: Multi-Provider Support
# ============================================================================


def test_openai_rate_limit_error_triggers_retry():
    """AC3.1: OpenAI RateLimitError triggers retry."""

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(side_effect=[MockRateLimitError("rate limited"), "success"])

    with patch("time.sleep"):
        result = invoke_with_retry(mock_fn, retries=3)

    assert result == "success"
    assert mock_fn.call_count == 2


def test_anthropic_rate_limit_error_triggers_retry():
    """AC3.2: Anthropic RateLimitError triggers retry."""

    class MockAnthropicRateLimitError(Exception):
        pass

    MockAnthropicRateLimitError.__module__ = "anthropic"
    MockAnthropicRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(side_effect=[MockAnthropicRateLimitError(), "success"])

    with patch("time.sleep"):
        result = invoke_with_retry(mock_fn, retries=3)

    assert result == "success"
    assert mock_fn.call_count == 2


def test_timeout_errors_trigger_retry():
    """AC3.3: Timeout errors trigger retry."""

    class MockTimeout(Exception):
        pass

    MockTimeout.__module__ = "requests.exceptions"
    MockTimeout.__name__ = "Timeout"

    mock_fn = Mock(side_effect=[MockTimeout("timeout"), "success"])

    with patch("time.sleep"):
        result = invoke_with_retry(mock_fn, retries=3)

    assert result == "success"
    assert mock_fn.call_count == 2


def test_connection_errors_trigger_retry():
    """AC3.4: Connection errors trigger retry."""

    class MockConnectionError(Exception):
        pass

    MockConnectionError.__module__ = "requests.exceptions"
    MockConnectionError.__name__ = "ConnectionError"

    mock_fn = Mock(side_effect=[MockConnectionError("connection failed"), "success"])

    with patch("time.sleep"):
        result = invoke_with_retry(mock_fn, retries=3)

    assert result == "success"
    assert mock_fn.call_count == 2


def test_authentication_error_does_not_retry():
    """AC3.5: Authentication error (4xx) fails immediately without retry."""

    class MockAuthenticationError(Exception):
        pass

    MockAuthenticationError.__module__ = "openai"
    MockAuthenticationError.__name__ = "AuthenticationError"

    mock_fn = Mock(side_effect=MockAuthenticationError("Invalid API key"))

    with pytest.raises(MockAuthenticationError):
        invoke_with_retry(mock_fn, retries=5)

    assert mock_fn.call_count == 1  # No retries


def test_bad_request_error_does_not_retry():
    """AC3.6: BadRequestError (400) fails immediately without retry."""

    class MockBadRequestError(Exception):
        pass

    MockBadRequestError.__module__ = "openai"
    MockBadRequestError.__name__ = "BadRequestError"

    mock_fn = Mock(side_effect=MockBadRequestError("Invalid request"))

    with pytest.raises(MockBadRequestError):
        invoke_with_retry(mock_fn, retries=5)

    assert mock_fn.call_count == 1


def test_http_429_status_code_triggers_retry():
    """AC3.7: HTTP 429 status code triggers retry."""

    class HTTPError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code
            super().__init__(f"HTTP {status_code}")

    mock_fn = Mock(side_effect=[HTTPError(429), "success"])

    with patch("time.sleep"):
        result = invoke_with_retry(mock_fn, retries=3)

    assert result == "success"
    assert mock_fn.call_count == 2


def test_http_5xx_status_codes_trigger_retry():
    """AC3.8: HTTP 5xx status codes trigger retry."""

    class HTTPError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code
            super().__init__(f"HTTP {status_code}")

    for status in [500, 502, 503, 504]:
        mock_fn = Mock(side_effect=[HTTPError(status), "success"])

        with patch("time.sleep"):
            result = invoke_with_retry(mock_fn, retries=3)

        assert result == "success"
        assert mock_fn.call_count == 2


# ============================================================================
# AC4: Configuration
# ============================================================================


def test_global_config_applied_by_default():
    """AC4.1: Global config from RETRY_CONFIG used by default."""
    from quantagent.default_config import RETRY_CONFIG

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    # Use actual RETRY_CONFIG value to avoid brittleness
    expected_retries = int(RETRY_CONFIG["max_retries"])
    mock_fn = Mock(side_effect=[MockRateLimitError()] * (expected_retries - 1) + ["success"])

    with patch("time.sleep"):
        result = invoke_with_retry(mock_fn)

    assert result == "success"
    assert mock_fn.call_count == expected_retries


def test_per_call_config_overrides_global():
    """AC4.2: Per-call config overrides global config."""

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(side_effect=[MockRateLimitError()] * 5)

    with patch("time.sleep"):
        with pytest.raises(RuntimeError, match="Max retries \\(2\\) exceeded"):
            invoke_with_retry(mock_fn, retries=2)

    assert mock_fn.call_count == 2  # Not 5


def test_partial_override_mix_global_and_local():
    """AC4.3: Partial override uses mix of global and local config."""

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(side_effect=[MockRateLimitError(), "success"])

    with patch("time.sleep") as mock_sleep:
        result = invoke_with_retry(mock_fn, base_wait=10.0, jitter=False)

    assert result == "success"
    assert mock_fn.call_count == 2
    mock_sleep.assert_called_once_with(10.0)  # Local base_wait used


# ============================================================================
# AC5: Backwards Compatibility
# ============================================================================


def test_legacy_signature_works():
    """AC5.1: Legacy signature with wait_sec works and maps to base_wait."""

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(side_effect=[MockRateLimitError(), "success"])

    with patch("time.sleep") as mock_sleep:
        result = invoke_with_retry(mock_fn, retries=3, wait_sec=4, jitter=False)

    assert result == "success"
    mock_sleep.assert_called_once_with(4.0)  # Verifies wait_sec → base_wait


def test_deprecation_warning_for_wait_sec(caplog):
    """AC5.2: Deprecation warning logged for wait_sec parameter."""
    mock_fn = Mock(return_value="success")

    with caplog.at_level(logging.WARNING):
        invoke_with_retry(mock_fn, wait_sec=4)

    assert "wait_sec parameter is deprecated" in caplog.text


# ============================================================================
# AC6: Edge Cases
# ============================================================================


def test_zero_retries():
    """AC6.1: Zero retries (retries=1) raises immediately on failure."""

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(side_effect=MockRateLimitError())

    with pytest.raises(RuntimeError):
        invoke_with_retry(mock_fn, retries=1)

    assert mock_fn.call_count == 1


def test_function_returns_none_successfully():
    """AC6.4: Function returning None is treated as success."""
    mock_fn = Mock(return_value=None)

    result = invoke_with_retry(mock_fn)

    assert result is None
    assert mock_fn.call_count == 1


def test_function_with_args_and_kwargs():
    """AC6.5: Function with positional and keyword arguments."""
    mock_fn = Mock(return_value="success")

    invoke_with_retry(mock_fn, "pos1", "pos2", key1="val1", key2="val2")

    mock_fn.assert_called_once_with("pos1", "pos2", key1="val1", key2="val2")


# ============================================================================
# AC7: Logging
# ============================================================================


def test_warning_on_each_retry(caplog):
    """AC7.1: WARNING log emitted on each retry."""

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(
        side_effect=[MockRateLimitError(), MockRateLimitError(), "success"]
    )

    with caplog.at_level(logging.WARNING):
        with patch("time.sleep"):
            invoke_with_retry(mock_fn, retries=5)

    assert "Retry 1/5" in caplog.text
    assert "Retry 2/5" in caplog.text


def test_error_on_final_failure(caplog):
    """AC7.2: ERROR log emitted on final failure."""

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(side_effect=MockRateLimitError("rate limited"))

    with caplog.at_level(logging.ERROR):
        with patch("time.sleep"):
            with pytest.raises(RuntimeError):
                invoke_with_retry(mock_fn, retries=5)

    assert "Max retries (5) exceeded" in caplog.text


def test_no_logs_on_success(caplog):
    """AC7.3: No WARNING/ERROR logs on first-try success."""
    mock_fn = Mock(return_value="success")

    with caplog.at_level(logging.WARNING):
        invoke_with_retry(mock_fn)

    # Should have no WARNING or ERROR logs
    assert len(caplog.records) == 0


# ============================================================================
# Additional Tests (Missing ACs)
# ============================================================================


def test_jitter_reproducible_with_seed():
    """AC2.4: Jitter calculations are reproducible with seed."""
    random.seed(42)
    waits1 = [
        _calculate_wait_time(
            attempt=0,
            base_wait=4.0,
            max_wait=60.0,
            exponential_base=2,
            jitter=True,
            jitter_factor=0.5,
        )
        for _ in range(10)
    ]

    random.seed(42)
    waits2 = [
        _calculate_wait_time(
            attempt=0,
            base_wait=4.0,
            max_wait=60.0,
            exponential_base=2,
            jitter=True,
            jitter_factor=0.5,
        )
        for _ in range(10)
    ]

    assert waits1 == waits2


def test_very_large_max_wait_applied():
    """AC6.2: Very large max_wait values are respected."""
    wait = _calculate_wait_time(
        attempt=12,  # 2 * 2^12 = 8192 > 3600
        base_wait=2.0,
        max_wait=3600.0,  # 1 hour
        exponential_base=2,
        jitter=False,
        jitter_factor=0.5,
    )
    assert wait == 3600.0  # Should be capped


def test_exponential_backoff_full_sequence():
    """Verify full exponential backoff sequence with real timing."""

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(side_effect=[MockRateLimitError()] * 3 + ["success"])

    with patch("time.sleep") as mock_sleep:
        result = invoke_with_retry(
            mock_fn, retries=5, base_wait=2.0, max_wait=60.0, jitter=False
        )

    assert result == "success"
    assert mock_sleep.call_count == 3
    expected_waits = [2.0, 4.0, 8.0]
    actual_waits = [call.args[0] for call in mock_sleep.call_args_list]
    assert actual_waits == expected_waits


def test_no_sleep_after_final_failure():
    """Verify no sleep occurs after the final failed attempt."""

    class MockRateLimitError(Exception):
        pass

    MockRateLimitError.__module__ = "openai"
    MockRateLimitError.__name__ = "RateLimitError"

    mock_fn = Mock(side_effect=MockRateLimitError())

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError):
            invoke_with_retry(mock_fn, retries=3)

    # Should sleep only between attempts (not after last)
    assert mock_sleep.call_count == 2  # retries=3 → 3 attempts → 2 sleeps


@pytest.mark.parametrize(
    "error_module,error_name",
    [
        ("openai", "RateLimitError"),
        ("anthropic", "RateLimitError"),
        ("requests.exceptions", "Timeout"),
        ("requests.exceptions", "ConnectionError"),
    ],
)
def test_retryable_errors_parametrized(error_module, error_name):
    """AC3: All documented retryable errors trigger retry (parametrized)."""

    class MockError(Exception):
        pass

    MockError.__module__ = error_module
    MockError.__name__ = error_name

    mock_fn = Mock(side_effect=[MockError(), "success"])

    with patch("time.sleep"):
        result = invoke_with_retry(mock_fn, retries=3)

    assert result == "success"
    assert mock_fn.call_count == 2


# ============================================================================
# Helper Function Tests
# ============================================================================


@pytest.mark.parametrize(
    "error_type,module,name,expected_retryable",
    [
        ("RateLimitError", "openai", "RateLimitError", True),
        ("RateLimitError", "anthropic", "RateLimitError", True),
        ("Timeout", "requests.exceptions", "Timeout", True),
        ("ConnectionError", "requests.exceptions", "ConnectionError", True),
        ("AuthenticationError", "openai", "AuthenticationError", False),
        ("BadRequestError", "openai", "BadRequestError", False),
    ],
)
def test_is_retryable_error_parametrized(error_type, module, name, expected_retryable):
    """Test _is_retryable_error correctly identifies retryable vs non-retryable errors."""

    class MockError(Exception):
        pass

    MockError.__module__ = module
    MockError.__name__ = name

    error = MockError()
    assert _is_retryable_error(error) is expected_retryable


@pytest.mark.parametrize(
    "status_code,expected_retryable",
    [
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
    ],
)
def test_is_retryable_error_status_codes(status_code, expected_retryable):
    """Test _is_retryable_error checks status_code attribute correctly."""

    class HTTPError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    assert _is_retryable_error(HTTPError(status_code)) is expected_retryable
