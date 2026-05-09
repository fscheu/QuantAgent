"""
Tests validating the removal of wait_sec deprecation (QuantAgent-lmn).

AC1: Function Signature Updated - wait_sec parameter not present
AC2: Deprecation Logic Removed - no code handling wait_sec
AC3: Tests Updated - no test uses wait_sec parameter
AC4: Deprecation Test Removed - test_deprecation_warning_for_wait_sec should not exist
AC5: Test Suite Passes - all tests pass without errors
AC6: No Regression - existing functionality using base_wait remains unchanged
"""

import inspect
from unittest.mock import Mock, patch

import pytest

from quantagent.agent_utils import invoke_with_retry


class TestAC1FunctionSignatureUpdated:
    """AC1: Function Signature Updated - wait_sec parameter should not be present."""

    def test_invoke_with_retry_does_not_have_wait_sec_parameter(self):
        """CRITICAL: Verify wait_sec parameter is completely removed from function signature."""
        sig = inspect.signature(invoke_with_retry)
        param_names = list(sig.parameters.keys())

        assert "wait_sec" not in param_names, (
            f"wait_sec parameter found in function signature. "
            f"Parameters: {param_names}"
        )

    def test_invoke_with_retry_accepts_base_wait_parameter(self):
        """Verify base_wait parameter exists in function signature."""
        sig = inspect.signature(invoke_with_retry)
        param_names = list(sig.parameters.keys())

        assert "base_wait" in param_names, (
            f"base_wait parameter NOT found in function signature. "
            f"Parameters: {param_names}"
        )

    def test_function_docstring_does_not_mention_wait_sec(self):
        """Verify docstring doesn't reference the deprecated parameter."""
        docstring = invoke_with_retry.__doc__

        assert docstring is not None, "Function has no docstring"
        assert (
            "wait_sec" not in docstring.lower()
        ), f"Docstring still mentions wait_sec: {docstring}"

    def test_function_docstring_documents_base_wait(self):
        """Verify docstring documents base_wait parameter correctly."""
        docstring = invoke_with_retry.__doc__

        assert docstring is not None
        assert "base_wait" in docstring, (
            f"Docstring doesn't document base_wait parameter: {docstring}"
        )


class TestAC2DeprecationLogicRemoved:
    """AC2: Deprecation Logic Removed - no code handling wait_sec parameter."""

    def test_invoke_with_retry_does_not_accept_wait_sec_kwarg(self):
        """Verify wait_sec is not in the explicit parameter list."""
        # Since **kwargs is used, wait_sec technically passes but gets ignored
        # The important test is that it's not in the signature
        sig = inspect.signature(invoke_with_retry)
        param_names = list(sig.parameters.keys())
        assert "wait_sec" not in param_names, "wait_sec should not be an explicit parameter"

    def test_no_wait_sec_handling_in_function_body(self):
        """Verify no code in function body handles wait_sec parameter."""
        # Read the source code
        source = inspect.getsource(invoke_with_retry)

        # Check that wait_sec parameter is not referenced (but wait_seconds logging var is OK)
        # Look for patterns like: "wait_sec", wait_sec =, wait_sec:
        import re
        wait_sec_pattern = r'\bwait_sec\b'  # Word boundary matches "wait_sec" but not "wait_seconds"
        matches = re.findall(wait_sec_pattern, source)

        assert len(matches) == 0, (
            "wait_sec parameter reference found in function body"
        )

    def test_no_deprecation_warning_emitted(self, caplog):
        """Verify no deprecation warning is emitted during execution."""
        import logging

        mock_fn = Mock(return_value="success")

        with caplog.at_level(logging.WARNING):
            invoke_with_retry(mock_fn, base_wait=2.0)

        # Filter for deprecation warnings
        deprecation_warnings = [
            record
            for record in caplog.records
            if "deprecated" in record.message.lower()
            and "wait_sec" in record.message.lower()
        ]

        assert (
            len(deprecation_warnings) == 0
        ), f"Deprecation warnings found: {deprecation_warnings}"


class TestAC3TestsUpdated:
    """AC3: Tests Updated - no test should use wait_sec parameter."""

    def test_all_retry_tests_use_base_wait_not_wait_sec(self):
        """Verify existing test file doesn't use wait_sec."""
        import tests.test_agent_utils_retry as test_module

        source = inspect.getsource(test_module)

        # Should not contain wait_sec in any test
        lines_with_wait_sec = [
            line for line in source.split("\n") if "wait_sec" in line
        ]

        assert (
            len(lines_with_wait_sec) == 0
        ), f"Found wait_sec usage in tests:\n{lines_with_wait_sec}"

    def test_base_wait_parameter_used_in_tests(self):
        """Verify tests use base_wait parameter instead."""
        import tests.test_agent_utils_retry as test_module

        source = inspect.getsource(test_module)

        # Should contain base_wait usage
        assert (
            "base_wait=" in source
        ), "Tests don't use base_wait parameter; they should"


class TestAC4DeprecationTestRemoved:
    """AC4: Deprecation Test Removed - test_deprecation_warning_for_wait_sec should not exist."""

    def test_no_deprecation_warning_test_exists(self):
        """Verify test_deprecation_warning_for_wait_sec does not exist."""
        import tests.test_agent_utils_retry as test_module

        test_names = [
            name for name in dir(test_module) if name.startswith("test_")
        ]

        deprecated_test_names = [
            name
            for name in test_names
            if "deprecation" in name and "wait_sec" in name
        ]

        assert (
            len(deprecated_test_names) == 0
        ), f"Deprecation warning tests still exist: {deprecated_test_names}"


class TestAC5TestSuitePasses:
    """AC5: Test Suite Passes - all tests should pass without errors."""

    def test_basic_retry_still_works_with_base_wait(self):
        """Verify basic retry functionality works with base_wait."""

        class MockRateLimitError(Exception):
            pass

        MockRateLimitError.__module__ = "openai"
        MockRateLimitError.__name__ = "RateLimitError"

        mock_fn = Mock(
            side_effect=[
                MockRateLimitError(),
                MockRateLimitError(),
                "success",
            ]
        )

        with patch("time.sleep"):
            result = invoke_with_retry(
                mock_fn, retries=5, base_wait=2.0, jitter=False
            )

        assert result == "success"
        assert mock_fn.call_count == 3

    def test_exponential_backoff_with_base_wait_parameter(self):
        """Verify exponential backoff works with explicit base_wait."""

        class MockRateLimitError(Exception):
            pass

        MockRateLimitError.__module__ = "openai"
        MockRateLimitError.__name__ = "RateLimitError"

        mock_fn = Mock(
            side_effect=[
                MockRateLimitError(),
                MockRateLimitError(),
                "success",
            ]
        )

        with patch("time.sleep") as mock_sleep:
            invoke_with_retry(
                mock_fn, retries=5, base_wait=1.0, jitter=False
            )

        # Verify exponential backoff: 1.0, 2.0 (1.0 * 2^1)
        assert mock_sleep.call_count == 2
        expected_waits = [1.0, 2.0]
        actual_waits = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_waits == expected_waits


class TestAC6NoRegression:
    """AC6: No Regression - behavior should remain unchanged from before."""

    def test_all_default_config_still_used(self):
        """Verify default config from RETRY_CONFIG is still used."""
        from quantagent.default_config import RETRY_CONFIG

        class MockRateLimitError(Exception):
            pass

        MockRateLimitError.__module__ = "openai"
        MockRateLimitError.__name__ = "RateLimitError"

        expected_retries = int(RETRY_CONFIG["max_retries"])
        mock_fn = Mock(
            side_effect=[MockRateLimitError()] * (expected_retries - 1)
            + ["success"]
        )

        with patch("time.sleep"):
            result = invoke_with_retry(mock_fn)

        assert result == "success"
        assert mock_fn.call_count == expected_retries

    def test_configuration_override_still_works(self):
        """Verify per-call configuration overrides still work."""

        class MockRateLimitError(Exception):
            pass

        MockRateLimitError.__module__ = "openai"
        MockRateLimitError.__name__ = "RateLimitError"

        mock_fn = Mock(side_effect=[MockRateLimitError()] * 5)

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="Max retries \\(2\\) exceeded"):
                invoke_with_retry(mock_fn, retries=2)

        assert mock_fn.call_count == 2

    def test_jitter_parameter_still_works(self):
        """Verify jitter parameter still functions correctly."""

        class MockRateLimitError(Exception):
            pass

        MockRateLimitError.__module__ = "openai"
        MockRateLimitError.__name__ = "RateLimitError"

        mock_fn = Mock(side_effect=[MockRateLimitError(), "success"])

        with patch("time.sleep") as mock_sleep:
            invoke_with_retry(mock_fn, base_wait=2.0, jitter=False)

        # With jitter=False, sleep should be exactly base_wait
        mock_sleep.assert_called_once_with(2.0)

    def test_max_wait_parameter_still_works(self):
        """Verify max_wait parameter still caps wait times."""

        class MockRateLimitError(Exception):
            pass

        MockRateLimitError.__module__ = "openai"
        MockRateLimitError.__name__ = "RateLimitError"

        # Simulate 4 failures that would exceed max_wait
        mock_fn = Mock(
            side_effect=[
                MockRateLimitError(),
                MockRateLimitError(),
                MockRateLimitError(),
                "success",
            ]
        )

        with patch("time.sleep") as mock_sleep:
            invoke_with_retry(
                mock_fn,
                retries=5,
                base_wait=2.0,
                max_wait=5.0,
                jitter=False,
            )

        # Expected: 2.0, 4.0, 5.0 (capped from 8.0)
        actual_waits = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_waits == [2.0, 4.0, 5.0]


class TestCodeSearchValidation:
    """Comprehensive search validation for wait_sec removal."""

    def test_agent_utils_file_has_no_wait_sec(self):
        """Verify agent_utils.py source has no wait_sec parameter references."""
        import re

        from quantagent import agent_utils

        source = inspect.getsource(agent_utils)

        # Use word boundary to exclude "wait_seconds" (the logging variable)
        wait_sec_pattern = r'\bwait_sec\b'
        matches = re.findall(wait_sec_pattern, source)

        assert len(matches) == 0, (
            "wait_sec parameter found in agent_utils.py source"
        )

    def test_all_agent_files_migrated_to_base_wait(self):
        """Verify all agent implementation files use base_wait."""
        import quantagent.decision_agent
        import quantagent.indicator_agent
        import quantagent.pattern_agent
        import quantagent.trend_agent

        for module in [
            quantagent.decision_agent,
            quantagent.indicator_agent,
            quantagent.pattern_agent,
            quantagent.trend_agent,
        ]:
            source = inspect.getsource(module)

            # Should not have wait_sec
            wait_sec_count = source.count("wait_sec")
            assert wait_sec_count == 0, (
                f"{module.__name__} has {wait_sec_count} wait_sec references"
            )

            # Should have base_wait if it calls invoke_with_retry
            if "invoke_with_retry" in source:
                # This module uses invoke_with_retry
                # If it uses explicit config, should use base_wait
                if "base_wait" in source or "retries=" in source:
                    assert "base_wait" in source, (
                        f"{module.__name__} uses invoke_with_retry "
                        f"but doesn't document base_wait usage"
                    )


class TestNoWaitSecInCodebase:
    """Final validation: no wait_sec anywhere in the codebase."""

    def test_grep_for_wait_sec_in_quantagent(self):
        """Use grep to verify no wait_sec parameter in quantagent/ directory."""
        import subprocess
        from pathlib import Path

        # Use word boundary to exclude "wait_seconds" (the logging variable)
        # Run from current repo root instead of hardcoded worktree
        repo_root = Path(__file__).parent.parent
        result = subprocess.run(
            ["grep", "-r", "-w", "wait_sec", "quantagent/", "--include=*.py"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )

        # Should return nothing (exit code 1 when no matches with -w flag)
        assert (
            result.returncode == 1
        ), f"Found wait_sec parameter in quantagent/:\n{result.stdout}"

    def test_grep_for_wait_sec_in_tests(self):
        """Use grep to verify no wait_sec in test_agent_utils_retry.py."""
        import subprocess
        from pathlib import Path

        # Check only the original test file (not this new validation test file)
        # Run from current repo root instead of hardcoded worktree
        repo_root = Path(__file__).parent.parent
        result = subprocess.run(
            ["grep", "-r", "-w", "wait_sec", "tests/test_agent_utils_retry.py"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )

        # Should return nothing (exit code 1 when no matches)
        assert result.returncode == 1, f"Found wait_sec parameter in test_agent_utils_retry.py:\n{result.stdout}"
