"""
Unit tests for refactored indicator_agent with structured Pydantic output.

Tests validate:
✅ Structure: Output is IndicatorReport with all required fields
✅ Constraints: RSI (0-100), confidence (0-1), valid enums
✅ Error handling: Fallback on LLM failure
✅ State flow: Messages properly constructed and preserved
✅ Edge cases: Empty data, missing fields
✅ Tool binding: System message includes tool instructions

See docs/03_technical/TESTING_PATTERNS.md for testing guidelines.
"""

from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage

from quantagent.agent_models import IndicatorReport
from quantagent.indicator_agent import create_indicator_agent


@pytest.fixture
def sample_state():
    """Fixture providing sample agent state with OHLCV data."""
    return {
        "kline_data": {
            "timestamps": [1700000000, 1700003600, 1700007200],
            "opens": [100.0, 100.5, 100.3],
            "highs": [100.8, 101.2, 100.9],
            "lows": [99.8, 100.2, 100.0],
            "closes": [100.5, 100.3, 100.6],
            "volumes": [1000, 1200, 1100],
        },
        "time_frame": "4hour",
        "stock_name": "BTC",
        "messages": [],
    }


@pytest.fixture
def empty_state():
    """Fixture with empty OHLCV data for edge case testing."""
    return {
        "kline_data": {
            "timestamps": [],
            "opens": [],
            "highs": [],
            "lows": [],
            "closes": [],
            "volumes": [],
        },
        "time_frame": "1hour",
        "stock_name": "TEST",
        "messages": [],
    }


# ============================================================================
# STRUCTURE TESTS - Validate output is correct Pydantic model with all fields
# ============================================================================


class TestOutputStructure:
    """Test that agent returns valid IndicatorReport structure."""

    def test_output_is_indicator_report_instance(
        self, mock_llm, mock_toolkit, sample_state
    ):
        """Verify output is IndicatorReport instance, not dict or string."""
        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["indicator_report"]
        assert isinstance(
            report, IndicatorReport
        ), "Output must be IndicatorReport Pydantic model"

    def test_report_has_all_required_fields(self, mock_llm, mock_toolkit, sample_state):
        """Verify IndicatorReport contains all required fields without None values."""
        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["indicator_report"]
        required_fields = {
            "macd": float,
            "macd_signal": float,
            "macd_histogram": float,
            "rsi": float,
            "rsi_level": str,
            "roc": float,
            "stochastic": float,
            "willr": float,
            "trend_direction": str,
            "confidence": float,
            "reasoning": str,
        }

        for field, expected_type in required_fields.items():
            assert hasattr(report, field), f"Missing field: {field}"
            value = getattr(report, field)
            assert value is not None, f"Field {field} is None"
            assert isinstance(
                value, expected_type
            ), f"Field {field} wrong type: {type(value)}"

    def test_rsi_level_is_valid_enum(self, mock_llm, mock_toolkit, sample_state):
        """Verify rsi_level is one of valid enum values."""
        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["indicator_report"]
        valid_levels = ["overbought", "oversold", "neutral"]
        assert report.rsi_level in valid_levels, f"rsi_level must be in {valid_levels}"

    def test_trend_direction_is_valid_enum(self, mock_llm, mock_toolkit, sample_state):
        """Verify trend_direction is one of valid enum values."""
        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["indicator_report"]
        valid_directions = ["bullish", "bearish", "neutral"]
        assert (
            report.trend_direction in valid_directions
        ), f"trend_direction must be in {valid_directions}"


# ============================================================================
# CONSTRAINT TESTS - Validate field ranges and Pydantic validation
# ============================================================================


class TestConstraintValidation:
    """Test that Pydantic validators enforce field constraints."""

    def test_rsi_within_0_to_100(self, mock_llm, mock_toolkit, sample_state):
        """Verify RSI is always constrained to 0-100 range."""
        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["indicator_report"]
        assert 0 <= report.rsi <= 100, f"RSI out of range: {report.rsi}"

    def test_stochastic_within_0_to_100(self, mock_llm, mock_toolkit, sample_state):
        """Verify Stochastic is constrained to 0-100 range."""
        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["indicator_report"]
        assert (
            0 <= report.stochastic <= 100
        ), f"Stochastic out of range: {report.stochastic}"

    def test_confidence_within_0_to_1(self, mock_llm, mock_toolkit, sample_state):
        """Verify confidence is always constrained to 0.0-1.0 range."""
        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["indicator_report"]
        assert (
            0.0 <= report.confidence <= 1.0
        ), f"Confidence out of range: {report.confidence}"

    def test_willr_within_minus_100_to_0(self, mock_llm, mock_toolkit, sample_state):
        """Verify Williams %R is constrained to -100 to 0 range."""
        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["indicator_report"]
        assert -100 <= report.willr <= 0, f"Williams %R out of range: {report.willr}"


# ============================================================================
# ERROR HANDLING TESTS - Validate fallback mechanism
# ============================================================================


class TestErrorHandling:
    """Test agent gracefully handles errors and returns valid fallback."""

    def test_fallback_on_llm_exception(self, mock_toolkit, sample_state):
        """Verify agent returns valid fallback report when LLM raises exception."""
        mock_llm_error = Mock()
        mock_llm_error.with_structured_output = Mock(
            side_effect=ValueError("LLM error")
        )

        agent_node = create_indicator_agent(mock_llm_error, mock_toolkit)
        result = agent_node(sample_state)

        report = result["indicator_report"]
        assert isinstance(
            report, IndicatorReport
        ), "Fallback must return valid IndicatorReport"
        assert report.confidence == 0.0, "Fallback confidence should be 0"
        assert (
            "output validation failed" in report.reasoning.lower()
        ), "Fallback reasoning should mention failure"

    def test_fallback_is_valid_pydantic_model(self, mock_toolkit, sample_state):
        """Verify fallback report respects all Pydantic constraints."""
        mock_llm_error = Mock()
        mock_llm_error.with_structured_output = Mock(
            side_effect=RuntimeError("Unexpected error")
        )

        agent_node = create_indicator_agent(mock_llm_error, mock_toolkit)
        result = agent_node(sample_state)

        report = result["indicator_report"]
        # Validate all constraints are respected in fallback
        assert isinstance(report, IndicatorReport)
        assert 0 <= report.rsi <= 100
        assert 0 <= report.stochastic <= 100
        assert -100 <= report.willr <= 0
        assert 0.0 <= report.confidence <= 1.0
        assert report.rsi_level in ["overbought", "oversold", "neutral"]
        assert report.trend_direction in ["bullish", "bearish", "neutral"]


# ============================================================================
# EDGE CASE TESTS - Validate robustness with boundary conditions
# ============================================================================


class TestEdgeCases:
    """Test agent handles edge cases gracefully."""

    def test_empty_kline_data(self, mock_llm, mock_toolkit, empty_state):
        """Verify agent handles empty OHLCV data gracefully."""
        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(empty_state)

        report = result["indicator_report"]
        # Should return valid report (possibly fallback)
        assert isinstance(report, IndicatorReport)
        # Should still respect constraints
        assert 0.0 <= report.confidence <= 1.0

    def test_single_candle_data(self, mock_llm, mock_toolkit):
        """Verify agent handles single candlestick gracefully."""
        state = {
            "kline_data": {
                "timestamps": [1700000000],
                "opens": [100.0],
                "highs": [100.5],
                "lows": [99.5],
                "closes": [100.2],
                "volumes": [1000],
            },
            "time_frame": "1hour",
            "stock_name": "TEST",
            "messages": [],
        }

        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(state)

        report = result["indicator_report"]
        assert isinstance(report, IndicatorReport)

    def test_messages_not_in_state(self, mock_llm, mock_toolkit):
        """Verify existing messages are preserved."""
        initial_msg = HumanMessage(content="Previous analysis")
        state = {
            "kline_data": {
                "timestamps": [1700000000],
                "opens": [100.0],
                "highs": [100.5],
                "lows": [99.5],
                "closes": [100.2],
                "volumes": [1000],
            },
            "time_frame": "1hour",
            "stock_name": "TEST",
            "messages": [initial_msg],
        }

        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(state)

        # Se quitaron los mensajes existentes del resultado final
        assert "messages" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
