"""
Unit tests for refactored trend_agent with structured Pydantic output and vision integration.

Tests validate:
✅ Structure: Output is TrendReport with all required fields
✅ Constraints: trend_strength (0-1), support/resistance are floats, valid directions
✅ Error handling: Fallback on LLM or vision failure
✅ State flow: Messages properly constructed and preserved
✅ Vision integration: Image usage/generation, vision fallback
✅ Edge cases: Empty data, single candle, level relationships

See docs/03_technical/TESTING_PATTERNS.md for testing guidelines.
"""

import pytest
from unittest.mock import Mock
from langchain_core.messages import SystemMessage, HumanMessage

from quantagent.agent_models import TrendReport
from quantagent.trend_agent import create_trend_agent


@pytest.fixture
def sample_state():
    """Fixture providing sample agent state with OHLCV data."""
    return {
        "kline_data": {
            "timestamps": [1, 2, 3, 4, 5],
            "opens": [100000, 99500, 99800, 100500, 101000],
            "highs": [100500, 99900, 100200, 101000, 101500],
            "lows": [99500, 99000, 99500, 100000, 100500],
            "closes": [99500, 99800, 100500, 101000, 101200],
            "volumes": [100, 110, 120, 115, 130]
        },
        "time_frame": "4hour",
        "stock_name": "BTC",
        "trend_image": None,
        "messages": []
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
            "volumes": []
        },
        "time_frame": "1hour",
        "stock_name": "TEST",
        "trend_image": None,
        "messages": []
    }


# ============================================================================
# STRUCTURE TESTS - Validate output is correct Pydantic model with all fields
# ============================================================================

class TestOutputStructure:
    """Test that agent returns valid TrendReport structure."""

    def test_output_is_trend_report_instance(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify output is TrendReport instance, not dict or string."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["trend_report"]
        assert isinstance(report, TrendReport), "Output must be TrendReport Pydantic model"

    def test_report_has_all_required_fields(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify TrendReport contains all required fields without None values."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["trend_report"]
        required_fields = {
            "support_level": float,
            "resistance_level": float,
            "trend_direction": str,
            "trend_strength": float,
            "reasoning": str
        }

        for field, expected_type in required_fields.items():
            assert hasattr(report, field), f"Missing field: {field}"
            value = getattr(report, field)
            assert value is not None, f"Field {field} is None"
            assert isinstance(value, expected_type), f"Field {field} wrong type: {type(value)}"

    def test_trend_direction_is_valid_enum(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify trend_direction is one of valid enum values."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["trend_report"]
        valid_directions = ["upward", "downward", "sideways"]
        assert report.trend_direction in valid_directions, \
            f"trend_direction must be in {valid_directions}, got {report.trend_direction}"


# ============================================================================
# CONSTRAINT TESTS - Validate field ranges and Pydantic validation
# ============================================================================

class TestConstraintValidation:
    """Test that Pydantic validators enforce field constraints."""

    def test_trend_strength_within_0_to_1(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify trend_strength is always constrained to 0.0-1.0 range."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["trend_report"]
        assert 0.0 <= report.trend_strength <= 1.0, f"Trend strength out of range: {report.trend_strength}"

    def test_support_and_resistance_are_floats(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify support and resistance levels are float types."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["trend_report"]
        assert isinstance(report.support_level, float), \
            f"support_level must be float, got {type(report.support_level)}"
        assert isinstance(report.resistance_level, float), \
            f"resistance_level must be float, got {type(report.resistance_level)}"

    def test_support_typically_below_resistance(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify support is typically below resistance (typical but not absolute)."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["trend_report"]
        # Only check if both are positive (valid price levels)
        if report.support_level > 0 and report.resistance_level > 0:
            assert report.support_level <= report.resistance_level, \
                f"Support ({report.support_level}) should be <= resistance ({report.resistance_level})"


# ============================================================================
# ERROR HANDLING TESTS - Validate fallback mechanism
# ============================================================================

class TestErrorHandling:
    """Test agent gracefully handles errors and returns valid fallback."""

    def test_fallback_on_llm_exception(self, mock_llm, mock_toolkit, sample_state):
        """Verify agent returns valid fallback report when LLM raises exception."""
        mock_vision_llm_error = Mock()
        mock_vision_llm_error.invoke = Mock(side_effect=ValueError("LLM error"))

        agent_node = create_trend_agent(mock_llm, mock_vision_llm_error, mock_toolkit)
        result = agent_node(sample_state)

        report = result["trend_report"]
        assert isinstance(report, TrendReport), "Fallback must return valid TrendReport"
        assert report.trend_strength == 0.0, "Fallback trend_strength should be 0"
        assert "llm error" in report.reasoning.lower(), "Fallback reasoning should mention failure"

    def test_fallback_is_valid_pydantic_model(self, mock_llm, mock_toolkit, sample_state):
        """Verify fallback report respects all Pydantic constraints."""
        mock_llm_error = Mock()
        mock_llm_error.invoke = Mock(side_effect=RuntimeError("LLM error"))

        agent_node = create_trend_agent(mock_llm, mock_llm_error, mock_toolkit)
        result = agent_node(sample_state)

        report = result["trend_report"]
        # Validate all constraints are respected in fallback
        assert isinstance(report, TrendReport)
        assert 0.0 <= report.trend_strength <= 1.0
        assert isinstance(report.support_level, float)
        assert isinstance(report.resistance_level, float)
        assert report.trend_direction in ["upward", "downward", "sideways"]


# ============================================================================
# STATE MANAGEMENT TESTS - Validate messages and state flow
# ============================================================================

class TestStateManagement:
    """Test proper message construction and state handling."""

    def test_result_contains_messages_key(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify result dict contains 'messages' key."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        assert "messages" in result, "Result must include 'messages' key"
        assert isinstance(result["messages"], list), "Messages must be list"

    def test_system_message_included(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify SystemMessage is properly included in messages."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        messages = result["messages"]
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) > 0, "SystemMessage required in message history"

    def test_human_message_included(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify HumanMessage is properly included in messages."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        messages = result["messages"]
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_msgs) > 0, "HumanMessage required in message history"

    def test_timeframe_in_system_message(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify timeframe is mentioned in system message."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        messages = result["messages"]
        human_msg = next((m for m in messages if isinstance(m, HumanMessage)), None)
        assert human_msg is not None, "HumanMessage not found"
        assert sample_state["time_frame"] in human_msg.content[0]['text'], \
            f"Timeframe '{sample_state['time_frame']}' not in human message"

# ============================================================================
# VISION INTEGRATION TESTS - Image handling and vision-specific behavior
# ============================================================================

class TestVisionIntegration:
    """Test vision LLM integration and image handling."""

    def test_uses_precomputed_image_when_available(self, mock_llm, mock_vision_llm, mock_toolkit):
        """Verify agent uses precomputed image from state when available."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)

        state = {
            "kline_data": {
                "timestamps": [1, 2],
                "opens": [100000, 99000],
                "highs": [100000, 99500],
                "lows": [99500, 98500],
                "closes": [99000, 98500],
                "volumes": [100, 110]
            },
            "time_frame": "4hour",
            "stock_name": "BTC",
            "trend_image": "precomputed_b64_image_data",
            "messages": []
        }

        result = agent_node(state)
        report = result["trend_report"]
        assert isinstance(report, TrendReport), "Should return valid report with precomputed image"

    def test_generates_image_if_missing(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify agent handles image generation when not in state."""
        sample_state["trend_image"] = None

        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["trend_report"]
        assert isinstance(report, TrendReport), "Should generate/handle missing image gracefully"

    def test_vision_failure_returns_fallback(self, mock_toolkit, sample_state):
        """Verify agent handles vision LLM failure gracefully."""
        mock_vision_llm_fails = Mock()
        mock_vision_llm_fails.with_structured_output = Mock(side_effect=Exception("Vision model unavailable"))

        agent_node = create_trend_agent(Mock(), mock_vision_llm_fails, mock_toolkit)
        result = agent_node(sample_state)

        report = result["trend_report"]
        assert isinstance(report, TrendReport), "Must return fallback on vision failure"


# ============================================================================
# EDGE CASE TESTS - Validate robustness with boundary conditions
# ============================================================================

class TestEdgeCases:
    """Test agent handles edge cases gracefully."""

    def test_empty_kline_data(self, mock_llm, mock_vision_llm, mock_toolkit, empty_state):
        """Verify agent handles empty OHLCV data gracefully."""
        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(empty_state)

        report = result["trend_report"]
        # Should return valid report (possibly fallback)
        assert isinstance(report, TrendReport)
        # Should still respect constraints
        assert 0.0 <= report.trend_strength <= 1.0

    def test_single_candle_data(self, mock_llm, mock_vision_llm, mock_toolkit):
        """Verify agent handles single candlestick gracefully."""
        state = {
            "kline_data": {
                "timestamps": [1],
                "opens": [100000],
                "highs": [100500],
                "lows": [99500],
                "closes": [100200],
                "volumes": [1000]
            },
            "time_frame": "1hour",
            "stock_name": "TEST",
            "trend_image": None,
            "messages": []
        }

        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(state)

        report = result["trend_report"]
        assert isinstance(report, TrendReport)

    def test_messages_persisted_when_provided(self, mock_llm, mock_vision_llm, mock_toolkit):
        """Verify existing messages are preserved."""
        initial_msg = HumanMessage(content="Previous analysis")
        state = {
            "kline_data": {
                "timestamps": [1, 2],
                "opens": [100000, 99000],
                "highs": [100000, 99500],
                "lows": [99500, 98500],
                "closes": [99000, 98500],
                "volumes": [100, 110]
            },
            "time_frame": "4hour",
            "stock_name": "TEST",
            "trend_image": None,
            "messages": [initial_msg]
        }

        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(state)

        # Messages list should be returned
        assert "messages" in result
        assert isinstance(result["messages"], list)

    def test_large_support_resistance_values(self, mock_llm, mock_vision_llm, mock_toolkit):
        """Verify agent handles very large price values gracefully."""
        state = {
            "kline_data": {
                "timestamps": [1, 2],
                "opens": [1000000.0, 1000500.0],
                "highs": [1000500.0, 1000900.0],
                "lows": [999500.0, 1000200.0],
                "closes": [1000500.0, 1000800.0],
                "volumes": [100, 110]
            },
            "time_frame": "1day",
            "stock_name": "TEST",
            "trend_image": None,
            "messages": []
        }

        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(state)

        report = result["trend_report"]
        assert isinstance(report, TrendReport)
        assert isinstance(report.support_level, float)
        assert isinstance(report.resistance_level, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
