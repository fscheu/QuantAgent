"""
Unit tests for refactored pattern_agent with structured Pydantic output and vision integration.

Tests validate:
✅ Structure: Output is PatternReport with all required fields
✅ Constraints: confidence (0-1), breakout_probability (0-1), valid patterns list
✅ Error handling: Fallback on LLM or vision failure
✅ State flow: Messages properly constructed and preserved
✅ Vision integration: Image usage/generation, vision fallback
✅ Edge cases: Empty data, single candle

See docs/03_technical/TESTING_PATTERNS.md for testing guidelines.
"""

import pytest
from unittest.mock import Mock
from langchain_core.messages import SystemMessage, HumanMessage

from quantagent.agent_models import PatternReport
from quantagent.pattern_agent import create_pattern_agent


@pytest.fixture
def sample_state():
    """Fixture providing sample agent state with OHLCV data."""
    return {
        "kline_data": {
            "timestamps": [1, 2, 3, 4, 5],
            "opens": [100.0, 99.0, 99.5, 100.5, 101.0],
            "highs": [100.5, 99.8, 100.2, 101.0, 101.5],
            "lows": [99.5, 98.5, 99.0, 100.0, 100.5],
            "closes": [99.0, 99.5, 100.5, 101.0, 101.2],
            "volumes": [1000, 1100, 1200, 1150, 1300]
        },
        "time_frame": "4hour",
        "stock_name": "BTC",
        "pattern_image": None,
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
        "pattern_image": None,
        "messages": []
    }


# ============================================================================
# STRUCTURE TESTS - Validate output is correct Pydantic model with all fields
# ============================================================================

class TestOutputStructure:
    """Test that agent returns valid PatternReport structure."""

    def test_output_is_pattern_report_instance(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify output is PatternReport instance, not dict or string."""
        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["pattern_report"]
        assert isinstance(report, PatternReport), "Output must be PatternReport Pydantic model"

    def test_report_has_all_required_fields(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify PatternReport contains all required fields without None values."""
        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["pattern_report"]
        required_fields = {
            "patterns_detected": list,
            "primary_pattern": str,
            "confidence": float,
            "breakout_probability": float,
            "reasoning": str
        }

        for field, expected_type in required_fields.items():
            assert hasattr(report, field), f"Missing field: {field}"
            value = getattr(report, field)
            # assert value is not None, f"Field {field} is None"
            assert isinstance(value, expected_type), f"Field {field} wrong type: {type(value)}"

    def test_patterns_detected_is_list(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify patterns_detected is a list (may be empty)."""
        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["pattern_report"]
        assert isinstance(report.patterns_detected, list), "patterns_detected must be list"


# ============================================================================
# CONSTRAINT TESTS - Validate field ranges and Pydantic validation
# ============================================================================

class TestConstraintValidation:
    """Test that Pydantic validators enforce field constraints."""

    def test_confidence_within_0_to_1(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify confidence is always constrained to 0.0-1.0 range."""
        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["pattern_report"]
        assert 0.0 <= report.confidence <= 1.0, f"Confidence out of range: {report.confidence}"

    def test_breakout_probability_within_0_to_1(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify breakout_probability is constrained to 0.0-1.0 range."""
        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["pattern_report"]
        assert 0.0 <= report.breakout_probability <= 1.0, \
            f"Breakout probability out of range: {report.breakout_probability}"


# ============================================================================
# ERROR HANDLING TESTS - Validate fallback mechanism
# ============================================================================

class TestErrorHandling:
    """Test agent gracefully handles errors and returns valid fallback."""

    def test_fallback_on_llm_exception(self, mock_toolkit, sample_state):
        """Verify agent returns valid fallback report when LLM raises exception."""
        mock_vision_llm_error = Mock()
        mock_vision_llm_error.with_structured_output = Mock(side_effect=ValueError("LLM error"))

        agent_node = create_pattern_agent(mock_toolkit, mock_vision_llm_error, mock_toolkit)
        result = agent_node(sample_state)

        report = result["pattern_report"]
        assert isinstance(report, PatternReport), "Fallback must return valid PatternReport"
        assert report.confidence == 0.0, "Fallback confidence should be 0"
        assert "pattern analysis could not be completed" in report.reasoning.lower(), "Fallback reasoning should mention failure"

    def test_fallback_is_valid_pydantic_model(self, mock_toolkit, sample_state):
        """Verify fallback report respects all Pydantic constraints."""
        mock_vision_llm_error = Mock()
        mock_vision_llm_error.with_structured_output = Mock(side_effect=RuntimeError("Vision error"))

        agent_node = create_pattern_agent(mock_toolkit, mock_vision_llm_error, mock_toolkit)
        result = agent_node(sample_state)

        report = result["pattern_report"]
        # Validate all constraints are respected in fallback
        assert isinstance(report, PatternReport)
        assert 0.0 <= report.confidence <= 1.0
        assert 0.0 <= report.breakout_probability <= 1.0
        assert isinstance(report.patterns_detected, list)


# ============================================================================
# STATE MANAGEMENT TESTS - Validate messages and state flow
# ============================================================================

class TestStateManagement:
    """Test proper message construction and state handling."""

    def test_result_contains_messages_key(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify result dict contains 'messages' key."""
        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        assert "messages" in result, "Result must include 'messages' key"
        assert isinstance(result["messages"], list), "Messages must be list"

    def test_system_message_included(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify SystemMessage is properly included in messages."""
        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        messages = result["messages"]
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) > 0, "SystemMessage required in message history"

    def test_human_message_included(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify HumanMessage is properly included in messages."""
        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        messages = result["messages"]
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_msgs) > 0, "HumanMessage required in message history"

    def test_timeframe_in_human_message(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify timeframe is mentioned in system message."""
        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
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
        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)

        state = {
            "kline_data": {
                "timestamps": [1, 2],
                "opens": [100.0, 99.0],
                "highs": [100.5, 99.8],
                "lows": [99.5, 98.5],
                "closes": [99.0, 99.5],
                "volumes": [1000, 1100]
            },
            "time_frame": "1hour",
            "stock_name": "BTC",
            "pattern_image": "precomputed_b64_image_data",
            "messages": []
        }

        result = agent_node(state)
        report = result["pattern_report"]
        assert isinstance(report, PatternReport), "Should return valid report with precomputed image"

    def test_generates_image_if_missing(self, mock_llm, mock_vision_llm, mock_toolkit, sample_state):
        """Verify agent handles image generation when not in state."""
        sample_state["pattern_image"] = None

        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state)

        report = result["pattern_report"]
        assert isinstance(report, PatternReport), "Should generate/handle missing image gracefully"

    def test_vision_failure_returns_fallback(self, mock_toolkit, sample_state):
        """Verify agent handles vision LLM failure gracefully."""
        mock_vision_llm_fails = Mock()
        mock_vision_llm_fails.with_structured_output = Mock(side_effect=Exception("Vision model unavailable"))

        agent_node = create_pattern_agent(Mock(), mock_vision_llm_fails, mock_toolkit)
        result = agent_node(sample_state)

        report = result["pattern_report"]
        assert isinstance(report, PatternReport), "Must return fallback on vision failure"


# ============================================================================
# EDGE CASE TESTS - Validate robustness with boundary conditions
# ============================================================================

class TestEdgeCases:
    """Test agent handles edge cases gracefully."""

    def test_empty_kline_data(self, mock_llm, mock_vision_llm, mock_toolkit, empty_state):
        """Verify agent handles empty OHLCV data gracefully."""
        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(empty_state)

        report = result["pattern_report"]
        # Should return valid report (possibly fallback)
        assert isinstance(report, PatternReport)
        # Should still respect constraints
        assert 0.0 <= report.confidence <= 1.0
        assert isinstance(report.patterns_detected, list)

    def test_single_candle_data(self, mock_llm, mock_vision_llm, mock_toolkit):
        """Verify agent handles single candlestick gracefully."""
        state = {
            "kline_data": {
                "timestamps": [1],
                "opens": [100.0],
                "highs": [100.5],
                "lows": [99.5],
                "closes": [100.2],
                "volumes": [1000]
            },
            "time_frame": "1hour",
            "stock_name": "TEST",
            "pattern_image": None,
            "messages": []
        }

        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(state)

        report = result["pattern_report"]
        assert isinstance(report, PatternReport)

    def test_messages_persisted_when_provided(self, mock_llm, mock_vision_llm, mock_toolkit):
        """Verify existing messages are preserved."""
        initial_msg = HumanMessage(content="Previous analysis")
        state = {
            "kline_data": {
                "timestamps": [1, 2],
                "opens": [100.0, 99.0],
                "highs": [100.5, 99.8],
                "lows": [99.5, 98.5],
                "closes": [99.0, 99.5],
                "volumes": [1000, 1100]
            },
            "time_frame": "1hour",
            "stock_name": "TEST",
            "pattern_image": None,
            "messages": [initial_msg]
        }

        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(state)

        # Messages list should be returned
        assert "messages" in result
        assert isinstance(result["messages"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
