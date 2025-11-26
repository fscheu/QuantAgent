"""
Unit tests for refactored decision_agent with structured Pydantic output.

Tests verify:
- Agent consumes structured Pydantic reports (IndicatorReport, PatternReport, TrendReport)
- Produces valid TradingDecision output with with_structured_output pattern
- Decision logic correctly combines report signals
- Error handling with fallback HOLD decision
"""

import json
import pytest
from unittest.mock import Mock

from quantagent.agent_models import (
    IndicatorReport, PatternReport, TrendReport, TradingDecision
)
from quantagent.decision_agent import create_final_trade_decider


@pytest.fixture
def sample_indicator_report():
    """Fixture providing sample IndicatorReport."""
    return IndicatorReport(
        macd=0.8,
        macd_signal=0.6,
        macd_histogram=0.2,
        rsi=70.0,
        rsi_level="overbought",
        roc=2.5,
        stochastic=75.0,
        willr=-25.0,
        trend_direction="bullish",
        confidence=0.8,
        reasoning="Strong bullish momentum with MACD crossover"
    )


@pytest.fixture
def sample_pattern_report():
    """Fixture providing sample PatternReport."""
    return PatternReport(
        patterns_detected=["double_bottom", "bullish_flag"],
        primary_pattern="double_bottom",
        confidence=0.75,
        breakout_probability=0.65,
        reasoning="Clear double bottom with bullish flag"
    )


@pytest.fixture
def sample_trend_report():
    """Fixture providing sample TrendReport."""
    return TrendReport(
        support_level=44000.0,
        resistance_level=46000.0,
        trend_direction="upward",
        trend_strength=0.75,
        reasoning="Upward trend with strong support bounce"
    )


# @pytest.fixture
# def mock_llm():
#     """Fixture providing mock LLM."""
#     return MockLLM()


class TestDecisionAgentOutput:
    """Test structured trading decision output."""

    def test_output_is_pydantic_model(self, mock_llm, sample_indicator_report,
                                     sample_pattern_report, sample_trend_report):
        """Verify output is TradingDecision instance."""
        agent_node = create_final_trade_decider(mock_llm)

        state = {
            "indicator_report": sample_indicator_report,
            "pattern_report": sample_pattern_report,
            "trend_report": sample_trend_report,
            "time_frame": "4hour",
            "stock_name": "BTC",
            "messages": []
        }

        result = agent_node(state)
        decision = result["final_trade_decision"]
        assert isinstance(decision, TradingDecision)

    def test_decision_has_all_fields(self, mock_llm, sample_indicator_report,
                                    sample_pattern_report, sample_trend_report):
        """Verify TradingDecision contains all required fields."""
        agent_node = create_final_trade_decider(mock_llm)

        state = {
            "indicator_report": sample_indicator_report,
            "pattern_report": sample_pattern_report,
            "trend_report": sample_trend_report,
            "time_frame": "4hour",
            "stock_name": "BTC",
            "messages": []
        }

        result = agent_node(state)
        decision = result["final_trade_decision"]

        assert hasattr(decision, "decision")
        assert hasattr(decision, "confidence")
        assert hasattr(decision, "reasoning")
        assert hasattr(decision, "risk_level")
        assert hasattr(decision, "entry_price")
        assert hasattr(decision, "stop_loss")
        assert hasattr(decision, "take_profit")


class TestDecisionValues:
    """Test trading decision values."""

    def test_decision_is_valid_choice(self, mock_llm, sample_indicator_report,
                                     sample_pattern_report, sample_trend_report):
        """Verify decision is LONG, SHORT, or HOLD."""
        agent_node = create_final_trade_decider(mock_llm)

        state = {
            "indicator_report": sample_indicator_report,
            "pattern_report": sample_pattern_report,
            "trend_report": sample_trend_report,
            "time_frame": "4hour",
            "stock_name": "BTC",
            "messages": []
        }

        result = agent_node(state)
        decision = result["final_trade_decision"]
        assert decision.decision in ["LONG", "SHORT", "HOLD"]

    def test_confidence_in_valid_range(self, mock_llm, sample_indicator_report,
                                      sample_pattern_report, sample_trend_report):
        """Verify confidence is in 0.0-1.0 range."""
        agent_node = create_final_trade_decider(mock_llm)

        state = {
            "indicator_report": sample_indicator_report,
            "pattern_report": sample_pattern_report,
            "trend_report": sample_trend_report,
            "time_frame": "4hour",
            "stock_name": "BTC",
            "messages": []
        }

        result = agent_node(state)
        decision = result["final_trade_decision"]
        assert 0.0 <= decision.confidence <= 1.0

    def test_risk_level_valid_values(self, mock_llm, sample_indicator_report,
                                    sample_pattern_report, sample_trend_report):
        """Verify risk_level is one of valid values."""
        agent_node = create_final_trade_decider(mock_llm)

        state = {
            "indicator_report": sample_indicator_report,
            "pattern_report": sample_pattern_report,
            "trend_report": sample_trend_report,
            "time_frame": "4hour",
            "stock_name": "BTC",
            "messages": []
        }

        result = agent_node(state)
        decision = result["final_trade_decision"]
        assert decision.risk_level in ["low", "medium", "high"]


class TestDecisionLogic:
    """Test decision logic with different signal combinations."""

    def test_bullish_alignment(self, mock_llm_custom):
        """Test decision when all signals align bullish."""
        llm = mock_llm_custom({
            "decision": "LONG",
            "confidence": 0.85,
            "reasoning": "All three reports aligned bullish",
            "risk_level": "low",
            "entry_price": 45000.0,
            "stop_loss": 44800.0,
            "take_profit": 45500.0
        })

        agent_node = create_final_trade_decider(llm)

        indicator = IndicatorReport(
            macd=1.0, macd_signal=0.8, macd_histogram=0.2,
            rsi=70.0, rsi_level="overbought",
            roc=3.0, stochastic=80.0, willr=-20.0,
            trend_direction="bullish", confidence=0.9,
            reasoning="Strong bullish"
        )
        pattern = PatternReport(
            patterns_detected=["bullish_flag"],
            primary_pattern="bullish_flag",
            confidence=0.8, breakout_probability=0.75,
            reasoning="Bullish pattern"
        )
        trend = TrendReport(
            support_level=44500.0, resistance_level=46000.0,
            trend_direction="upward", trend_strength=0.85,
            reasoning="Strong uptrend"
        )

        state = {
            "indicator_report": indicator,
            "pattern_report": pattern,
            "trend_report": trend,
            "time_frame": "4hour",
            "stock_name": "BTC",
            "messages": []
        }

        result = agent_node(state)
        decision = result["final_trade_decision"]
        assert decision.decision == "LONG"
        assert decision.confidence > 0.75

    def test_bearish_alignment(self, mock_llm_custom):
        """Test decision when all signals align bearish."""
        llm = mock_llm_custom({
            "decision": "SHORT",
            "confidence": 0.82,
            "reasoning": "All three reports aligned bearish",
            "risk_level": "medium",
            "entry_price": 44500.0,
            "stop_loss": 45000.0,
            "take_profit": 43500.0
        })

        agent_node = create_final_trade_decider(llm)

        indicator = IndicatorReport(
            macd=-0.8, macd_signal=-0.6, macd_histogram=-0.2,
            rsi=30.0, rsi_level="oversold",
            roc=-2.5, stochastic=20.0, willr=-80.0,
            trend_direction="bearish", confidence=0.85,
            reasoning="Strong bearish"
        )
        pattern = PatternReport(
            patterns_detected=["bearish_flag"],
            primary_pattern="bearish_flag",
            confidence=0.75, breakout_probability=0.65,
            reasoning="Bearish pattern"
        )
        trend = TrendReport(
            support_level=43500.0, resistance_level=45000.0,
            trend_direction="downward", trend_strength=0.8,
            reasoning="Downtrend confirmed"
        )

        state = {
            "indicator_report": indicator,
            "pattern_report": pattern,
            "trend_report": trend,
            "time_frame": "4hour",
            "stock_name": "BTC",
            "messages": []
        }

        result = agent_node(state)
        decision = result["final_trade_decision"]
        assert decision.decision == "SHORT"


class TestErrorHandling:
    """Test error handling and fallback behavior."""

    def test_malformed_json_creates_hold_decision(self, mock_llm, sample_indicator_report,
                                                  sample_pattern_report, sample_trend_report):
        """Verify malformed JSON creates HOLD fallback decision."""
        # Create llm that will fail to parse (simulate error in with_structured_output)
        from unittest.mock import Mock
        llm = Mock()
        llm.with_structured_output = Mock(side_effect=ValueError("LLM parsing failed"))

        agent_node = create_final_trade_decider(llm)

        state = {
            "indicator_report": sample_indicator_report,
            "pattern_report": sample_pattern_report,
            "trend_report": sample_trend_report,
            "time_frame": "4hour",
            "stock_name": "BTC",
            "messages": []
        }

        result = agent_node(state)
        decision = result["final_trade_decision"]
        assert isinstance(decision, TradingDecision)
        assert decision.confidence == 0.0
        assert decision.decision == "HOLD"

    def test_markdown_json_parsing(self, mock_llm_custom, sample_indicator_report,
                                  sample_pattern_report, sample_trend_report):
        """Verify structured output directly returns TradingDecision."""
        # Use custom mock with specific decision values
        llm = mock_llm_custom({
            "decision": "LONG",
            "confidence": 0.7,
            "reasoning": "Test decision",
            "risk_level": "low",
            "entry_price": 45000.0,
            "stop_loss": 44500.0,
            "take_profit": 46000.0
        })

        agent_node = create_final_trade_decider(llm)

        state = {
            "indicator_report": sample_indicator_report,
            "pattern_report": sample_pattern_report,
            "trend_report": sample_trend_report,
            "time_frame": "4hour",
            "stock_name": "BTC",
            "messages": []
        }

        result = agent_node(state)
        decision = result["final_trade_decision"]
        assert decision.decision == "LONG"
        assert decision.confidence == 0.7


class TestStateConsumption:
    """Test that agent correctly consumes state."""

    def test_consumes_structured_reports(self, mock_llm, sample_indicator_report,
                                        sample_pattern_report, sample_trend_report):
        """Verify agent consumes Pydantic model reports."""
        agent_node = create_final_trade_decider(mock_llm)

        state = {
            "indicator_report": sample_indicator_report,
            "pattern_report": sample_pattern_report,
            "trend_report": sample_trend_report,
            "time_frame": "4hour",
            "stock_name": "BTC",
            "messages": []
        }

        # Should not raise any errors
        result = agent_node(state)
        assert "final_trade_decision" in result
        assert isinstance(result["final_trade_decision"], TradingDecision)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
