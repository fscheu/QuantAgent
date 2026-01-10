"""
Tests for message state management across multi-agent graph execution.

Validates QuantAgent-h7d fix: Analysis agents (indicator, pattern, trend) do NOT
add messages to shared state, while Decision Agent DOES add messages.

See docs/03_design/MESSAGE_STATE_MANAGEMENT.md for design rationale.
"""

import pytest

from quantagent.agent_models import (IndicatorReport, PatternReport,
                                     TradingDecision, TrendReport)


class TestMessageStateManagement:
    """Test that message management follows design in MESSAGE_STATE_MANAGEMENT.md."""

    def test_indicator_agent_does_not_add_messages(
        self, mock_llm, mock_toolkit, sample_state_inicial
    ):
        """Verify Indicator Agent does NOT add messages to state."""
        from quantagent.indicator_agent import create_indicator_agent

        agent_node = create_indicator_agent(mock_llm, mock_toolkit)
        result = agent_node(sample_state_inicial)

        # Analysis agents communicate via structured reports, not messages
        assert (
            "messages" not in result
        ), "Indicator agent must NOT add messages to state"
        assert (
            "indicator_report" in result
        ), "Indicator agent must return indicator_report"
        assert isinstance(result["indicator_report"], IndicatorReport)

    def test_pattern_agent_does_not_add_messages(
        self, mock_llm, mock_vision_llm, mock_toolkit, sample_state_inicial
    ):
        """Verify Pattern Agent does NOT add messages to state."""
        from quantagent.pattern_agent import create_pattern_agent

        agent_node = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state_inicial)

        # Analysis agents communicate via structured reports, not messages
        assert "messages" not in result, "Pattern agent must NOT add messages to state"
        assert "pattern_report" in result, "Pattern agent must return pattern_report"
        assert isinstance(result["pattern_report"], PatternReport)

    def test_trend_agent_does_not_add_messages(
        self, mock_llm, mock_vision_llm, mock_toolkit, sample_state_inicial
    ):
        """Verify Trend Agent does NOT add messages to state."""
        from quantagent.trend_agent import create_trend_agent

        agent_node = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)
        result = agent_node(sample_state_inicial)

        # Analysis agents communicate via structured reports, not messages
        assert "messages" not in result, "Trend agent must NOT add messages to state"
        assert "trend_report" in result, "Trend agent must return trend_report"
        assert isinstance(result["trend_report"], TrendReport)

    def test_decision_agent_adds_messages(
        self, mock_llm, mock_vision_llm, mock_toolkit, sample_state_inicial
    ):
        """Verify Decision Agent DOES add messages to state."""
        from quantagent.decision_agent import create_final_trade_decider
        from quantagent.indicator_agent import create_indicator_agent
        from quantagent.pattern_agent import create_pattern_agent
        from quantagent.trend_agent import create_trend_agent

        # Build complete state with all analysis reports
        state = sample_state_inicial.copy()
        state.update(create_indicator_agent(mock_llm, mock_toolkit)(state))
        state.update(
            create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)(state)
        )
        state.update(create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)(state))

        # Decision agent should add messages
        decision_node = create_final_trade_decider(mock_llm)
        result = decision_node(state)

        # Decision agent adds messages to enable follow-up conversations
        assert "messages" in result, "Decision agent must add messages to state"
        assert isinstance(result["messages"], list), "messages must be list"
        assert len(result["messages"]) > 0, "messages must not be empty"
        assert (
            "final_trade_decision" in result
        ), "Decision agent must return final_trade_decision"

    def test_parallel_agents_do_not_conflict(
        self, mock_llm, mock_vision_llm, mock_toolkit, sample_state_inicial
    ):
        """
        Verify parallel analysis agents don't cause INVALID_CONCURRENT_GRAPH_UPDATE.

        This was the original bug: all agents were adding messages to state,
        causing LangGraph to error when parallel nodes updated 'messages' differently.
        """
        from quantagent.indicator_agent import create_indicator_agent
        from quantagent.pattern_agent import create_pattern_agent
        from quantagent.trend_agent import create_trend_agent

        state = sample_state_inicial.copy()

        # Simulate parallel execution (all agents run on same initial state)
        indicator_result = create_indicator_agent(mock_llm, mock_toolkit)(state)
        pattern_result = create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)(
            state
        )
        trend_result = create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)(
            state
        )

        # None of them should add 'messages' key
        assert "messages" not in indicator_result
        assert "messages" not in pattern_result
        assert "messages" not in trend_result

        # All should have their respective reports
        assert "indicator_report" in indicator_result
        assert "pattern_report" in pattern_result
        assert "trend_report" in trend_result

    def test_full_pipeline_message_flow(
        self, mock_llm, mock_vision_llm, mock_toolkit, sample_state_inicial
    ):
        """
        Verify message flow through complete pipeline:
        - Analysis agents: no messages added
        - Decision agent: adds messages
        """
        from quantagent.decision_agent import create_final_trade_decider
        from quantagent.indicator_agent import create_indicator_agent
        from quantagent.pattern_agent import create_pattern_agent
        from quantagent.trend_agent import create_trend_agent

        state = sample_state_inicial.copy()

        # Step 1: Indicator agent (no messages)
        state.update(create_indicator_agent(mock_llm, mock_toolkit)(state))
        messages_after_indicator = state.get("messages", [])
        assert len(messages_after_indicator) == 0, "No messages after indicator agent"

        # Step 2: Pattern agent (no messages)
        state.update(
            create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)(state)
        )
        messages_after_pattern = state.get("messages", [])
        assert len(messages_after_pattern) == 0, "No messages after pattern agent"

        # Step 3: Trend agent (no messages)
        state.update(create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)(state))
        messages_after_trend = state.get("messages", [])
        assert len(messages_after_trend) == 0, "No messages after trend agent"

        # Step 4: Decision agent (adds messages)
        state.update(create_final_trade_decider(mock_llm)(state))
        messages_after_decision = state.get("messages", [])
        assert (
            len(messages_after_decision) > 0
        ), "Decision agent must add messages to state"

        # Verify decision output
        assert "final_trade_decision" in state
        assert isinstance(state["final_trade_decision"], TradingDecision)

    def test_indicator_agent_fallback_does_not_add_messages(
        self, mock_toolkit, sample_state_inicial
    ):
        """
        Verify Indicator Agent fallback path does NOT add messages.

        Tests the error handling path (lines 86-99 in indicator_agent.py)
        to ensure fallback reports also exclude messages from state.
        """
        from unittest.mock import Mock

        from quantagent.indicator_agent import create_indicator_agent

        # Mock LLM to raise exception at invoke level (after with_structured_output)
        failing_llm = Mock()
        failing_llm.bind_tools = Mock(return_value=failing_llm)

        # Create a mock structured LLM that raises on invoke
        mock_structured = Mock()
        mock_structured.invoke = Mock(side_effect=Exception("Simulated LLM failure"))
        failing_llm.with_structured_output = Mock(return_value=mock_structured)

        agent_node = create_indicator_agent(failing_llm, mock_toolkit)
        result = agent_node(sample_state_inicial)

        # Even in fallback, should NOT add messages
        assert (
            "messages" not in result
        ), "Fallback indicator agent must NOT add messages"
        assert "indicator_report" in result
        assert isinstance(result["indicator_report"], IndicatorReport)
        # Verify it's actually the fallback (confidence should be 0.0)
        assert result["indicator_report"].confidence == 0.0
        assert "failed" in result["indicator_report"].reasoning.lower()

    def test_pattern_agent_fallback_does_not_add_messages(
        self, mock_llm, mock_toolkit, sample_state_inicial
    ):
        """
        Verify Pattern Agent fallback path does NOT add messages.

        Tests the error handling path (lines 161-168 in pattern_agent.py)
        to ensure fallback reports also exclude messages from state.
        """
        from unittest.mock import Mock

        from quantagent.pattern_agent import create_pattern_agent

        # Mock vision LLM to raise exception
        failing_vision_llm = Mock()
        failing_vision_llm.invoke = Mock(side_effect=Exception("Vision LLM failed"))

        agent_node = create_pattern_agent(mock_llm, failing_vision_llm, mock_toolkit)
        result = agent_node(sample_state_inicial)

        # Even in fallback, should NOT add messages
        assert "messages" not in result, "Fallback pattern agent must NOT add messages"
        assert "pattern_report" in result
        assert isinstance(result["pattern_report"], PatternReport)
        # Verify it's actually the fallback
        assert result["pattern_report"].primary_pattern == "failed to analyze"

    def test_trend_agent_fallback_does_not_add_messages(
        self, mock_llm, mock_toolkit, sample_state_inicial
    ):
        """
        Verify Trend Agent fallback path does NOT add messages.

        Tests the error handling path (lines 111-125 in trend_agent.py)
        to ensure fallback reports also exclude messages from state.
        """
        from unittest.mock import Mock

        from quantagent.trend_agent import create_trend_agent

        # Mock vision LLM to raise exception
        failing_vision_llm = Mock()
        failing_vision_llm.invoke = Mock(side_effect=Exception("Vision LLM failed"))

        agent_node = create_trend_agent(mock_llm, failing_vision_llm, mock_toolkit)
        result = agent_node(sample_state_inicial)

        # Even in fallback, should NOT add messages
        assert "messages" not in result, "Fallback trend agent must NOT add messages"
        assert "trend_report" in result
        assert isinstance(result["trend_report"], TrendReport)
        # Verify it's actually the fallback (trend_strength should be 0.0)
        assert result["trend_report"].trend_strength == 0.0

    def test_decision_agent_fallback_still_adds_messages(
        self, mock_llm, mock_vision_llm, mock_toolkit, sample_state_inicial
    ):
        """
        Verify Decision Agent fallback path STILL adds messages.

        Tests the error handling path (lines 174-179 in decision_agent.py)
        to ensure even fallback decisions include messages for follow-up.
        """
        from unittest.mock import Mock

        from quantagent.decision_agent import create_final_trade_decider
        from quantagent.indicator_agent import create_indicator_agent
        from quantagent.pattern_agent import create_pattern_agent
        from quantagent.trend_agent import create_trend_agent

        # Build state with valid reports using working mocks
        state = sample_state_inicial.copy()
        state.update(create_indicator_agent(mock_llm, mock_toolkit)(state))
        state.update(
            create_pattern_agent(mock_llm, mock_vision_llm, mock_toolkit)(state)
        )
        state.update(create_trend_agent(mock_llm, mock_vision_llm, mock_toolkit)(state))

        # Mock decision LLM to raise exception at invoke level
        failing_decision_llm = Mock()
        mock_structured = Mock()
        mock_structured.invoke = Mock(side_effect=Exception("Decision LLM failed"))
        failing_decision_llm.with_structured_output = Mock(return_value=mock_structured)

        decision_node = create_final_trade_decider(failing_decision_llm)
        result = decision_node(state)

        # Even in fallback, Decision Agent MUST add messages
        assert "messages" in result, "Fallback decision agent MUST add messages"
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0
        assert "final_trade_decision" in result
        # Verify it's actually the fallback (decision should be HOLD)
        assert result["final_trade_decision"].decision == "HOLD"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
