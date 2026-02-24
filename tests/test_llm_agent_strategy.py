"""Unit tests for LLMAgentStrategy."""

from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import Mock

import pytest

from quantagent.models import ActivePosition, OrderSide
from quantagent.strategy.llm_agent_strategy import LLMAgentStrategy


@pytest.fixture
def mock_trading_graph():
    """Fixture for mock TradingGraph."""
    graph = Mock()
    graph.graph = Mock()
    return graph


@pytest.fixture
def llm_strategy(mock_trading_graph):
    """Fixture for LLMAgentStrategy."""
    return LLMAgentStrategy(mock_trading_graph)


@pytest.fixture
def sample_kline_data() -> List[Dict]:
    """Fixture for sample kline data."""
    return [
        {
            "timestamp": datetime(2024, 1, 1, 0, 0) + timedelta(hours=i * 4),
            "open": 100 + i,
            "high": 105 + i,
            "low": 99 + i,
            "close": 103 + i,
            "volume": 1000,
        }
        for i in range(30)
    ]


class TestGenerateSignalLong:
    """Test LLM signal generation for LONG."""

    def test_generate_long_signal(
        self, llm_strategy, mock_trading_graph, sample_kline_data
    ):
        """AC2.1: LLMAgentStrategy generates valid LONG signal."""
        # Mock graph response
        mock_trading_graph.graph.invoke.return_value = {
            "final_trade_decision": "LONG with 0.75 confidence",
            "reasoning": "Strong bullish momentum",
        }

        signal = llm_strategy.generate_signal(sample_kline_data, "BTCUSDT", "4h", 100.0)

        assert signal is not None
        assert signal.decision == "LONG"
        assert signal.confidence == 0.75
        assert signal.entry_price == 100.0
        assert signal.stop_loss == 98.0  # 2% below
        assert signal.take_profit == 103.0  # 3% above
        assert signal.reasoning == "Strong bullish momentum"
        assert signal.trailing_stop_pct == 0.05

    def test_generate_short_signal(
        self, llm_strategy, mock_trading_graph, sample_kline_data
    ):
        """LLMAgentStrategy generates valid SHORT signal."""
        mock_trading_graph.graph.invoke.return_value = {
            "final_trade_decision": "SHORT",
            "decision_report": {"reasoning": "Bearish trend confirmed"},
        }

        signal = llm_strategy.generate_signal(sample_kline_data, "ETHUSDT", "4h", 100.0)

        assert signal is not None
        assert signal.decision == "SHORT"
        assert signal.confidence == 1.0  # Default when not specified
        assert signal.stop_loss == 102.0  # 2% above
        assert signal.take_profit == 97.0  # 3% below

    def test_generate_hold_signal_returns_none(
        self, llm_strategy, mock_trading_graph, sample_kline_data
    ):
        """AC2.4: Strategy returns None when HOLD."""
        mock_trading_graph.graph.invoke.return_value = {
            "final_trade_decision": "HOLD",
        }

        signal = llm_strategy.generate_signal(sample_kline_data, "BTCUSDT", "4h", 100.0)

        assert signal is None


class TestParseDecision:
    """Test decision string parsing."""

    def test_parse_long_with_confidence(self, llm_strategy):
        """Parse LONG with confidence."""
        decision, confidence = llm_strategy._parse_decision("LONG with 0.85 confidence")

        assert decision == "LONG"
        assert confidence == 0.85

    def test_parse_short_simple(self, llm_strategy):
        """Parse simple SHORT."""
        decision, confidence = llm_strategy._parse_decision("SHORT")

        assert decision == "SHORT"
        assert confidence == 1.0

    def test_parse_hold(self, llm_strategy):
        """Parse HOLD."""
        decision, confidence = llm_strategy._parse_decision("HOLD - wait for signal")

        assert decision == "HOLD"
        assert confidence == 1.0

    def test_parse_confidence_clamped(self, llm_strategy):
        """Confidence is clamped to [0, 1]."""
        decision, confidence = llm_strategy._parse_decision("LONG confidence 1.5")

        assert decision == "LONG"
        assert confidence == 1.0  # Clamped


class TestShouldReevaluate:
    """Test re-evaluation logic."""

    def test_should_not_reevaluate(self, llm_strategy):
        """AC2.2: LLMAgentStrategy does not re-evaluate."""
        position = Mock(spec=ActivePosition)
        position.side = OrderSide.BUY

        should_reeval = llm_strategy.should_reevaluate(position, 105.0)

        assert should_reeval is False


class TestUsesExistingGraph:
    """Test that LLMAgentStrategy uses TradingGraph."""

    def test_invokes_graph(self, llm_strategy, mock_trading_graph, sample_kline_data):
        """AC2.2: Strategy invokes TradingGraph internally."""
        mock_trading_graph.graph.invoke.return_value = {
            "final_trade_decision": "LONG",
        }

        llm_strategy.generate_signal(sample_kline_data, "BTCUSDT", "4h", 100.0)

        # Verify graph was invoked
        mock_trading_graph.graph.invoke.assert_called_once()

        # Verify state structure
        call_args = mock_trading_graph.graph.invoke.call_args[0][0]
        assert call_args["kline_data"] == sample_kline_data
        assert call_args["stock_name"] == "BTCUSDT"
        assert call_args["time_frame"] == "4h"

    def test_invokes_graph_with_thread_id(
        self, llm_strategy, mock_trading_graph, sample_kline_data
    ):
        """Verify strategy passes thread_id config when provided (for checkpointing)."""
        mock_trading_graph.graph.invoke.return_value = {
            "final_trade_decision": "LONG with 0.8 confidence",
            "reasoning": "Bullish trend",
        }

        thread_id = "backtest_123_BTC_2024-01-01T00:00:00"
        signal = llm_strategy.generate_signal(
            sample_kline_data, "BTC", "4h", 50000.0, thread_id=thread_id
        )

        # Verify graph was invoked with correct config
        mock_trading_graph.graph.invoke.assert_called_once()
        call_args = mock_trading_graph.graph.invoke.call_args

        # Verify state (first positional arg)
        state = call_args[0][0]
        assert state["stock_name"] == "BTC"
        assert state["time_frame"] == "4h"
        assert state["kline_data"] == sample_kline_data

        # Verify config was passed as keyword arg with thread_id
        config = call_args[1]["config"]
        assert config is not None
        assert "configurable" in config
        assert config["configurable"]["thread_id"] == thread_id

        # Verify signal was generated correctly
        assert signal is not None
        assert signal.decision == "LONG"
        assert signal.confidence == 0.8

    def test_invokes_graph_without_thread_id(
        self, llm_strategy, mock_trading_graph, sample_kline_data
    ):
        """Verify strategy invokes graph without config when thread_id not provided."""
        mock_trading_graph.graph.invoke.return_value = {
            "final_trade_decision": "SHORT",
        }

        signal = llm_strategy.generate_signal(
            sample_kline_data, "ETH", "1h", 3000.0, thread_id=None
        )

        # Verify graph was invoked
        mock_trading_graph.graph.invoke.assert_called_once()
        call_args = mock_trading_graph.graph.invoke.call_args

        # Verify config is None (second positional arg or keyword arg)
        config = call_args[1].get("config") if len(call_args) > 1 else None
        assert config is None

        # Verify signal was generated
        assert signal is not None
        assert signal.decision == "SHORT"
