"""Additional tests for TradingStrategy - Constraint validation and error handling."""

from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from quantagent.models import ActivePosition, ExitPolicy, OrderSide
from quantagent.strategy.base import TradingSignal, TradingStrategy
from quantagent.strategy.llm_agent_strategy import LLMAgentStrategy
from quantagent.strategy.rsi_strategy import RSIMeanReversionStrategy


class TestTradingSignalConstraints:
    """Test Pydantic constraints validation for TradingSignal."""

    def test_confidence_must_be_between_0_and_1(self):
        """Confidence must be validated as 0.0 <= c <= 1.0."""
        # Valid: 0.0
        signal = TradingSignal(decision="LONG", confidence=0.0)
        assert signal.confidence == 0.0

        # Valid: 1.0
        signal = TradingSignal(decision="SHORT", confidence=1.0)
        assert signal.confidence == 1.0

        # Invalid: > 1.0
        with pytest.raises(ValidationError):
            TradingSignal(decision="LONG", confidence=1.5)

        # Invalid: < 0.0
        with pytest.raises(ValidationError):
            TradingSignal(decision="LONG", confidence=-0.1)

    def test_decision_must_be_valid_string(self):
        """Decision accepts any string but should be LONG/SHORT/HOLD by convention."""
        # Valid values
        for decision in ["LONG", "SHORT", "HOLD"]:
            signal = TradingSignal(decision=decision, confidence=0.5)
            assert signal.decision == decision

        # Edge case: lowercase (allowed but not conventional)
        signal = TradingSignal(decision="long", confidence=0.5)
        assert signal.decision == "long"

        # Edge case: invalid decision (allowed by Pydantic but wrong semantically)
        # This is caught at runtime, not by Pydantic
        signal = TradingSignal(decision="INVALID", confidence=0.5)
        assert signal.decision == "INVALID"

    def test_optional_fields_can_be_none(self):
        """Optional fields (SL, TP, entry_price) can be None."""
        signal = TradingSignal(decision="HOLD", confidence=0.5)

        assert signal.entry_price is None
        assert signal.stop_loss is None
        assert signal.take_profit is None
        assert signal.trailing_stop_pct is None
        assert signal.max_hold_candles is None

    def test_exit_policy_default_is_trailing_stop(self):
        """Default exit_policy is TRAILING_STOP."""
        signal = TradingSignal(decision="LONG", confidence=0.8)
        assert signal.exit_policy == ExitPolicy.TRAILING_STOP


class TestLLMAgentStrategyErrorHandling:
    """Test error handling and fallback in LLMAgentStrategy."""

    @pytest.fixture
    def sample_kline_data(self) -> List[Dict]:
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

    def test_graph_returns_none(self, sample_kline_data):
        """Handle case when graph.invoke returns None."""
        mock_graph = Mock()
        mock_graph.graph = Mock()
        mock_graph.graph.invoke.return_value = None

        strategy = LLMAgentStrategy(mock_graph)

        # Should handle gracefully (likely returns None or raises)
        # This tests robustness
        with pytest.raises(AttributeError):
            strategy.generate_signal(sample_kline_data, "BTCUSDT", "4h", 100.0)

        # Either returns None (HOLD) or handles the error
        # Current implementation will raise AttributeError on .get()
        # This is expected behavior - no fallback in current design

    def test_graph_returns_empty_dict(self, sample_kline_data):
        """Handle case when graph returns empty dict."""
        mock_graph = Mock()
        mock_graph.graph = Mock()
        mock_graph.graph.invoke.return_value = {}

        strategy = LLMAgentStrategy(mock_graph)

        signal = strategy.generate_signal(sample_kline_data, "BTCUSDT", "4h", 100.0)

        # Default is "HOLD" when no decision found
        assert signal is None

    def test_graph_returns_malformed_decision(self, sample_kline_data):
        """Handle case when decision string is malformed."""
        mock_graph = Mock()
        mock_graph.graph = Mock()
        mock_graph.graph.invoke.return_value = {
            "final_trade_decision": "UNKNOWN_ACTION with confidence maybe 0.7"
        }

        strategy = LLMAgentStrategy(mock_graph)

        signal = strategy.generate_signal(sample_kline_data, "BTCUSDT", "4h", 100.0)

        # Parser should default to HOLD for unknown decisions
        assert signal is None

    def test_parse_decision_with_invalid_float(self):
        """Parser handles strings with invalid floats gracefully."""
        mock_graph = Mock()
        strategy = LLMAgentStrategy(mock_graph)

        # Case: string has "confidence" but no valid number
        decision, confidence = strategy._parse_decision("LONG with confidence: high")
        assert decision == "LONG"
        assert confidence == 1.0  # Default when parsing fails

        # Case: multiple numbers, only first valid one used
        decision, confidence = strategy._parse_decision("LONG 0.8 0.9 confidence")
        assert decision == "LONG"
        assert confidence == 0.8  # First valid float

    def test_parse_decision_edge_cases(self):
        """Test parsing edge cases."""
        mock_graph = Mock()
        strategy = LLMAgentStrategy(mock_graph)

        # Case: number > 1.0 (should be ignored)
        decision, confidence = strategy._parse_decision("LONG with 99.5 accuracy")
        assert decision == "LONG"
        assert confidence == 1.0  # No valid confidence found

        # Case: decision word appears multiple times
        decision, confidence = strategy._parse_decision("LONG LONG LONG 0.75")
        assert decision == "LONG"
        assert confidence == 0.75


class TestTrailingStopEdgeCases:
    """Test edge cases in trailing stop logic."""

    def test_short_trailing_stop_with_declining_price(self):
        """
        Test SHORT trailing stop updates lowest_price_seen correctly.
        Covers line 161 in base.py.
        """
        from decimal import Decimal


        class TestStrategy(TradingStrategy):
            def generate_signal(self, kline_data, symbol, timeframe, current_price):
                return None

            def should_reevaluate(self, position, current_price):
                return False

        strategy = TestStrategy()

        # Create SHORT position
        position = ActivePosition(
            id=1,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            entry_price=Decimal("100.0"),
            stop_loss=Decimal("105.0"),
            take_profit=Decimal("90.0"),
            quantity=Decimal("1.0"),
            decision_timestamp=datetime.utcnow(),
            candles_since_entry=0,
            exit_policy=ExitPolicy.TRAILING_STOP,
            trailing_stop_pct=0.05,
            prediction_horizon=3,
            candles_direction=[],
            is_active=True,
            lowest_price_seen=None,
        )

        import pandas as pd

        ohlc_data = pd.DataFrame(
            {"open": [100], "high": [105], "low": [95], "close": [98], "volume": [1000]}
        )

        # First check: price drops to 95 (profitable for SHORT)
        should_exit, reason = strategy.should_exit(position, 95.0, ohlc_data)

        # Should NOT exit yet (price moving in our favor)
        assert should_exit is False
        assert position.lowest_price_seen == 95.0  # Updated

        # Second check: price rises to 100 (trailing stop = 95 * 1.05 = 99.75)
        should_exit, reason = strategy.should_exit(position, 100.0, ohlc_data)

        # Should exit via trailing stop
        assert should_exit is True
        assert reason == "TRAILING_STOP"

    def test_trailing_stop_with_none_pct_does_not_crash(self):
        """Trailing stop disabled when trailing_stop_pct is None."""
        from decimal import Decimal


        strategy = RSIMeanReversionStrategy()

        position = ActivePosition(
            id=1,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=Decimal("100.0"),
            stop_loss=Decimal("95.0"),
            take_profit=Decimal("110.0"),
            quantity=Decimal("1.0"),
            decision_timestamp=datetime.utcnow(),
            candles_since_entry=0,
            exit_policy=ExitPolicy.TRAILING_STOP,
            trailing_stop_pct=None,  # Explicitly None
            prediction_horizon=3,
            candles_direction=[],
            is_active=True,
        )

        import pandas as pd

        ohlc_data = pd.DataFrame(
            {"open": [100], "high": [105], "low": [99], "close": [103], "volume": [1000]}
        )

        # Should not crash, trailing stop just disabled
        should_exit, reason = strategy.should_exit(position, 105.0, ohlc_data)

        assert should_exit is False  # Between SL and TP
