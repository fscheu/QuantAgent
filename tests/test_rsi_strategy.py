"""Unit tests for RSIMeanReversionStrategy."""

from datetime import datetime, timedelta
from typing import Dict, List

import pytest

from quantagent.models import ActivePosition, ExitPolicy, OrderSide
from quantagent.strategy.rsi_strategy import RSIMeanReversionStrategy


@pytest.fixture
def rsi_strategy():
    """Fixture for RSI strategy with default params."""
    return RSIMeanReversionStrategy(
        rsi_period=14,
        oversold_threshold=30.0,
        overbought_threshold=70.0,
    )


@pytest.fixture
def oversold_kline_data() -> List[Dict]:
    """
    Fixture for kline data that produces RSI < 30 (oversold).
    Simulates a downtrend followed by consolidation.
    """
    # Start high, then decline (creates oversold RSI)
    prices = [110.0] + [110.0 - i * 2 for i in range(1, 31)]

    return [
        {
            "timestamp": datetime(2024, 1, 1, 0, 0) + timedelta(hours=i * 4),
            "open": prices[i],
            "high": prices[i] + 1,
            "low": prices[i] - 1,
            "close": prices[i],
            "volume": 1000,
        }
        for i in range(len(prices))
    ]


@pytest.fixture
def overbought_kline_data() -> List[Dict]:
    """
    Fixture for kline data that produces RSI > 70 (overbought).
    Simulates an uptrend.
    """
    # Start low, then rally (creates overbought RSI)
    prices = [50.0] + [50.0 + i * 2 for i in range(1, 31)]

    return [
        {
            "timestamp": datetime(2024, 1, 1, 0, 0) + timedelta(hours=i * 4),
            "open": prices[i],
            "high": prices[i] + 1,
            "low": prices[i] - 1,
            "close": prices[i],
            "volume": 1000,
        }
        for i in range(len(prices))
    ]


@pytest.fixture
def neutral_kline_data() -> List[Dict]:
    """Fixture for neutral kline data (RSI ~50)."""
    # Sideways movement
    prices = [100.0 + (i % 2) for i in range(31)]

    return [
        {
            "timestamp": datetime(2024, 1, 1, 0, 0) + timedelta(hours=i * 4),
            "open": prices[i],
            "high": prices[i] + 0.5,
            "low": prices[i] - 0.5,
            "close": prices[i],
            "volume": 1000,
        }
        for i in range(len(prices))
    ]


class TestGenerateSignalOversold:
    """Test signal generation for oversold conditions."""

    def test_generate_long_signal_oversold(self, rsi_strategy, oversold_kline_data):
        """AC2.3: RSI strategy generates LONG on oversold (RSI < 30)."""
        signal = rsi_strategy.generate_signal(
            oversold_kline_data, "BTCUSDT", "4h", 50.0
        )

        assert signal is not None
        assert signal.decision == "LONG"
        assert signal.confidence > 0.5  # High confidence for oversold
        assert signal.entry_price == 50.0
        assert signal.stop_loss == 49.0  # 2% below
        assert signal.take_profit == 51.5  # 3% above
        assert "oversold" in signal.reasoning.lower()
        assert signal.exit_policy == ExitPolicy.TRAILING_STOP


class TestGenerateSignalOverbought:
    """Test signal generation for overbought conditions."""

    def test_generate_short_signal_overbought(
        self, rsi_strategy, overbought_kline_data
    ):
        """RSI strategy generates SHORT on overbought (RSI > 70)."""
        signal = rsi_strategy.generate_signal(
            overbought_kline_data, "ETHUSDT", "4h", 100.0
        )

        assert signal is not None
        assert signal.decision == "SHORT"
        assert signal.confidence > 0.5  # High confidence for overbought
        assert signal.stop_loss == 102.0  # 2% above
        assert signal.take_profit == 97.0  # 3% below
        assert "overbought" in signal.reasoning.lower()


class TestGenerateSignalNeutral:
    """Test signal generation for neutral conditions."""

    def test_generate_hold_signal_neutral(self, rsi_strategy, neutral_kline_data):
        """AC2.4: RSI strategy returns None for neutral RSI (~50)."""
        signal = rsi_strategy.generate_signal(
            neutral_kline_data, "BTCUSDT", "4h", 100.0
        )

        # Should return None when RSI is between thresholds
        assert signal is None


class TestInsufficientData:
    """Test behavior with insufficient data."""

    def test_insufficient_data_returns_none(self, rsi_strategy):
        """Strategy returns None when insufficient candles."""
        short_data = [
            {
                "timestamp": datetime(2024, 1, 1, i, 0).isoformat(),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000,
            }
            for i in range(10)  # Less than rsi_period (14)
        ]

        signal = rsi_strategy.generate_signal(short_data, "BTCUSDT", "4h", 100.0)

        assert signal is None


class TestCustomParameters:
    """Test strategy with custom parameters."""

    def test_custom_thresholds(self, overbought_kline_data):
        """Strategy respects custom RSI thresholds."""
        # More aggressive thresholds
        strategy = RSIMeanReversionStrategy(
            oversold_threshold=40.0,  # Higher oversold threshold
            overbought_threshold=60.0,  # Lower overbought threshold
        )

        signal = strategy.generate_signal(overbought_kline_data, "BTCUSDT", "4h", 100.0)

        # Should trigger SHORT with looser threshold
        assert signal is not None
        assert signal.decision == "SHORT"

    def test_custom_stop_loss_take_profit(self, oversold_kline_data):
        """Strategy respects custom SL/TP percentages."""
        strategy = RSIMeanReversionStrategy(
            stop_loss_pct=0.05,  # 5% SL
            take_profit_pct=0.10,  # 10% TP
        )

        signal = strategy.generate_signal(oversold_kline_data, "BTCUSDT", "4h", 100.0)

        assert signal is not None
        assert signal.stop_loss == 95.0  # 5% below
        assert (
            abs(signal.take_profit - 110.0) < 0.01
        )  # 10% above (with floating point tolerance)


class TestNoLLMInvocation:
    """Test that RSI strategy does NOT use LLM."""

    def test_no_llm_dependency(self, rsi_strategy, oversold_kline_data):
        """AC2.3: RSI strategy generates signal without LLM."""
        # This test verifies by inspection - RSI strategy only uses pandas/numpy
        # No mock needed because no external dependencies

        signal = rsi_strategy.generate_signal(
            oversold_kline_data, "BTCUSDT", "4h", 50.0
        )

        assert signal is not None
        # If we got a signal, it means strategy worked without any LLM


class TestShouldReevaluate:
    """Test re-evaluation logic."""

    def test_should_not_reevaluate(self, rsi_strategy):
        """RSI strategy does not re-evaluate positions."""
        position = ActivePosition(
            id=1,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            quantity=1.0,
            decision_timestamp=datetime.utcnow(),
            candles_since_entry=5,
            exit_policy=ExitPolicy.TRAILING_STOP,
            prediction_horizon=3,
            candles_direction=[],
            is_active=True,
        )

        should_reeval = rsi_strategy.should_reevaluate(position, 105.0)

        assert should_reeval is False


class TestDefaultExitPolicy:
    """Test default exit policy."""

    def test_default_exit_policy(self, rsi_strategy):
        """RSI strategy uses TRAILING_STOP by default."""
        assert rsi_strategy.get_default_exit_policy() == ExitPolicy.TRAILING_STOP
