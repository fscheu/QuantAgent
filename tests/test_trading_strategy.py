"""Unit tests for TradingStrategy base class (Template Method Pattern)."""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
import pytest

from quantagent.models import ActivePosition, ExitPolicy, OrderSide
from quantagent.strategy.base import TradingSignal, TradingStrategy


class DummyStrategy(TradingStrategy):
    """Minimal concrete strategy for testing base class."""

    def generate_signal(
        self,
        kline_data: List[Dict],
        symbol: str,
        timeframe: str,
        current_price: float,
    ) -> Optional[TradingSignal]:
        """Return dummy signal."""
        return TradingSignal(
            decision="LONG",
            confidence=0.8,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )

    def should_reevaluate(self, position: ActivePosition, current_price: float) -> bool:
        """Never re-evaluate."""
        return False


@pytest.fixture
def dummy_strategy():
    """Fixture for dummy strategy."""
    return DummyStrategy()


@pytest.fixture
def long_position():
    """Fixture for LONG position."""
    return ActivePosition(
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
        trailing_stop_pct=0.05,
        prediction_horizon=3,
        candles_direction=[],
        is_active=True,
    )


@pytest.fixture
def short_position():
    """Fixture for SHORT position."""
    return ActivePosition(
        id=2,
        symbol="ETHUSDT",
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
    )


@pytest.fixture
def ohlc_data():
    """Fixture for OHLC DataFrame."""
    return pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [99, 100, 101],
            "close": [103, 104, 105],
            "volume": [1000, 1100, 1200],
        }
    )


class TestShouldExitStopLoss:
    """Test stop loss exit condition."""

    def test_long_stop_loss_triggered(self, dummy_strategy, long_position, ohlc_data):
        """AC1.3: Stop loss triggered for LONG position."""
        current_price = 94.0  # Below SL of 95.0
        should_exit, reason = dummy_strategy.should_exit(
            long_position, current_price, ohlc_data
        )

        assert should_exit is True
        assert reason == "STOP_LOSS"

    def test_short_stop_loss_triggered(self, dummy_strategy, short_position, ohlc_data):
        """Stop loss triggered for SHORT position."""
        current_price = 106.0  # Above SL of 105.0
        should_exit, reason = dummy_strategy.should_exit(
            short_position, current_price, ohlc_data
        )

        assert should_exit is True
        assert reason == "STOP_LOSS"

    def test_long_stop_loss_not_triggered(
        self, dummy_strategy, long_position, ohlc_data
    ):
        """Stop loss NOT triggered for LONG."""
        current_price = 96.0  # Above SL of 95.0
        should_exit, reason = dummy_strategy.should_exit(
            long_position, current_price, ohlc_data
        )

        assert should_exit is False
        assert reason is None


class TestShouldExitTakeProfit:
    """Test take profit exit condition."""

    def test_long_take_profit_triggered(self, dummy_strategy, long_position, ohlc_data):
        """AC1.4: Take profit triggered for LONG position."""
        current_price = 111.0  # Above TP of 110.0
        should_exit, reason = dummy_strategy.should_exit(
            long_position, current_price, ohlc_data
        )

        assert should_exit is True
        assert reason == "TAKE_PROFIT"

    def test_short_take_profit_triggered(
        self, dummy_strategy, short_position, ohlc_data
    ):
        """Take profit triggered for SHORT position."""
        current_price = 89.0  # Below TP of 90.0
        should_exit, reason = dummy_strategy.should_exit(
            short_position, current_price, ohlc_data
        )

        assert should_exit is True
        assert reason == "TAKE_PROFIT"

    def test_long_take_profit_not_triggered(
        self, dummy_strategy, long_position, ohlc_data
    ):
        """Take profit NOT triggered for LONG."""
        current_price = 109.0  # Below TP of 110.0
        should_exit, reason = dummy_strategy.should_exit(
            long_position, current_price, ohlc_data
        )

        assert should_exit is False
        assert reason is None


class TestShouldExitTrailingStop:
    """Test trailing stop exit condition."""

    def test_long_trailing_stop_updates_highest(
        self, dummy_strategy, long_position, ohlc_data
    ):
        """AC1.5: Trailing stop updates highest price seen."""
        # Initial highest is None
        assert long_position.highest_price_seen is None

        # Price moves up
        current_price = 110.0
        should_exit, reason = dummy_strategy.should_exit(
            long_position, current_price, ohlc_data
        )

        # Highest should be updated (but TP triggers first)
        assert should_exit is True
        assert reason == "TAKE_PROFIT"  # TP at 110.0 triggers first

        # Reset for trailing test
        long_position.take_profit = Decimal("120.0")  # Move TP higher
        long_position.highest_price_seen = None

        current_price = 110.0
        should_exit, reason = dummy_strategy.should_exit(
            long_position, current_price, ohlc_data
        )

        assert should_exit is False
        assert long_position.highest_price_seen == 110.0

    def test_long_trailing_stop_triggered(
        self, dummy_strategy, long_position, ohlc_data
    ):
        """AC1.6: Trailing stop triggered for LONG position."""
        # Setup: price went up to 110, then drops
        long_position.highest_price_seen = Decimal("110.0")
        long_position.take_profit = Decimal("120.0")  # Move TP higher to not interfere

        # Trailing stop = 110 * (1 - 0.05) = 104.5
        current_price = 104.0  # Below trailing stop
        should_exit, reason = dummy_strategy.should_exit(
            long_position, current_price, ohlc_data
        )

        assert should_exit is True
        assert reason == "TRAILING_STOP"

    def test_short_trailing_stop_triggered(
        self, dummy_strategy, short_position, ohlc_data
    ):
        """Trailing stop triggered for SHORT position."""
        # Setup: price went down to 90, then rises
        short_position.lowest_price_seen = Decimal("90.0")
        short_position.take_profit = Decimal("80.0")  # Move TP lower to not interfere

        # Trailing stop = 90 * (1 + 0.05) = 94.5
        current_price = 95.0  # Above trailing stop
        should_exit, reason = dummy_strategy.should_exit(
            short_position, current_price, ohlc_data
        )

        assert should_exit is True
        assert reason == "TRAILING_STOP"

    def test_trailing_stop_disabled_when_no_pct(
        self, dummy_strategy, long_position, ohlc_data
    ):
        """Trailing stop disabled if trailing_stop_pct is None."""
        long_position.trailing_stop_pct = None
        long_position.take_profit = Decimal("120.0")

        current_price = 100.0
        should_exit, reason = dummy_strategy.should_exit(
            long_position, current_price, ohlc_data
        )

        assert should_exit is False
        assert reason is None


class TestShouldExitTimeBased:
    """Test time-based exit condition."""

    def test_time_based_exit_triggered(self, dummy_strategy, long_position, ohlc_data):
        """Time-based exit triggered when max_hold_candles reached."""
        long_position.exit_policy = ExitPolicy.TIME_BASED
        long_position.max_hold_candles = 3
        long_position.candles_since_entry = 3
        long_position.take_profit = Decimal("120.0")  # Move TP to not interfere

        current_price = 105.0  # Between SL and TP
        should_exit, reason = dummy_strategy.should_exit(
            long_position, current_price, ohlc_data
        )

        assert should_exit is True
        assert reason == "TIME_EXPIRED"

    def test_time_based_exit_not_triggered_before_limit(
        self, dummy_strategy, long_position, ohlc_data
    ):
        """Time-based exit NOT triggered before max_hold_candles."""
        long_position.exit_policy = ExitPolicy.TIME_BASED
        long_position.max_hold_candles = 3
        long_position.candles_since_entry = 2
        long_position.take_profit = Decimal("120.0")

        current_price = 105.0
        should_exit, reason = dummy_strategy.should_exit(
            long_position, current_price, ohlc_data
        )

        assert should_exit is False
        assert reason is None


class TestShouldExitActivePosition:
    """Test position remains active when no exit conditions met."""

    def test_position_remains_active(self, dummy_strategy, long_position, ohlc_data):
        """AC1.7: Position remains active when no exit conditions met."""
        long_position.take_profit = Decimal("120.0")

        current_price = 105.0  # Between SL (95) and TP (120)
        should_exit, reason = dummy_strategy.should_exit(
            long_position, current_price, ohlc_data
        )

        assert should_exit is False
        assert reason is None


class TestDefaultExitPolicy:
    """Test default exit policy."""

    def test_default_exit_policy(self, dummy_strategy):
        """Default exit policy is TRAILING_STOP."""
        assert dummy_strategy.get_default_exit_policy() == ExitPolicy.TRAILING_STOP
