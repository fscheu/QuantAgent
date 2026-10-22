"""Unit tests for TradingScheduler position monitoring integration."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from quantagent.models import ExitPolicy, OrderSide, TradeSignal
from quantagent.strategy.rsi_strategy import RSIMeanReversionStrategy
from quantagent.trading.scheduler import TradingScheduler


class DummyActivePosition:
    """Mock ActivePosition for testing."""

    def __init__(
        self,
        symbol="BTC-USD",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=105.0,
        candles_since_entry=0,
        max_hold_candles=None,
        signal_id=None,
        trade_id=None,
    ):
        self.symbol = symbol
        self.side = side
        self.entry_price = Decimal(str(entry_price))
        self.stop_loss = Decimal(str(stop_loss))
        self.take_profit = Decimal(str(take_profit))
        self.exit_policy = ExitPolicy.TIME_BASED
        self.candles_since_entry = candles_since_entry
        self.max_hold_candles = max_hold_candles
        self.signal_id = signal_id
        self.trade_id = trade_id
        self.is_active = True
        self.close_reason = None
        self.closed_at = None


@pytest.fixture
def mock_scheduler():
    """Create a scheduler with mocked dependencies."""
    scheduler = Mock(spec=TradingScheduler)
    scheduler.db = Mock()
    scheduler.position_monitor = Mock()
    scheduler.order_manager = Mock()
    scheduler.environment = Mock(value="paper")

    scheduler.strategy = RSIMeanReversionStrategy()
    scheduler._execute_position_exit = TradingScheduler._execute_position_exit.__get__(
        scheduler, TradingScheduler
    )

    return scheduler


def test_should_exit_stop_loss_long(mock_scheduler):
    """AC-1: Stop loss triggered for LONG position."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, stop_loss=98.0, take_profit=105.0
    )

    should_exit, reason = mock_scheduler.strategy.should_exit(position, 97.0, None)

    assert should_exit is True
    assert reason == "STOP_LOSS"


def test_should_exit_stop_loss_short(mock_scheduler):
    """AC-1: Stop loss triggered for SHORT position."""
    position = DummyActivePosition(
        side=OrderSide.SELL, entry_price=100.0, stop_loss=102.0, take_profit=95.0
    )

    should_exit, reason = mock_scheduler.strategy.should_exit(position, 103.0, None)

    assert should_exit is True
    assert reason == "STOP_LOSS"


def test_should_exit_take_profit_long(mock_scheduler):
    """AC-2: Take profit triggered for LONG position."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, stop_loss=98.0, take_profit=105.0
    )

    should_exit, reason = mock_scheduler.strategy.should_exit(position, 106.0, None)

    assert should_exit is True
    assert reason == "TAKE_PROFIT"


def test_should_exit_take_profit_short(mock_scheduler):
    """AC-2: Take profit triggered for SHORT position."""
    position = DummyActivePosition(
        side=OrderSide.SELL, entry_price=100.0, stop_loss=102.0, take_profit=95.0
    )

    should_exit, reason = mock_scheduler.strategy.should_exit(position, 94.0, None)

    assert should_exit is True
    assert reason == "TAKE_PROFIT"


def test_should_exit_max_hold(mock_scheduler):
    """AC-3: Max hold candles exceeded."""
    position = DummyActivePosition(
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=105.0,
        candles_since_entry=25,
        max_hold_candles=20,
    )

    should_exit, reason = mock_scheduler.strategy.should_exit(position, 101.0, None)

    assert should_exit is True
    assert reason == "TIME_EXPIRED"


def test_should_exit_max_hold_at_limit(mock_scheduler):
    """AC-3: Max hold candles at exact limit triggers exit."""
    position = DummyActivePosition(
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=105.0,
        candles_since_entry=20,
        max_hold_candles=20,
    )

    should_exit, reason = mock_scheduler.strategy.should_exit(position, 101.0, None)

    assert should_exit is True
    assert reason == "TIME_EXPIRED"


def test_should_exit_no_exit(mock_scheduler):
    """AC-4: No exit conditions met."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, stop_loss=98.0, take_profit=105.0
    )

    should_exit, reason = mock_scheduler.strategy.should_exit(position, 101.0, None)

    assert should_exit is False
    assert reason is None


def test_should_exit_no_max_hold_set(mock_scheduler):
    """AC-5: No exit when max_hold_candles is None."""
    position = DummyActivePosition(
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=105.0,
        candles_since_entry=1000,
        max_hold_candles=None,
    )

    should_exit, reason = mock_scheduler.strategy.should_exit(position, 101.0, None)

    assert should_exit is False
    assert reason is None


def test_should_exit_stop_loss_exact_price_long(mock_scheduler):
    """AC-6: Stop loss at exact price (boundary condition, LONG)."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, stop_loss=98.0, take_profit=105.0
    )

    should_exit, reason = mock_scheduler.strategy.should_exit(position, 98.0, None)

    assert should_exit is True
    assert reason == "STOP_LOSS"


def test_should_exit_take_profit_exact_price_long(mock_scheduler):
    """AC-6: Take profit at exact price (boundary condition, LONG)."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, stop_loss=98.0, take_profit=105.0
    )

    should_exit, reason = mock_scheduler.strategy.should_exit(position, 105.0, None)

    assert should_exit is True
    assert reason == "TAKE_PROFIT"


# ============================================================================
# Unit Tests: _execute_position_exit
# ============================================================================


def test_execute_position_exit_closes_position(mock_scheduler):
    """AC-7: _execute_position_exit closes position."""
    position = DummyActivePosition(side=OrderSide.BUY, entry_price=100.0)

    mock_scheduler._execute_position_exit(position, "stop_loss", 97.0, "BTC-USD")

    # Verify position was closed via PositionMonitor
    mock_scheduler.position_monitor.close_position.assert_called_once_with(
        position, reason="stop_loss", exit_price=97.0
    )


def test_execute_position_exit_closes_trade(mock_scheduler):
    """An exit closes the tracked trade rather than sending a reversal signal."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, signal_id=123, trade_id=456
    )

    mock_scheduler._execute_position_exit(position, "stop_loss", 97.0, "BTC-USD")

    mock_scheduler.order_manager.close_trade.assert_called_once_with(
        456, 97.0, environment=mock_scheduler.environment
    )
    assert mock_scheduler.order_manager.close_trade.return_value.trigger_signal_id == 123


def test_execute_position_exit_updates_trade_record(mock_scheduler):
    """AC-9: _execute_position_exit updates Trade record with exit reason."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, trade_id=456
    )
    position.closed_at = "2024-01-01T12:00:00Z"

    mock_trade = Mock()
    mock_scheduler.db.query.return_value.filter.return_value.first.return_value = (
        mock_trade
    )

    mock_scheduler._execute_position_exit(position, "take_profit", 105.0, "BTC-USD")

    # Verify Trade was updated
    assert mock_trade.exit_signal == "take_profit"
    assert mock_trade.closed_at == "2024-01-01T12:00:00Z"
    mock_scheduler.db.commit.assert_called_once()


def test_execute_position_exit_short_position(mock_scheduler):
    """AC-10: _execute_position_exit for SHORT position uses LONG exit signal."""
    position = DummyActivePosition(
        side=OrderSide.SELL, entry_price=100.0, signal_id=789
    )

    mock_scheduler._execute_position_exit(position, "stop_loss", 103.0, "ETH-USD")

    mock_scheduler.order_manager.close_symbol.assert_called_once_with(
        "ETH-USD", 103.0, environment=mock_scheduler.environment
    )


def test_execute_position_exit_no_trade_id(mock_scheduler):
    """AC-11: _execute_position_exit handles missing trade_id gracefully."""
    position = DummyActivePosition(signal_id=321)

    # Should not raise, just skip Trade update
    mock_scheduler._execute_position_exit(position, "stop_loss", 97.0, "BTC-USD")

    # Position closed through the portfolio-aware symbol close path.
    mock_scheduler.position_monitor.close_position.assert_called_once()
    mock_scheduler.order_manager.close_symbol.assert_called_once_with(
        "BTC-USD", 97.0, environment=mock_scheduler.environment
    )
    assert mock_scheduler.order_manager.close_symbol.return_value.trigger_signal_id == 321
    mock_scheduler.db.query.assert_not_called()


def test_execute_position_exit_order_failure_rolls_back(mock_scheduler):
    """AC-12: _execute_position_exit rolls back on order failure."""
    position = DummyActivePosition(side=OrderSide.BUY, entry_price=100.0)
    mock_scheduler.order_manager.close_symbol.side_effect = Exception(
        "Broker error"
    )

    with pytest.raises(Exception, match="Exit failed"):
        mock_scheduler._execute_position_exit(position, "stop_loss", 97.0, "BTC-USD")

    # Verify rollback was called
    mock_scheduler.db.rollback.assert_called_once()
