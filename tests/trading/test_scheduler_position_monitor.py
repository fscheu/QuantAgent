"""Unit tests for TradingScheduler position monitoring integration."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from quantagent.models import OrderSide, TradeSignal
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

    # Bind real methods to the mock
    scheduler._check_exit_conditions = TradingScheduler._check_exit_conditions.__get__(
        scheduler, TradingScheduler
    )
    scheduler._execute_position_exit = TradingScheduler._execute_position_exit.__get__(
        scheduler, TradingScheduler
    )

    return scheduler


# ============================================================================
# Unit Tests: _check_exit_conditions
# ============================================================================


def test_check_exit_conditions_stop_loss_long(mock_scheduler):
    """AC-1: Stop loss triggered for LONG position."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, stop_loss=98.0, take_profit=105.0
    )

    should_exit, reason = mock_scheduler._check_exit_conditions(position, 97.0)

    assert should_exit is True
    assert reason == "stop_loss"


def test_check_exit_conditions_stop_loss_short(mock_scheduler):
    """AC-1: Stop loss triggered for SHORT position."""
    position = DummyActivePosition(
        side=OrderSide.SELL, entry_price=100.0, stop_loss=102.0, take_profit=95.0
    )

    should_exit, reason = mock_scheduler._check_exit_conditions(position, 103.0)

    assert should_exit is True
    assert reason == "stop_loss"


def test_check_exit_conditions_take_profit_long(mock_scheduler):
    """AC-2: Take profit triggered for LONG position."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, stop_loss=98.0, take_profit=105.0
    )

    should_exit, reason = mock_scheduler._check_exit_conditions(position, 106.0)

    assert should_exit is True
    assert reason == "take_profit"


def test_check_exit_conditions_take_profit_short(mock_scheduler):
    """AC-2: Take profit triggered for SHORT position."""
    position = DummyActivePosition(
        side=OrderSide.SELL, entry_price=100.0, stop_loss=102.0, take_profit=95.0
    )

    should_exit, reason = mock_scheduler._check_exit_conditions(position, 94.0)

    assert should_exit is True
    assert reason == "take_profit"


def test_check_exit_conditions_max_hold(mock_scheduler):
    """AC-3: Max hold candles exceeded."""
    position = DummyActivePosition(
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=105.0,
        candles_since_entry=25,
        max_hold_candles=20,
    )

    should_exit, reason = mock_scheduler._check_exit_conditions(position, 101.0)

    assert should_exit is True
    assert reason == "max_hold"


def test_check_exit_conditions_max_hold_at_limit(mock_scheduler):
    """AC-3: Max hold candles at exact limit triggers exit."""
    position = DummyActivePosition(
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=105.0,
        candles_since_entry=20,
        max_hold_candles=20,
    )

    should_exit, reason = mock_scheduler._check_exit_conditions(position, 101.0)

    assert should_exit is True
    assert reason == "max_hold"


def test_check_exit_conditions_no_exit(mock_scheduler):
    """AC-4: No exit conditions met."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, stop_loss=98.0, take_profit=105.0
    )

    should_exit, reason = mock_scheduler._check_exit_conditions(position, 101.0)

    assert should_exit is False
    assert reason is None


def test_check_exit_conditions_no_max_hold_set(mock_scheduler):
    """AC-5: No exit when max_hold_candles is None."""
    position = DummyActivePosition(
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=105.0,
        candles_since_entry=1000,
        max_hold_candles=None,
    )

    should_exit, reason = mock_scheduler._check_exit_conditions(position, 101.0)

    assert should_exit is False
    assert reason is None


def test_check_exit_conditions_stop_loss_exact_price_long(mock_scheduler):
    """AC-6: Stop loss at exact price (boundary condition, LONG)."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, stop_loss=98.0, take_profit=105.0
    )

    should_exit, reason = mock_scheduler._check_exit_conditions(position, 98.0)

    assert should_exit is True
    assert reason == "stop_loss"


def test_check_exit_conditions_take_profit_exact_price_long(mock_scheduler):
    """AC-6: Take profit at exact price (boundary condition, LONG)."""
    position = DummyActivePosition(
        side=OrderSide.BUY, entry_price=100.0, stop_loss=98.0, take_profit=105.0
    )

    should_exit, reason = mock_scheduler._check_exit_conditions(position, 105.0)

    assert should_exit is True
    assert reason == "take_profit"


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


def test_execute_position_exit_creates_exit_order(mock_scheduler):
    """AC-8: _execute_position_exit creates exit order with opposite signal."""
    position = DummyActivePosition(side=OrderSide.BUY, entry_price=100.0, signal_id=123)

    mock_scheduler._execute_position_exit(position, "stop_loss", 97.0, "BTC-USD")

    # Verify exit order was placed (opposite signal)
    mock_scheduler.order_manager.execute_decision.assert_called_once()
    call_kwargs = mock_scheduler.order_manager.execute_decision.call_args.kwargs
    assert call_kwargs["symbol"] == "BTC-USD"
    assert call_kwargs["decision"] == TradeSignal.SHORT  # Opposite of BUY
    assert call_kwargs["confidence"] == 1.0
    assert call_kwargs["current_price"] == 97.0
    assert call_kwargs["trigger_signal_id"] == 123


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

    # Verify exit order uses LONG (opposite of SELL)
    call_kwargs = mock_scheduler.order_manager.execute_decision.call_args.kwargs
    assert call_kwargs["decision"] == TradeSignal.LONG
    assert call_kwargs["current_price"] == 103.0


def test_execute_position_exit_no_trade_id(mock_scheduler):
    """AC-11: _execute_position_exit handles missing trade_id gracefully."""
    position = DummyActivePosition(side=OrderSide.BUY, entry_price=100.0, trade_id=None)

    # Should not raise, just skip Trade update
    mock_scheduler._execute_position_exit(position, "stop_loss", 97.0, "BTC-USD")

    # Position closed and order executed, but no Trade query
    mock_scheduler.position_monitor.close_position.assert_called_once()
    mock_scheduler.order_manager.execute_decision.assert_called_once()
    mock_scheduler.db.query.assert_not_called()


def test_execute_position_exit_order_failure_rolls_back(mock_scheduler):
    """AC-12: _execute_position_exit rolls back on order failure."""
    position = DummyActivePosition(side=OrderSide.BUY, entry_price=100.0)
    mock_scheduler.order_manager.execute_decision.side_effect = Exception(
        "Broker error"
    )

    with pytest.raises(Exception, match="Exit failed"):
        mock_scheduler._execute_position_exit(position, "stop_loss", 97.0, "BTC-USD")

    # Verify rollback was called
    mock_scheduler.db.rollback.assert_called_once()
