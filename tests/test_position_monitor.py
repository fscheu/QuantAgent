"""Tests for PositionMonitor.

Note: These tests require PostgreSQL due to JSON column usage in models.
They are marked as integration tests to exclude from default pytest run.
"""

import os
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.models import (
    ActivePosition,
    Base,
    ExitPolicy,
    OrderSide,
)
from quantagent.trading.position_monitor import PositionMonitor


@pytest.fixture
def db_session():
    """Create database session for testing using DATABASE_URL if available."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        engine = create_engine(database_url)
    else:
        # Fallback to SQLite for local development
        engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    db = TestSession()
    yield db
    db.close()


@pytest.fixture
def position_monitor(db_session):
    """Create PositionMonitor instance."""
    return PositionMonitor(db_session)

def test_open_position(position_monitor, db_session):
    """Test creating a new active position."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.TRAILING_STOP,
        trailing_stop_pct=0.05,
    )

    assert position.id is not None
    assert position.symbol == "BTCUSDT"
    assert position.side == OrderSide.BUY
    assert position.entry_price == 100.0
    assert position.stop_loss == 95.0
    assert position.take_profit == 110.0
    assert position.quantity == Decimal("1.0")
    assert position.exit_policy == ExitPolicy.TRAILING_STOP
    assert position.trailing_stop_pct == 0.05
    assert position.is_active is True
    assert position.candles_since_entry == 0
    assert position.candles_direction == []

def test_get_active_position(position_monitor):
    """Test retrieving active position by symbol."""
    position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    active = position_monitor.get_active_position("BTCUSDT")
    assert active is not None
    assert active.symbol == "BTCUSDT"
    assert active.is_active is True

    no_position = position_monitor.get_active_position("ETHUSDT")
    assert no_position is None

def test_update_candle_tracking_up(position_monitor):
    """Test tracking candle direction - up."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.TIME_BASED,
        prediction_horizon=3,
    )

    position_monitor.update_candle_tracking(
        position, current_price=102.0, prev_close=100.0
    )

    assert position.candles_since_entry == 1
    assert position.candles_direction == ["up"]

def test_update_candle_tracking_down(position_monitor):
    """Test tracking candle direction - down."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.TIME_BASED,
        prediction_horizon=3,
    )

    position_monitor.update_candle_tracking(
        position, current_price=98.0, prev_close=100.0
    )

    assert position.candles_since_entry == 1
    assert position.candles_direction == ["down"]

def test_update_candle_tracking_respects_horizon(position_monitor):
    """Test that candle tracking respects prediction horizon."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.TIME_BASED,
        prediction_horizon=3,
    )

    for i, price in enumerate([102.0, 104.0, 106.0, 108.0], start=1):
        position_monitor.update_candle_tracking(
            position, current_price=price, prev_close=price - 2
        )
        assert position.candles_since_entry == i

    assert len(position.candles_direction) == 3

def test_close_position_long_accurate(position_monitor):
    """Test closing LONG position with accurate predictions."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.TIME_BASED,
        prediction_horizon=3,
    )

    position.candles_direction = ["up", "up", "down"]

    position_monitor.close_position(position, reason="TIME_EXPIRED", exit_price=108.0)

    assert position.is_active is False
    assert position.close_reason == "TIME_EXPIRED"
    assert position.closed_at is not None
    assert position.accuracy == pytest.approx(2 / 3, rel=1e-2)

def test_close_position_short_accurate(position_monitor):
    """Test closing SHORT position with accurate predictions."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        entry_price=100.0,
        stop_loss=105.0,
        take_profit=90.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.TIME_BASED,
        prediction_horizon=3,
    )

    position.candles_direction = ["down", "down", "up"]

    position_monitor.close_position(position, reason="TAKE_PROFIT", exit_price=90.0)

    assert position.is_active is False
    assert position.close_reason == "TAKE_PROFIT"
    assert position.accuracy == pytest.approx(2 / 3, rel=1e-2)

def test_close_position_no_tracking(position_monitor):
    """Test closing position with no candle tracking."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    position_monitor.close_position(position, reason="STOP_LOSS", exit_price=95.0)

    assert position.is_active is False
    assert position.accuracy is None

def test_only_one_active_position_per_symbol(position_monitor, db_session):
    """Test constraint: only one active position per symbol."""
    position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    active_positions = (
        db_session.query(ActivePosition)
        .filter(
            ActivePosition.symbol == "BTCUSDT", ActivePosition.is_active.is_(True)
        )
        .all()
    )

    assert len(active_positions) == 1

    pos1 = position_monitor.get_active_position("BTCUSDT")
    position_monitor.close_position(pos1, reason="TEST", exit_price=105.0)

    position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        entry_price=105.0,
        stop_loss=110.0,
        take_profit=95.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    active_positions_after = (
        db_session.query(ActivePosition)
        .filter(
            ActivePosition.symbol == "BTCUSDT", ActivePosition.is_active.is_(True)
        )
        .all()
    )

    assert len(active_positions_after) == 1
    assert active_positions_after[0].side == OrderSide.SELL
