"""Additional tests for PositionMonitor - Constraints & Edge Cases."""

from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.database import Base
from quantagent.models import (ActivePosition, Environment, ExitPolicy,
                               OrderSide)
from quantagent.trading.position_monitor import PositionMonitor


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def position_monitor(db_session):
    """Create PositionMonitor instance."""
    return PositionMonitor(db_session)


# ============================================================================
# CONSTRAINT VALIDATION TESTS (siguiendo TESTING_PATTERNS.md)
# ============================================================================


def test_active_position_has_required_fields(position_monitor):
    """Validate ActivePosition model has all required fields per design."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    # Required fields per docs/03_design/QuantAgent-nu7-DS-active-position-monitoring.md
    required_attrs = [
        "id",
        "symbol",
        "side",
        "entry_price",
        "stop_loss",
        "take_profit",
        "quantity",
        "decision_timestamp",
        "candles_since_entry",
        "exit_policy",
        "prediction_horizon",
        "candles_direction",
        "is_active",
        "environment",
    ]

    for attr in required_attrs:
        assert hasattr(position, attr), f"Missing required field: {attr}"
        assert getattr(position, attr) is not None or attr in [
            "max_hold_candles",
            "trailing_stop_pct",
            "highest_price_seen",
            "lowest_price_seen",
            "trade_id",
            "signal_id",
            "closed_at",
            "close_reason",
            "accuracy",
        ], f"Field {attr} should not be None"


def test_default_values_are_correct(position_monitor):
    """Validate default values match design specifications."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    # Defaults per design
    assert position.is_active is True
    assert position.candles_since_entry == 0
    assert position.prediction_horizon == 3
    assert position.candles_direction == []
    assert position.environment == Environment.BACKTEST
    assert position.closed_at is None
    assert position.close_reason is None
    assert position.accuracy is None


def test_position_with_all_optional_fields(position_monitor):
    """Validate position creation with all optional fields set."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.TRAILING_STOP,
        trailing_stop_pct=0.05,
        max_hold_candles=10,
        prediction_horizon=5,
        trade_id=123,
        signal_id=456,
    )

    assert position.trailing_stop_pct == 0.05
    assert position.max_hold_candles == 10
    assert position.prediction_horizon == 5
    assert position.trade_id == 123
    assert position.signal_id == 456


# ============================================================================
# INVARIANTS TESTS
# ============================================================================


def test_candles_since_entry_never_decrements(position_monitor):
    """Invariant: candles_since_entry should only increment, never decrement."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.TIME_BASED,
        prediction_horizon=5,
    )

    prev_value = position.candles_since_entry

    for i in range(1, 6):
        position_monitor.update_candle_tracking(
            position, current_price=100.0 + i, prev_close=100.0 + i - 1
        )
        current_value = position.candles_since_entry
        assert (
            current_value > prev_value
        ), f"candles_since_entry decremented: {prev_value} -> {current_value}"
        prev_value = current_value


def test_candles_direction_never_exceeds_prediction_horizon(position_monitor):
    """Invariant: len(candles_direction) <= prediction_horizon."""
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

    # Try to add 10 candles
    for i in range(1, 11):
        position_monitor.update_candle_tracking(
            position, current_price=100.0 + i, prev_close=100.0 + i - 1
        )
        assert (
            len(position.candles_direction) <= position.prediction_horizon
        ), f"candles_direction exceeded horizon: {len(position.candles_direction)} > {position.prediction_horizon}"


def test_accuracy_is_between_zero_and_one(position_monitor):
    """Invariant: accuracy should be in [0.0, 1.0] range."""
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

    # Add varied directions
    position.candles_direction = ["up", "down", "up"]

    position_monitor.close_position(position, reason="TIME_EXPIRED", exit_price=108.0)

    assert position.accuracy is not None
    assert (
        0.0 <= position.accuracy <= 1.0
    ), f"accuracy out of range: {position.accuracy}"


def test_closed_position_has_closed_at_timestamp(position_monitor):
    """Invariant: closed position must have closed_at timestamp."""
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
    assert position.closed_at is not None
    assert position.close_reason == "STOP_LOSS"


# ============================================================================
# EDGE CASES & ERROR HANDLING
# ============================================================================


def test_close_already_closed_position_is_idempotent(position_monitor):
    """Edge case: closing an already closed position should not error."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    # Close once
    position_monitor.close_position(position, reason="TAKE_PROFIT", exit_price=110.0)

    # Close again (should be idempotent)
    position_monitor.close_position(position, reason="MANUAL", exit_price=110.0)

    assert position.is_active is False
    assert position.close_reason == "MANUAL"  # Reason updated


def test_update_tracking_on_closed_position_still_increments(position_monitor):
    """Edge case: tracking on closed position (shouldn't happen but shouldn't crash)."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    position_monitor.close_position(position, reason="TAKE_PROFIT", exit_price=110.0)

    # Try to update tracking (shouldn't crash)
    position_monitor.update_candle_tracking(
        position, current_price=111.0, prev_close=110.0
    )

    # Should still increment (no guard in implementation)
    assert position.candles_since_entry == 1


def test_zero_quantity_position(position_monitor):
    """Edge case: position with zero quantity."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("0.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    assert position.quantity == Decimal("0.0")
    assert position.is_active is True


def test_candle_tracking_with_equal_prices(position_monitor):
    """Edge case: current_price == prev_close (flat candle)."""
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

    # Flat candle (same price)
    position_monitor.update_candle_tracking(
        position, current_price=100.0, prev_close=100.0
    )

    # Should be classified as "down" per implementation logic
    assert position.candles_direction == ["down"]


def test_accuracy_calculation_all_correct_predictions(position_monitor):
    """Edge case: 100% accurate predictions."""
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

    position.candles_direction = ["up", "up", "up"]

    position_monitor.close_position(position, reason="TIME_EXPIRED", exit_price=110.0)

    assert position.accuracy == 1.0


def test_accuracy_calculation_all_wrong_predictions(position_monitor):
    """Edge case: 0% accurate predictions."""
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

    # All wrong for LONG (expected "up", got "down")
    position.candles_direction = ["down", "down", "down"]

    position_monitor.close_position(position, reason="TIME_EXPIRED", exit_price=95.0)

    assert position.accuracy == 0.0


def test_accuracy_calculation_short_with_up_candles(position_monitor):
    """Edge case: SHORT position with up candles (all wrong)."""
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

    # All wrong for SHORT (expected "down", got "up")
    position.candles_direction = ["up", "up", "up"]

    position_monitor.close_position(position, reason="STOP_LOSS", exit_price=105.0)

    assert position.accuracy == 0.0


def test_get_active_position_returns_most_recent_if_multiple(
    position_monitor, db_session
):
    """Edge case: if DB has multiple active positions (shouldn't happen but test query behavior)."""
    # Create 2 positions for same symbol (violates constraint but test DB query)
    pos1 = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    # Manually create another active position (bypassing monitor)
    pos2 = ActivePosition(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        entry_price=105.0,
        stop_loss=110.0,
        take_profit=95.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
        decision_timestamp=pos1.decision_timestamp,
        is_active=True,
        candles_direction=[],
    )
    db_session.add(pos2)
    db_session.commit()

    # get_active_position should return first match
    active = position_monitor.get_active_position("BTCUSDT")
    assert active is not None
    assert active.id == pos1.id  # Returns first one


# ============================================================================
# VALIDATION OF PERSISTENCE
# ============================================================================


def test_position_persists_after_session_close(db_session):
    """Validate position is actually persisted to DB."""
    monitor = PositionMonitor(db_session)

    position = monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    position_id = position.id

    # Query directly from DB
    db_position = db_session.query(ActivePosition).filter_by(id=position_id).first()

    assert db_position is not None
    assert db_position.symbol == "BTCUSDT"
    assert db_position.is_active is True


def test_closed_position_not_returned_by_get_active(position_monitor):
    """Validate closed positions are not returned by get_active_position."""
    position = position_monitor.open_position(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=Decimal("1.0"),
        exit_policy=ExitPolicy.SL_TP_ONLY,
    )

    position_monitor.close_position(position, reason="MANUAL", exit_price=105.0)

    # Should not be returned by get_active_position
    active = position_monitor.get_active_position("BTCUSDT")
    assert active is None


# ============================================================================
# TESTS FOR DIFFERENT EXIT POLICIES
# ============================================================================


def test_all_exit_policies_can_be_set(position_monitor):
    """Validate all ExitPolicy enum values are valid."""
    for policy in [
        ExitPolicy.SL_TP_ONLY,
        ExitPolicy.TIME_BASED,
        ExitPolicy.REEVALUATE,
        ExitPolicy.TRAILING_STOP,
    ]:
        position = position_monitor.open_position(
            symbol=f"TEST_{policy.value}",
            side=OrderSide.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            quantity=Decimal("1.0"),
            exit_policy=policy,
        )

        assert position.exit_policy == policy
