from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, Mock

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quantagent.models import ActivePosition, Environment, OrderSide, SchedulerHeartbeat, Signal, Trade, TradeSignal
from quantagent.settings import SchedulerSettings
from quantagent.strategy.base import TradingSignal as StrategyTradingSignal
from quantagent.trading.position_monitor import PositionMonitor
from quantagent.trading.scheduler import TradingScheduler


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    for table in (
        Trade.__table__,
        ActivePosition.__table__,
        SchedulerHeartbeat.__table__,
        Signal.__table__,
    ):
        table.create(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _make_scheduler(
    session, *, closes: list[float], decision: str = "HOLD", take_profit: float | None = 110.0,
    trailing_stop_pct: float | None = None,
) -> TradingScheduler:
    config = SchedulerSettings(
        enabled=True,
        interval_hours=1.0,
        assets=["BTC"],
        environment="paper",
        timeframe="1h",
        lookback_hours=12,
    )
    data_provider = MagicMock()
    data_provider.get_ohlc.return_value = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="h"),
            "open": closes,
            "high": [price + 1 for price in closes],
            "low": [price - 1 for price in closes],
            "close": closes,
            "volume": [1_000_000] * len(closes),
        }
    )
    strategy = Mock()
    strategy.should_exit.return_value = (False, None)
    strategy.generate_signal.return_value = StrategyTradingSignal(
        decision=decision,
        confidence=0.9,
        entry_price=closes[-1],
        stop_loss=97.5,
        take_profit=take_profit,
        reasoning="test signal",
        trailing_stop_pct=trailing_stop_pct,
    )
    order_manager = MagicMock()
    order_manager.execute_decision.return_value = Mock(
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        symbol="BTC",
    )
    return TradingScheduler(
        trading_graph=Mock(),
        order_manager=order_manager,
        data_provider=data_provider,
        db_session=session,
        scheduler_settings=config,
        strategy=strategy,
    )


def _create_trade(session, *, side: OrderSide = OrderSide.BUY) -> Trade:
    trade = Trade(
        symbol="BTC",
        entry_price=100,
        quantity=1,
        side=side,
        environment=Environment.PAPER,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def _create_active_position(
    session,
    *,
    trade_id: int,
    side: OrderSide = OrderSide.BUY,
    stop_loss: float = 98.0,
    take_profit: float = 105.0,
) -> ActivePosition:
    position = PositionMonitor(session).open_position(
        symbol="BTC",
        side=side,
        entry_price=100.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        quantity=Decimal("1.0"),
        exit_policy="sl_tp_only",
        trade_id=trade_id,
        prediction_horizon=3,
    )
    position.environment = Environment.PAPER
    # La entrada es la primera vela del fixture: el guard de brwj solo evalúa velas posteriores.
    position.decision_timestamp = datetime(2024, 1, 1)
    session.commit()
    session.refresh(position)
    return position


def test_scheduler_hold_path_updates_tracking_and_continues_analysis():
    """Integration AC: normal hold updates tracking and still runs strategy analysis."""
    SessionLocal = _make_session()
    with SessionLocal() as session:
        trade = _create_trade(session)
        position = _create_active_position(session, trade_id=trade.id)
        scheduler = _make_scheduler(session, closes=[100.0, 101.0], decision="HOLD")

        stats = scheduler.run_once()

        session.refresh(position)
        assert stats["processed"] == 1
        assert stats["errors"] == 0
        assert position.is_active is True
        assert position.candles_since_entry == 1
        assert position.candles_direction == ["up"]
        scheduler.strategy.generate_signal.assert_called_once()
        scheduler.order_manager.execute_decision.assert_not_called()


def test_scheduler_opens_position_with_signal_exit_policy_and_trailing_stop():
    SessionLocal = _make_session()
    with SessionLocal() as session:
        scheduler = _make_scheduler(
            session, closes=[100.0, 101.0], decision="LONG", trailing_stop_pct=0.05,
        )

        scheduler.run_once()

        position = session.query(ActivePosition).one()
        assert position.trailing_stop_pct == 0.05
        assert position.exit_policy.value == "trailing_stop"


def test_scheduler_uses_three_percent_take_profit_default():
    SessionLocal = _make_session()
    with SessionLocal() as session:
        scheduler = _make_scheduler(
            session, closes=[100.0, 101.0], decision="LONG", take_profit=None,
        )

        scheduler.run_once()

        position = session.query(ActivePosition).one()
        assert position.take_profit == Decimal("104.03000000")


def test_scheduler_stop_loss_exit_skips_llm_and_persists_trade_exit_reason():
    """Integration AC: stop-loss exit closes the active position before any new analysis."""
    SessionLocal = _make_session()
    with SessionLocal() as session:
        trade = _create_trade(session)
        position = _create_active_position(
            session,
            trade_id=trade.id,
            stop_loss=98.0,
            take_profit=105.0,
        )
        scheduler = _make_scheduler(session, closes=[100.0, 97.0], decision="LONG")

        stats = scheduler.run_once()

        session.refresh(position)
        session.refresh(trade)
        assert stats["processed"] == 1
        assert stats["errors"] == 0
        assert position.is_active is False
        assert position.close_reason == "STOP_LOSS"
        assert trade.exit_signal == "STOP_LOSS"
        assert trade.closed_at is not None
        scheduler.strategy.generate_signal.assert_not_called()
        scheduler.order_manager.close_trade.assert_called_once_with(
            trade.id, 97.0, environment=scheduler.environment
        )


def test_scheduler_intrabar_stop_matches_backtest_stop_price():
    SessionLocal = _make_session()
    with SessionLocal() as session:
        trade = _create_trade(session)
        position = _create_active_position(session, trade_id=trade.id, stop_loss=90.0)
        scheduler = _make_scheduler(session, closes=[100.0, 95.0], decision="LONG")
        scheduler.data_provider.get_ohlc.return_value.loc[1, "low"] = 89.0

        scheduler.run_once()

        session.refresh(position)
        session.refresh(trade)
        assert position.is_active is False
        assert trade.exit_signal == "STOP_LOSS"
        scheduler.order_manager.close_trade.assert_called_once_with(
            trade.id, 90.0, environment=scheduler.environment
        )


def test_scheduler_take_profit_exit_skips_llm_and_persists_trade_exit_reason():
    """Integration AC: take-profit exit closes the active position before any new analysis."""
    SessionLocal = _make_session()
    with SessionLocal() as session:
        trade = _create_trade(session)
        position = _create_active_position(
            session,
            trade_id=trade.id,
            stop_loss=98.0,
            take_profit=105.0,
        )
        scheduler = _make_scheduler(session, closes=[100.0, 106.0], decision="LONG")

        stats = scheduler.run_once()

        session.refresh(position)
        session.refresh(trade)
        heartbeat = session.query(SchedulerHeartbeat).one()
        assert stats["processed"] == 1
        assert stats["errors"] == 0
        assert heartbeat.status == "completed"
        assert position.is_active is False
        assert position.close_reason == "TAKE_PROFIT"
        assert trade.exit_signal == "TAKE_PROFIT"
        assert trade.closed_at is not None
        scheduler.strategy.generate_signal.assert_not_called()
        scheduler.order_manager.close_trade.assert_called_once_with(
            trade.id, 106.0, environment=scheduler.environment
        )
