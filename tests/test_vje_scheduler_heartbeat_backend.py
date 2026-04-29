from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.streamlit.services.db import DbHandle
from quantagent.models import Environment, Order, OrderSide, SchedulerHeartbeat, Signal, Trade
from quantagent.settings import SchedulerSettings
from quantagent.strategy.base import TradingSignal as StrategyTradingSignal
from quantagent.trading.scheduler import TradingScheduler


def _make_session(table_names: list[str]):
    engine = create_engine("sqlite:///:memory:")
    for name in table_names:
        table = {
            "signals": Signal.__table__,
            "orders": Order.__table__,
            "trades": Trade.__table__,
            "scheduler_heartbeats": SchedulerHeartbeat.__table__,
        }[name]
        table.create(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _make_scheduler(session, assets: list[str] | None = None) -> TradingScheduler:
    config = SchedulerSettings(
        enabled=True,
        interval_hours=1.0,
        assets=assets or ["BTC"],
        environment="paper",
        timeframe="1h",
        lookback_hours=12,
    )
    data_provider = MagicMock()
    data_provider.get_ohlc.return_value = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000_000, 1_100_000, 1_200_000],
        }
    )
    strategy = Mock()
    strategy.generate_signal.return_value = StrategyTradingSignal(
        decision="LONG",
        confidence=0.9,
        entry_price=101.0,
        stop_loss=97.5,
        take_profit=110.0,
        reasoning="test signal",
    )
    order_manager = MagicMock()
    order_manager.execute_decision.return_value = object()
    return TradingScheduler(
        trading_graph=Mock(),
        order_manager=order_manager,
        data_provider=data_provider,
        db_session=session,
        scheduler_settings=config,
        strategy=strategy,
    )


def test_upsert_heartbeat_start_updates_existing_row_for_environment():
    SessionLocal = _make_session(["signals", "orders", "trades", "scheduler_heartbeats"])
    with SessionLocal() as session:
        scheduler = _make_scheduler(session, ["BTC", "ETH"])
        first_started = datetime.utcnow() - timedelta(minutes=5)
        second_started = datetime.utcnow()

        first = scheduler._upsert_heartbeat_start(first_started)
        second = scheduler._upsert_heartbeat_start(second_started)

        rows = session.query(SchedulerHeartbeat).all()
        assert len(rows) == 1
        assert first.id == second.id
        assert rows[0].timestamp == second_started
        assert rows[0].status == "running"
        assert rows[0].assets == ["BTC", "ETH"]


def test_upsert_heartbeat_complete_sets_last_trade_id():
    SessionLocal = _make_session(["signals", "orders", "trades", "scheduler_heartbeats"])
    with SessionLocal() as session:
        session.add(
            Trade(
                symbol="BTC",
                entry_price=100,
                quantity=1,
                side=OrderSide.BUY,
                environment=Environment.PAPER,
            )
        )
        session.commit()

        scheduler = _make_scheduler(session)
        heartbeat = scheduler._upsert_heartbeat_start(datetime.utcnow())
        scheduler._upsert_heartbeat_complete(
            heartbeat,
            {"processed": 1, "errors": 0, "duration_seconds": 3.2, "total": 1},
        )

        refreshed = session.query(SchedulerHeartbeat).one()
        assert refreshed.status == "completed"
        assert refreshed.completed_at is not None
        assert refreshed.last_trade_id == 1
        assert refreshed.stats["processed"] == 1


def test_analyze_and_trade_continues_when_heartbeat_start_fails():
    SessionLocal = _make_session(["signals", "orders", "trades", "scheduler_heartbeats"])
    with SessionLocal() as session:
        scheduler = _make_scheduler(session)
        scheduler._upsert_heartbeat_start = Mock(return_value=None)
        scheduler._upsert_heartbeat_complete = Mock()

        stats = scheduler.analyze_and_trade()

        assert stats["processed"] == 1
        assert stats["errors"] == 0
        scheduler._upsert_heartbeat_complete.assert_called_once()
        assert scheduler._upsert_heartbeat_complete.call_args.args[0] is None


def test_db_handle_returns_latest_heartbeat_dict():
    SessionLocal = _make_session(["signals", "orders", "trades", "scheduler_heartbeats"])
    started = datetime.utcnow() - timedelta(minutes=10)
    completed = datetime.utcnow() - timedelta(minutes=9)
    with SessionLocal() as session:
        session.add(
            SchedulerHeartbeat(
                timestamp=started,
                completed_at=completed,
                status="completed",
                environment=Environment.PAPER,
                assets=["BTC"],
                stats={"processed": 1, "errors": 0, "duration_seconds": 60.0, "total": 1},
                last_trade_id=7,
            )
        )
        session.commit()

    db = DbHandle(ok=True, error=None, SessionLocal=SessionLocal)
    heartbeat = db.get_latest_heartbeat("paper")

    assert heartbeat is not None
    assert heartbeat["status"] == "completed"
    assert heartbeat["environment"] == "paper"
    assert heartbeat["assets"] == ["BTC"]
    assert heartbeat["stats"]["processed"] == 1
    assert heartbeat["last_trade_id"] == 7


def test_db_handle_recent_heartbeats_are_limited_and_sorted_desc():
    SessionLocal = _make_session(["signals", "orders", "trades", "scheduler_heartbeats"])
    base = datetime.utcnow()
    with SessionLocal() as session:
        session.add_all(
            [
                SchedulerHeartbeat(
                    timestamp=base - timedelta(minutes=offset),
                    completed_at=base - timedelta(minutes=offset - 1),
                    status="completed",
                    environment=Environment.PAPER,
                    assets=["BTC"],
                    stats={"processed": 1, "errors": 0, "duration_seconds": 5.0, "total": 1},
                )
                for offset in range(12)
            ]
        )
        session.commit()

    db = DbHandle(ok=True, error=None, SessionLocal=SessionLocal)
    recent = db.get_recent_heartbeats("paper", limit=10)

    assert len(recent) == 10
    timestamps = [item["timestamp"] for item in recent]
    assert timestamps == sorted(timestamps, reverse=True)


def test_db_handle_missing_heartbeat_table_fails_closed():
    SessionLocal = _make_session(["signals", "orders", "trades"])
    db = DbHandle(ok=True, error=None, SessionLocal=SessionLocal)

    assert db.get_latest_heartbeat("paper") is None
    assert db.get_recent_heartbeats("paper", limit=10) == []
