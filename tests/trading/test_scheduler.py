from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
from quantagent.trading.scheduler import TradingScheduler

from quantagent.models import Environment
from quantagent.settings import SchedulerSettings
from quantagent.strategy.base import TradingSignal as StrategyTradingSignal


class DummySession:
    def __init__(self):
        self._objects = []
        self.committed = False
        self.rolled_back = False

    def add(self, obj):
        self._objects.append(obj)

    def flush(self):
        for idx, obj in enumerate(self._objects, start=1):
            if getattr(obj, "id", None) is None:
                setattr(obj, "id", idx)

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            setattr(obj, "id", len(self._objects))
        return obj

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class DummyScheduler:
    def __init__(self):
        self.jobs = []
        self.started = False
        self.shutdown_called = False

    def add_job(self, func, **kwargs):
        self.jobs.append({"func": func, **kwargs})

    def start(self):
        self.started = True

    def shutdown(self, wait=True):
        self.shutdown_called = True
        self.started = False


def _sample_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000_000, 1_100_000, 1_200_000],
        }
    )


def test_scheduler_settings_validation_errors():
    with pytest.raises(ValueError):
        SchedulerSettings(interval_hours=0)
    with pytest.raises(ValueError):
        SchedulerSettings(assets=[])


def test_trading_scheduler_runs_signal_and_executes_order():
    config = SchedulerSettings(
        enabled=True,
        interval_hours=1.0,
        assets=["BTC"],
        environment="paper",
        timeframe="1h",
        lookback_hours=12,
    )

    session = DummySession()
    data_provider = MagicMock()
    data_provider.get_ohlc.return_value = _sample_df()

    order_manager = MagicMock()
    order_manager.execute_decision.return_value = object()

    strategy = Mock()
    strategy.generate_signal.return_value = StrategyTradingSignal(
        decision="LONG",
        confidence=0.9,
        entry_price=101.0,
        stop_loss=97.5,
        take_profit=110.0,
        reasoning="Test",
    )

    scheduler = TradingScheduler(
        trading_graph=Mock(),
        order_manager=order_manager,
        data_provider=data_provider,
        db_session=session,
        scheduler_settings=config,
        scheduler_factory=DummyScheduler,
        strategy=strategy,
    )

    stats = scheduler.run_once()

    assert stats["processed"] == 1
    order_manager.execute_decision.assert_called_once()
    kwargs = order_manager.execute_decision.call_args.kwargs
    assert kwargs["environment"] == Environment.PAPER
    assert kwargs["symbol"] == "BTC"
    assert session.rolled_back is False


def test_trading_scheduler_hold_signal_skips_execution():
    config = SchedulerSettings(
        enabled=True,
        interval_hours=1.0,
        assets=["BTC"],
        environment="paper",
    )
    session = DummySession()
    data_provider = MagicMock()
    data_provider.get_ohlc.return_value = _sample_df()

    order_manager = MagicMock()
    strategy = Mock()
    strategy.generate_signal.return_value = None

    scheduler = TradingScheduler(
        trading_graph=Mock(),
        order_manager=order_manager,
        data_provider=data_provider,
        db_session=session,
        scheduler_settings=config,
        scheduler_factory=DummyScheduler,
        strategy=strategy,
    )

    stats = scheduler.run_once()
    assert stats["processed"] == 1
    order_manager.execute_decision.assert_not_called()


def test_scheduler_start_and_stop_toggle_state():
    config = SchedulerSettings(
        enabled=True,
        interval_hours=1.0,
        assets=["BTC"],
        environment="paper",
    )
    session = DummySession()
    scheduler = TradingScheduler(
        trading_graph=Mock(),
        order_manager=Mock(),
        data_provider=Mock(),
        db_session=session,
        scheduler_settings=config,
        scheduler_factory=DummyScheduler,
        strategy=Mock(),
    )

    assert scheduler.start(immediate=False) is True
    assert scheduler.is_running is True
    assert scheduler.scheduler.started is True

    scheduler.stop()
    assert scheduler.is_running is False
    assert scheduler.scheduler.shutdown_called is True
