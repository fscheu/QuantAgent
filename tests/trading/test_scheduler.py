from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from quantagent.models import Environment, OrderSide, TradeSignal
from quantagent.settings import SchedulerSettings
from quantagent.strategy.base import TradingSignal as StrategyTradingSignal
from quantagent.trading.scheduler import (
    TradingScheduler,
)


def _make_mock_order(side=OrderSide.BUY, quantity="1.0", symbol="BTC-USD"):
    """Create a mock order with necessary attributes for PositionMonitor."""
    mock_order = Mock()
    mock_order.side = side
    mock_order.quantity = quantity
    mock_order.symbol = symbol
    return mock_order


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

    def query(self, model):
        """Return a mock query object that returns no results."""
        mock_q = Mock()
        mock_q.filter.return_value = mock_q
        mock_q.order_by.return_value = mock_q
        mock_q.first.return_value = None
        mock_q.all.return_value = []
        return mock_q


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


# ============================================================================
# AC-8 & AC-9: Configuration Validation
# ============================================================================


def test_scheduler_settings_validation_interval_zero():
    """AC-8: Configuration Validation - Invalid Interval."""
    with pytest.raises(ValueError, match="interval_hours must be > 0"):
        SchedulerSettings(interval_hours=0)


def test_scheduler_settings_validation_negative_interval():
    """AC-8: Configuration Validation - Invalid Interval (negative)."""
    with pytest.raises(ValueError, match="interval_hours must be > 0"):
        SchedulerSettings(interval_hours=-0.5)


def test_scheduler_settings_validation_empty_assets():
    """AC-9: Configuration Validation - Empty Assets List."""
    with pytest.raises(ValueError, match="assets list cannot be empty"):
        SchedulerSettings(assets=[])


# ============================================================================
# AC-1: Scheduler Start (Happy Path)
# ============================================================================


def test_scheduler_start_happy_path():
    """AC-1: Scheduler Start - Happy Path."""
    config = SchedulerSettings(
        enabled=True,
        interval_hours=1.0,
        assets=["BTC", "SPX"],
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

    result = scheduler.start(immediate=False)

    assert result is True
    assert scheduler.is_running is True
    assert scheduler.scheduler.started is True
    # Verify job was registered
    assert len(scheduler.scheduler.jobs) == 1
    assert scheduler.scheduler.jobs[0]["id"] == TradingScheduler.JOB_ID


def test_scheduler_start_disabled_config():
    """AC-1: Scheduler Start - When disabled in config."""
    config = SchedulerSettings(
        enabled=False,
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

    result = scheduler.start()
    assert result is False
    assert scheduler.is_running is False


# ============================================================================
# AC-2: Scheduler Stop (Graceful Shutdown)
# ============================================================================


def test_scheduler_stop_graceful_shutdown():
    """AC-2: Scheduler Stop - Graceful Shutdown."""
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

    scheduler.start(immediate=False)
    assert scheduler.is_running is True

    scheduler.stop()

    assert scheduler.is_running is False
    assert scheduler.scheduler.shutdown_called is True


def test_scheduler_stop_idempotent():
    """AC-16: Idempotency - Stop Without Start."""
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

    # Should not raise exception
    scheduler.stop()
    assert scheduler.is_running is False


# ============================================================================
# AC-3: Analysis Cycle - LONG Signal
# ============================================================================


def test_trading_scheduler_long_signal_executes_order():
    """AC-3: Analysis Cycle - LONG Signal."""
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
    order_manager.execute_decision.return_value = _make_mock_order()

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
    assert stats["errors"] == 0
    order_manager.execute_decision.assert_called_once()
    kwargs = order_manager.execute_decision.call_args.kwargs
    assert kwargs["environment"] == Environment.PAPER
    assert kwargs["symbol"] == "BTC"
    assert kwargs["decision"] == TradeSignal.LONG


# ============================================================================
# AC-4: Analysis Cycle - SHORT Signal
# ============================================================================


def test_trading_scheduler_short_signal_executes_order():
    """AC-4: Analysis Cycle - SHORT Signal."""
    config = SchedulerSettings(
        enabled=True,
        interval_hours=1.0,
        assets=["SPX"],
        environment="paper",
        timeframe="1h",
        lookback_hours=12,
    )

    session = DummySession()
    data_provider = MagicMock()
    data_provider.get_ohlc.return_value = _sample_df()

    order_manager = MagicMock()
    order_manager.execute_decision.return_value = _make_mock_order()

    strategy = Mock()
    strategy.generate_signal.return_value = StrategyTradingSignal(
        decision="SHORT",
        confidence=0.75,
        entry_price=101.0,
        stop_loss=105.0,
        take_profit=92.5,
        reasoning="Test SHORT signal",
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
    assert stats["errors"] == 0
    order_manager.execute_decision.assert_called_once()
    kwargs = order_manager.execute_decision.call_args.kwargs
    assert kwargs["environment"] == Environment.PAPER
    assert kwargs["symbol"] == "SPX"
    assert kwargs["decision"] == TradeSignal.SHORT


# ============================================================================
# AC-5: Analysis Cycle - HOLD Signal (No Action)
# ============================================================================


def test_trading_scheduler_hold_signal_skips_execution():
    """AC-5: Analysis Cycle - HOLD Signal (No Action)."""
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
    assert stats["errors"] == 0
    order_manager.execute_decision.assert_not_called()


# ============================================================================
# AC-6: Error Handling - Transient (API Timeout)
# ============================================================================


def test_scheduler_transient_error_continue_with_next_asset():
    """AC-6: Error Handling - Transient (API Timeout)."""
    config = SchedulerSettings(
        enabled=True,
        interval_hours=1.0,
        assets=["BTC", "SPX"],
        environment="paper",
        timeframe="1h",
        lookback_hours=12,
    )

    session = DummySession()
    data_provider = MagicMock()
    # First asset fails with timeout, second succeeds
    data_provider.get_ohlc.side_effect = [
        TimeoutError("API timeout"),
        _sample_df(),
    ]

    order_manager = MagicMock()
    order_manager.execute_decision.return_value = _make_mock_order()

    strategy = Mock()
    strategy.generate_signal.return_value = StrategyTradingSignal(
        decision="LONG",
        confidence=0.8,
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

    # Should have processed BTC (failed) and SPX (succeeded)
    assert stats["processed"] == 1  # Only 1 succeeded
    assert stats["errors"] == 1  # 1 failed
    assert stats["total"] == 2
    # Only SPX should have been executed
    order_manager.execute_decision.assert_called_once()
    kwargs = order_manager.execute_decision.call_args.kwargs
    assert kwargs["symbol"] == "SPX"


# ============================================================================
# AC-7: Error Handling - Analysis Failure
# ============================================================================


def test_scheduler_analysis_failure_continues_processing():
    """AC-7: Error Handling - Analysis Failure."""
    config = SchedulerSettings(
        enabled=True,
        interval_hours=1.0,
        assets=["BTC", "ETH"],
        environment="paper",
        timeframe="1h",
        lookback_hours=12,
    )

    session = DummySession()
    data_provider = MagicMock()
    data_provider.get_ohlc.return_value = _sample_df()

    order_manager = MagicMock()
    order_manager.execute_decision.return_value = _make_mock_order()

    strategy = Mock()
    # First call raises exception, second succeeds
    strategy.generate_signal.side_effect = [
        Exception("Model error"),
        StrategyTradingSignal(
            decision="LONG",
            confidence=0.8,
            entry_price=101.0,
            stop_loss=97.5,
            take_profit=110.0,
            reasoning="Test",
        ),
    ]

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
    assert stats["errors"] == 1
    assert stats["total"] == 2
    # Only ETH should execute
    order_manager.execute_decision.assert_called_once()
    kwargs = order_manager.execute_decision.call_args.kwargs
    assert kwargs["symbol"] == "ETH"


# ============================================================================
# AC-11: Environment Tagging - Database Records
# ============================================================================


def test_scheduler_environment_tagging_in_execution():
    """AC-11: Environment Tagging - Database Records."""
    config = SchedulerSettings(
        enabled=True,
        interval_hours=1.0,
        assets=["BTC", "SPX"],
        environment="paper",
        timeframe="1h",
        lookback_hours=12,
    )

    session = DummySession()
    data_provider = MagicMock()
    data_provider.get_ohlc.return_value = _sample_df()

    order_manager = MagicMock()
    order_manager.execute_decision.return_value = _make_mock_order()

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

    scheduler.run_once()

    # Verify environment is tagged as 'paper' in all calls
    calls = order_manager.execute_decision.call_args_list
    for call in calls:
        kwargs = call.kwargs
        assert kwargs["environment"] == Environment.PAPER


# ============================================================================
# AC-15: Idempotency - Double Start
# ============================================================================


def test_scheduler_double_start_idempotent():
    """AC-15: Idempotency - Double Start."""
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

    # First start
    result1 = scheduler.start(immediate=False)
    assert result1 is True
    assert scheduler.is_running is True
    job_count_after_first = len(scheduler.scheduler.jobs)

    # Second start (should be no-op)
    result2 = scheduler.start(immediate=False)
    assert result2 is False
    assert scheduler.is_running is True
    # Job count should remain the same (no duplicate job registered)
    job_count_after_second = len(scheduler.scheduler.jobs)
    assert job_count_after_first == job_count_after_second


# ============================================================================
# Integration: Multiple Assets in Cycle
# ============================================================================


def test_scheduler_processes_multiple_assets():
    """Test that scheduler processes all configured assets in single cycle."""
    config = SchedulerSettings(
        enabled=True,
        interval_hours=1.0,
        assets=["BTC", "ETH", "SPX"],
        environment="paper",
        timeframe="1h",
        lookback_hours=12,
    )

    session = DummySession()
    data_provider = MagicMock()
    data_provider.get_ohlc.return_value = _sample_df()

    order_manager = MagicMock()
    order_manager.execute_decision.return_value = _make_mock_order()

    strategy = Mock()
    strategy.generate_signal.return_value = StrategyTradingSignal(
        decision="LONG",
        confidence=0.8,
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

    assert stats["processed"] == 3
    assert stats["errors"] == 0
    assert stats["total"] == 3
    assert order_manager.execute_decision.call_count == 3

    # Verify all assets were processed
    symbols_processed = [
        call.kwargs["symbol"] for call in order_manager.execute_decision.call_args_list
    ]
    assert set(symbols_processed) == {"BTC", "ETH", "SPX"}


# ============================================================================
# Integration: Cycle Tracking and Stats
# ============================================================================


def test_scheduler_tracks_last_run_stats():
    """Test that scheduler tracks last run statistics."""
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
    order_manager.execute_decision.return_value = _make_mock_order()

    strategy = Mock()
    strategy.generate_signal.return_value = StrategyTradingSignal(
        decision="LONG",
        confidence=0.8,
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

    assert scheduler.last_run_stats is None

    scheduler.run_once()

    assert scheduler.last_run_stats is not None
    assert scheduler.last_run_stats["processed"] == 1
    assert scheduler.last_run_stats["errors"] == 0
    assert "duration_seconds" in scheduler.last_run_stats
