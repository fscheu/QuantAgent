"""Tests for QuantAgent-4w4 strategy-driven backtest lookback windows."""

import math
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from quantagent.backtesting.backtest import Backtest
from quantagent.models import ActivePosition
from quantagent.strategy.base import TradingStrategy
from quantagent.strategy.triple_screen_strategy import TripleScreenStrategy


class DummyStrategy(TradingStrategy):
    def __init__(self, required_history_bars: int = 30):
        self._required_history_bars = required_history_bars
        self.generate_signal_calls = 0

    @property
    def required_history_bars(self) -> int:
        return self._required_history_bars

    def generate_signal(
        self,
        kline_data,
        symbol: str,
        timeframe: str,
        current_price: float,
        **kwargs,
    ):
        self.generate_signal_calls += 1
        return None

    def should_reevaluate(self, position: ActivePosition, current_price: float) -> bool:
        return False


@pytest.fixture
def backtest_factory():
    def _build(timeframe: str = "1d", strategy: TradingStrategy | None = None):
        with (
            patch("quantagent.backtesting.backtest.DataProvider"),
            patch("quantagent.backtesting.backtest.StrategyAssembler") as mock_assembler,
            patch("quantagent.backtesting.backtest.PositionMonitor") as mock_monitor,
        ):
            mock_resolved = Mock()
            mock_assembler.from_snapshot.return_value = mock_resolved

            mock_components = Mock()
            mock_components.graph = Mock()
            mock_components.portfolio_manager = Mock()
            mock_components.position_sizer = Mock()
            mock_components.risk_manager = Mock()
            mock_components.broker = Mock()
            mock_components.order_manager = Mock()
            mock_assembler.build_components.return_value = mock_components

            monitor_instance = Mock()
            monitor_instance.get_active_position.return_value = None
            mock_monitor.return_value = monitor_instance

            strategy = strategy or DummyStrategy()
            return Backtest(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
                assets=["BTC"],
                timeframe=timeframe,
                initial_capital=100000.0,
                db_session=Mock(),
                strategy=strategy,
            )

    return _build


def _make_dataframe(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [1000.0] * rows,
        }
    )


def test_4w4_required_history_bars_default():
    strategy = TripleScreenStrategy()

    assert strategy.required_history_bars == 30


def test_4w4_required_history_bars_override():
    strategy = DummyStrategy(required_history_bars=300)

    assert strategy.required_history_bars == 300


@pytest.mark.parametrize(
    ("timeframe", "bars", "expected"),
    [
        ("1d", 252, 365),
        ("1d", 303, 439),
        ("1h", 303, math.ceil(303 / 6.5 * 7 / 5)),
        ("4h", 303, math.ceil(303 * 4 / 6.5 * 7 / 5)),
        ("15m", 303, 606),
    ],
)
def test_4w4_bars_to_calendar_days(backtest_factory, timeframe, bars, expected):
    backtest = backtest_factory(timeframe=timeframe)

    assert backtest._bars_to_calendar_days(bars) == expected


def test_4w4_engine_requests_sufficient_bars(backtest_factory):
    strategy = DummyStrategy(required_history_bars=300)
    backtest = backtest_factory(timeframe="1d", strategy=strategy)
    backtest.data_provider.get_ohlc.return_value = pd.DataFrame()

    current_date = datetime(2024, 12, 31)
    backtest._analyze_and_trade("BTC", current_date)

    start_date = backtest.data_provider.get_ohlc.call_args.kwargs["start_date"]
    requested_days = (current_date - start_date).days

    assert requested_days == math.ceil(300 * 365 / 252)


def test_4w4_insufficient_data_guard(backtest_factory, caplog):
    strategy = DummyStrategy(required_history_bars=300)
    backtest = backtest_factory(timeframe="1d", strategy=strategy)
    backtest.data_provider.get_ohlc.return_value = _make_dataframe(10)

    with caplog.at_level("WARNING"):
        backtest._analyze_and_trade("BTC", datetime(2024, 12, 31))

    assert "need 300" in caplog.text
    assert strategy.generate_signal_calls == 0


def test_4w4_non_positive_required_history_bars_falls_back_to_30(backtest_factory):
    strategy = DummyStrategy(required_history_bars=0)
    backtest = backtest_factory(timeframe="1d", strategy=strategy)
    backtest.data_provider.get_ohlc.return_value = pd.DataFrame()

    current_date = datetime(2024, 12, 31)
    backtest._analyze_and_trade("BTC", current_date)

    start_date = backtest.data_provider.get_ohlc.call_args.kwargs["start_date"]
    requested_days = (current_date - start_date).days

    assert requested_days == math.ceil(30 * 365 / 252)


def test_4w4_no_spurious_warnings(backtest_factory, caplog):
    strategy = DummyStrategy(required_history_bars=30)
    backtest = backtest_factory(timeframe="1d", strategy=strategy)
    backtest.data_provider.get_ohlc.return_value = _make_dataframe(35)

    with caplog.at_level("WARNING"):
        backtest._analyze_and_trade("BTC", datetime(2024, 12, 31))

    assert "Insufficient data" not in caplog.text
    assert strategy.generate_signal_calls == 1
