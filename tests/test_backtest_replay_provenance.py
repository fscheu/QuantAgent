from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from quantagent.backtesting.backtest import Backtest
from quantagent.models import BacktestRun, Environment, Signal, TradeSignal
from quantagent.strategy.assembler import StrategyAssembler


@pytest.fixture
def replay_dependencies():
    with (
        patch("quantagent.backtesting.backtest.DataProvider") as mock_data_provider_cls,
        patch("quantagent.backtesting.backtest.StrategyAssembler") as mock_assembler,
        patch("quantagent.backtesting.backtest.PositionMonitor") as mock_monitor_cls,
    ):
        data_provider = Mock()
        data_provider.get_ohlc.return_value = pd.DataFrame(
            {"close": [100.0, 101.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="1d"),
        )
        mock_data_provider_cls.return_value = data_provider

        mock_assembler.from_snapshot.return_value = Mock()
        mock_assembler.make_thread_id.side_effect = (
            lambda run_id, symbol, ts: f"backtest_{run_id}_{symbol}_{ts.isoformat()}"
        )
        mock_assembler.config_snapshot.return_value = {
            "initial_cash": 100000.0,
            "base_position_pct": 0.05,
            "max_daily_loss_pct": 0.05,
            "max_position_pct": 0.10,
            "slippage_pct": 0.01,
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0.1,
            "use_checkpointing": False,
            "universe": [],
        }

        components = Mock()
        components.graph = Mock()
        components.portfolio_manager = Mock(cash=100000.0)
        components.portfolio_manager.get_total_value.return_value = 100000.0
        components.position_sizer = Mock()
        components.risk_manager = Mock()
        components.broker = Mock()
        components.order_manager = Mock()
        components.order_manager.execute_decision.return_value = Mock(
            id=501,
            filled_quantity=1.0,
        )
        mock_assembler.build_components.return_value = components

        monitor = Mock()
        monitor.get_active_position.return_value = None
        mock_monitor_cls.return_value = monitor

        yield SimpleNamespace(
            data_provider=data_provider,
            assembler=mock_assembler,
            components=components,
            monitor=monitor,
        )


class TestBacktestReplayProvenance:
    def test_analyze_and_trade_persists_thread_provenance(
        self, db_session, replay_dependencies
    ):
        ts = datetime(2024, 1, 1)
        strategy = Mock()
        strategy.required_history_bars = 2
        strategy.should_exit.return_value = (False, None)
        strategy.generate_signal.return_value = SimpleNamespace(
            decision="LONG",
            confidence=0.75,
            reasoning="Momentum breakout",
            entry_price=101.0,
            stop_loss=99.0,
            take_profit=104.0,
            exit_policy=SimpleNamespace(value="sl_tp_only"),
            trailing_stop_pct=None,
            max_hold_candles=None,
        )

        backtest = Backtest(
            start_date=ts,
            end_date=ts,
            assets=["BTC"],
            timeframe="1d",
            initial_capital=100000.0,
            config={"market_hours_filter": False},
            db_session=db_session,
            strategy=strategy,
        )
        backtest._create_backtest_run("source")

        backtest._analyze_and_trade("BTC", ts)

        signal = db_session.query(Signal).one()
        expected_thread_id = StrategyAssembler.make_thread_id(
            backtest.backtest_run_id, "BTC", ts
        )

        assert signal.thread_id == expected_thread_id
        assert "thread_id" not in strategy.generate_signal.call_args.kwargs

    def test_run_replay_uses_only_source_run_signals(
        self, db_session, replay_dependencies
    ):
        ts = datetime(2024, 1, 1)
        source_run = BacktestRun(
            name="source",
            timeframe="1d",
            assets=["BTC"],
            start_date=ts,
            end_date=ts,
            config_snapshot={"initial_cash": 100000.0},
        )
        overlapping_run = BacktestRun(
            name="overlap",
            timeframe="1d",
            assets=["BTC"],
            start_date=ts,
            end_date=ts,
            config_snapshot={"initial_cash": 100000.0},
        )
        db_session.add_all([source_run, overlapping_run])
        db_session.commit()

        source_signal = Signal(
            symbol="BTC",
            signal=TradeSignal.LONG,
            confidence=0.9,
            timeframe="1d",
            generated_at=ts,
            environment=Environment.BACKTEST,
            analysis_summary="source run signal",
            thread_id=StrategyAssembler.make_thread_id(source_run.id, "BTC", ts),
        )
        overlapping_signal = Signal(
            symbol="BTC",
            signal=TradeSignal.SHORT,
            confidence=0.2,
            timeframe="1d",
            generated_at=ts,
            environment=Environment.BACKTEST,
            analysis_summary="overlapping run signal",
            thread_id=StrategyAssembler.make_thread_id(overlapping_run.id, "BTC", ts),
        )
        db_session.add_all([source_signal, overlapping_signal])
        db_session.commit()

        strategy = Mock()
        strategy.should_exit.return_value = (False, None)
        strategy.generate_signal = Mock()

        backtest = Backtest(
            start_date=ts,
            end_date=ts,
            assets=["BTC"],
            timeframe="1d",
            initial_capital=100000.0,
            config={"market_hours_filter": False},
            db_session=db_session,
            strategy=strategy,
        )

        metrics = backtest.run_replay(source_run_id=source_run.id, name="Replay")
        replay_run = db_session.get(BacktestRun, backtest.backtest_run_id)
        execute_call = replay_dependencies.components.order_manager.execute_decision.call_args

        assert metrics.total_trades == 0
        assert backtest.agent_invocations == 0
        strategy.generate_signal.assert_not_called()
        assert execute_call.kwargs["trigger_signal_id"] == source_signal.id
        assert replay_run is not None
        assert replay_run.replay_source_run_id == source_run.id
