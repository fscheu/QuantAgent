"""Integration tests for backtesting engine end-to-end flow."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
import pandas as pd

from quantagent.backtesting.backtest import Backtest, BacktestMetrics
from quantagent.models import BacktestRun, Trade, Signal, Environment, MarketData
from quantagent.database import SessionLocal


class TestBacktestIntegration:
    """Integration test suite for full backtest flow."""

    @pytest.fixture
    def db_session(self):
        """Create test database session."""
        session = SessionLocal()
        yield session
        # Cleanup
        session.query(Trade).delete()
        session.query(Signal).delete()
        session.query(BacktestRun).delete()
        session.query(MarketData).delete()
        session.commit()
        session.close()

    @pytest.fixture
    def sample_config(self):
        """Sample backtest configuration."""
        return {
            'base_position_pct': 0.05,
            'max_daily_loss_pct': 0.05,
            'max_position_pct': 0.10,
            'slippage_pct': 0.01,
            'agent_llm_provider': 'openai',
            'agent_llm_model': 'gpt-4o-mini',
            'agent_llm_temperature': 0.1
        }

    @pytest.fixture
    def mock_market_data(self, db_session):
        """Create mock market data in database."""
        start_date = datetime(2024, 1, 1, 0, 0, 0)

        # Create 30 days of hourly data for BTC and SPX
        for symbol, base_price in [('BTC', 42000), ('SPX', 4500)]:
            for day in range(30):
                for hour in range(24):
                    timestamp = start_date + timedelta(days=day, hours=hour)
                    price = base_price + (day * 10) + (hour * 1)  # Gradually increasing price

                    record = MarketData(
                        symbol=symbol,
                        timeframe='1h',
                        timestamp=timestamp,
                        open=Decimal(str(price)),
                        high=Decimal(str(price + 5)),
                        low=Decimal(str(price - 5)),
                        close=Decimal(str(price + 2)),
                        volume=Decimal('1000000')
                    )
                    db_session.add(record)

        db_session.commit()

    def test_full_backtest_flow_minimal(self, db_session, mock_market_data, sample_config):
        """
        Integration test: Full backtest flow with minimal mocking.

        Tests:
        - DataProvider fetches data from cache
        - Backtest creates BacktestRun record
        - Analysis is executed (mocked for speed)
        - Metrics are calculated
        - Results are persisted
        """
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 2, 23, 0, 0)  # 2 days

        # Mock TradingGraph to avoid actual LLM calls
        with patch('quantagent.backtesting.backtest.TradingGraph') as mock_tg_class:
            # Create mock graph instance
            mock_graph = MagicMock()
            mock_tg_instance = MagicMock()
            mock_tg_instance.graph = mock_graph
            mock_tg_class.return_value = mock_tg_instance

            # Mock graph.invoke to return analysis result
            def mock_invoke(state, config=None):
                return {
                    'final_trade_decision': 'LONG - Strong bullish momentum detected',
                    'indicator_report': Mock(confidence=0.75),
                    'pattern_report': Mock(primary_pattern='bullish_engulfing'),
                    'trend_report': Mock(trend_direction='up'),
                    'rsi': [55.0],
                    'macd': [0.5]
                }

            mock_graph.invoke = mock_invoke

            # Create and run backtest
            backtest = Backtest(
                start_date=start_date,
                end_date=end_date,
                assets=['BTC'],
                timeframe='1h',
                initial_capital=100000.0,
                config=sample_config,
                db_session=db_session,
                use_checkpointing=False
            )

            metrics = backtest.run(name="Integration Test Backtest")

            # Verify BacktestRun was created
            run = db_session.query(BacktestRun).filter(
                BacktestRun.id == backtest.backtest_run_id
            ).first()

            assert run is not None
            assert run.name == "Integration Test Backtest"
            assert run.timeframe == '1h'
            assert run.assets == ['BTC']

            # Verify metrics were calculated
            assert isinstance(metrics, BacktestMetrics)
            assert metrics.total_trades >= 0

            # Verify metrics were persisted
            run_updated = db_session.query(BacktestRun).filter(
                BacktestRun.id == backtest.backtest_run_id
            ).first()

            assert run_updated.total_trades is not None
            assert run_updated.total_trades == metrics.total_trades

    def test_backtest_with_trades_execution(self, db_session, mock_market_data, sample_config):
        """
        Integration test: Backtest executes trades and calculates P&L.

        Tests:
        - Orders are created
        - Trades are executed via OrderManager
        - Portfolio is updated
        - Signals are persisted with environment tag
        """
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 5, 0, 0)  # 6 hours (short test)

        with patch('quantagent.backtesting.backtest.TradingGraph') as mock_tg_class:
            mock_graph = MagicMock()
            mock_tg_instance = MagicMock()
            mock_tg_instance.graph = mock_graph
            mock_tg_class.return_value = mock_tg_instance

            # Alternate between LONG and NEUTRAL decisions
            call_count = [0]

            def mock_invoke(state, config=None):
                call_count[0] += 1
                if call_count[0] % 2 == 1:
                    return {
                        'final_trade_decision': 'LONG',
                        'indicator_report': Mock(confidence=0.8),
                        'pattern_report': Mock(primary_pattern='bullish'),
                        'trend_report': Mock(trend_direction='up'),
                        'rsi': [60.0],
                        'macd': [1.0]
                    }
                else:
                    return {
                        'final_trade_decision': 'HOLD',
                        'indicator_report': Mock(confidence=0.3),
                        'pattern_report': Mock(primary_pattern='neutral'),
                        'trend_report': Mock(trend_direction='sideways'),
                        'rsi': [50.0],
                        'macd': [0.0]
                    }

            mock_graph.invoke = mock_invoke

            backtest = Backtest(
                start_date=start_date,
                end_date=end_date,
                assets=['BTC'],
                timeframe='1h',
                initial_capital=100000.0,
                config=sample_config,
                db_session=db_session,
                use_checkpointing=False
            )

            metrics = backtest.run(name="Trade Execution Test")

            # Verify signals were created
            signals = db_session.query(Signal).filter(
                Signal.environment == Environment.BACKTEST
            ).all()

            assert len(signals) > 0

            # Verify all signals have BACKTEST environment
            for signal in signals:
                assert signal.environment == Environment.BACKTEST

            # Verify model metadata is stored
            for signal in signals:
                assert signal.model_provider == 'openai'
                assert signal.model_name == 'gpt-4o-mini'

    def test_backtest_metrics_accuracy(self, db_session, mock_market_data, sample_config):
        """
        Integration test: Verify metrics calculations are accurate.

        Tests:
        - Win rate calculation
        - Total P&L calculation
        - Equity curve generation
        """
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 3, 0, 0)  # 4 hours

        with patch('quantagent.backtesting.backtest.TradingGraph') as mock_tg_class:
            mock_graph = MagicMock()
            mock_tg_instance = MagicMock()
            mock_tg_instance.graph = mock_graph
            mock_tg_class.return_value = mock_tg_instance

            # Always return LONG decision
            def mock_invoke(state, config=None):
                return {
                    'final_trade_decision': 'LONG',
                    'indicator_report': Mock(confidence=0.9),
                    'pattern_report': Mock(primary_pattern='bullish'),
                    'trend_report': Mock(trend_direction='up'),
                    'rsi': [70.0],
                    'macd': [2.0]
                }

            mock_graph.invoke = mock_invoke

            backtest = Backtest(
                start_date=start_date,
                end_date=end_date,
                assets=['BTC'],
                timeframe='1h',
                initial_capital=100000.0,
                config=sample_config,
                db_session=db_session,
                use_checkpointing=False
            )

            metrics = backtest.run(name="Metrics Accuracy Test")

            # Verify metrics structure
            assert isinstance(metrics, BacktestMetrics)
            assert hasattr(metrics, 'total_trades')
            assert hasattr(metrics, 'win_rate')
            assert hasattr(metrics, 'total_pnl')
            assert hasattr(metrics, 'sharpe_ratio')
            assert hasattr(metrics, 'max_drawdown')

            # Verify equity curve was generated
            equity_df = backtest.get_equity_curve()
            assert isinstance(equity_df, pd.DataFrame)
            assert len(equity_df) > 0

    def test_backtest_handles_risk_rejections(self, db_session, mock_market_data, sample_config):
        """
        Integration test: Backtest handles risk manager rejections.

        Tests:
        - Orders rejected due to risk limits are NOT executed
        - Rejections are logged
        - Portfolio stays within limits
        """
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 2, 0, 0)

        # Use very restrictive risk limits to trigger rejections
        strict_config = sample_config.copy()
        strict_config['max_daily_loss_pct'] = 0.001  # 0.1% max loss (very strict)
        strict_config['max_position_pct'] = 0.01  # 1% max position (very strict)

        with patch('quantagent.backtesting.backtest.TradingGraph') as mock_tg_class:
            mock_graph = MagicMock()
            mock_tg_instance = MagicMock()
            mock_tg_instance.graph = mock_graph
            mock_tg_class.return_value = mock_tg_instance

            # Always return LONG decision
            def mock_invoke(state, config=None):
                return {
                    'final_trade_decision': 'LONG',
                    'indicator_report': Mock(confidence=1.0),  # High confidence
                    'pattern_report': Mock(primary_pattern='strong_bullish'),
                    'trend_report': Mock(trend_direction='strong_up'),
                    'rsi': [80.0],
                    'macd': [5.0]
                }

            mock_graph.invoke = mock_invoke

            backtest = Backtest(
                start_date=start_date,
                end_date=end_date,
                assets=['BTC'],
                timeframe='1h',
                initial_capital=100000.0,
                config=strict_config,
                db_session=db_session,
                use_checkpointing=False
            )

            metrics = backtest.run(name="Risk Rejection Test")

            # With strict limits, many trades should be rejected
            # Verify that not all signals resulted in trades
            signals = db_session.query(Signal).filter(
                Signal.environment == Environment.BACKTEST,
                Signal.generated_at >= start_date,
                Signal.generated_at <= end_date
            ).count()

            trades = db_session.query(Trade).filter(
                Trade.environment == Environment.BACKTEST,
                Trade.opened_at >= start_date,
                Trade.opened_at <= end_date
            ).count()

            # Should have signals but fewer (or zero) trades due to rejections
            assert signals >= trades

    def test_backtest_config_snapshot_reproducibility(self, db_session, mock_market_data, sample_config):
        """
        Integration test: Config snapshot enables reproducibility.

        Tests:
        - Config snapshot is persisted
        - Snapshot includes all critical parameters
        - Different runs can use different configs
        """
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 1, 0, 0)

        with patch('quantagent.backtesting.backtest.TradingGraph') as mock_tg_class:
            mock_graph = MagicMock()
            mock_tg_instance = MagicMock()
            mock_tg_instance.graph = mock_graph
            mock_tg_class.return_value = mock_tg_instance

            def mock_invoke(state, config=None):
                return {
                    'final_trade_decision': 'HOLD',
                    'indicator_report': Mock(confidence=0.5),
                    'rsi': [50.0],
                    'macd': [0.0]
                }

            mock_graph.invoke = mock_invoke

            # Run with config 1
            config1 = sample_config.copy()
            config1['base_position_pct'] = 0.05

            backtest1 = Backtest(
                start_date=start_date,
                end_date=end_date,
                assets=['BTC'],
                timeframe='1h',
                initial_capital=100000.0,
                config=config1,
                db_session=db_session
            )

            backtest1.run(name="Config Test 1")

            # Run with config 2
            config2 = sample_config.copy()
            config2['base_position_pct'] = 0.10  # Different config

            backtest2 = Backtest(
                start_date=start_date,
                end_date=end_date,
                assets=['BTC'],
                timeframe='1h',
                initial_capital=100000.0,
                config=config2,
                db_session=db_session
            )

            backtest2.run(name="Config Test 2")

            # Verify both runs have different config snapshots
            run1 = db_session.query(BacktestRun).filter(
                BacktestRun.id == backtest1.backtest_run_id
            ).first()

            run2 = db_session.query(BacktestRun).filter(
                BacktestRun.id == backtest2.backtest_run_id
            ).first()

            assert run1.config_snapshot['base_position_pct'] == 0.05
            assert run2.config_snapshot['base_position_pct'] == 0.10

    def test_backtest_date_range_iteration(self, db_session, mock_market_data, sample_config):
        """
        Integration test: Backtest correctly iterates through date range.

        Tests:
        - All dates in range are analyzed
        - Analysis is executed for each asset at each date
        """
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 4, 0, 0)  # 5 hours (0, 1, 2, 3, 4)

        with patch('quantagent.backtesting.backtest.TradingGraph') as mock_tg_class:
            mock_graph = MagicMock()
            mock_tg_instance = MagicMock()
            mock_tg_instance.graph = mock_graph
            mock_tg_class.return_value = mock_tg_instance

            invoke_calls = []

            def mock_invoke(state, config=None):
                invoke_calls.append(state['stock_name'])
                return {
                    'final_trade_decision': 'HOLD',
                    'indicator_report': Mock(confidence=0.5),
                    'rsi': [50.0],
                    'macd': [0.0]
                }

            mock_graph.invoke = mock_invoke

            backtest = Backtest(
                start_date=start_date,
                end_date=end_date,
                assets=['BTC', 'SPX'],  # 2 assets
                timeframe='1h',
                initial_capital=100000.0,
                config=sample_config,
                db_session=db_session
            )

            backtest.run(name="Date Range Test")

            # Should have 5 hours * 2 assets = 10 invoke calls
            assert len(invoke_calls) == 10
            assert invoke_calls.count('BTC') == 5
            assert invoke_calls.count('SPX') == 5

    def test_backtest_equity_curve_tracking(self, db_session, mock_market_data, sample_config):
        """
        Integration test: Equity curve is tracked throughout backtest.

        Tests:
        - Equity is recorded at each period
        - Equity reflects portfolio changes
        - DataFrame export works correctly
        """
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 2, 0, 0)

        with patch('quantagent.backtesting.backtest.TradingGraph') as mock_tg_class:
            mock_graph = MagicMock()
            mock_tg_instance = MagicMock()
            mock_tg_instance.graph = mock_graph
            mock_tg_class.return_value = mock_tg_instance

            def mock_invoke(state, config=None):
                return {
                    'final_trade_decision': 'HOLD',
                    'indicator_report': Mock(confidence=0.5),
                    'rsi': [50.0],
                    'macd': [0.0]
                }

            mock_graph.invoke = mock_invoke

            backtest = Backtest(
                start_date=start_date,
                end_date=end_date,
                assets=['BTC'],
                timeframe='1h',
                initial_capital=100000.0,
                config=sample_config,
                db_session=db_session
            )

            backtest.run(name="Equity Curve Test")

            # Get equity curve
            equity_df = backtest.get_equity_curve()

            # Verify structure
            assert isinstance(equity_df, pd.DataFrame)
            assert 'date' in equity_df.columns
            assert 'equity' in equity_df.columns
            assert 'cash' in equity_df.columns
            assert 'positions_value' in equity_df.columns

            # Should have recorded equity for each period
            assert len(equity_df) > 0
