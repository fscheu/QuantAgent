"""Tests for Backtest engine."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
import pandas as pd

from quantagent.backtesting.backtest import Backtest, BacktestMetrics
from quantagent.models import BacktestRun, Trade, Signal, Environment, OrderSide
from quantagent.database import SessionLocal


class TestBacktest:
    """Test suite for Backtest engine."""

    @pytest.fixture
    def db_session(self):
        """Create test database session."""
        session = SessionLocal()
        yield session
        # Cleanup
        session.query(Trade).delete()
        session.query(Signal).delete()
        session.query(BacktestRun).delete()
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
    def sample_dates(self):
        """Sample date range for testing."""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 7, 0, 0, 0)
        return start, end

    # Structure & Type Validation Tests

    def test_backtest_initialization(self, db_session, sample_dates, sample_config):
        """Verify Backtest initializes with correct attributes."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        assert backtest.start_date == start
        assert backtest.end_date == end
        assert backtest.assets == ['BTC']
        assert backtest.timeframe == '1h'
        assert backtest.initial_capital == 100000.0
        assert backtest.config == sample_config

    def test_backtest_creates_run_record(self, db_session, sample_dates, sample_config):
        """Verify backtest creates BacktestRun record in database."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC', 'SPX'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        backtest._create_backtest_run(name="Test Backtest")

        # Check database
        run = db_session.query(BacktestRun).filter(BacktestRun.id == backtest.backtest_run_id).first()

        assert run is not None
        assert run.name == "Test Backtest"
        assert run.timeframe == '1h'
        assert run.assets == ['BTC', 'SPX']
        assert run.start_date == start
        assert run.end_date == end

    def test_backtest_config_snapshot_includes_all_params(self, db_session, sample_dates, sample_config):
        """Verify config snapshot includes all required parameters."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        snapshot = backtest._build_config_snapshot()

        required_fields = [
            'initial_capital',
            'base_position_pct',
            'max_daily_loss_pct',
            'max_position_pct',
            'slippage_pct',
            'model_provider',
            'model_name',
            'temperature'
        ]

        for field in required_fields:
            assert field in snapshot

    # Date Range Generation Tests

    def test_get_date_range_hourly(self, db_session, sample_config):
        """Verify date range generation for hourly timeframe."""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 5, 0, 0)

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        dates = backtest._get_date_range()

        # Should have 6 periods (0, 1, 2, 3, 4, 5)
        assert len(dates) == 6
        assert dates[0] == start
        assert dates[-1] == end

    def test_get_date_range_daily(self, db_session, sample_dates, sample_config):
        """Verify date range generation for daily timeframe."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1d',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        dates = backtest._get_date_range()

        # Should have 7 days (Jan 1-7)
        assert len(dates) == 7
        assert dates[0] == start
        assert dates[-1] == end

    def test_get_date_range_4hour(self, db_session, sample_config):
        """Verify date range generation for 4-hour timeframe."""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 0)

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='4h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        dates = backtest._get_date_range()

        # Should have 4 periods (0, 4, 8, 12)
        assert len(dates) == 4
        assert dates[1] == start + timedelta(hours=4)

    # Metrics Calculation Tests

    def test_calculate_metrics_with_no_trades(self, db_session, sample_dates, sample_config):
        """Verify metrics calculation handles zero trades correctly."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        metrics = backtest._calculate_metrics()

        assert isinstance(metrics, BacktestMetrics)
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.total_pnl == 0.0

    def test_calculate_metrics_win_rate(self, db_session, sample_dates, sample_config):
        """Verify win rate calculation is correct."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        # Create test trades: 3 wins, 2 losses
        for i, pnl in enumerate([100, 50, 75, -40, -30]):
            trade = Trade(
                symbol='BTC',
                entry_price=Decimal('42000'),
                exit_price=Decimal('42000') + Decimal(str(pnl)),
                quantity=Decimal('0.1'),
                side=OrderSide.BUY,
                pnl=Decimal(str(pnl)),
                opened_at=start + timedelta(hours=i),
                environment=Environment.BACKTEST
            )
            db_session.add(trade)
        db_session.commit()

        metrics = backtest._calculate_metrics()

        assert metrics.total_trades == 5
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 2
        assert metrics.win_rate == 0.6  # 3/5 = 60%

    def test_calculate_metrics_profit_factor(self, db_session, sample_dates, sample_config):
        """Verify profit factor calculation is correct."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        # Create test trades: wins = 100 + 50 = 150, losses = 40 + 30 = 70
        for i, pnl in enumerate([100, 50, -40, -30]):
            trade = Trade(
                symbol='BTC',
                entry_price=Decimal('42000'),
                quantity=Decimal('0.1'),
                side=OrderSide.BUY,
                pnl=Decimal(str(pnl)),
                opened_at=start + timedelta(hours=i),
                environment=Environment.BACKTEST
            )
            db_session.add(trade)
        db_session.commit()

        metrics = backtest._calculate_metrics()

        # Profit factor = 150 / 70 ≈ 2.14
        assert 2.0 < metrics.profit_factor < 2.2

    def test_calculate_metrics_total_pnl(self, db_session, sample_dates, sample_config):
        """Verify total P&L calculation is correct."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        # Create test trades
        trade_pnls = [100, -50, 75, -25, 200]
        for i, pnl in enumerate(trade_pnls):
            trade = Trade(
                symbol='BTC',
                entry_price=Decimal('42000'),
                quantity=Decimal('0.1'),
                side=OrderSide.BUY,
                pnl=Decimal(str(pnl)),
                opened_at=start + timedelta(hours=i),
                environment=Environment.BACKTEST
            )
            db_session.add(trade)
        db_session.commit()

        metrics = backtest._calculate_metrics()

        # Total P&L = 100 - 50 + 75 - 25 + 200 = 300
        assert metrics.total_pnl == 300.0

    def test_calculate_metrics_total_return_pct(self, db_session, sample_dates, sample_config):
        """Verify total return percentage calculation is correct."""
        start, end = sample_dates

        initial_capital = 100000.0
        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=initial_capital,
            config=sample_config,
            db_session=db_session
        )

        # Create trade with 5000 profit
        trade = Trade(
            symbol='BTC',
            entry_price=Decimal('42000'),
            quantity=Decimal('0.1'),
            side=OrderSide.BUY,
            pnl=Decimal('5000'),
            opened_at=start,
            environment=Environment.BACKTEST
        )
        db_session.add(trade)
        db_session.commit()

        metrics = backtest._calculate_metrics()

        # Return % = (5000 / 100000) * 100 = 5%
        assert metrics.total_return_pct == 5.0

    def test_calculate_metrics_avg_win_loss(self, db_session, sample_dates, sample_config):
        """Verify average win/loss calculation is correct."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        # Wins: 100, 200 (avg = 150), Losses: -50, -30 (avg = 40)
        for i, pnl in enumerate([100, 200, -50, -30]):
            trade = Trade(
                symbol='BTC',
                entry_price=Decimal('42000'),
                quantity=Decimal('0.1'),
                side=OrderSide.BUY,
                pnl=Decimal(str(pnl)),
                opened_at=start + timedelta(hours=i),
                environment=Environment.BACKTEST
            )
            db_session.add(trade)
        db_session.commit()

        metrics = backtest._calculate_metrics()

        assert metrics.avg_win == 150.0
        assert metrics.avg_loss == 40.0

    def test_calculate_metrics_largest_win_loss(self, db_session, sample_dates, sample_config):
        """Verify largest win/loss calculation is correct."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        # Largest win: 500, Largest loss: -200
        for i, pnl in enumerate([100, 500, 50, -200, -30]):
            trade = Trade(
                symbol='BTC',
                entry_price=Decimal('42000'),
                quantity=Decimal('0.1'),
                side=OrderSide.BUY,
                pnl=Decimal(str(pnl)),
                opened_at=start + timedelta(hours=i),
                environment=Environment.BACKTEST
            )
            db_session.add(trade)
        db_session.commit()

        metrics = backtest._calculate_metrics()

        assert metrics.largest_win == 500.0
        assert metrics.largest_loss == -200.0

    # Max Drawdown Tests

    def test_calculate_max_drawdown_no_data(self, db_session, sample_dates, sample_config):
        """Verify max drawdown returns 0 with no equity data."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        max_dd = backtest._calculate_max_drawdown()

        assert max_dd == 0.0

    def test_calculate_max_drawdown_with_equity_curve(self, db_session, sample_dates, sample_config):
        """Verify max drawdown calculation with equity curve data."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        # Create equity curve: 100k -> 110k -> 95k -> 105k
        # Peak: 110k, Trough: 95k, Drawdown: (110-95)/110 = 13.6%
        backtest.equity_curve = [
            {'date': start, 'equity': 100000, 'cash': 100000, 'positions_value': 0},
            {'date': start + timedelta(hours=1), 'equity': 110000, 'cash': 110000, 'positions_value': 0},
            {'date': start + timedelta(hours=2), 'equity': 95000, 'cash': 95000, 'positions_value': 0},
            {'date': start + timedelta(hours=3), 'equity': 105000, 'cash': 105000, 'positions_value': 0}
        ]

        max_dd = backtest._calculate_max_drawdown()

        # Should be approximately 0.136 (13.6%)
        assert 0.13 < max_dd < 0.14

    # Sharpe Ratio Tests

    def test_calculate_sharpe_ratio_no_data(self, db_session, sample_dates, sample_config):
        """Verify Sharpe ratio returns 0 with no equity data."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        sharpe = backtest._calculate_sharpe_ratio()

        assert sharpe == 0.0

    def test_get_periods_per_year_by_timeframe(self, db_session, sample_dates, sample_config):
        """Verify periods per year calculation for different timeframes."""
        start, end = sample_dates

        # Test different timeframes
        timeframes = {
            '1h': 252 * 6.5,
            '4h': 252 * 1.625,
            '1d': 252,
            '1w': 52
        }

        for timeframe, expected_periods in timeframes.items():
            backtest = Backtest(
                start_date=start,
                end_date=end,
                assets=['BTC'],
                timeframe=timeframe,
                initial_capital=100000.0,
                config=sample_config,
                db_session=db_session
            )

            periods = backtest._get_periods_per_year()
            assert periods == expected_periods

    # Decision Parsing Tests

    def test_parse_decision_long(self, db_session, sample_dates, sample_config):
        """Verify LONG decision parsing."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        assert backtest._parse_decision("LONG") == backtest._parse_decision("long")
        assert backtest._parse_decision("BUY signal detected") == backtest._parse_decision("LONG")

    def test_parse_decision_short(self, db_session, sample_dates, sample_config):
        """Verify SHORT decision parsing."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        from quantagent.models import TradeSignal
        assert backtest._parse_decision("SHORT") == TradeSignal.SHORT
        assert backtest._parse_decision("SELL signal detected") == TradeSignal.SHORT

    def test_parse_decision_neutral(self, db_session, sample_dates, sample_config):
        """Verify NEUTRAL/HOLD decision parsing."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        from quantagent.models import TradeSignal
        assert backtest._parse_decision("HOLD") == TradeSignal.NEUTRAL
        assert backtest._parse_decision("No clear signal") == TradeSignal.NEUTRAL

    # DataFrame Conversion Tests

    def test_df_to_kline_data_conversion(self, db_session, sample_dates, sample_config):
        """Verify DataFrame to kline_data dict conversion."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        df = pd.DataFrame({
            'timestamp': [start],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000.0]
        })

        kline_data = backtest._df_to_kline_data(df)

        assert isinstance(kline_data, dict)
        required_keys = ['timestamps', 'opens', 'highs', 'lows', 'closes', 'volumes']
        for key in required_keys:
            assert key in kline_data

    def test_get_equity_curve_returns_dataframe(self, db_session, sample_dates, sample_config):
        """Verify get_equity_curve returns DataFrame."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        backtest.equity_curve = [
            {'date': start, 'equity': 100000, 'cash': 100000, 'positions_value': 0}
        ]

        df = backtest.get_equity_curve()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'equity' in df.columns

    # Edge Cases

    def test_backtest_handles_empty_asset_list(self, db_session, sample_dates, sample_config):
        """Verify backtest handles empty asset list gracefully."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=[],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        assert backtest.assets == []

    def test_backtest_uses_default_config_when_none_provided(self, db_session, sample_dates):
        """Verify backtest uses default config when none provided."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            db_session=db_session
        )

        assert isinstance(backtest.config, dict)

    def test_extract_confidence_with_missing_report(self, db_session, sample_dates, sample_config):
        """Verify confidence extraction defaults when report missing."""
        start, end = sample_dates

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=['BTC'],
            timeframe='1h',
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session
        )

        result = {}
        confidence = backtest._extract_confidence(result)

        # Should default to 0.5
        assert confidence == 0.5
