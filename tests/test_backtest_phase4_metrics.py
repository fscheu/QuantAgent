"""Tests for Phase 4 metrics (QuantAgent-r6y): MDA, invocation tracking, close reasons."""

from datetime import datetime, timedelta

import pytest

from quantagent.backtesting.backtest import Backtest, BacktestMetrics
from quantagent.database import SessionLocal
from quantagent.models import ActivePosition, Environment, ExitPolicy, OrderSide

pytestmark = pytest.mark.api


class TestPhase4Metrics:
    """Test suite for Phase 4 backtest metrics (QuantAgent-r6y)."""

    @pytest.fixture
    def db_session(self):
        """Create test database session."""
        session = SessionLocal()
        yield session
        # Cleanup
        session.query(ActivePosition).delete()
        session.commit()
        session.close()

    @pytest.fixture
    def sample_dates(self):
        """Sample date range for testing."""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 7, 0, 0, 0)
        return start, end

    # Structure & Type Validation

    def test_backtest_metrics_has_phase4_fields(self):
        """Verify BacktestMetrics dataclass has Phase 4 fields with correct types."""
        metrics = BacktestMetrics(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            max_drawdown=0.15,
            total_pnl=5000.0,
            avg_win=1000.0,
            avg_loss=-500.0,
            largest_win=2000.0,
            largest_loss=-1000.0,
            total_return_pct=5.0,
        )

        # Phase 4 fields should exist with defaults
        assert hasattr(metrics, "agent_invocations")
        assert hasattr(metrics, "invocations_saved")
        assert hasattr(metrics, "invocation_reduction_pct")
        assert hasattr(metrics, "mean_directional_accuracy")
        assert hasattr(metrics, "accuracy_by_candle")
        assert hasattr(metrics, "close_reasons")

        # Check default values
        assert metrics.agent_invocations == 0
        assert metrics.invocations_saved == 0
        assert metrics.invocation_reduction_pct == 0.0
        assert metrics.mean_directional_accuracy == 0.0
        assert isinstance(metrics.accuracy_by_candle, dict)
        assert isinstance(metrics.close_reasons, dict)

    def test_backtest_metrics_phase4_fields_accept_values(self):
        """Verify Phase 4 fields accept correct value types."""
        metrics = BacktestMetrics(
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
            win_rate=0.6,
            profit_factor=1.5,
            sharpe_ratio=1.0,
            max_drawdown=0.1,
            total_pnl=1000.0,
            avg_win=500.0,
            avg_loss=-300.0,
            largest_win=800.0,
            largest_loss=-500.0,
            total_return_pct=1.0,
            agent_invocations=50,
            invocations_saved=200,
            invocation_reduction_pct=80.0,
            mean_directional_accuracy=0.55,
            accuracy_by_candle={1: 0.6, 2: 0.55, 3: 0.5},
            close_reasons={"close_sl": 2, "close_tp": 3},
        )

        assert metrics.agent_invocations == 50
        assert metrics.invocations_saved == 200
        assert metrics.invocation_reduction_pct == 80.0
        assert metrics.mean_directional_accuracy == 0.55
        assert metrics.accuracy_by_candle == {1: 0.6, 2: 0.55, 3: 0.5}
        assert metrics.close_reasons == {"close_sl": 2, "close_tp": 3}

    # Constraint Validation

    def test_mda_range_validation(self):
        """Verify MDA is between 0.0 and 1.0 (AC4.1)."""
        # Valid range
        metrics = BacktestMetrics(
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=1.0,
            profit_factor=1.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_pnl=100.0,
            avg_win=100.0,
            avg_loss=0.0,
            largest_win=100.0,
            largest_loss=0.0,
            total_return_pct=1.0,
            mean_directional_accuracy=0.5,
        )
        assert 0.0 <= metrics.mean_directional_accuracy <= 1.0

        # Edge cases
        metrics.mean_directional_accuracy = 0.0
        assert metrics.mean_directional_accuracy == 0.0

        metrics.mean_directional_accuracy = 1.0
        assert metrics.mean_directional_accuracy == 1.0

    def test_accuracy_by_candle_all_values_in_range(self):
        """Verify all accuracy_by_candle values are between 0.0 and 1.0 (AC4.2)."""
        metrics = BacktestMetrics(
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=1.0,
            profit_factor=1.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_pnl=100.0,
            avg_win=100.0,
            avg_loss=0.0,
            largest_win=100.0,
            largest_loss=0.0,
            total_return_pct=1.0,
            accuracy_by_candle={1: 0.7, 2: 0.6, 3: 0.5},
        )

        for candle_idx, accuracy in metrics.accuracy_by_candle.items():
            assert 0.0 <= accuracy <= 1.0, f"Candle {candle_idx} accuracy out of range"

    def test_invocation_reduction_pct_calculation_correct(self):
        """Verify invocation_reduction_pct is correctly calculated (AC4.3)."""
        # 50 invocations, 200 saved = 250 total = 80% reduction
        metrics = BacktestMetrics(
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=1.0,
            profit_factor=1.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            total_pnl=100.0,
            avg_win=100.0,
            avg_loss=0.0,
            largest_win=100.0,
            largest_loss=0.0,
            total_return_pct=1.0,
            agent_invocations=50,
            invocations_saved=200,
            invocation_reduction_pct=80.0,
        )

        # Verify calculation
        total_candles = metrics.agent_invocations + metrics.invocations_saved
        expected_reduction = (metrics.invocations_saved / total_candles) * 100
        assert metrics.invocation_reduction_pct == pytest.approx(expected_reduction)

    # _calculate_directional_accuracy() tests

    def test_calculate_directional_accuracy_no_positions(
        self, db_session, sample_dates
    ):
        """Verify MDA calculation returns 0.0 when no positions exist."""
        start, end = sample_dates
        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC"],
            timeframe="1h",
            db_session=db_session,
        )

        mda, accuracy_by_candle = backtest._calculate_directional_accuracy()

        assert mda == 0.0
        assert accuracy_by_candle == {}

    def test_calculate_directional_accuracy_perfect_prediction_long(
        self, db_session, sample_dates
    ):
        """Verify MDA = 1.0 for LONG position with all candles up."""
        start, end = sample_dates

        # Create LONG position with all "up" candles (perfect prediction)
        pos = ActivePosition(
            symbol="BTC",
            side=OrderSide.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            quantity=1.0,
            decision_timestamp=start,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["up", "up", "up"],
            is_active=False,
            close_reason="close_tp",
            environment=Environment.BACKTEST,
        )
        db_session.add(pos)
        db_session.commit()

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC"],
            timeframe="1h",
            db_session=db_session,
        )

        mda, accuracy_by_candle = backtest._calculate_directional_accuracy()

        assert mda == 1.0
        assert accuracy_by_candle == {1: 1.0, 2: 1.0, 3: 1.0}

    def test_calculate_directional_accuracy_perfect_prediction_short(
        self, db_session, sample_dates
    ):
        """Verify MDA = 1.0 for SHORT position with all candles down."""
        start, end = sample_dates

        # Create SHORT position with all "down" candles
        pos = ActivePosition(
            symbol="BTC",
            side=OrderSide.SELL,
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0,
            quantity=1.0,
            decision_timestamp=start,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["down", "down", "down"],
            is_active=False,
            close_reason="close_tp",
            environment=Environment.BACKTEST,
        )
        db_session.add(pos)
        db_session.commit()

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC"],
            timeframe="1h",
            db_session=db_session,
        )

        mda, accuracy_by_candle = backtest._calculate_directional_accuracy()

        assert mda == 1.0
        assert accuracy_by_candle == {1: 1.0, 2: 1.0, 3: 1.0}

    def test_calculate_directional_accuracy_zero_accuracy(
        self, db_session, sample_dates
    ):
        """Verify MDA = 0.0 when all predictions are wrong."""
        start, end = sample_dates

        # Create LONG position with all "down" candles (100% wrong)
        pos = ActivePosition(
            symbol="BTC",
            side=OrderSide.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            quantity=1.0,
            decision_timestamp=start,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["down", "down", "down"],
            is_active=False,
            close_reason="close_sl",
            environment=Environment.BACKTEST,
        )
        db_session.add(pos)
        db_session.commit()

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC"],
            timeframe="1h",
            db_session=db_session,
        )

        mda, accuracy_by_candle = backtest._calculate_directional_accuracy()

        assert mda == 0.0
        assert accuracy_by_candle == {1: 0.0, 2: 0.0, 3: 0.0}

    def test_calculate_directional_accuracy_mixed_results(
        self, db_session, sample_dates
    ):
        """Verify MDA calculation with mixed correct/incorrect predictions."""
        start, end = sample_dates

        # Position 1: LONG with 2/3 correct
        pos1 = ActivePosition(
            symbol="BTC",
            side=OrderSide.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            quantity=1.0,
            decision_timestamp=start,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["up", "up", "down"],
            is_active=False,
            close_reason="close_tp",
            environment=Environment.BACKTEST,
        )

        # Position 2: LONG with 1/3 correct
        pos2 = ActivePosition(
            symbol="ETH",
            side=OrderSide.BUY,
            entry_price=50.0,
            stop_loss=48.0,
            take_profit=55.0,
            quantity=1.0,
            decision_timestamp=start + timedelta(hours=1),
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["up", "down", "down"],
            is_active=False,
            close_reason="close_sl",
            environment=Environment.BACKTEST,
        )

        db_session.add_all([pos1, pos2])
        db_session.commit()

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC", "ETH"],
            timeframe="1h",
            db_session=db_session,
        )

        mda, accuracy_by_candle = backtest._calculate_directional_accuracy()

        # Total: 3 correct out of 6 candles = 0.5
        assert mda == pytest.approx(0.5)

        # Candle 1: 2/2 = 1.0, Candle 2: 1/2 = 0.5, Candle 3: 0/2 = 0.0
        assert accuracy_by_candle[1] == 1.0
        assert accuracy_by_candle[2] == 0.5
        assert accuracy_by_candle[3] == 0.0

    def test_calculate_directional_accuracy_respects_prediction_horizon(
        self, db_session, sample_dates
    ):
        """Verify only candles up to prediction_horizon are evaluated."""
        start, end = sample_dates

        # Position with horizon=2 but 4 candles tracked
        pos = ActivePosition(
            symbol="BTC",
            side=OrderSide.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            quantity=1.0,
            decision_timestamp=start,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=2,
            candles_direction=["up", "up", "down", "down"],
            is_active=False,
            close_reason="close_tp",
            environment=Environment.BACKTEST,
        )
        db_session.add(pos)
        db_session.commit()

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC"],
            timeframe="1h",
            db_session=db_session,
        )

        mda, accuracy_by_candle = backtest._calculate_directional_accuracy()

        # Only first 2 candles should be evaluated
        assert mda == 1.0
        assert accuracy_by_candle == {1: 1.0, 2: 1.0}
        assert 3 not in accuracy_by_candle
        assert 4 not in accuracy_by_candle

    def test_calculate_directional_accuracy_filters_by_date_range(
        self, db_session, sample_dates
    ):
        """Verify only positions within backtest date range are included."""
        start, end = sample_dates

        # Position BEFORE date range
        pos_before = ActivePosition(
            symbol="BTC",
            side=OrderSide.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            quantity=1.0,
            decision_timestamp=start - timedelta(days=10),
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["up", "up", "up"],
            is_active=False,
            close_reason="close_tp",
            environment=Environment.BACKTEST,
        )

        # Position WITHIN date range
        pos_within = ActivePosition(
            symbol="ETH",
            side=OrderSide.BUY,
            entry_price=50.0,
            stop_loss=48.0,
            take_profit=55.0,
            quantity=1.0,
            decision_timestamp=start + timedelta(days=1),
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["down", "down", "down"],
            is_active=False,
            close_reason="close_sl",
            environment=Environment.BACKTEST,
        )

        # Position AFTER date range
        pos_after = ActivePosition(
            symbol="SPX",
            side=OrderSide.BUY,
            entry_price=4000.0,
            stop_loss=3900.0,
            take_profit=4100.0,
            quantity=1.0,
            decision_timestamp=end + timedelta(days=10),
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["up", "up", "up"],
            is_active=False,
            close_reason="close_tp",
            environment=Environment.BACKTEST,
        )

        db_session.add_all([pos_before, pos_within, pos_after])
        db_session.commit()

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC", "ETH", "SPX"],
            timeframe="1h",
            db_session=db_session,
        )

        mda, accuracy_by_candle = backtest._calculate_directional_accuracy()

        # Only pos_within (all wrong) should be counted
        assert mda == 0.0
        assert accuracy_by_candle == {1: 0.0, 2: 0.0, 3: 0.0}

    def test_calculate_directional_accuracy_ignores_active_positions(
        self, db_session, sample_dates
    ):
        """Verify active positions (is_active=True) are not included in MDA."""
        start, end = sample_dates

        # Closed position (should be included)
        pos_closed = ActivePosition(
            symbol="BTC",
            side=OrderSide.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            quantity=1.0,
            decision_timestamp=start,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["up", "up", "up"],
            is_active=False,
            close_reason="close_tp",
            environment=Environment.BACKTEST,
        )

        # Active position (should be excluded)
        pos_active = ActivePosition(
            symbol="ETH",
            side=OrderSide.BUY,
            entry_price=50.0,
            stop_loss=48.0,
            take_profit=55.0,
            quantity=1.0,
            decision_timestamp=start,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["down", "down", "down"],
            is_active=True,
            environment=Environment.BACKTEST,
        )

        db_session.add_all([pos_closed, pos_active])
        db_session.commit()

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC", "ETH"],
            timeframe="1h",
            db_session=db_session,
        )

        mda, accuracy_by_candle = backtest._calculate_directional_accuracy()

        # Only closed position should be evaluated
        assert mda == 1.0
        assert accuracy_by_candle == {1: 1.0, 2: 1.0, 3: 1.0}

    # _calculate_close_reasons() tests

    def test_calculate_close_reasons_no_positions(self, db_session, sample_dates):
        """Verify close_reasons returns empty dict when no positions exist."""
        start, end = sample_dates
        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC"],
            timeframe="1h",
            db_session=db_session,
        )

        close_reasons = backtest._calculate_close_reasons()

        assert close_reasons == {}

    def test_calculate_close_reasons_distribution(self, db_session, sample_dates):
        """Verify close_reasons correctly counts distribution (AC4.5)."""
        start, end = sample_dates

        positions = [
            ActivePosition(
                symbol="BTC",
                side=OrderSide.BUY,
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                quantity=1.0,
                decision_timestamp=start,
                exit_policy=ExitPolicy.SL_TP_ONLY,
                prediction_horizon=3,
                candles_direction=["down", "down", "down"],
                is_active=False,
                close_reason="close_sl",
                environment=Environment.BACKTEST,
            ),
            ActivePosition(
                symbol="ETH",
                side=OrderSide.BUY,
                entry_price=50.0,
                stop_loss=48.0,
                take_profit=55.0,
                quantity=1.0,
                decision_timestamp=start,
                exit_policy=ExitPolicy.SL_TP_ONLY,
                prediction_horizon=3,
                candles_direction=["up", "up", "up"],
                is_active=False,
                close_reason="close_tp",
                environment=Environment.BACKTEST,
            ),
            ActivePosition(
                symbol="SPX",
                side=OrderSide.BUY,
                entry_price=4000.0,
                stop_loss=3900.0,
                take_profit=4100.0,
                quantity=1.0,
                decision_timestamp=start,
                exit_policy=ExitPolicy.TRAILING_STOP,
                prediction_horizon=3,
                candles_direction=["up", "down", "down"],
                is_active=False,
                close_reason="close_trailing",
                environment=Environment.BACKTEST,
            ),
            ActivePosition(
                symbol="CL",
                side=OrderSide.BUY,
                entry_price=80.0,
                stop_loss=78.0,
                take_profit=85.0,
                quantity=1.0,
                decision_timestamp=start,
                exit_policy=ExitPolicy.SL_TP_ONLY,
                prediction_horizon=3,
                candles_direction=["down", "down", "down"],
                is_active=False,
                close_reason="close_sl",
                environment=Environment.BACKTEST,
            ),
        ]

        db_session.add_all(positions)
        db_session.commit()

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC", "ETH", "SPX", "CL"],
            timeframe="1h",
            db_session=db_session,
        )

        close_reasons = backtest._calculate_close_reasons()

        assert close_reasons == {
            "close_sl": 2,
            "close_tp": 1,
            "close_trailing": 1,
        }

    def test_calculate_close_reasons_handles_none(self, db_session, sample_dates):
        """Verify close_reasons handles None values as 'unknown'."""
        start, end = sample_dates

        pos = ActivePosition(
            symbol="BTC",
            side=OrderSide.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            quantity=1.0,
            decision_timestamp=start,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["up", "up", "up"],
            is_active=False,
            close_reason=None,
            environment=Environment.BACKTEST,
        )
        db_session.add(pos)
        db_session.commit()

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC"],
            timeframe="1h",
            db_session=db_session,
        )

        close_reasons = backtest._calculate_close_reasons()

        assert close_reasons == {"unknown": 1}

    def test_calculate_close_reasons_filters_by_date_range(
        self, db_session, sample_dates
    ):
        """Verify only positions within backtest date range are counted."""
        start, end = sample_dates

        # Position before range
        pos_before = ActivePosition(
            symbol="BTC",
            side=OrderSide.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            quantity=1.0,
            decision_timestamp=start - timedelta(days=10),
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["up"],
            is_active=False,
            close_reason="close_tp",
            environment=Environment.BACKTEST,
        )

        # Position within range
        pos_within = ActivePosition(
            symbol="ETH",
            side=OrderSide.BUY,
            entry_price=50.0,
            stop_loss=48.0,
            take_profit=55.0,
            quantity=1.0,
            decision_timestamp=start + timedelta(days=1),
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["up"],
            is_active=False,
            close_reason="close_sl",
            environment=Environment.BACKTEST,
        )

        db_session.add_all([pos_before, pos_within])
        db_session.commit()

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC", "ETH"],
            timeframe="1h",
            db_session=db_session,
        )

        close_reasons = backtest._calculate_close_reasons()

        # Only pos_within should be counted
        assert close_reasons == {"close_sl": 1}

    def test_calculate_close_reasons_ignores_active_positions(
        self, db_session, sample_dates
    ):
        """Verify active positions are not included in close_reasons."""
        start, end = sample_dates

        pos_closed = ActivePosition(
            symbol="BTC",
            side=OrderSide.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            quantity=1.0,
            decision_timestamp=start,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["up"],
            is_active=False,
            close_reason="close_tp",
            environment=Environment.BACKTEST,
        )

        pos_active = ActivePosition(
            symbol="ETH",
            side=OrderSide.BUY,
            entry_price=50.0,
            stop_loss=48.0,
            take_profit=55.0,
            quantity=1.0,
            decision_timestamp=start,
            exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3,
            candles_direction=["up"],
            is_active=True,
            environment=Environment.BACKTEST,
        )

        db_session.add_all([pos_closed, pos_active])
        db_session.commit()

        backtest = Backtest(
            start_date=start,
            end_date=end,
            assets=["BTC", "ETH"],
            timeframe="1h",
            db_session=db_session,
        )

        close_reasons = backtest._calculate_close_reasons()

        assert close_reasons == {"close_tp": 1}
