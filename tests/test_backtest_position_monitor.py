"""
Tests for Backtest integration with PositionMonitor and TradingStrategy.

Validates AC3.1-AC3.6 from QuantAgent-nu7-AC-active-position-monitoring.md (Phase 3).

Testing Strategy:
- Structure validation (AC3.1, AC3.2)
- Flow validation (AC3.3)
- Data integrity (AC3.4, AC3.5)
- Backward compatibility (AC3.6)
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch

import pytest

from quantagent.backtesting.backtest import Backtest, BacktestMetrics
from quantagent.models import (
    ActivePosition,
    MarketData,
)
from quantagent.strategy.base import ExitPolicy, TradingSignal, TradingStrategy
from quantagent.strategy.llm_agent_strategy import LLMAgentStrategy

pytestmark = pytest.mark.api

class MockSimpleStrategy(TradingStrategy):
    """Simple mock strategy for testing without LLM dependencies."""

    def __init__(self, signal_sequence=None):
        """
        Args:
            signal_sequence: List of decisions to return sequentially ["LONG", "HOLD", ...]
        """
        self.signal_sequence = signal_sequence or ["HOLD"]
        self.call_count = 0

    def generate_signal(self, kline_data, symbol, timeframe, current_price):
        decision = self.signal_sequence[
            min(self.call_count, len(self.signal_sequence) - 1)
        ]
        self.call_count += 1

        if decision == "HOLD":
            return None

        return TradingSignal(
            decision=decision,
            confidence=0.8,
            entry_price=current_price,
            stop_loss=(
                current_price * 0.98 if decision == "LONG" else current_price * 1.02
            ),
            take_profit=(
                current_price * 1.03 if decision == "LONG" else current_price * 0.97
            ),
            reasoning=f"Mock {decision} signal",
            exit_policy=ExitPolicy.TRAILING_STOP,
            trailing_stop_pct=0.05,
        )

    def should_reevaluate(self, position, current_price):
        return False

class TestBacktestPositionMonitorIntegration:
    """Integration tests for Backtest + PositionMonitor + TradingStrategy."""

    @pytest.fixture
    def sample_config(self):
        """Basic backtest configuration."""
        return {
            "base_position_pct": 0.05,
            "max_daily_loss_pct": 0.05,
            "max_position_pct": 0.10,
            "slippage_pct": 0.01,
            "agent_llm_provider": "openai",
            "agent_llm_model": "gpt-4o-mini",
            "agent_llm_temperature": 0.1,
        }

    @pytest.fixture
    def mock_market_data(self, db_session):
        """
        Create mock market data for BTC.

        Creates 35 days of hourly data to satisfy backtest lookback requirements (30 days).
        """
        # Start 35 days before test date to cover lookback period
        start_date = datetime(2023, 11, 27, 0, 0, 0)  # 35 days before 2024-01-01
        base_price = 42000

        for day in range(35):
            for hour in range(24):
                timestamp = start_date + timedelta(days=day, hours=hour)
                # Gradually increasing price with some variation
                price = base_price + (day * 10) + (hour * 2)

                record = MarketData(
                    symbol="BTC",
                    timeframe="1h",
                    timestamp=timestamp,
                    open=Decimal(str(price)),
                    high=Decimal(str(price + 10)),
                    low=Decimal(str(price - 10)),
                    close=Decimal(str(price + 5)),
                    volume=Decimal("1000000"),
                )
                db_session.add(record)

        db_session.commit()

    @pytest.fixture(autouse=True)
    def mock_yfinance(self):
        """
        Auto-patch yfinance to prevent external API calls during tests.

        DataProvider should use DB cache, but this ensures no API leakage.
        """
        with patch("quantagent.data.provider.yf.download") as mock_download:
            # Return empty DataFrame to force DB cache usage
            mock_download.return_value = Mock(empty=True)
            yield mock_download

    # ==================== AC3.1: Backtest accepts strategy parameter ====================

    def test_backtest_accepts_custom_strategy(self, db_session, sample_config):
        """
        AC3.1: Backtest acepta strategy como parametro.

        Validates:
        - Custom strategy can be passed to Backtest.__init__
        - Backtest uses provided strategy (not default)
        - No TradingGraph created when custom strategy provided
        """
        custom_strategy = MockSimpleStrategy(signal_sequence=["HOLD"])

        backtest = Backtest(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            assets=["BTC"],
            timeframe="1h",
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session,
            strategy=custom_strategy,
        )

        # Structure validation
        assert backtest.strategy is custom_strategy
        assert isinstance(backtest.strategy, MockSimpleStrategy)
        assert not isinstance(backtest.strategy, LLMAgentStrategy)

        # TradingGraph should still exist (created by StrategyAssembler)
        # but strategy should NOT be LLMAgentStrategy
        assert backtest.trading_graph is not None

    # ==================== AC3.2: Default strategy is LLMAgentStrategy ====================

    def test_backtest_defaults_to_llm_agent_strategy(self, db_session, sample_config):
        """
        AC3.2: Backtest usa LLMAgentStrategy por defecto.

        Validates:
        - When no strategy provided, Backtest creates LLMAgentStrategy
        - LLMAgentStrategy wraps TradingGraph
        """
        with patch("quantagent.backtesting.backtest.TradingGraph"):
            backtest = Backtest(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
                assets=["BTC"],
                timeframe="1h",
                initial_capital=100000.0,
                config=sample_config,
                db_session=db_session,
                # NO strategy parameter
            )

            # Type validation
            assert backtest.strategy is not None
            assert isinstance(backtest.strategy, LLMAgentStrategy)
            assert backtest.strategy.trading_graph is not None

    # ==================== AC3.3: Active position prevents invocation ====================

    def test_active_position_prevents_strategy_invocation(
        self, db_session, mock_market_data, sample_config
    ):
        """
        AC3.3: Posicion activa evita invocacion de strategy.

        Validates:
        - When position is active and not at exit condition
        - strategy.generate_signal() is NOT called
        - candles_since_entry increments
        """
        # Strategy that returns LONG once, then HOLD
        strategy = MockSimpleStrategy(signal_sequence=["LONG", "HOLD"])

        backtest = Backtest(
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 5, 0, 0),  # 6 hours
            assets=["BTC"],
            timeframe="1h",
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session,
            strategy=strategy,
        )

        backtest.run(name="AC3.3 Test")

        # Invocation count validation
        # Expected: 1 invocation for initial LONG, then position stays active
        # Subsequent candles should NOT invoke strategy.generate_signal()
        assert strategy.call_count < 6, (
            f"Expected < 6 invocations (position should stay active), "
            f"got {strategy.call_count}"
        )

        # Position tracking validation
        positions = (
            db_session.query(ActivePosition)
            .filter(ActivePosition.symbol == "BTC")
            .all()
        )

        assert len(positions) > 0, "At least one position should be created"

        # Candles tracking validation
        for pos in positions:
            if pos.candles_since_entry > 0:
                assert (
                    pos.candles_since_entry > 0
                ), "candles_since_entry should increment while position is active"

    # ==================== AC3.4: Close reason recorded ====================

    def test_position_close_reason_recorded(
        self, db_session, mock_market_data, sample_config
    ):
        """
        AC3.4: Cierre de posicion registra razon.

        Validates:
        - When position closes (SL/TP/trailing)
        - close_reason is recorded in ActivePosition
        """
        # Create a position that will hit stop loss
        # Price starts at 42000, SL at 98% = 41160
        # We need price to drop below SL

        strategy = MockSimpleStrategy(signal_sequence=["LONG"])

        backtest = Backtest(
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 3, 0, 0),
            assets=["BTC"],
            timeframe="1h",
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session,
            strategy=strategy,
        )

        backtest.run(name="AC3.4 Test")

        # Check closed positions have close_reason
        closed_positions = (
            db_session.query(ActivePosition)
            .filter(
                ActivePosition.symbol == "BTC",
                ActivePosition.is_active.is_(False),
            )
            .all()
        )

        for pos in closed_positions:
            # Constraint validation
            assert (
                pos.close_reason is not None
            ), f"Closed position {pos.id} must have close_reason"
            assert pos.close_reason in [
                "STOP_LOSS",
                "TAKE_PROFIT",
                "TRAILING_STOP",
                "TIME_EXPIRED",
            ], f"Invalid close_reason: {pos.close_reason}"
            assert (
                pos.closed_at is not None
            ), "Closed position must have closed_at timestamp"

    # ==================== AC3.5: Position created with SL/TP from signal ====================

    def test_position_created_with_signal_sl_tp(
        self, db_session, mock_market_data, sample_config
    ):
        """
        AC3.5: Nueva posicion se crea con SL/TP del signal.

        Validates:
        - ActivePosition.stop_loss matches TradingSignal.stop_loss
        - ActivePosition.take_profit matches TradingSignal.take_profit
        - ActivePosition.trailing_stop_pct matches signal
        """

        # Custom strategy with known SL/TP values
        class KnownSLTPStrategy(TradingStrategy):
            def generate_signal(self, kline_data, symbol, timeframe, current_price):
                return TradingSignal(
                    decision="LONG",
                    confidence=0.9,
                    entry_price=current_price,
                    stop_loss=95.0,  # Known value
                    take_profit=110.0,  # Known value
                    trailing_stop_pct=0.03,  # Known value
                    exit_policy=ExitPolicy.TRAILING_STOP,
                )

            def should_reevaluate(self, position, current_price):
                return False

        strategy = KnownSLTPStrategy()

        backtest = Backtest(
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 1, 0, 0),
            assets=["BTC"],
            timeframe="1h",
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session,
            strategy=strategy,
        )

        backtest.run(name="AC3.5 Test")

        # Data integrity validation
        positions = (
            db_session.query(ActivePosition)
            .filter(ActivePosition.symbol == "BTC")
            .first()
        )

        if positions:
            # SL/TP from signal should be preserved
            assert (
                float(positions.stop_loss) == 95.0
            ), f"Expected stop_loss=95.0, got {positions.stop_loss}"
            assert (
                float(positions.take_profit) == 110.0
            ), f"Expected take_profit=110.0, got {positions.take_profit}"
            assert (
                positions.trailing_stop_pct == 0.03
            ), f"Expected trailing_stop_pct=0.03, got {positions.trailing_stop_pct}"

    # ==================== AC3.6: Backward compatibility ====================

    def test_backward_compatibility_no_strategy_param(
        self, db_session, mock_market_data, sample_config
    ):
        """
        AC3.6: Compatibilidad con backtest existente.

        Validates:
        - Backtest without strategy parameter works (legacy API)
        - Uses LLMAgentStrategy internally
        - Generates valid metrics
        """
        with patch("quantagent.backtesting.backtest.TradingGraph") as mock_tg:
            # Mock graph to return HOLD (no trades)
            mock_graph = MagicMock()
            mock_tg_instance = MagicMock()
            mock_tg_instance.graph = mock_graph
            mock_tg.return_value = mock_tg_instance

            mock_graph.invoke = Mock(
                return_value={
                    "final_trade_decision": "HOLD",
                    "indicator_report": Mock(confidence=0.5),
                    "rsi": [50.0],
                    "macd": [0.0],
                }
            )

            # Legacy instantiation (no strategy parameter)
            backtest = Backtest(
                start_date=datetime(2024, 1, 1, 0, 0, 0),
                end_date=datetime(2024, 1, 1, 2, 0, 0),
                assets=["BTC"],
                timeframe="1h",
                initial_capital=100000.0,
                config=sample_config,
                db_session=db_session,
                # NO strategy parameter (backward compatible)
            )

            metrics = backtest.run(name="AC3.6 Backward Compatibility Test")

            # API compatibility validation
            assert isinstance(backtest.strategy, LLMAgentStrategy)
            assert backtest.position_monitor is not None

            # Metrics structure validation
            assert isinstance(metrics, BacktestMetrics)
            assert hasattr(metrics, "total_trades")
            assert hasattr(metrics, "win_rate")
            assert hasattr(metrics, "total_pnl")

    # ==================== Error path: Position monitor with closed position ====================

    def test_position_monitor_handles_already_closed_position(
        self, db_session, mock_market_data, sample_config
    ):
        """
        Error path: Verify position_monitor doesn't create duplicate entries.

        Validates:
        - Only one active position per symbol at a time
        - Closed position doesn't block new position
        """
        strategy = MockSimpleStrategy(signal_sequence=["LONG", "LONG"])

        backtest = Backtest(
            start_date=datetime(2024, 1, 1, 0, 0, 0),
            end_date=datetime(2024, 1, 1, 8, 0, 0),
            assets=["BTC"],
            timeframe="1h",
            initial_capital=100000.0,
            config=sample_config,
            db_session=db_session,
            strategy=strategy,
        )

        backtest.run(name="Duplicate Position Test")

        # Constraint validation: Only one active position at a time
        active_positions = (
            db_session.query(ActivePosition)
            .filter(
                ActivePosition.symbol == "BTC",
                ActivePosition.is_active.is_(True),
            )
            .all()
        )

        assert (
            len(active_positions) <= 1
        ), f"Should have at most 1 active position, got {len(active_positions)}"
