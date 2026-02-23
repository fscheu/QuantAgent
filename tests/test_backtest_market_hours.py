"""Integration tests for backtest market hours filtering."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from quantagent.backtesting.backtest import Backtest


class TestBacktestMarketHours:
    """Test backtest integration with market hours filtering."""

    @pytest.fixture
    def mock_components(self):
        """Mock external dependencies."""
        with (
            patch("quantagent.backtesting.backtest.DataProvider"),
            patch(
                "quantagent.backtesting.backtest.StrategyAssembler"
            ) as mock_assembler,
            patch("quantagent.backtesting.backtest.SessionLocal"),
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

            yield

    def test_backtest_filtering_enabled_by_default(self, mock_components):
        """Test that filtering is enabled by default."""
        backtest = Backtest(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            assets=["SPX"],
            timeframe="4h",
            initial_capital=100000.0,
        )

        assert backtest.market_hours_filter is True
        assert backtest._market_calendar is not None

    def test_backtest_filtering_can_be_disabled(self, mock_components):
        """Test that filtering can be disabled via config."""
        backtest = Backtest(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            assets=["SPX"],
            timeframe="4h",
            initial_capital=100000.0,
            config={"market_hours_filter": False},
        )

        assert backtest.market_hours_filter is False
        assert backtest._market_calendar is None

    def test_get_date_range_for_asset_crypto_no_filtering(self, mock_components):
        """Test crypto assets get all timestamps."""
        backtest = Backtest(
            start_date=datetime(2024, 1, 1, 0, 0),
            end_date=datetime(2024, 1, 2, 0, 0),
            assets=["BTC"],
            timeframe="4h",
            initial_capital=100000.0,
        )

        btc_periods = backtest._get_date_range_for_asset("BTC")
        all_periods = backtest._get_date_range()

        assert len(btc_periods) == len(all_periods)

    def test_get_date_range_for_asset_equity_filtering(self, mock_components):
        """Test equity assets get filtered timestamps."""
        backtest = Backtest(
            start_date=datetime(2024, 1, 1, 0, 0),
            end_date=datetime(2024, 1, 8, 0, 0),
            assets=["SPX"],
            timeframe="4h",
            initial_capital=100000.0,
        )

        spx_periods = backtest._get_date_range_for_asset("SPX")
        all_periods = backtest._get_date_range()

        assert len(spx_periods) < len(all_periods)
        assert len(spx_periods) > 0

    def test_mixed_assets_different_filtering(self, mock_components):
        """Test mixed assets get different filtering."""
        backtest = Backtest(
            start_date=datetime(2024, 1, 1, 0, 0),
            end_date=datetime(2024, 1, 8, 0, 0),
            assets=["BTC", "SPX"],
            timeframe="4h",
            initial_capital=100000.0,
        )

        btc_periods = backtest._get_date_range_for_asset("BTC")
        spx_periods = backtest._get_date_range_for_asset("SPX")

        assert len(btc_periods) > len(spx_periods)

    def test_filtering_disabled_all_periods(self, mock_components):
        """Test with filtering disabled all assets get all periods."""
        backtest = Backtest(
            start_date=datetime(2024, 1, 1, 0, 0),
            end_date=datetime(2024, 1, 8, 0, 0),
            assets=["SPX"],
            timeframe="4h",
            initial_capital=100000.0,
            config={"market_hours_filter": False},
        )

        spx_periods = backtest._get_date_range_for_asset("SPX")
        all_periods = backtest._get_date_range()

        assert len(spx_periods) == len(all_periods)
