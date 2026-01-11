"""Tests for market calendar functionality."""

from datetime import datetime

import pytest

from quantagent.data.asset_types import AssetType
from quantagent.data.market_calendar import MarketCalendar


class TestMarketCalendar:
    """Test market calendar integration."""

    def test_calendar_loads(self):
        """Test that calendars load successfully."""
        calendar = MarketCalendar()
        nyse = calendar.get_calendar("NYSE")
        assert nyse is not None

        cme = calendar.get_calendar("CME_Equity")
        assert cme is not None

    def test_schedule_structure(self):
        """Test schedule returns valid DataFrame."""
        calendar = MarketCalendar()
        schedule = calendar.get_schedule(
            "NYSE",
            datetime(2024, 1, 2),
            datetime(2024, 1, 5),
        )

        assert schedule is not None
        assert "market_open" in schedule.columns
        assert "market_close" in schedule.columns
        assert len(schedule) > 0

    def test_schedule_caching(self):
        """Test schedule caching works."""
        calendar = MarketCalendar()

        schedule1 = calendar.get_schedule(
            "NYSE", datetime(2024, 1, 2), datetime(2024, 1, 5)
        )
        schedule2 = calendar.get_schedule(
            "NYSE", datetime(2024, 1, 2), datetime(2024, 1, 5)
        )

        assert schedule1 is schedule2

    def test_calendar_fallback(self):
        """Test graceful fallback on invalid exchange."""
        calendar = MarketCalendar()
        result = calendar.get_calendar("INVALID_EXCHANGE_XYZ")
        assert result is None

    def test_crypto_no_filtering(self):
        """Test crypto timestamps are not filtered."""
        calendar = MarketCalendar()
        timestamps = [
            datetime(2024, 1, 1, 3, 0),
            datetime(2024, 1, 6, 12, 0),
            datetime(2024, 1, 7, 23, 0),
        ]

        filtered = calendar.filter_to_trading_hours(timestamps, AssetType.CRYPTO)

        assert len(filtered) == len(timestamps)
        assert filtered == timestamps

    def test_equity_weekend_filtering(self):
        """Test weekends are filtered for US equity."""
        calendar = MarketCalendar()
        timestamps = [
            datetime(2024, 1, 5, 15, 0),
            datetime(2024, 1, 6, 15, 0),
            datetime(2024, 1, 7, 15, 0),
            datetime(2024, 1, 8, 15, 0),
        ]

        filtered = calendar.filter_to_trading_hours(timestamps, AssetType.US_EQUITY)

        assert datetime(2024, 1, 5, 15, 0) in filtered
        assert datetime(2024, 1, 8, 15, 0) in filtered
        assert datetime(2024, 1, 6, 15, 0) not in filtered
        assert datetime(2024, 1, 7, 15, 0) not in filtered

    def test_equity_outside_hours_filtering(self):
        """Test outside hours are filtered for US equity."""
        calendar = MarketCalendar()
        timestamps = [
            datetime(2024, 1, 3, 13, 0),
            datetime(2024, 1, 3, 15, 0),
            datetime(2024, 1, 3, 20, 0),
            datetime(2024, 1, 3, 22, 0),
        ]

        filtered = calendar.filter_to_trading_hours(timestamps, AssetType.US_EQUITY)

        assert datetime(2024, 1, 3, 15, 0) in filtered
        assert datetime(2024, 1, 3, 20, 0) in filtered
        assert datetime(2024, 1, 3, 13, 0) not in filtered
        assert datetime(2024, 1, 3, 22, 0) not in filtered

    def test_unknown_no_filtering(self):
        """Test unknown assets are not filtered."""
        calendar = MarketCalendar()
        timestamps = [datetime(2024, 1, 6, 12, 0)]

        filtered = calendar.filter_to_trading_hours(timestamps, AssetType.UNKNOWN)

        assert len(filtered) == len(timestamps)
