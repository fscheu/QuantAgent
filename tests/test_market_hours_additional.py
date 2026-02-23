"""Additional meaningful tests for QuantAgent-s92 - Holiday and constraint validation."""

from datetime import datetime

from quantagent.data.asset_types import AssetType
from quantagent.data.market_calendar import MarketCalendar


class TestHolidayFiltering:
    """Test holiday filtering behavior (AC3.4)."""

    def test_july_4th_holiday_filtered(self):
        """Test July 4th is filtered for US equity."""
        calendar = MarketCalendar()

        timestamps = [
            datetime(2024, 7, 3, 15, 0),
            datetime(2024, 7, 4, 15, 0),
            datetime(2024, 7, 5, 15, 0),
        ]

        filtered = calendar.filter_to_trading_hours(timestamps, AssetType.US_EQUITY)

        assert datetime(2024, 7, 3, 15, 0) in filtered
        assert datetime(2024, 7, 5, 15, 0) in filtered
        assert datetime(2024, 7, 4, 15, 0) not in filtered

    def test_new_years_day_holiday_filtered(self):
        """Test New Year's Day is filtered for US equity."""
        calendar = MarketCalendar()

        timestamps = [
            datetime(2024, 12, 31, 15, 0),
            datetime(2025, 1, 1, 15, 0),
            datetime(2025, 1, 2, 15, 0),
        ]

        filtered = calendar.filter_to_trading_hours(timestamps, AssetType.US_EQUITY)

        assert datetime(2024, 12, 31, 15, 0) in filtered
        assert datetime(2025, 1, 2, 15, 0) in filtered
        assert datetime(2025, 1, 1, 15, 0) not in filtered

    def test_crypto_trades_on_holidays(self):
        """Test crypto trades on US holidays (24/7)."""
        calendar = MarketCalendar()

        timestamps = [
            datetime(2024, 7, 4, 15, 0),
            datetime(2025, 1, 1, 15, 0),
        ]

        filtered = calendar.filter_to_trading_hours(timestamps, AssetType.CRYPTO)

        assert len(filtered) == len(timestamps)


class TestDSTTransitions:
    """Test DST transition handling (AC6.4)."""

    def test_march_dst_spring_forward(self):
        """Test March DST transition (spring forward)."""
        calendar = MarketCalendar()

        timestamps = [
            datetime(2024, 3, 8, 15, 0),
            datetime(2024, 3, 11, 15, 0),
        ]

        filtered = calendar.filter_to_trading_hours(timestamps, AssetType.US_EQUITY)

        assert len(filtered) == 2

    def test_november_dst_fall_back(self):
        """Test November DST transition (fall back)."""
        calendar = MarketCalendar()

        timestamps = [
            datetime(2024, 11, 1, 15, 0),
            datetime(2024, 11, 4, 15, 0),
        ]

        filtered = calendar.filter_to_trading_hours(timestamps, AssetType.US_EQUITY)

        assert len(filtered) == 2


class TestScheduleStructureValidation:
    """Test schedule DataFrame structure (AC2.2 enhanced)."""

    def test_schedule_has_timezone_aware_timestamps(self):
        """Test schedule timestamps are timezone-aware."""
        calendar = MarketCalendar()
        schedule = calendar.get_schedule(
            "NYSE",
            datetime(2024, 1, 2),
            datetime(2024, 1, 5),
        )

        assert schedule is not None
        market_open = schedule.iloc[0]["market_open"]
        market_close = schedule.iloc[0]["market_close"]

        assert market_open.tzinfo is not None
        assert market_close.tzinfo is not None

    def test_schedule_market_close_after_open(self):
        """Test market close is always after market open."""
        calendar = MarketCalendar()
        schedule = calendar.get_schedule(
            "NYSE",
            datetime(2024, 1, 2),
            datetime(2024, 1, 31),
        )

        for _, row in schedule.iterrows():
            assert row["market_close"] > row["market_open"]

    def test_schedule_no_weekend_days(self):
        """Test schedule doesn't include weekends."""
        calendar = MarketCalendar()
        schedule = calendar.get_schedule(
            "NYSE",
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )

        for date in schedule.index:
            weekday = date.weekday()
            assert weekday < 5


class TestErrorConstraints:
    """Test error handling and constraints."""

    def test_empty_timestamp_list_returns_empty(self):
        """Test empty input returns empty output."""
        calendar = MarketCalendar()

        filtered = calendar.filter_to_trading_hours([], AssetType.US_EQUITY)

        assert filtered == []

    def test_none_exchange_returns_none_calendar(self):
        """Test None exchange returns None calendar."""
        calendar = MarketCalendar()

        result = calendar.get_calendar(None)

        assert result is None

    def test_invalid_exchange_name_returns_none(self):
        """Test invalid exchange name returns None gracefully."""
        calendar = MarketCalendar()

        result = calendar.get_calendar("INVALID_NONEXISTENT_EXCHANGE_XYZ")

        assert result is None

    def test_future_timestamps_handled(self):
        """Test future timestamps are handled correctly."""
        calendar = MarketCalendar()

        timestamps = [
            datetime(2026, 1, 5, 15, 0),
            datetime(2026, 1, 6, 15, 0),
        ]

        filtered = calendar.filter_to_trading_hours(timestamps, AssetType.US_EQUITY)

        assert isinstance(filtered, list)


class TestSingletonBehavior:
    """Test singleton pattern for market calendar."""

    def test_get_market_calendar_returns_same_instance(self):
        """Test get_market_calendar returns singleton."""
        from quantagent.data.market_calendar import get_market_calendar

        cal1 = get_market_calendar()
        cal2 = get_market_calendar()

        assert cal1 is cal2

    def test_singleton_maintains_cache(self):
        """Test singleton maintains cache across calls."""
        from quantagent.data.market_calendar import get_market_calendar

        cal1 = get_market_calendar()
        cal1.get_schedule("NYSE", datetime(2024, 1, 2), datetime(2024, 1, 5))
        initial_cache_size = len(cal1._schedule_cache)

        cal2 = get_market_calendar()
        final_cache_size = len(cal2._schedule_cache)

        assert initial_cache_size == final_cache_size
        assert initial_cache_size > 0
