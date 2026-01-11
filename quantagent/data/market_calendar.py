"""Market calendar integration for trading hours filtering."""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import pandas_market_calendars as mcal

from quantagent.data.asset_types import AssetType

logger = logging.getLogger(__name__)


ASSET_TYPE_TO_EXCHANGE: dict[AssetType, Optional[str]] = {
    AssetType.US_EQUITY: "NYSE",
    AssetType.US_FUTURES: "CME_Equity",
    AssetType.EUROPEAN: "EUREX",
    AssetType.CRYPTO: None,
    AssetType.UNKNOWN: None,
}


class MarketCalendar:
    """
    Wrapper around pandas_market_calendars for trading hours determination.

    Provides caching and fallback behavior for market hours lookups.
    """

    def __init__(self):
        """Initialize calendar cache."""
        self._calendars: dict[str, Optional[mcal.MarketCalendar]] = {}
        self._schedule_cache: dict[tuple, pd.DataFrame] = {}

    def get_calendar(self, exchange: str) -> Optional[mcal.MarketCalendar]:
        """
        Get or create calendar for exchange.

        Args:
            exchange: Exchange name (e.g., "NYSE", "CME_Equity")

        Returns:
            Market calendar or None if not available
        """
        if exchange is None:
            return None

        if exchange not in self._calendars:
            try:
                self._calendars[exchange] = mcal.get_calendar(exchange)
            except Exception as e:
                logger.warning(f"Could not load calendar for {exchange}: {e}")
                self._calendars[exchange] = None

        return self._calendars[exchange]

    def get_schedule(
        self,
        exchange: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Get trading schedule for date range.

        Args:
            exchange: Exchange name
            start_date: Start of range
            end_date: End of range

        Returns:
            DataFrame with market_open, market_close columns or None
        """
        cache_key = (exchange, start_date.date(), end_date.date())

        if cache_key in self._schedule_cache:
            return self._schedule_cache[cache_key]

        calendar = self.get_calendar(exchange)
        if calendar is None:
            return None

        try:
            schedule = calendar.schedule(
                start_date=start_date,
                end_date=end_date,
            )
            self._schedule_cache[cache_key] = schedule
            return schedule
        except Exception as e:
            logger.warning(f"Could not get schedule for {exchange}: {e}")
            return None

    def filter_to_trading_hours(
        self,
        timestamps: list[datetime],
        asset_type: AssetType,
    ) -> list[datetime]:
        """
        Filter timestamps to only include trading hours.

        Args:
            timestamps: List of timestamps to filter
            asset_type: Asset type for calendar selection

        Returns:
            Filtered list of timestamps during trading hours
        """
        if asset_type in (AssetType.CRYPTO, AssetType.UNKNOWN):
            return timestamps

        if not timestamps:
            return []

        exchange = ASSET_TYPE_TO_EXCHANGE.get(asset_type)
        if exchange is None:
            return timestamps

        schedule = self.get_schedule(
            exchange,
            min(timestamps),
            max(timestamps),
        )

        if schedule is None or schedule.empty:
            logger.warning("No schedule available, returning unfiltered timestamps")
            return timestamps

        filtered = []
        for ts in timestamps:
            if self._timestamp_in_schedule(ts, schedule):
                filtered.append(ts)

        logger.info(
            f"Filtered {len(timestamps)} -> {len(filtered)} timestamps "
            f"({100 * len(filtered) / len(timestamps):.1f}% retained)"
        )

        return filtered

    def _timestamp_in_schedule(
        self,
        timestamp: datetime,
        schedule: pd.DataFrame,
    ) -> bool:
        """Check if timestamp falls within any trading session in schedule."""
        if timestamp.tzinfo is None:
            ts = pd.Timestamp(timestamp, tz="UTC")
        else:
            ts = pd.Timestamp(timestamp)

        for _, row in schedule.iterrows():
            market_open = row["market_open"]
            market_close = row["market_close"]

            if ts >= market_open and ts <= market_close:
                return True

        return False


_market_calendar: Optional[MarketCalendar] = None


def get_market_calendar() -> MarketCalendar:
    """Get singleton market calendar instance."""
    global _market_calendar
    if _market_calendar is None:
        _market_calendar = MarketCalendar()
    return _market_calendar
