"""Data provider with caching layer for market data."""

import logging
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from quantagent.data.sources.base import CandleSource, to_utc
from quantagent.data.sources.yahoo import YahooSource
from quantagent.models import MarketData

logger = logging.getLogger(__name__)


class DataProvider:
    """
    Data provider with cache-aside pattern for market data.

    Pattern:
    1. Check database for existing data
    2. Identify gaps in cached data
    3. Fetch missing data from the candle source
    4. Store fetched data in database
    5. Return complete dataset

    Benefits:
    - 10x faster backtesting (local DB vs API)
    - Fewer API calls (cached data)
    - Reproducible results (same data every run)
    - Offline capability (after first fetch)
    """

    # Pandas frequency mapping
    PANDAS_FREQ_MAPPING = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w",
        "1mo": "1mo",
    }

    def __init__(
        self,
        db_session: Session,
        source: Optional[CandleSource] = None,
        offline: bool = False,
        now: Callable[[], datetime] = datetime.utcnow,
        source_name: Optional[str] = None,
    ):
        """
        Initialize DataProvider.

        Args:
            db_session: SQLAlchemy database session
            source: CandleSource that serves the gaps (YahooSource if None)
            now: Clock returning UTC without tzinfo, also used by the scheduler.
            offline: If True, never call the source for gaps -- only ever
                return what's already cached. For deterministic, fixture-based
                backtests (e.g. the `backtest run` CLI), hitting a live API
                defeats the point (reproducibility, no network dependency) and,
                worse, the naive gap-detection below re-triggers a request for
                any pre-fixture lookback window on nearly every candle.
        """
        self.db = db_session
        self.source_name = source.name if source is not None else (source_name or "legacy")
        self.source = source if source is not None else YahooSource()
        self.offline = offline
        self.now = now

    def get_ohlc(
        self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Get OHLC data, checking cache first, then API.

        Args:
            symbol: Trading symbol (e.g., "BTC", "SPX", "CL")
            timeframe: Timeframe (e.g., "1h", "4h", "1d")
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(
            f"Fetching OHLC data for {symbol} ({timeframe}) from {start_date} to {end_date}"
        )

        # 1. Query database for existing data
        cached = (
            self.db.query(MarketData)
            .filter(
                and_(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe,
                    MarketData.source == self.source_name,
                    MarketData.timestamp >= start_date,
                    MarketData.timestamp <= end_date,
                )
            )
            .order_by(MarketData.timestamp)
            .all()
        )

        logger.debug(f"Found {len(cached)} cached records")

        # 2. Convert cached data to DataFrame
        cached_df = self._rows_to_df(cached)

        # 3. Identify gaps in cached data
        missing_ranges = self._find_gaps(cached_df, start_date, end_date, timeframe)
        if cached and not missing_ranges:
            last_cached = max(row.timestamp for row in cached)
            missing_ranges = [(last_cached, end_date)]

        # 4. Fetch missing data from API
        if missing_ranges and self.offline:
            logger.debug(
                f"Offline mode: skipping {len(missing_ranges)} gap(s) for {symbol} ({timeframe})"
            )
        elif missing_ranges:
            logger.info(
                f"Found {len(missing_ranges)} gap(s) in cached data, fetching from API"
            )

            now = self.now()
            earliest_start = self.source.earliest(timeframe, now)
            latest_end = self.source.latest(now)

            for gap_start, gap_end in missing_ranges:
                if earliest_start is not None:
                    gap_start = max(gap_start, earliest_start)
                gap_end = min(gap_end, latest_end)
                if gap_start >= gap_end:
                    logger.warning(
                        f"{symbol} ({timeframe}): gap outside what "
                        f"{self.source.name} serves; not calling the source"
                    )
                    continue

                try:
                    api_data = self.source.fetch(symbol, timeframe, gap_start, gap_end)

                    if not api_data.empty:
                        api_data = api_data.copy()
                        # DB stores UTC without tzinfo
                        ts = pd.to_datetime(api_data["timestamp"])
                        if ts.dt.tz is not None:
                            ts = to_utc(ts, timeframe).dt.tz_localize(None)
                        api_data["timestamp"] = ts
                        overlap = cached_df.merge(
                            api_data[["timestamp", "close"]], on="timestamp", suffixes=("_cached", "_new")
                        )
                        if not overlap.empty and (
                            (overlap["close_cached"] - overlap["close_new"]).abs()
                            / overlap["close_cached"].abs().clip(lower=1e-12)
                        ).max() > 1e-4:
                            self.db.query(MarketData).filter(
                                MarketData.symbol == symbol,
                                MarketData.timeframe == timeframe,
                                MarketData.source == self.source_name,
                            ).delete(synchronize_session=False)
                            self.db.commit()
                            api_data = self.source.fetch(symbol, timeframe, start_date, end_date).copy()
                            ts = pd.to_datetime(api_data["timestamp"])
                            if ts.dt.tz is not None:
                                ts = to_utc(ts, timeframe).dt.tz_localize(None)
                            api_data["timestamp"] = ts
                            cached_df = pd.DataFrame(columns=cached_df.columns)
                        duration = (
                            pd.DateOffset(months=1) if timeframe == "1mo"
                            else pd.Timedelta(weeks=1) if timeframe == "1w"
                            else pd.Timedelta(self._to_pandas_freq(timeframe))
                        )
                        api_data = api_data[ts + duration <= latest_end]

                        # Store in database
                        self._store_data(symbol, timeframe, api_data)

                        # Add to cached data
                        cached_df = pd.concat([cached_df, api_data], ignore_index=True)

                        logger.info(
                            f"Fetched and cached {len(api_data)} records for gap {gap_start} to {gap_end}"
                        )
                    else:
                        logger.warning(
                            f"No data returned from API for gap {gap_start} to {gap_end}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error fetching data for gap {gap_start} to {gap_end}: {e}"
                    )
                    # Continue with partial data rather than failing completely

        # 5. Return complete, sorted dataset
        if cached_df.empty:
            logger.warning(
                f"No data available for {symbol} ({timeframe}) from {start_date} to {end_date}"
            )
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        result = cached_df.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Returning {len(result)} total records")

        return result

    def _store_data(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """
        Store DataFrame rows in database.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            df: DataFrame with OHLCV data
        """
        records_stored = 0

        for _, row in df.iterrows():
            # Check if record already exists (avoid duplicates)
            existing = (
                self.db.query(MarketData)
                .filter(
                    and_(
                        MarketData.symbol == symbol,
                        MarketData.timeframe == timeframe,
                        MarketData.timestamp == row["timestamp"],
                        MarketData.source == self.source_name,
                    )
                )
                .first()
            )

            if existing:
                continue

            record = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                source=self.source_name,
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )

            self.db.add(record)
            records_stored += 1

        self.db.commit()
        logger.debug(f"Stored {records_stored} new records")

    def _find_gaps(
        self, df: pd.DataFrame, start_date: datetime, end_date: datetime, timeframe: str
    ) -> list[tuple[datetime, datetime]]:
        """
        Find missing date ranges in cached data.

        Args:
            df: Cached data DataFrame
            start_date: Requested start date
            end_date: Requested end date
            timeframe: Timeframe

        Returns:
            List of (gap_start, gap_end) tuples
        """
        if df.empty:
            return [(start_date, end_date)]

        gaps = []

        # Convert timestamps to datetime and ensure timezone-naive
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Remove timezone info if present to ensure consistent comparisons
        if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df = df.sort_values("timestamp")

        # Gap before first cached date
        first_cached = df["timestamp"].min()
        # Convert pandas Timestamp to Python datetime for comparison
        first_cached = (
            first_cached.to_pydatetime()
            if hasattr(first_cached, "to_pydatetime")
            else first_cached
        )
        if first_cached > start_date:
            gaps.append((start_date, first_cached))

        # Gaps between cached dates (simple approach: check if there's a significant gap)
        # For MVP, we'll just check if the entire range is covered
        # A more sophisticated approach would check for missing candles

        # Gap after last cached date
        last_cached = df["timestamp"].max()
        # Convert pandas Timestamp to Python datetime for comparison
        last_cached = (
            last_cached.to_pydatetime()
            if hasattr(last_cached, "to_pydatetime")
            else last_cached
        )
        if last_cached < end_date:
            gaps.append((last_cached, end_date))

        return gaps

    def _rows_to_df(self, rows: list) -> pd.DataFrame:
        """
        Convert SQLAlchemy rows to DataFrame.

        Args:
            rows: List of MarketData model instances

        Returns:
            DataFrame with OHLCV data
        """
        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        data = [
            {
                "timestamp": row.timestamp,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }
            for row in rows
        ]

        return pd.DataFrame(data)

    def _to_pandas_freq(self, timeframe: str) -> str:
        """Convert timeframe to pandas frequency."""
        return self.PANDAS_FREQ_MAPPING.get(timeframe, "1h")

    def get_cache_stats(self, symbol: Optional[str] = None) -> dict:
        """
        Get cache statistics.

        Args:
            symbol: Optional symbol to filter by

        Returns:
            Dictionary with cache statistics
        """
        query = self.db.query(MarketData)

        if symbol:
            query = query.filter(MarketData.symbol == symbol)

        total_records = query.count()

        if total_records == 0:
            return {"total_records": 0, "symbols": [], "earliest": None, "latest": None}

        # Get unique symbols
        symbols_query = self.db.query(MarketData.symbol.distinct())
        if symbol:
            symbols_query = symbols_query.filter(MarketData.symbol == symbol)
        symbols = [s[0] for s in symbols_query.all()]

        # Get date range
        from sqlalchemy import func

        earliest = query.with_entities(func.min(MarketData.timestamp)).scalar()
        latest = query.with_entities(func.max(MarketData.timestamp)).scalar()

        return {
            "total_records": total_records,
            "symbols": symbols,
            "earliest": earliest,
            "latest": latest,
        }

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """
        Clear cached data.

        Args:
            symbol: Optional symbol to filter by (clears all if None)

        Returns:
            Number of records deleted
        """
        query = self.db.query(MarketData)

        if symbol:
            query = query.filter(MarketData.symbol == symbol)
            logger.info(f"Clearing cache for symbol {symbol}")
        else:
            logger.info("Clearing all cached data")

        count = query.count()
        query.delete()
        self.db.commit()

        logger.info(f"Deleted {count} cached records")
        return count
