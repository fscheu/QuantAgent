# Design: Backtest Market Hours Filtering

**Issue ID:** QuantAgent-s92
**Type:** DS (Design)
**Created:** 2026-01-05
**Related:** [QuantAgent-s92-RQ-backtest-market-hours.md](../01_requirements/QuantAgent-s92-RQ-backtest-market-hours.md)

---

## Overview

This document specifies the technical design for adding market hours awareness to the backtest engine, filtering out non-trading periods to improve efficiency.

---

## Current Implementation Analysis

### File: `quantagent/backtesting/backtest.py`

```python
def _get_date_range(self) -> List[datetime]:
    """
    Get list of dates to backtest.
    For hourly/intraday: every N hours
    For daily: every day
    """
    dates = []
    current = self.start_date

    # Determine step size based on timeframe
    if self.timeframe in ["1h", "4h"]:
        step_hours = int(self.timeframe.replace("h", ""))
        step = timedelta(hours=step_hours)
    elif self.timeframe == "1d":
        step = timedelta(days=1)
    elif self.timeframe == "1w":
        step = timedelta(weeks=1)
    else:
        step = timedelta(hours=1)

    while current <= self.end_date:
        dates.append(current)
        current += step

    return dates
```

**Issues:**
- No awareness of market hours
- Generates all intervals regardless of trading schedule
- Same behavior for crypto and traditional markets

### File: `quantagent/data/provider.py`

```python
SYMBOL_MAPPING = {
    "BTC": "BTC-USD",      # Crypto - 24/7
    "SPX": "^GSPC",        # US Equity Index
    "CL": "CL=F",          # Crude Oil Futures
    "DAX": "^GDAXI",       # European Index
    "ES": "ES=F",          # S&P 500 Futures
    "NQ": "NQ=F",          # NASDAQ Futures
    "QQQ": "QQQ",          # ETF
    "GC": "GC=F",          # Gold Futures
    "VIX": "^VIX",         # Volatility Index
    "DXY": "DX-Y.NYB",     # Dollar Index
}
```

**Missing:** Asset type classification for trading hours

---

## Proposed Design

### 1. Asset Type Enumeration

**New file: `quantagent/data/asset_types.py`**

```python
"""Asset type classifications for trading hours determination."""

from enum import Enum
from typing import Optional


class AssetType(Enum):
    """Classification of assets by trading schedule."""

    CRYPTO = "crypto"           # 24/7 trading
    US_EQUITY = "us_equity"     # NYSE/NASDAQ hours
    US_FUTURES = "us_futures"   # CME/NYMEX extended hours
    EUROPEAN = "european"       # European exchange hours
    UNKNOWN = "unknown"         # Default to 24/7


# Default asset type mappings based on symbol patterns
ASSET_TYPE_MAPPING: dict[str, AssetType] = {
    # Crypto
    "BTC": AssetType.CRYPTO,
    "ETH": AssetType.CRYPTO,
    "BTC-USD": AssetType.CRYPTO,
    "ETH-USD": AssetType.CRYPTO,

    # US Equity Indices
    "SPX": AssetType.US_EQUITY,
    "^GSPC": AssetType.US_EQUITY,
    "QQQ": AssetType.US_EQUITY,
    "VIX": AssetType.US_EQUITY,
    "^VIX": AssetType.US_EQUITY,

    # US Futures
    "ES": AssetType.US_FUTURES,
    "ES=F": AssetType.US_FUTURES,
    "NQ": AssetType.US_FUTURES,
    "NQ=F": AssetType.US_FUTURES,
    "CL": AssetType.US_FUTURES,
    "CL=F": AssetType.US_FUTURES,
    "GC": AssetType.US_FUTURES,
    "GC=F": AssetType.US_FUTURES,
    "DXY": AssetType.US_FUTURES,
    "DX-Y.NYB": AssetType.US_FUTURES,

    # European
    "DAX": AssetType.EUROPEAN,
    "^GDAXI": AssetType.EUROPEAN,
}


def get_asset_type(symbol: str) -> AssetType:
    """
    Get asset type for a symbol.

    Args:
        symbol: Trading symbol (e.g., "BTC", "SPX")

    Returns:
        AssetType classification
    """
    # Direct lookup
    if symbol in ASSET_TYPE_MAPPING:
        return ASSET_TYPE_MAPPING[symbol]

    # Pattern-based inference
    symbol_upper = symbol.upper()

    # Crypto patterns
    if symbol_upper.endswith("-USD") or symbol_upper in ("BTC", "ETH", "SOL", "XRP"):
        return AssetType.CRYPTO

    # Futures patterns
    if symbol_upper.endswith("=F") or symbol_upper in ("ES", "NQ", "CL", "GC", "SI", "HG"):
        return AssetType.US_FUTURES

    # Index patterns
    if symbol_upper.startswith("^"):
        return AssetType.US_EQUITY

    # Default to unknown (will use 24/7 schedule)
    return AssetType.UNKNOWN
```

### 2. Market Calendar Module

**New file: `quantagent/data/market_calendar.py`**

```python
"""Market calendar integration for trading hours filtering."""

import logging
from datetime import datetime, time
from typing import Optional
from functools import lru_cache

import pandas_market_calendars as mcal
import pandas as pd

from quantagent.data.asset_types import AssetType, get_asset_type

logger = logging.getLogger(__name__)


# Exchange mapping by asset type
ASSET_TYPE_TO_EXCHANGE: dict[AssetType, str] = {
    AssetType.US_EQUITY: "NYSE",
    AssetType.US_FUTURES: "CME_Equity",  # CME equity futures calendar
    AssetType.EUROPEAN: "EUREX",
    AssetType.CRYPTO: None,  # No calendar needed
    AssetType.UNKNOWN: None,  # No calendar needed
}


class MarketCalendar:
    """
    Wrapper around pandas_market_calendars for trading hours determination.

    Provides caching and fallback behavior for market hours lookups.
    """

    def __init__(self):
        """Initialize calendar cache."""
        self._calendars: dict[str, mcal.MarketCalendar] = {}
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

    def is_market_open(
        self,
        timestamp: datetime,
        asset_type: AssetType,
    ) -> bool:
        """
        Check if market is open at given timestamp.

        Args:
            timestamp: Timestamp to check
            asset_type: Asset type for calendar selection

        Returns:
            True if market is open or asset trades 24/7
        """
        # 24/7 assets always trade
        if asset_type in (AssetType.CRYPTO, AssetType.UNKNOWN):
            return True

        exchange = ASSET_TYPE_TO_EXCHANGE.get(asset_type)
        if exchange is None:
            return True  # Default to always open

        # Get schedule for the day
        schedule = self.get_schedule(
            exchange,
            timestamp.replace(hour=0, minute=0, second=0),
            timestamp.replace(hour=23, minute=59, second=59),
        )

        if schedule is None or schedule.empty:
            logger.debug(f"No schedule for {timestamp.date()}, assuming closed")
            return False

        # Check if timestamp falls within any session
        ts_utc = pd.Timestamp(timestamp, tz='UTC') if timestamp.tzinfo is None else pd.Timestamp(timestamp)

        for _, row in schedule.iterrows():
            market_open = row['market_open']
            market_close = row['market_close']

            if market_open <= ts_utc <= market_close:
                return True

        return False

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
        # 24/7 assets - no filtering
        if asset_type in (AssetType.CRYPTO, AssetType.UNKNOWN):
            return timestamps

        if not timestamps:
            return []

        exchange = ASSET_TYPE_TO_EXCHANGE.get(asset_type)
        if exchange is None:
            return timestamps

        # Get schedule for full range
        schedule = self.get_schedule(
            exchange,
            min(timestamps),
            max(timestamps),
        )

        if schedule is None or schedule.empty:
            logger.warning(f"No schedule available, returning unfiltered timestamps")
            return timestamps

        # Build set of valid timestamps
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
        # Convert to pandas Timestamp with UTC
        if timestamp.tzinfo is None:
            ts = pd.Timestamp(timestamp, tz='UTC')
        else:
            ts = pd.Timestamp(timestamp)

        for _, row in schedule.iterrows():
            market_open = row['market_open']
            market_close = row['market_close']

            # Include timestamps that overlap with trading hours
            # For multi-hour candles, include if any part overlaps
            if ts >= market_open and ts <= market_close:
                return True

        return False


# Singleton instance
_market_calendar: Optional[MarketCalendar] = None


def get_market_calendar() -> MarketCalendar:
    """Get singleton market calendar instance."""
    global _market_calendar
    if _market_calendar is None:
        _market_calendar = MarketCalendar()
    return _market_calendar
```

### 3. Modified Backtest Class

**File: `quantagent/backtesting/backtest.py`**

```python
# Add imports at top
from quantagent.data.asset_types import AssetType, get_asset_type
from quantagent.data.market_calendar import get_market_calendar

class Backtest:
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        assets: List[str],
        timeframe: str = "1h",
        initial_capital: float = 100000.0,
        config: Optional[Dict] = None,
        db_session: Optional[Session] = None,
        use_checkpointing: bool = False,
    ):
        # ... existing init code ...

        # Market hours filtering (new)
        self.market_hours_filter = self.config.get("market_hours_filter", True)
        self._market_calendar = get_market_calendar() if self.market_hours_filter else None

        # Cache asset types for each symbol
        self._asset_types: Dict[str, AssetType] = {
            asset: get_asset_type(asset) for asset in assets
        }

    def _get_date_range(self) -> List[datetime]:
        """
        Get list of dates to backtest.

        For hourly/intraday: every N hours (filtered by market hours if enabled)
        For daily: every day (weekdays only for non-crypto if enabled)
        """
        dates = []
        current = self.start_date

        # Determine step size based on timeframe
        if self.timeframe in ["1h", "4h"]:
            step_hours = int(self.timeframe.replace("h", ""))
            step = timedelta(hours=step_hours)
        elif self.timeframe == "1d":
            step = timedelta(days=1)
        elif self.timeframe == "1w":
            step = timedelta(weeks=1)
        else:
            step = timedelta(hours=1)

        while current <= self.end_date:
            dates.append(current)
            current += step

        return dates

    def _get_date_range_for_asset(self, asset: str) -> List[datetime]:
        """
        Get date range filtered by market hours for specific asset.

        Args:
            asset: Asset symbol

        Returns:
            List of valid trading timestamps for this asset
        """
        # Get full date range
        all_dates = self._get_date_range()

        # No filtering if disabled or no calendar
        if not self.market_hours_filter or self._market_calendar is None:
            return all_dates

        # Get asset type
        asset_type = self._asset_types.get(asset, AssetType.UNKNOWN)

        # Filter by market hours
        return self._market_calendar.filter_to_trading_hours(all_dates, asset_type)

    def run(self, name: Optional[str] = None) -> BacktestMetrics:
        """Run backtest and return metrics."""
        logger.info(f"Starting backtest: {self.start_date} to {self.end_date}")
        logger.info(f"Assets: {self.assets}, Timeframe: {self.timeframe}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"Market hours filter: {self.market_hours_filter}")

        # Create backtest run record
        self._create_backtest_run(name)

        # Process each asset with its filtered date range
        total_periods = 0
        for asset in self.assets:
            asset_dates = self._get_date_range_for_asset(asset)
            total_periods += len(asset_dates)

            asset_type = self._asset_types.get(asset, AssetType.UNKNOWN)
            logger.info(
                f"Asset {asset} ({asset_type.value}): {len(asset_dates)} analysis periods"
            )

        logger.info(f"Backtesting {total_periods} total analysis periods")

        # Track progress across all periods
        periods_completed = 0

        # Loop through assets (outer) and their dates (inner)
        for asset in self.assets:
            asset_dates = self._get_date_range_for_asset(asset)

            for i, current_date in enumerate(asset_dates):
                self.current_date = current_date

                # Reset daily P&L tracking at start of each day
                if i == 0 or current_date.date() != asset_dates[i - 1].date():
                    self.risk_manager.reset_daily_tracker()

                try:
                    self._analyze_and_trade(asset, current_date)
                except Exception as e:
                    logger.error(
                        f"Error analyzing {asset} at {current_date}: {e}",
                        exc_info=True
                    )
                    continue

                # Record equity at end of period
                self._record_equity(current_date)

                periods_completed += 1

                # Log progress
                if periods_completed % 100 == 0 or periods_completed == total_periods:
                    progress = (periods_completed / total_periods) * 100
                    logger.info(f"Progress: {progress:.1f}% ({periods_completed}/{total_periods})")

        # Calculate metrics
        metrics = self._calculate_metrics()
        self._update_backtest_run(metrics)

        logger.info(
            f"Backtest complete: {metrics.total_trades} trades, "
            f"Win rate: {metrics.win_rate:.2%}"
        )
        logger.info(
            f"Total P&L: ${metrics.total_pnl:,.2f} ({metrics.total_return_pct:.2%})"
        )

        return metrics
```

### 4. Configuration Updates

**File: `quantagent/default_config.py`**

```python
# Add to existing config:

# Market hours filtering for backtests
MARKET_HOURS_CONFIG = {
    "enabled": True,                    # Enable market hours filtering by default
    "fallback_to_24_7": True,          # Use 24/7 schedule for unknown assets
    "include_extended_hours": False,    # Exclude pre-market/after-hours for MVP
}
```

---

## Data Flow Diagram

```
                    Backtest.run()
                         |
                         v
            +------------------------+
            | For each asset:        |
            | - Get asset type       |
            | - Get filtered dates   |
            +------------------------+
                         |
         +---------------+---------------+
         |                               |
         v                               v
    +-----------+               +---------------+
    | CRYPTO    |               | US_EQUITY/    |
    | (24/7)    |               | US_FUTURES    |
    +-----------+               +---------------+
         |                               |
         v                               v
    All timestamps              MarketCalendar
    retained                    .filter_to_trading_hours()
         |                               |
         |                               v
         |                      +-------------------+
         |                      | pandas_market_    |
         |                      | calendars lookup  |
         |                      +-------------------+
         |                               |
         |                               v
         |                      Filtered timestamps
         |                      (trading hours only)
         |                               |
         +---------------+---------------+
                         |
                         v
              _analyze_and_trade()
              (for each valid timestamp)
```

---

## Module Structure

### New Files

```
quantagent/
  data/
    asset_types.py      # AssetType enum and classification
    market_calendar.py  # MarketCalendar wrapper class
```

### Modified Files

```
quantagent/
  backtesting/
    backtest.py         # Add filtering to _get_date_range()
  default_config.py     # Add MARKET_HOURS_CONFIG
requirements.txt        # Add pandas_market_calendars
```

---

## API Changes

### Backtest Constructor

No signature changes. New optional config keys:

```python
config = {
    # ... existing keys ...
    "market_hours_filter": True,  # Enable/disable filtering
}
```

### New Public Functions

```python
# quantagent/data/asset_types.py
def get_asset_type(symbol: str) -> AssetType:
    """Get asset type classification for symbol."""

# quantagent/data/market_calendar.py
def get_market_calendar() -> MarketCalendar:
    """Get singleton market calendar instance."""
```

---

## Performance Considerations

### Caching Strategy

1. **Calendar objects** - Cached per exchange (singleton pattern)
2. **Schedule DataFrames** - Cached by (exchange, start_date, end_date) tuple
3. **Asset type lookups** - Cached at Backtest init time

### Memory Impact

- Schedule cache: ~10KB per month per exchange
- Calendar objects: ~1MB total for all supported exchanges
- Negligible compared to backtest data

### Timing

| Operation | Expected Time |
|-----------|---------------|
| Calendar load (first time) | ~100ms |
| Schedule lookup (cached) | ~1ms |
| Filter 1000 timestamps | ~10ms |
| Total overhead per backtest | < 100ms |

---

## Error Handling

### Calendar Load Failure

```python
try:
    calendar = mcal.get_calendar(exchange)
except Exception as e:
    logger.warning(f"Could not load calendar for {exchange}: {e}")
    # Fallback: treat as 24/7
    return timestamps
```

### Unknown Exchange

```python
if exchange not in supported_exchanges:
    logger.warning(f"Unknown exchange {exchange}, using 24/7 schedule")
    return timestamps
```

### Schedule API Error

```python
try:
    schedule = calendar.schedule(start_date, end_date)
except Exception as e:
    logger.warning(f"Schedule lookup failed: {e}")
    return timestamps  # Fallback to no filtering
```

---

## Timezone Handling

### Design Decisions

1. **Internal processing**: All timestamps converted to UTC for comparison
2. **Calendar schedules**: Returned in UTC by pandas_market_calendars
3. **User input**: Assumed to be in local timezone (converted to UTC internally)
4. **Display**: Converted back to user timezone for logging

### Implementation

```python
# Convert user timestamp to UTC for comparison
if timestamp.tzinfo is None:
    ts_utc = pd.Timestamp(timestamp, tz='UTC')
else:
    ts_utc = pd.Timestamp(timestamp).tz_convert('UTC')

# Compare with schedule times (already in UTC)
if market_open <= ts_utc <= market_close:
    return True
```

---

## Testing Strategy

### Unit Tests

```
tests/
  test_asset_types.py
    - test_crypto_classification
    - test_equity_classification
    - test_futures_classification
    - test_unknown_defaults_to_unknown
    - test_pattern_inference

  test_market_calendar.py
    - test_calendar_load
    - test_schedule_caching
    - test_is_market_open_crypto
    - test_is_market_open_equity
    - test_filter_to_trading_hours
    - test_fallback_on_error
    - test_timezone_handling
```

### Integration Tests

```
tests/
  test_backtest_market_hours.py
    - test_backtest_with_filtering_enabled
    - test_backtest_with_filtering_disabled
    - test_backtest_mixed_assets
    - test_backtest_unknown_asset_no_filter
```

### Mock Strategy

```python
@patch('quantagent.data.market_calendar.mcal.get_calendar')
def test_filter_to_trading_hours(mock_get_calendar):
    # Create mock calendar with known schedule
    mock_cal = Mock()
    mock_cal.schedule.return_value = pd.DataFrame({
        'market_open': [pd.Timestamp('2024-01-02 14:30', tz='UTC')],
        'market_close': [pd.Timestamp('2024-01-02 21:00', tz='UTC')],
    })
    mock_get_calendar.return_value = mock_cal

    # Test filtering
    calendar = MarketCalendar()
    filtered = calendar.filter_to_trading_hours(
        timestamps=[...],
        asset_type=AssetType.US_EQUITY,
    )

    assert len(filtered) < len(timestamps)
```

---

## Backwards Compatibility

### Preserved Behavior

1. **Default on**: Filtering enabled by default (improvement, not breaking)
2. **Opt-out**: `config={'market_hours_filter': False}` restores old behavior
3. **Crypto unchanged**: BTC/ETH backtests produce identical results
4. **API unchanged**: No changes to Backtest constructor signature

### Potential Differences

1. **Fewer analysis periods**: Traditional market backtests will run fewer iterations
2. **Different timing**: Daily metrics may differ slightly due to period changes
3. **Log output**: Additional logging for filtered periods

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Calendar library changes | Medium | Pin version in requirements.txt |
| Incorrect timezone handling | High | Comprehensive timezone tests |
| Performance regression | Low | Caching at multiple levels |
| Unknown asset misclassification | Medium | Fallback to 24/7 (conservative) |
| DST transition bugs | Medium | Use UTC internally, test DST dates |

---

## Dependencies

### New External

```
pandas_market_calendars>=4.0.0,<5.0.0
```

### Transitive (via pandas_market_calendars)

- pandas (already present)
- pytz (already present)
- exchange_calendars (optional, not required)

---

## References

- [Requirements Document](../01_requirements/QuantAgent-s92-RQ-backtest-market-hours.md)
- [pandas_market_calendars Documentation](https://pandas-market-calendars.readthedocs.io/)
- [Existing backtesting engine docs](./backtesting_engine.md)
- Current code: `quantagent/backtesting/backtest.py`
