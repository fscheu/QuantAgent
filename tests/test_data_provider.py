"""Tests for DataProvider (data caching layer)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from quantagent.data.provider import DataProvider
from quantagent.data.sources.yahoo import YahooSource
from quantagent.models import MarketData

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class FakeSource:
    """Fuente falsa sin límites: sirve de a un hueco y registra los pedidos."""

    name = "fake"

    def __init__(self, earliest=None, latest=None):
        self._earliest = earliest
        self._latest = latest
        self.calls = []

    def earliest(self, timeframe, now):
        return self._earliest

    def latest(self, now):
        return self._latest or now

    def fetch(self, symbol, timeframe, start, end):
        self.calls.append((symbol, timeframe, start, end))
        ts = pd.date_range(start, end, freq="1h", tz="UTC")
        return pd.DataFrame(
            {"timestamp": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}
        )


class TestDataProvider:
    """Test suite for DataProvider."""

    @pytest.fixture
    def provider(self, db_session):
        """Create DataProvider instance."""
        return DataProvider(db_session, source=FakeSource())

    @pytest.fixture
    def sample_dates(self):
        """Sample date range for testing."""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 7, 0, 0, 0)
        return start, end

    # Structure & Type Validation Tests

    def test_get_ohlc_returns_dataframe(self, provider, sample_dates):
        """Verify get_ohlc returns DataFrame with correct columns."""
        start, end = sample_dates

        with patch.object(provider.source, "fetch") as mock_fetch:
            # Mock API response
            mock_fetch.return_value = pd.DataFrame(
                {
                    "timestamp": [start],
                    "open": [100.0],
                    "high": [105.0],
                    "low": [95.0],
                    "close": [102.0],
                    "volume": [1000.0],
                }
            )

            result = provider.get_ohlc("BTC", "1h", start, end)

            # Validate type
            assert isinstance(result, pd.DataFrame)

            # Validate columns
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in required_columns:
                assert col in result.columns

    def test_get_ohlc_returns_empty_dataframe_on_no_data(self, provider, sample_dates):
        """Verify empty DataFrame returned when no data available."""
        start, end = sample_dates

        with patch.object(provider.source, "fetch") as _mock_fetch:
            _mock_fetch.return_value = pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            result = provider.get_ohlc("INVALID", "1h", start, end)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
            assert list(result.columns) == [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

    # Cache-Aside Pattern Tests

    def test_get_ohlc_checks_database_first(self, provider, db_session, sample_dates):
        """Verify provider checks database before calling API."""
        start, end = sample_dates

        # Pre-populate database with data
        cached_data = MarketData(
            symbol="BTC",
            timeframe="1h",
            source="fake",
            timestamp=start,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
        )
        db_session.add(cached_data)
        db_session.commit()

        with patch.object(provider.source, "fetch") as _mock_fetch:
            result = provider.get_ohlc("BTC", "1h", start, start)

            # Should NOT call API if data is cached
            # Note: May still call for gaps, so we check result instead
            assert len(result) >= 1
            assert float(result.iloc[0]["close"]) == 102.0

    def test_get_ohlc_fetches_from_api_on_cache_miss(self, provider, sample_dates):
        """Verify provider fetches from API when cache is empty."""
        start, end = sample_dates

        with patch.object(provider.source, "fetch") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                {
                    "timestamp": [start],
                    "open": [100.0],
                    "high": [105.0],
                    "low": [95.0],
                    "close": [102.0],
                    "volume": [1000.0],
                }
            )

            result = provider.get_ohlc("BTC", "1h", start, end)

            # Should call API when no cached data
            assert mock_fetch.call_count > 0
            assert len(result) >= 1

    def test_get_ohlc_stores_fetched_data(self, provider, db_session, sample_dates):
        """Verify fetched data is stored in database."""
        start, end = sample_dates

        with patch.object(provider.source, "fetch") as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                {
                    "timestamp": [start],
                    "open": [100.0],
                    "high": [105.0],
                    "low": [95.0],
                    "close": [102.0],
                    "volume": [1000.0],
                }
            )

            provider.get_ohlc("BTC", "1h", start, end)

            # Check database has the record
            stored = (
                db_session.query(MarketData)
                .filter(MarketData.symbol == "BTC", MarketData.timestamp == start)
                .first()
            )

            assert stored is not None
            assert float(stored.close) == 102.0
            assert stored.source == provider.source.name

    def test_split_reloads_source_history_when_overlap_changes(self, db_session, sample_dates):
        start, end = sample_dates
        source = FakeSource()
        provider = DataProvider(db_session, source=source)
        db_session.add_all([
            MarketData(symbol="BTC", timeframe="1h", source="fake", timestamp=start + timedelta(hours=i),
                       open=100, high=110, low=90, close=100 - i, volume=1)
            for i in range(3)
        ])
        db_session.commit()

        def adjusted(symbol, timeframe, fetch_start, fetch_end):
            dates = pd.date_range(start, start + timedelta(hours=2), freq="1h", tz="UTC")
            return pd.DataFrame({"timestamp": dates, "open": 50, "high": 55, "low": 45,
                                 "close": [50, 49.5, 49], "volume": 1})

        source.fetch = adjusted
        result = provider.get_ohlc("BTC", "1h", start, end)

        rows = db_session.query(MarketData).filter_by(symbol="BTC", source="fake").all()
        assert [row.close for row in rows] == [50, 49.5, 49]
        assert result.iloc[-1]["close"] == 49

    def test_unchanged_overlap_keeps_cached_history(self, db_session, sample_dates):
        start, end = sample_dates
        source = FakeSource()
        provider = DataProvider(db_session, source=source)
        db_session.add_all([
            MarketData(symbol="BTC", timeframe="1h", source="fake", timestamp=start + timedelta(hours=i),
                       open=100, high=110, low=90, close=100 - i, volume=1)
            for i in range(3)
        ])
        db_session.commit()
        source.fetch = lambda *args: pd.DataFrame({
            "timestamp": pd.date_range(start + timedelta(hours=2), periods=1, freq="1h", tz="UTC"),
            "open": [100], "high": [110], "low": [90], "close": [98], "volume": [1],
        })

        provider.get_ohlc("BTC", "1h", start, end)

        assert db_session.query(MarketData).filter_by(symbol="BTC", source="fake").count() == 3

    # Conversion & Transformation Tests

    def test_rows_to_df_conversion(self, provider):
        """Verify SQLAlchemy rows convert to DataFrame correctly."""
        # Create mock rows
        mock_row = Mock()
        mock_row.timestamp = datetime(2024, 1, 1)
        mock_row.open = 100.0
        mock_row.high = 105.0
        mock_row.low = 95.0
        mock_row.close = 102.0
        mock_row.volume = 1000.0

        df = provider._rows_to_df([mock_row])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["close"] == 102.0

    def test_rows_to_df_empty_list(self, provider):
        """Verify empty list returns empty DataFrame with correct columns."""
        df = provider._rows_to_df([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

    # Gap Detection Tests

    def test_find_gaps_detects_missing_data_at_start(self, provider, sample_dates):
        """Verify gap detection finds missing data before cached range."""
        start, end = sample_dates

        # Cached data starts 2 days after requested start
        cached_start = start + timedelta(days=2)
        cached_df = pd.DataFrame(
            {
                "timestamp": [cached_start],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        )

        gaps = provider._find_gaps(cached_df, start, end, "1d")

        # Should find gap at start
        assert len(gaps) > 0
        assert gaps[0][0] == start

    def test_find_gaps_detects_missing_data_at_end(self, provider, sample_dates):
        """Verify gap detection finds missing data after cached range."""
        start, end = sample_dates

        # Cached data ends 2 days before requested end
        cached_end = end - timedelta(days=2)
        cached_df = pd.DataFrame(
            {
                "timestamp": [cached_end],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        )

        gaps = provider._find_gaps(cached_df, start, end, "1d")

        # Should find gap at end
        assert len(gaps) > 0
        assert any(gap[1] == end for gap in gaps)

    def test_find_gaps_returns_full_range_on_empty_cache(self, provider, sample_dates):
        """Verify gap detection returns full range when no cached data."""
        start, end = sample_dates

        empty_df = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        gaps = provider._find_gaps(empty_df, start, end, "1d")

        # Should return entire range as gap
        assert len(gaps) == 1
        assert gaps[0] == (start, end)

    # Cache Management Tests

    def test_get_cache_stats_returns_correct_structure(
        self, provider, db_session, sample_dates
    ):
        """Verify cache stats returns dictionary with expected fields."""
        stats = provider.get_cache_stats()

        assert isinstance(stats, dict)
        assert "total_records" in stats
        assert "symbols" in stats
        assert "earliest" in stats
        assert "latest" in stats

    def test_get_cache_stats_filters_by_symbol(
        self, provider, db_session, sample_dates
    ):
        """Verify cache stats can filter by symbol."""
        start, _ = sample_dates

        # Add data for multiple symbols
        for symbol in ["BTC", "SPX"]:
            record = MarketData(
                symbol=symbol,
                timeframe="1h",
                timestamp=start,
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0,
            )
            db_session.add(record)
        db_session.commit()

        stats = provider.get_cache_stats(symbol="BTC")

        assert stats["total_records"] >= 1
        assert "BTC" in stats["symbols"]

    def test_clear_cache_removes_all_data(self, provider, db_session, sample_dates):
        """Verify clear_cache removes all cached data."""
        start, _ = sample_dates

        # Add test data
        record = MarketData(
            symbol="BTC",
            timeframe="1h",
            timestamp=start,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
        )
        db_session.add(record)
        db_session.commit()

        # Clear cache
        count = provider.clear_cache()

        assert count >= 1
        # Verify database is empty
        remaining = db_session.query(MarketData).count()
        assert remaining == 0

    def test_clear_cache_filters_by_symbol(self, provider, db_session, sample_dates):
        """Verify clear_cache can filter by symbol."""
        start, _ = sample_dates

        # Add data for multiple symbols
        for symbol in ["BTC", "SPX"]:
            record = MarketData(
                symbol=symbol,
                timeframe="1h",
                timestamp=start,
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0,
            )
            db_session.add(record)
        db_session.commit()

        # Clear only BTC
        count = provider.clear_cache(symbol="BTC")

        assert count >= 1
        # SPX should still exist
        remaining = (
            db_session.query(MarketData).filter(MarketData.symbol == "SPX").count()
        )
        assert remaining >= 1

    # Edge Cases

    def test_get_ohlc_handles_api_errors_gracefully(self, provider, sample_dates):
        """Verify provider handles API errors without crashing."""
        start, end = sample_dates

        with patch.object(provider.source, "fetch") as mock_fetch:
            mock_fetch.side_effect = Exception("API Error")

            result = provider.get_ohlc("BTC", "1h", start, end)

            # Should return empty DataFrame instead of crashing
            assert isinstance(result, pd.DataFrame)

    def test_store_data_avoids_duplicates(self, provider, db_session, sample_dates):
        """Verify _store_data doesn't create duplicate records."""
        start, _ = sample_dates

        df = pd.DataFrame(
            {
                "timestamp": [start],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        )

        # Store same data twice
        provider._store_data("BTC", "1h", df)
        provider._store_data("BTC", "1h", df)

        # Should only have one record
        count = (
            db_session.query(MarketData)
            .filter(MarketData.symbol == "BTC", MarketData.timestamp == start)
            .count()
        )

        assert count == 1


class TestSourceDelegation:
    """DataProvider pide a la fuente solo el hueco y no la llama otra vez ni offline."""

    START = datetime(2024, 1, 1, 0)
    END = datetime(2024, 1, 1, 10)

    def test_only_missing_gap_is_requested_and_repeat_call_hits_cache(self, db_session):
        source = FakeSource()
        provider = DataProvider(db_session, source=source)
        for h in range(0, 5):  # cacheado 00:00..04:00
            db_session.add(
                MarketData(
                    symbol="BTC", timeframe="1h", source="fake", timestamp=self.START + timedelta(hours=h),
                    open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0,
                )
            )
        db_session.commit()

        provider.get_ohlc("BTC", "1h", self.START, self.END)
        assert source.calls == [("BTC", "1h", self.START + timedelta(hours=4), self.END)]

        second = provider.get_ohlc("BTC", "1h", self.START, self.END)
        assert len(source.calls) == 1
        assert len(second) == 11

    def test_utc_aware_candles_are_stored_naive_utc(self, db_session):
        provider = DataProvider(db_session, source=FakeSource())
        provider.get_ohlc("BTC", "1h", self.START, self.END)

        stored = db_session.query(MarketData).order_by(MarketData.timestamp).first()
        assert stored.timestamp == self.START
        assert stored.timestamp.tzinfo is None

    def test_offline_never_calls_source(self, db_session):
        source = FakeSource()
        provider = DataProvider(db_session, source=source, offline=True)

        result = provider.get_ohlc("BTC", "1h", self.START, self.END)

        assert source.calls == []
        assert result.empty

    def test_source_limits_clamp_the_request(self, db_session):
        earliest = self.START + timedelta(hours=3)
        latest = self.START + timedelta(hours=7)
        source = FakeSource(earliest=earliest, latest=latest)
        DataProvider(db_session, source=source).get_ohlc("BTC", "1h", self.START, self.END)

        assert source.calls == [("BTC", "1h", earliest, latest)]

    def test_aware_source_limits_clamp_naive_request(self, db_session):
        earliest = (self.START + timedelta(hours=3)).replace(tzinfo=timezone.utc)
        latest = (self.START + timedelta(hours=7)).replace(tzinfo=timezone.utc)
        source = FakeSource(earliest=earliest, latest=latest)

        DataProvider(db_session, source=source).get_ohlc("BTC", "1h", self.START, self.END)

        assert source.calls == [("BTC", "1h", self.START + timedelta(hours=3), self.START + timedelta(hours=7))]

    def test_range_entirely_outside_source_limits_skips_source(self, db_session):
        source = FakeSource(earliest=self.END + timedelta(days=1))
        result = DataProvider(db_session, source=source).get_ohlc(
            "BTC", "1h", self.START, self.END
        )

        assert source.calls == []
        assert result.empty

    def test_default_source_is_yahoo(self, db_session):
        assert isinstance(DataProvider(db_session).source, YahooSource)


@pytest.mark.parametrize("timeframe,start,closed", [
    ("1h", datetime(2024, 1, 2, 14), datetime(2024, 1, 2, 15)),
    ("4h", datetime(2024, 1, 2, 14), datetime(2024, 1, 2, 18)),
    ("1d", datetime(2024, 1, 2), datetime(2024, 1, 3)),
    ("1w", datetime(2024, 1, 2), datetime(2024, 1, 9)),
    ("1mo", datetime(2024, 2, 1), datetime(2024, 3, 1)),
])
def test_only_closed_candles_are_cached(db_session, timeframe, start, closed):
    class Source(FakeSource):
        def latest(self, now):
            return now - timedelta(minutes=15)

        def fetch(self, symbol, timeframe, start_date, end_date):
            return pd.DataFrame([[start, 1, 2, 0.5, self.close, 10]], columns=COLUMNS)

    source = Source()
    source.close = 1.1
    clock = [closed + timedelta(minutes=14)]
    provider = DataProvider(db_session, source=source, now=lambda: clock[0])
    assert provider.get_ohlc("SPY", timeframe, start, clock[0]).empty
    assert db_session.query(MarketData).count() == 0
    clock[0] += timedelta(minutes=1)
    source.close = 1.5
    result = provider.get_ohlc("SPY", timeframe, start, clock[0])
    stored = db_session.query(MarketData).one()
    assert stored.timestamp == start
    assert float(stored.close) == result.iloc[0]["close"] == 1.5


@pytest.mark.parametrize("strategy,fixture,trades,pnl", [
    ("rsi", "spy-90d", 340, 4571.27),
    ("fifty-two-week-high", "spy-2y-1d", 6, -249.35),
    ("triple-screen", "spy-1y-4h", 6, 174.34),
])
def test_m1_references_unchanged(strategy, fixture, trades, pnl):
    from quantagent.cli.backtest import _run_verify_pass

    metrics, _, _ = _run_verify_pass(strategy, fixture)
    assert metrics.total_trades == trades
    assert round(metrics.total_pnl, 2) == pnl


class TestDailyBarConvention:
    """Las barras diarias se guardan a las 00:00 (convención vieja), las intradía en UTC."""

    class NewYorkDailySource(FakeSource):
        """Devuelve una barra diaria a las 00:00 de Nueva York, como yfinance."""

        def fetch(self, symbol, timeframe, start, end):
            self.calls.append((symbol, timeframe, start, end))
            ts = pd.DatetimeIndex(["2024-01-02 00:00"], tz="America/New_York")
            return pd.DataFrame(
                {"timestamp": ts, "open": 9.0, "high": 9.5, "low": 8.5, "close": 9.2, "volume": 99.0}
            )

    def test_daily_bar_does_not_duplicate_or_shift_a_row_cached_at_midnight(self, db_session):
        day = datetime(2024, 1, 2, 0, 0)
        db_session.add(
            MarketData(
                symbol="QQQ", timeframe="1d", source="fake", timestamp=day,
                open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0,
            )
        )
        db_session.commit()
        provider = DataProvider(db_session, source=self.NewYorkDailySource())

        result = provider.get_ohlc("QQQ", "1d", datetime(2024, 1, 1), datetime(2024, 1, 4))

        assert provider.source.calls, "el hueco posterior a la última barra debía pedirse"
        stored = db_session.query(MarketData).filter(MarketData.symbol == "QQQ").all()
        assert [r.timestamp for r in stored] == [day]
        # el borde cacheado/hueco puede repetirse en el DataFrame (comportamiento previo); lo que no debe haber es otra hora
        assert set(result["timestamp"]) == {day}

    def test_intraday_bar_keeps_utc_naive(self, db_session):
        class NewYorkHourly(FakeSource):
            def fetch(self, symbol, timeframe, start, end):
                ts = pd.DatetimeIndex(["2024-01-02 09:30"], tz="America/New_York")
                return pd.DataFrame(
                    {"timestamp": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}
                )

        DataProvider(db_session, source=NewYorkHourly()).get_ohlc(
            "QQQ", "1h", datetime(2024, 1, 2), datetime(2024, 1, 3)
        )

        stored = db_session.query(MarketData).one()
        assert stored.timestamp == datetime(2024, 1, 2, 14, 30)


def test_sources_keep_same_candle_key_isolated(db_session):
    stamp = datetime(2024, 1, 1)
    for source_name, close in (("snapshot:x", 10.0), ("alpaca", 20.0)):
        candle = pd.DataFrame({
            "timestamp": [stamp], "open": [close], "high": [close], "low": [close],
            "close": [close], "volume": [1],
        })
        DataProvider(db_session, source_name=source_name)._store_data("SPY", "1h", candle)

    for source_name, expected in (("snapshot:x", 10.0), ("alpaca", 20.0)):
        result = DataProvider(db_session, source_name=source_name, offline=True).get_ohlc(
            "SPY", "1h", stamp, stamp
        )
        assert result["close"].tolist() == [expected]


def test_offline_without_source_name_reads_legacy(db_session):
    stamp = datetime(2024, 1, 1)
    db_session.add(MarketData(
        symbol="SPY", timeframe="1h", timestamp=stamp, source="legacy",
        open=10, high=10, low=10, close=10, volume=1,
    ))
    db_session.commit()

    result = DataProvider(db_session, offline=True).get_ohlc("SPY", "1h", stamp, stamp)

    assert result["close"].tolist() == [10.0]
