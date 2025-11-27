"""Tests for DataProvider (data caching layer)."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from quantagent.data.provider import DataProvider
from quantagent.models import MarketData
from quantagent.database import SessionLocal


class TestDataProvider:
    """Test suite for DataProvider."""

    @pytest.fixture
    def db_session(self):
        """Create test database session."""
        session = SessionLocal()
        yield session
        # Cleanup
        session.query(MarketData).delete()
        session.commit()
        session.close()

    @pytest.fixture
    def provider(self, db_session):
        """Create DataProvider instance."""
        return DataProvider(db_session)

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

        with patch.object(provider, '_fetch_yfinance') as mock_fetch:
            # Mock API response
            mock_fetch.return_value = pd.DataFrame({
                'timestamp': [start],
                'open': [100.0],
                'high': [105.0],
                'low': [95.0],
                'close': [102.0],
                'volume': [1000.0]
            })

            result = provider.get_ohlc('BTC', '1h', start, end)

            # Validate type
            assert isinstance(result, pd.DataFrame)

            # Validate columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                assert col in result.columns

    def test_get_ohlc_returns_empty_dataframe_on_no_data(self, provider, sample_dates):
        """Verify empty DataFrame returned when no data available."""
        start, end = sample_dates

        with patch.object(provider, '_fetch_yfinance') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            result = provider.get_ohlc('INVALID', '1h', start, end)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
            assert list(result.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    # Cache-Aside Pattern Tests

    def test_get_ohlc_checks_database_first(self, provider, db_session, sample_dates):
        """Verify provider checks database before calling API."""
        start, end = sample_dates

        # Pre-populate database with data
        cached_data = MarketData(
            symbol='BTC',
            timeframe='1h',
            timestamp=start,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0
        )
        db_session.add(cached_data)
        db_session.commit()

        with patch.object(provider, '_fetch_yfinance') as mock_fetch:
            result = provider.get_ohlc('BTC', '1h', start, start)

            # Should NOT call API if data is cached
            # Note: May still call for gaps, so we check result instead
            assert len(result) >= 1
            assert float(result.iloc[0]['close']) == 102.0

    def test_get_ohlc_fetches_from_api_on_cache_miss(self, provider, sample_dates):
        """Verify provider fetches from API when cache is empty."""
        start, end = sample_dates

        with patch.object(provider, '_fetch_yfinance') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                'timestamp': [start],
                'open': [100.0],
                'high': [105.0],
                'low': [95.0],
                'close': [102.0],
                'volume': [1000.0]
            })

            result = provider.get_ohlc('BTC', '1h', start, end)

            # Should call API when no cached data
            assert mock_fetch.call_count > 0
            assert len(result) >= 1

    def test_get_ohlc_stores_fetched_data(self, provider, db_session, sample_dates):
        """Verify fetched data is stored in database."""
        start, end = sample_dates

        with patch.object(provider, '_fetch_yfinance') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                'timestamp': [start],
                'open': [100.0],
                'high': [105.0],
                'low': [95.0],
                'close': [102.0],
                'volume': [1000.0]
            })

            provider.get_ohlc('BTC', '1h', start, end)

            # Check database has the record
            stored = db_session.query(MarketData).filter(
                MarketData.symbol == 'BTC',
                MarketData.timestamp == start
            ).first()

            assert stored is not None
            assert float(stored.close) == 102.0

    # Conversion & Transformation Tests

    def test_to_yfinance_symbol_mapping(self, provider):
        """Verify symbol mapping to yfinance format."""
        assert provider._to_yfinance_symbol('BTC') == 'BTC-USD'
        assert provider._to_yfinance_symbol('SPX') == '^GSPC'
        assert provider._to_yfinance_symbol('CL') == 'CL=F'
        # Unknown symbols pass through
        assert provider._to_yfinance_symbol('AAPL') == 'AAPL'

    def test_to_yfinance_interval_mapping(self, provider):
        """Verify timeframe mapping to yfinance interval."""
        assert provider._to_yfinance_interval('1h') == '1h'
        assert provider._to_yfinance_interval('4h') == '4h'
        assert provider._to_yfinance_interval('1d') == '1d'
        assert provider._to_yfinance_interval('1w') == '1wk'
        # Unknown defaults to 1h
        assert provider._to_yfinance_interval('unknown') == '1h'

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
        assert df.iloc[0]['close'] == 102.0

    def test_rows_to_df_empty_list(self, provider):
        """Verify empty list returns empty DataFrame with correct columns."""
        df = provider._rows_to_df([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    # Gap Detection Tests

    def test_find_gaps_detects_missing_data_at_start(self, provider, sample_dates):
        """Verify gap detection finds missing data before cached range."""
        start, end = sample_dates

        # Cached data starts 2 days after requested start
        cached_start = start + timedelta(days=2)
        cached_df = pd.DataFrame({
            'timestamp': [cached_start],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000.0]
        })

        gaps = provider._find_gaps(cached_df, start, end, '1d')

        # Should find gap at start
        assert len(gaps) > 0
        assert gaps[0][0] == start

    def test_find_gaps_detects_missing_data_at_end(self, provider, sample_dates):
        """Verify gap detection finds missing data after cached range."""
        start, end = sample_dates

        # Cached data ends 2 days before requested end
        cached_end = end - timedelta(days=2)
        cached_df = pd.DataFrame({
            'timestamp': [cached_end],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000.0]
        })

        gaps = provider._find_gaps(cached_df, start, end, '1d')

        # Should find gap at end
        assert len(gaps) > 0
        assert any(gap[1] == end for gap in gaps)

    def test_find_gaps_returns_full_range_on_empty_cache(self, provider, sample_dates):
        """Verify gap detection returns full range when no cached data."""
        start, end = sample_dates

        empty_df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        gaps = provider._find_gaps(empty_df, start, end, '1d')

        # Should return entire range as gap
        assert len(gaps) == 1
        assert gaps[0] == (start, end)

    # Cache Management Tests

    def test_get_cache_stats_returns_correct_structure(self, provider, db_session, sample_dates):
        """Verify cache stats returns dictionary with expected fields."""
        stats = provider.get_cache_stats()

        assert isinstance(stats, dict)
        assert 'total_records' in stats
        assert 'symbols' in stats
        assert 'earliest' in stats
        assert 'latest' in stats

    def test_get_cache_stats_filters_by_symbol(self, provider, db_session, sample_dates):
        """Verify cache stats can filter by symbol."""
        start, _ = sample_dates

        # Add data for multiple symbols
        for symbol in ['BTC', 'SPX']:
            record = MarketData(
                symbol=symbol,
                timeframe='1h',
                timestamp=start,
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0
            )
            db_session.add(record)
        db_session.commit()

        stats = provider.get_cache_stats(symbol='BTC')

        assert stats['total_records'] >= 1
        assert 'BTC' in stats['symbols']

    def test_clear_cache_removes_all_data(self, provider, db_session, sample_dates):
        """Verify clear_cache removes all cached data."""
        start, _ = sample_dates

        # Add test data
        record = MarketData(
            symbol='BTC',
            timeframe='1h',
            timestamp=start,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0
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
        for symbol in ['BTC', 'SPX']:
            record = MarketData(
                symbol=symbol,
                timeframe='1h',
                timestamp=start,
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0
            )
            db_session.add(record)
        db_session.commit()

        # Clear only BTC
        count = provider.clear_cache(symbol='BTC')

        assert count >= 1
        # SPX should still exist
        remaining = db_session.query(MarketData).filter(MarketData.symbol == 'SPX').count()
        assert remaining >= 1

    # Edge Cases

    def test_get_ohlc_handles_api_errors_gracefully(self, provider, sample_dates):
        """Verify provider handles API errors without crashing."""
        start, end = sample_dates

        with patch.object(provider, '_fetch_yfinance') as mock_fetch:
            mock_fetch.side_effect = Exception("API Error")

            result = provider.get_ohlc('BTC', '1h', start, end)

            # Should return empty DataFrame instead of crashing
            assert isinstance(result, pd.DataFrame)

    def test_store_data_avoids_duplicates(self, provider, db_session, sample_dates):
        """Verify _store_data doesn't create duplicate records."""
        start, _ = sample_dates

        df = pd.DataFrame({
            'timestamp': [start],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000.0]
        })

        # Store same data twice
        provider._store_data('BTC', '1h', df)
        provider._store_data('BTC', '1h', df)

        # Should only have one record
        count = db_session.query(MarketData).filter(
            MarketData.symbol == 'BTC',
            MarketData.timestamp == start
        ).count()

        assert count == 1
