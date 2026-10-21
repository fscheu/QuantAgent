import importlib
from datetime import datetime

import pytest

from quantagent import settings
from quantagent.backtesting.backtest import Backtest
from quantagent.data.sources import build_source
from quantagent.data.sources.alpaca import AlpacaSource
from quantagent.data.sources.yahoo import YahooSource
from quantagent.models import MarketData
from quantagent.strategy.rsi_strategy import RSIMeanReversionStrategy


def test_default_without_variable(monkeypatch):
    with monkeypatch.context() as patch:
        patch.delenv("QUANTAGENT_DATA_SOURCE", raising=False)
        patch.setattr("dotenv.load_dotenv", lambda **kwargs: None)
        importlib.reload(settings)
        assert isinstance(build_source(), YahooSource)
    importlib.reload(settings)


@pytest.mark.parametrize(
    "name,expected", [("yahoo", YahooSource), ("alpaca", AlpacaSource)]
)
def test_configured_source(monkeypatch, name, expected):
    monkeypatch.setattr(settings, "QUANTAGENT_DATA_SOURCE", name)
    assert isinstance(build_source(), expected)
    assert isinstance(build_source(name), expected)


def test_invalid_source(monkeypatch):
    monkeypatch.setattr(settings, "QUANTAGENT_DATA_SOURCE", "typo")
    with pytest.raises(ValueError, match="typo.*yahoo, alpaca"):
        build_source()


def test_backtest_stays_offline(db_session, monkeypatch):
    monkeypatch.setattr(settings, "QUANTAGENT_DATA_SOURCE", "invalid-for-paper")

    def forbidden(*args, **kwargs):
        pytest.fail("Offline backtest called the candle source")

    for method in ("fetch", "earliest", "latest"):
        monkeypatch.setattr(YahooSource, method, forbidden)
    start, end = datetime(2024, 1, 2), datetime(2024, 1, 3)
    backtest = Backtest(
        start,
        end,
        ["SPY"],
        config={"offline_data": True},
        db_session=db_session,
        strategy=RSIMeanReversionStrategy(),
    )
    assert backtest.data_provider.offline is True
    assert backtest.data_provider.get_ohlc("SPY", "1d", start, end).empty


def test_offline_backtest_queries_only_configured_source(db_session):
    for source, day, close in (
        ("snapshot:x", datetime(2024, 1, 2), 10),
        ("alpaca", datetime(2024, 1, 3), 20),
        ("legacy", datetime(2024, 1, 4), 30),
    ):
        db_session.add(MarketData(
            symbol="SPY", timeframe="1d", timestamp=day, source=source,
            open=close, high=close, low=close, close=close, volume=1,
        ))
    db_session.commit()

    backtest = Backtest(
        datetime(2024, 1, 1), datetime(2024, 1, 5), ["SPY"], timeframe="1d",
        config={"offline_data": True, "market_hours_filter": False, "source_name": "snapshot:x"},
        db_session=db_session, strategy=RSIMeanReversionStrategy(),
    )

    assert backtest.data_provider.source_name == "snapshot:x"
    assert backtest._get_date_range_for_asset("SPY") == [datetime(2024, 1, 2)]
    assert backtest._get_history_df("SPY", datetime(2024, 1, 5), 10)["close"].tolist() == [10.0]

    legacy = Backtest(
        datetime(2024, 1, 1), datetime(2024, 1, 5), ["SPY"], timeframe="1d",
        config={"offline_data": True, "market_hours_filter": False},
        db_session=db_session, strategy=RSIMeanReversionStrategy(),
    )
    assert legacy._get_date_range_for_asset("SPY") == [datetime(2024, 1, 4)]
    assert legacy._get_history_df("SPY", datetime(2024, 1, 5), 10)["close"].tolist() == [30.0]
