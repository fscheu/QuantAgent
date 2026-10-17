"""YahooSource: conversión de zona horaria y ventana intradía."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

from quantagent.data.sources.yahoo import YahooSource


def _history(index):
    return pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100},
        index=pd.DatetimeIndex(index, name="Datetime"),
    )


def test_fetch_converts_new_york_to_utc():
    idx = pd.DatetimeIndex(
        ["2024-01-02 09:30", "2024-01-02 10:30"], tz="America/New_York"
    )
    with patch("quantagent.data.sources.yahoo.yf.Ticker") as ticker:
        ticker.return_value.history.return_value = _history(idx)
        df = YahooSource().fetch(
            "QQQ", "1d", datetime(2024, 1, 1), datetime(2024, 1, 3)
        )

    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-02 14:30", tz="UTC")
    assert df["timestamp"].iloc[1] == pd.Timestamp("2024-01-02 15:30", tz="UTC")
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_earliest_and_latest():
    now = datetime(2026, 10, 17, 12, 0)
    src = YahooSource()
    assert src.earliest("1h", now) == now - timedelta(days=729)
    assert src.earliest("1d", now) is None
    assert src.latest(now) == now


def test_intraday_entirely_outside_window_skips_api():
    old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1000)
    with patch("quantagent.data.sources.yahoo.yf.Ticker") as ticker:
        df = YahooSource().fetch("QQQ", "1h", old, old + timedelta(days=5))
    ticker.assert_not_called()
    assert df.empty
