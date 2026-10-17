"""Fuente de velas Yahoo Finance (yfinance)."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from quantagent.data.sources.base import CANDLE_COLUMNS

logger = logging.getLogger(__name__)


class YahooSource:
    """CandleSource sobre yfinance."""

    name = "yahoo"

    SYMBOL_MAPPING = {
        "BTC": "BTC-USD",
        "SPX": "SPY",  # SPY ETF used as proxy for S&P 500 (^GSPC doesn't support intraday data)
        "CL": "CL=F",
        "DAX": "^GDAXI",
        "ES": "ES=F",
        "NQ": "NQ=F",
        "QQQ": "QQQ",
        "GC": "GC=F",
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
    }

    TIMEFRAME_MAPPING = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1wk",
        "1mo": "1mo",
    }

    # Yahoo rejects intraday requests whose start is older than 730 days
    # ("The requested range must be within the last 730 days") and returns an
    # empty frame. One day of margin keeps a clamped start inside the window.
    INTRADAY_WINDOW_DAYS = {"1h": 729, "4h": 729}

    def earliest(self, timeframe: str, now: datetime) -> Optional[datetime]:
        window_days = self.INTRADAY_WINDOW_DAYS.get(timeframe)
        if window_days is None:
            return None
        return now - timedelta(days=window_days)

    def latest(self, now: datetime) -> datetime:
        return now

    def fetch(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        yf_symbol = self.SYMBOL_MAPPING.get(symbol, symbol)
        yf_interval = self.TIMEFRAME_MAPPING.get(timeframe, "1h")

        earliest = self.earliest(timeframe, datetime.now())
        if earliest is not None:
            if end <= earliest:
                logger.warning(
                    f"{symbol} ({timeframe}): requested range {start} to {end} "
                    f"is entirely older than Yahoo's 730-day intraday window "
                    f"(earliest available: {earliest}); not calling the API"
                )
                return pd.DataFrame(columns=CANDLE_COLUMNS)
            if start < earliest:
                logger.warning(
                    f"{symbol} ({timeframe}): requested start {start} is older than "
                    f"Yahoo's 730-day intraday window; truncating start to {earliest}"
                )
                start = earliest

        logger.debug(f"Fetching {yf_symbol} with interval {yf_interval}")

        df = yf.Ticker(yf_symbol).history(
            start=start,
            end=end,
            interval=yf_interval,
            auto_adjust=True,
            actions=False,
        )
        if df.empty:
            return pd.DataFrame(columns=CANDLE_COLUMNS)

        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "timestamp"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "timestamp"})
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df[CANDLE_COLUMNS].copy()

        ts = pd.to_datetime(df["timestamp"])
        # Yahoo devuelve hora local del exchange; sin zona se asume UTC.
        ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
        df["timestamp"] = ts
        return df
