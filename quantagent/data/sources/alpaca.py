"""Alpaca market data source (SIP feed, bars only). Ticket QuantAgent-jpms.

Same three methods as the CandleSource interface (QuantAgent-qix2), defined
here without importing the protocol until that ticket is merged.
"""

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

DATA_HOST = "https://data.alpaca.markets"
BARS_PATH = "/v2/stocks/bars"
SIP_DELAY = timedelta(minutes=15)
EARLIEST = datetime(2016, 1, 1, tzinfo=timezone.utc)
TIMEFRAMES = {
    "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
    "1h": "1Hour", "4h": "4Hour", "1d": "1Day", "1w": "1Week", "1mo": "1Month",
}
MAX_RETRIES = 3
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class AlpacaSource:
    name = "alpaca"

    def __init__(self, session=None, now=None, sleep=time.sleep):
        self._session = session or requests.Session()
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep

    def _headers(self) -> dict:
        key, secret = os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError(
                "Faltan claves de Alpaca: definir APCA_API_KEY_ID y APCA_API_SECRET_KEY"
            )
        return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def earliest(self, timeframe, now=None):
        return EARLIEST

    def latest(self, now=None):
        return (now or self._now()) - SIP_DELAY

    def _get(self, params, headers):
        for attempt in range(MAX_RETRIES + 1):
            resp = self._session.get(
                DATA_HOST + BARS_PATH, params=params, headers=headers, timeout=30
            )
            if resp.status_code != 429 or attempt == MAX_RETRIES:
                resp.raise_for_status()
                return resp.json()
            self._sleep(2**attempt)

    def fetch(self, symbol, timeframe, start, end) -> pd.DataFrame:
        headers = self._headers()
        end = min(pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tzinfo is None
                  else pd.Timestamp(end), pd.Timestamp(self.latest()))
        start = pd.Timestamp(start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start
        params = {
            "symbols": symbol,
            "timeframe": TIMEFRAMES[timeframe],
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": "sip",
            "adjustment": "split",
            "limit": 10000,
        }
        bars = []
        while True:
            body = self._get(params, headers)
            bars += (body.get("bars") or {}).get(symbol, [])
            if not body.get("next_page_token"):
                break
            params["page_token"] = body["next_page_token"]
        df = pd.DataFrame(bars).rename(
            columns={"t": "timestamp", "o": "open", "h": "high", "l": "low",
                     "c": "close", "v": "volume"}
        )
        if df.empty:
            return pd.DataFrame(columns=COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df[COLUMNS].reset_index(drop=True)
