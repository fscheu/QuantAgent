"""Protocolo de fuente de velas, independiente de la caché."""

from datetime import datetime
from typing import Optional, Protocol

import pandas as pd

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

DAILY_TIMEFRAMES = {"1d", "1w", "1mo"}


def to_utc(ts: pd.Series, timeframe: str) -> pd.Series:
    """Pasa timestamps con zona a UTC.

    Las barras diarias o mayores que vienen a las 00:00 locales conservan esa
    fecha a las 00:00 UTC (convención de la caché); convertirlas las movería a
    las 04:00/05:00 y duplicaría las filas ya guardadas.
    """
    if timeframe not in DAILY_TIMEFRAMES:
        return ts.dt.tz_convert("UTC")
    local_midnight = ts.dt.tz_localize(None) == ts.dt.tz_localize(None).dt.normalize()
    wall_as_utc = ts.dt.tz_localize(None).dt.tz_localize("UTC")
    return wall_as_utc.where(local_midnight, ts.dt.tz_convert("UTC"))


class CandleSource(Protocol):
    """Proveedor de velas OHLCV. `timestamp` sale en UTC con zona."""

    name: str

    def fetch(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Velas entre start y end, con columnas `CANDLE_COLUMNS`."""
        ...

    def earliest(self, timeframe: str, now: datetime) -> Optional[datetime]:
        """Inicio más viejo que la fuente sirve para el timeframe; None si no hay límite."""
        ...

    def latest(self, now: datetime) -> datetime:
        """Fin más reciente que la fuente sirve a `now`."""
        ...
