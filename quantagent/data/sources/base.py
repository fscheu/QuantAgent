"""Protocolo de fuente de velas, independiente de la caché."""

from datetime import datetime
from typing import Optional, Protocol

import pandas as pd

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


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
