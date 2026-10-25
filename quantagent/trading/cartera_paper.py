"""Sesiones cerradas de mercado para ejecución paper de carteras."""

from datetime import datetime

import pandas as pd
import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("NYSE")


def _a_utc(ts: datetime | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def sesiones_cerradas_nyse(
    as_of: datetime | pd.Timestamp,
    n_sesiones: int = 253,
) -> pd.DatetimeIndex:
    """Índice de sesiones NYSE cerradas al momento `as_of` (UTC inyectado)."""
    ts_utc = _a_utc(as_of)
    dias = max(int(n_sesiones * 2) + 30, 10)
    sched = _NYSE.schedule(start_date=(ts_utc - pd.Timedelta(days=dias)).date(), end_date=ts_utc.date())
    cerradas = sched[sched["market_close"] <= ts_utc]
    return cerradas.index[-n_sesiones:]


def ultima_sesion_cerrada_nyse(as_of: datetime | pd.Timestamp) -> pd.Timestamp:
    """Fecha normalizada de la última rueda NYSE cerrada al momento `as_of`."""
    return sesiones_cerradas_nyse(as_of, n_sesiones=1)[-1]
