"""Criterio de evaluación de curvas de equity (holdout y sub-períodos)."""
import warnings
from typing import NamedTuple

import pandas as pd

from quantagent.backtesting.vbt_filter import equity_metrics

MIN_SESIONES = 30


class AniosPositivos(NamedTuple):
    positivos: int
    totales: int


def sharpe_por_anio(equity, fechas=None, risk_free_rate: float = 0.0) -> dict[int, float]:
    """Calcula Sharpe por año calendario. Años con <30 sesiones se descartan e informan."""
    if fechas is not None:
        vals = equity.values if isinstance(equity, pd.Series) else equity
        s = pd.Series(vals, index=pd.to_datetime(fechas))
    elif isinstance(equity, pd.Series):
        s = equity if isinstance(equity.index, pd.DatetimeIndex) else pd.Series(equity.values, index=pd.to_datetime(equity.index))
    else:
        raise ValueError("Se requieren fechas cuando equity no tiene DatetimeIndex")

    res = {}
    for anio, sub in s.sort_index().groupby(s.index.year):
        if len(sub) < MIN_SESIONES:
            warnings.warn(f"Año {anio} descartado: {len(sub)} sesiones (< {MIN_SESIONES})", UserWarning, stacklevel=2)
            continue
        ret = sub.pct_change().dropna()
        if len(ret) >= 2 and ret.std() == 0:
            res[int(anio)] = float("inf") if ret.mean() > 0 else (float("-inf") if ret.mean() < 0 else 0.0)
        else:
            res[int(anio)] = float(equity_metrics(sub, risk_free_rate)["sharpe"])
    return res


def anios_positivos(equity_o_sharpes, fechas=None, risk_free_rate: float = 0.0) -> AniosPositivos:
    """Devuelve cuántos años del tramo tienen Sharpe > 0."""
    sharpes = equity_o_sharpes if isinstance(equity_o_sharpes, dict) else sharpe_por_anio(equity_o_sharpes, fechas, risk_free_rate)
    positivos = sum(1 for s in sharpes.values() if s > 0)
    return AniosPositivos(positivos, len(sharpes))
