"""Criterio de evaluación de curvas de equity (holdout y sub-períodos)."""
import math
import warnings
from typing import NamedTuple

import pandas as pd

from quantagent.backtesting.vbt_filter import equity_metrics
from quantagent.backtesting.verdict import MAX_DRAWDOWN, MIN_TRADES
from quantagent.data.macro import serie_punto_en_el_tiempo

MIN_SESIONES = 30


class AniosPositivos(NamedTuple):
    positivos: int
    totales: int


class RecargoPaper(NamedTuple):
    tamano_pct: int
    semanas: int


def sharpes_en_exceso(cartera, benchmark, *, base_dir=None) -> tuple[float, float]:
    """Diagnóstico con DGS3MO de macro-2026-10, sin cambiar el corte ni el DSR.

    Ambas curvas deben tener las mismas fechas. Se resta la tasa conocida al
    inicio de cada intervalo: porcentaje anual / 100 × días reales / 365.25
    (interés simple, incluidos fines de semana). Se anualiza como equity_metrics.
    """
    if not cartera.index.equals(benchmark.index) or len(cartera) < 3:
        raise ValueError("Se requieren dos curvas con las mismas fechas y al menos tres puntos")
    fechas = cartera.index
    if not fechas.is_monotonic_increasing or not fechas.is_unique:
        raise ValueError("Las fechas deben ser crecientes y únicas")
    tasa = serie_punto_en_el_tiempo("macro-2026-10", "DGS3MO", fechas[:-1], base_dir)
    if tasa.isna().any():
        raise ValueError("DGS3MO sin cobertura al inicio de uno o más intervalos")
    dias = (fechas[1:] - fechas[:-1]).total_seconds().to_numpy() / 86400
    libre = tasa.to_numpy() / 100 * dias / 365.25
    ppy = (len(fechas) - 1) * 365.25 / dias.sum()
    resultado = []
    for curva in (cartera, benchmark):
        if curva.isna().any():
            raise ValueError("Curva sin cobertura en una o más fechas")
        exceso = pd.Series(curva.to_numpy()[1:] / curva.to_numpy()[:-1] - 1 - libre)
        resultado.append(0.0 if exceso.abs().max() < 1e-12 or exceso.std() == 0
                         else float(exceso.mean() / exceso.std() * math.sqrt(ppy)))
    return tuple(resultado)


def recargo_paper(dsr: float | None) -> RecargoPaper:
    """Devuelve (tamano_pct, semanas) de paper trading según el DSR."""
    if dsr is None or math.isnan(dsr) or dsr < 0.50:
        return RecargoPaper(50, 12)
    if dsr < 0.95:
        return RecargoPaper(75, 6)
    return RecargoPaper(100, 0)


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


def curva_benchmark(cierres, capital: float = 100_000.0) -> pd.Series:
    """Calcula la curva de equity con pesos iguales y rebalanceo mensual."""
    if cierres is None or (isinstance(cierres, (pd.DataFrame, pd.Series)) and cierres.empty):
        return pd.Series(dtype=float)
    df = (cierres.to_frame() if isinstance(cierres, pd.Series) else cierres.loc[:, ~cierres.columns.duplicated()]).sort_index()
    if len(df.columns) == 0 or len(df) == 0:
        return pd.Series(dtype=float)
    meses = df.index.to_period("M")
    equity = pd.Series(index=df.index, dtype=float)
    eq_actual = capital
    for m in meses.unique():
        sub = df[meses == m]
        loc = df.index.get_loc(sub.index[0])
        loc = loc.start if isinstance(loc, slice) else loc
        base = df.iloc[loc - 1] if loc > 0 else sub.iloc[0]
        rel = (sub / base).mean(axis=1)
        sub_eq = eq_actual * rel
        equity.loc[sub.index] = sub_eq
        eq_actual = float(sub_eq.iloc[-1])
    return equity


def sharpe_benchmark(cierres, risk_free_rate: float = 0.0) -> float:
    """Sharpe a tasa 0 de la cartera de activos con pesos iguales y rebalanceo mensual."""
    eq = curva_benchmark(cierres)
    return float(equity_metrics(eq, risk_free_rate)["sharpe"]) if not eq.empty else 0.0


def evaluar_etapa3_cartera(
    eq_cartera: pd.Series | None,
    sharpe_bench3: float,
    sharpe_cartera_etapa2: float | None = None,
    sharpe_bench2: float | None = None,
    max_dd_limite: float = MAX_DRAWDOWN,
    trades: int | None = None,
) -> tuple[bool, list[str]]:
    """Criterio de pase de la etapa 3 para carteras (DC 2026-10-19)."""
    if eq_cartera is None or eq_cartera.empty:
        return False, ["curva de cartera vacía"]
    m = equity_metrics(eq_cartera, 0.0)
    sharpe = float(m["sharpe"])
    dd = float(m["max_drawdown"])
    motivos = []
    if trades is not None and trades < MIN_TRADES:
        motivos.append(f"trades < {MIN_TRADES}")
    if sharpe <= 0:
        motivos.append("Sharpe de cartera <= 0")
    if sharpe <= sharpe_bench3:
        motivos.append(f"Sharpe de cartera ({sharpe:.2f}) <= benchmark ({sharpe_bench3:.2f})")
    if sharpe_cartera_etapa2 is None:
        motivos.append("sin Sharpe de etapa 2")
    if sharpe_bench2 is None:
        motivos.append("sin benchmark de etapa 2")
    if sharpe_cartera_etapa2 is not None and sharpe_bench2 is not None and sharpe_cartera_etapa2 <= sharpe_bench2:
        motivos.append(f"Sharpe etapa 2 ({sharpe_cartera_etapa2:.2f}) <= benchmark ({sharpe_bench2:.2f})")
    ap = anios_positivos(eq_cartera)
    if ap.positivos < 3:
        motivos.append(f"Sharpe positivo en {ap.positivos} de {ap.totales} años (< 3)")
    if dd > max_dd_limite:
        motivos.append(f"max_drawdown > {max_dd_limite}")
    return len(motivos) == 0, motivos

