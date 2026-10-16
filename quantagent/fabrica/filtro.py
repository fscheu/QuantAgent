"""Etapa 1 de la fábrica: walk-forward por activo con el motor del filtro (QuantAgent-d4cq.13)."""

import pandas as pd

from quantagent.backtesting.signals import bar_signals
from quantagent.backtesting.vbt_filter import equity_metrics, run_filter
from quantagent.cli.walkforward import construir_ventanas, seleccionar_ganador
from quantagent.strategy.registry import build_strategy, cli_aliases


def filtrar_activo(h, velas: pd.DataFrame, *, costos: dict, capital: float, train_years: int = 4, test_years: int = 2) -> dict:
    """Elige por ventana la combinación de mayor sharpe_tasa_0 en el ajuste y la corre en la prueba.

    `velas`: timestamp, open, high, low, close, volume; `costos`: commission_pct y slippage_pct.
    Las señales se calculan una vez por combinación sobre toda la serie y se recortan a cada tramo
    (válido porque la admisión verificó que no miran el futuro).
    """
    velas = velas.copy()
    velas.index = pd.DatetimeIndex(velas["timestamp"])
    nombre = cli_aliases().get(h.clase) or h.clase
    señales = [(c, bar_signals(build_strategy(nombre, **c), velas, h.activos[0], h.timeframe)) for c in h.combinaciones()]
    correr = lambda sig, a, b: run_filter(sig.loc[a:b], velas.loc[a:b], capital=capital, **costos)
    ventanas = construir_ventanas(velas.index[0].to_pydatetime(), velas.index[-1].to_pydatetime(), train_years, test_years)
    ganadores, retornos, trades = [], [], 0
    for a, b, c, d in ventanas:
        ajuste = [{**correr(sig, a, b), "params": params} for params, sig in señales]
        params = seleccionar_ganador(ajuste)["params"]
        r = correr(next(sig for p, sig in señales if p == params), c, d)
        ganadores.append({"prueba_desde": c, "prueba_hasta": d, "params": params})
        retornos.append(r["equity"].pct_change().dropna())
        trades += r["trades"]
    ret = pd.concat(retornos)
    equity = capital * (1 + ret).cumprod()
    return {"sharpe_tasa_0": equity_metrics(equity, 0.0)["sharpe"], "trades": trades, "ganadores": ganadores}
