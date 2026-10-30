"""Etapa 1 de la fábrica: walk-forward por activo con el motor del filtro (QuantAgent-d4cq.13)."""

from functools import partial
from math import isnan

import pandas as pd

from quantagent.backtesting.signals import bar_signals
from quantagent.backtesting.vbt_filter import equity_metrics, run_filter
from quantagent.cli.walkforward import construir_ventanas, seleccionar_ganador
from quantagent.fabrica.cartera import combinar, pesos
from quantagent.fabrica.transversal import MASCARAS, METRICAS, REGLAS, semanales, señales_transversales
from quantagent.strategy.registry import build_strategy, cli_aliases


def parametros_elegidos(ganadores) -> dict:
    """Más frecuente; en empate gana la ventana de prueba más reciente."""
    recientes = sorted(ganadores, key=lambda g: g["prueba_desde"], reverse=True)
    return max(recientes, key=lambda g: sum(
        otro["params"] == g["params"] for otro in ganadores
    ))["params"]


def veredicto_etapa1(sharpe_cartera, trades_totales) -> tuple:
    """Veredicto a nivel cartera (D3): mínimo 30 trades y Sharpe >= 0,10."""
    if trades_totales < 30:
        return "CORTA", "menos de 30 trades en la cartera"
    if sharpe_cartera is None or isnan(sharpe_cartera) or sharpe_cartera < 0.10:
        return "CORTA", "Sharpe de cartera bajo 0,10"
    return "PASA", f"Sharpe de cartera {sharpe_cartera:.2f}"


def filtrar_activo(h, velas: pd.DataFrame, *, costos: dict, capital: float, panel_cierres: pd.DataFrame | None = None, train_years: int = 4, test_years: int = 2) -> dict:
    """Elige por ventana la combinación de mayor sharpe_tasa_0 en el ajuste y la corre en la prueba.

    `velas`: timestamp, open, high, low, close, volume; `costos`: commission_pct y slippage_pct.
    Las señales se calculan una vez por combinación sobre toda la serie y se recortan a cada tramo
    (válido porque la admisión verificó que no miran el futuro).
    """
    velas = velas.copy()
    velas.index = pd.DatetimeIndex(velas["timestamp"])
    nombre = cli_aliases().get(h.clase) or h.clase
    timeframe = getattr(h, "timeframe", "1d")
    estrategia = build_strategy(nombre, **next(iter(h.combinaciones()), {}))
    if hasattr(estrategia, "posiciones_objetivo"):
        if panel_cierres is None:
            raise ValueError("La estrategia de posiciones objetivo requiere cierres sincronizados")
        cierres = panel_cierres.copy()
        cierres.index = pd.DatetimeIndex(cierres.index)
        objetivos = estrategia.posiciones_objetivo(cierres)
        ventanas = construir_ventanas(velas.index[0].to_pydatetime(), velas.index[-1].to_pydatetime(), train_years, test_years)
        por_activo = {}
        for activo in h.activos:
            objetivo = objetivos[activo].reindex(cierres.index).fillna(0.0)
            cambios = objetivo.diff().fillna(0.0).ne(0)
            retornos, trades, ganadores = [], 0, []
            for _, _, c, d in ventanas:
                peso = objetivo.loc[c:d]
                ret = cierres.loc[c:d, activo].pct_change().fillna(0.0) * peso.shift(1, fill_value=0.0)
                retornos.append(ret.iloc[1:])
                trades += int(cambios.loc[c:d].sum())
                ganadores.append({"prueba_desde": c, "prueba_hasta": d, "params": {}})
            ret = pd.concat(retornos)
            equity = capital * (1 + ret).cumprod()
            por_activo[activo] = {"sharpe_tasa_0": equity_metrics(equity, 0.0)["sharpe"], "trades": trades,
                                  "ganadores": ganadores, "retornos": ret}
        return {**por_activo[h.activos[0]], "por_activo": por_activo}
    señales = [(c, bar_signals(build_strategy(nombre, **c), velas, h.activos[0], timeframe)) for c in h.combinaciones()]
    def correr(sig, a, b):
        return run_filter(sig.loc[a:b], velas.loc[a:b], capital=capital, **costos)

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
    return {"sharpe_tasa_0": equity_metrics(equity, 0.0)["sharpe"], "trades": trades, "ganadores": ganadores, "retornos": ret}


def filtrar_pierna(h, pierna, paneles, *, costos: dict, capital: float, train_years: int = 4, test_years: int = 2) -> dict:
    """Como `filtrar_activo`, pero la combinación se elige por pierna transversal y no por activo (QuantAgent-d4cq.62).

    `paneles`: {activo: velas} con timestamp, open, high, low, close, volume. Por combinación se calcula una vez el
    panel 0/1 (válido porque la admisión verificó que no mira el futuro); en cada ventana gana la combinación de
    mayor `sharpe_tasa_0` de la curva de la pierna en el ajuste, con los pesos de `cartera.pesos`. Cada activo corre
    `run_filter` con su columna: entra al pasar de 0 a 1 y sale al pasar de 1 a 0. `retornos` trae columnas
    `(índice de la pierna, activo)` fuera de muestra, y `trades` es la suma de las entradas de todos los activos.
    """
    indice, activos = h.piernas.index(pierna), list(pierna.activos)
    velas = {}
    for activo in activos:
        v = paneles[activo].copy()
        v.index = pd.DatetimeIndex(v["timestamp"])
        velas[activo] = v
    cierres = pd.DataFrame({a: v["close"] for a, v in velas.items()})
    combinaciones = pierna.combinaciones() if pierna.parametros else [{}]
    candidatas = []
    for comb in combinaciones:
        panel = señales_transversales(cierres, partial(METRICAS[pierna.clase], **comb), mascara=MASCARAS.get(pierna.elegibilidad), **REGLAS.get(pierna.clase, {}))
        previo = panel.shift(1, fill_value=0)
        señales = {a: pd.DataFrame({"entry": ((panel[a] == 1) & (previo[a] == 0)).astype(int),
                                    "exit": (panel[a] == 0) & (previo[a] == 1)}) for a in activos}
        candidatas.append((comb, panel, señales))

    def correr(señales, panel, a, b):
        # activos que nacieron después de la ventana no tienen velas: no operan y pesan 0
        res = {act: run_filter(señales[act].reindex(velas[act].index).loc[a:b], velas[act].loc[a:b], capital=capital, **costos)
               for act in activos if len(velas[act].loc[a:b])}
        ret = pd.DataFrame({(indice, act): r["equity"].pct_change().fillna(0.0) for act, r in res.items()}).reindex(
            index=cierres.loc[a:b].index, columns=pd.MultiIndex.from_product([[indice], activos])).fillna(0.0)
        w = pesos(cierres.loc[a:b], {indice: activos}, seleccion={indice: panel.loc[a:b]}, semanales={indice} if semanales([pierna]) else ()).fillna(0.0)
        pierna_ret = combinar(ret, w)
        return res, ret, pierna_ret, capital * (1 + pierna_ret).cumprod()

    ventanas = construir_ventanas(cierres.index[0].to_pydatetime(), cierres.index[-1].to_pydatetime(), train_years, test_years)
    ganadores, retornos, de_pierna, trades = [], [], [], 0
    seleccion = pd.DataFrame(0, index=cierres.index, columns=activos)
    for a, b, c, d in ventanas:
        ajuste = [{"sharpe_tasa_0": equity_metrics(correr(sig, panel, a, b)[3], 0.0)["sharpe"], "params": comb}
                  for comb, panel, sig in candidatas]
        params = seleccionar_ganador(ajuste)["params"]
        _, panel, sig = next(x for x in candidatas if x[0] == params)
        res, ret, pierna_ret, _ = correr(sig, panel, c, d)
        seleccion.loc[c:d] = panel.loc[c:d]
        ganadores.append({"prueba_desde": c, "prueba_hasta": d, "params": params})
        retornos.append(ret.iloc[1:])
        de_pierna.append(pierna_ret.iloc[1:])
        trades += sum(r["trades"] for r in res.values())
    curva = capital * (1 + pd.concat(de_pierna)).cumprod()
    return {"sharpe_tasa_0": equity_metrics(curva, 0.0)["sharpe"], "trades": trades, "ganadores": ganadores,
            "retornos": pd.concat(retornos), "seleccion": seleccion}
