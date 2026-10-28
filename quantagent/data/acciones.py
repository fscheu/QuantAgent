"""Snapshot de acciones del S&P 500 2007-2018 (M5-03): Yahoo si cubre la pertenencia, si no Quandl WIKI.

Nunca se empalman dos fuentes dentro de un mismo símbolo; el manifiesto dice qué fuente se usó en cada uno.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from quantagent.data.snapshot import load_manifest, read_snapshot, write_snapshot
from quantagent.data.sp500 import miembros

COMPOSICION = "sp500-2026-10"
INICIO, FIN = pd.Timestamp("2007-01-01"), pd.Timestamp("2018-12-31")
DESDE_YAHOO = "2006-01-01"
TOLERANCIA = pd.Timedelta(days=7)
ALIAS_CSV = Path(__file__).resolve().parent / "tablas" / "alias_tickers.csv"
_WIKI_COLUMNAS = {
    "adj_open": "open",
    "adj_high": "high",
    "adj_low": "low",
    "adj_close": "close",
    "adj_volume": "volume",
}


def _sesiones() -> pd.DatetimeIndex:
    import pandas_market_calendars as mcal

    return mcal.get_calendar("NYSE").schedule(INICIO, FIN).index.normalize()


def _miembros_por_sesion(composicion: str, base_dir: Optional[Path]) -> list[set]:
    return [set(miembros(composicion, d, base_dir)) for d in _sesiones()]


def _alias() -> Dict[tuple, str]:
    tabla = pd.read_csv(ALIAS_CSV)
    return {(r.ticker_sp500, r.fuente): r.ticker_fuente for r in tabla.itertuples()}


def _descargar_yahoo(symbol: str, start: str) -> pd.DataFrame:
    from quantagent.cli.data import download_daily

    return download_daily(symbol, start)


def _leer_wiki(wiki_csv: Path, nombres: set) -> Dict[str, pd.DataFrame]:
    """Lee WIKI_PRICES.csv en chunks y arma un frame por ticker con las columnas de `read_snapshot`."""
    columnas = ["ticker", "date", *_WIKI_COLUMNAS]
    trozos = [c[c["ticker"].isin(nombres)] for c in pd.read_csv(wiki_csv, usecols=columnas, chunksize=1_000_000)]
    frames = {}
    for ticker, df in pd.concat(trozos).groupby("ticker"):
        df = df.assign(timestamp=pd.to_datetime(df["date"])).set_index("timestamp").sort_index()
        df = df.rename(columns=_WIKI_COLUMNAS)[list(_WIKI_COLUMNAS.values())]
        df["adj_close"] = df["close"]
        df["volume"] = df["volume"].astype("int64")
        frames[ticker] = df[["open", "high", "low", "close", "adj_close", "volume"]]
    return frames


def crear_snapshot_acciones(
    nombre: str,
    wiki_csv: Path,
    composicion: str = COMPOSICION,
    base_dir: Optional[Path] = None,
    yahoo: Callable[[str, str], pd.DataFrame] = _descargar_yahoo,
) -> dict:
    """Arma el snapshot de todos los tickers que fueron miembros del S&P 500 entre 2007 y 2018."""
    alias = _alias()
    por_sesion = _miembros_por_sesion(composicion, base_dir)
    sesiones = _sesiones()
    periodos = {}
    for fecha, tickers in zip(sesiones, por_sesion):
        for t in tickers:
            periodos[t] = (periodos[t][0] if t in periodos else fecha, fecha)

    frames: Dict[str, pd.DataFrame] = {}
    fuentes: Dict[str, str] = {}
    parcial_yahoo = {}
    for t, (ini, fin) in sorted(periodos.items()):
        df = yahoo(alias.get((t, "yahoo"), t), DESDE_YAHOO)
        if df.empty:
            continue
        df = df[df.index <= FIN]
        if df.empty:
            continue
        if df.index.min() <= ini + TOLERANCIA and df.index.max() >= fin - TOLERANCIA:
            frames[t], fuentes[t] = df, "yahoo"
        else:
            parcial_yahoo[t] = df

    pendientes = {alias.get((t, "wiki"), t): t for t in periodos if t not in frames}
    wiki = _leer_wiki(wiki_csv, set(pendientes)) if pendientes else {}
    for nombre_wiki, t in pendientes.items():
        if nombre_wiki in wiki:
            frames[t], fuentes[t] = wiki[nombre_wiki][lambda d: d.index <= FIN], "wiki"
        elif t in parcial_yahoo:
            frames[t], fuentes[t] = parcial_yahoo[t], "yahoo"
    return write_snapshot(nombre, frames, source="yahoo+wiki", base_dir=base_dir, fuentes=fuentes)


def reporte_cobertura(nombre: str, composicion: str = COMPOSICION, base_dir: Optional[Path] = None) -> dict:
    """Cobertura de precios sobre los miembro-días 2007-2018, tickers sin precio, cortes anticipados y saltos >50%."""
    manifiesto = load_manifest(nombre, base_dir)
    sesiones = _sesiones()
    por_sesion = _miembros_por_sesion(composicion, base_dir)
    por_anio = pd.DataFrame(0, index=sorted(set(sesiones.year)), columns=["miembro", "precio"])
    filas, sin_precio, terminan_antes, saltos = [], [], {}, []
    for t in sorted(set().union(*por_sesion)):
        miembro = np.array([t in s for s in por_sesion])
        por_anio["miembro"] += pd.Series(miembro, index=sesiones.year).groupby(level=0).sum()
        if t not in manifiesto["symbols"]:
            sin_precio.append(t)
            continue
        df = read_snapshot(nombre, t, base_dir, adjusted=False)
        con_precio = miembro & sesiones.isin(df.index)
        faltan = int((miembro & (sesiones > df.index.max())).sum())
        if faltan:
            terminan_antes[t] = faltan
        for fecha, ret in df["adj_close"].pct_change().loc[INICIO:FIN][lambda r: r.abs() > 0.5].items():
            saltos.append({"ticker": t, "fecha": fecha.date().isoformat(), "retorno": float(ret)})
        por_anio["precio"] += pd.Series(con_precio, index=sesiones.year).groupby(level=0).sum()
        filas.append(
            {
                "ticker": t,
                "fuente": manifiesto.get("fuentes", {}).get(t),
                "miembro_dias": int(miembro.sum()),
                "dias_con_precio": int(con_precio.sum()),
                "dias_faltan_al_final": faltan,
            }
        )
    miembro_total = sum(len(s) for s in por_sesion)
    return {
        "miembro_dias": miembro_total,
        "miembro_dias_con_precio": int(por_anio["precio"].sum()),
        "pct_global": float(por_anio["precio"].sum() / miembro_total * 100),
        "pct_por_anio": {int(a): float(r.precio / r.miembro * 100) for a, r in por_anio.iterrows()},
        "sin_precio": sin_precio,
        "terminan_antes": terminan_antes,
        "saltos": saltos,
        "por_ticker": filas,
    }
