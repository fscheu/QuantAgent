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
FIN_WIKI = pd.Timestamp("2018-03-27")
MIN_COINCIDENCIA = 0.9
ALIAS_CSV = Path(__file__).resolve().parent / "tablas" / "alias_tickers.csv"
# Tickers sin WIKI cuya serie de Yahoo es de otra empresa (reúso), revisados a mano en fnpr.1.8.
DESCARTADOS_YAHOO = frozenset({"BUD", "CCU", "EP", "EQ", "GENZ", "KG", "MI", "SII", "TT", "UST", "WB"})
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
    if trozos:
        for ticker, df in pd.concat(trozos).groupby("ticker"):
            df = df.assign(timestamp=pd.to_datetime(df["date"])).set_index("timestamp").sort_index()
            df = df.rename(columns=_WIKI_COLUMNAS)[list(_WIKI_COLUMNAS.values())]
            df["adj_close"] = df["close"]
            df["volume"] = df["volume"].astype("int64")
            frames[ticker] = df[["open", "high", "low", "close", "adj_close", "volume"]]
    return frames


def _coincidencia(yahoo: pd.DataFrame, wiki: pd.DataFrame) -> Optional[float]:
    """Fracción de sesiones 2007..2018-03 en que los retornos diarios de Yahoo y WIKI difieren menos de 0,5 %."""
    ret_y = yahoo["adj_close"].loc[INICIO:FIN_WIKI].pct_change()
    ret_w = wiki["close"].loc[INICIO:FIN_WIKI].pct_change()
    dif = (ret_y - ret_w).dropna()
    return float((dif.abs() < 0.005).mean()) if len(dif) >= 10 else None


def crear_snapshot_acciones(
    nombre: str,
    wiki_csv: Path,
    composicion: str = COMPOSICION,
    base_dir: Optional[Path] = None,
    yahoo: Callable[[str, str], pd.DataFrame] = _descargar_yahoo,
) -> dict:
    """Arma el snapshot de todos los tickers que fueron miembros del S&P 500 entre 2007 y 2018.

    Por ticker: Yahoo si cubre la pertenencia y coincide con WIKI (si WIKI lo tiene); si no, WIKI; si tampoco, el
    parcial de Yahoo. Una coincidencia < MIN_COINCIDENCIA indica reúso del ticker en Yahoo por otra empresa.
    """
    alias = _alias()
    periodos = {}
    for fecha, tickers in zip(_sesiones(), _miembros_por_sesion(composicion, base_dir)):
        for t in tickers:
            periodos[t] = (periodos[t][0] if t in periodos else fecha, fecha)
    wiki = _leer_wiki(wiki_csv, {alias.get((t, "wiki"), t) for t in periodos})

    frames: Dict[str, pd.DataFrame] = {}
    fuentes: Dict[str, str] = {}
    reuso: Dict[str, float] = {}
    yahoo_sin_wiki: Dict[str, dict] = {}
    for t, (ini, fin) in sorted(periodos.items()):
        df_w = wiki.get(alias.get((t, "wiki"), t))
        df_y = yahoo(alias.get((t, "yahoo"), t), DESDE_YAHOO)
        df_y = df_y[df_y.index <= FIN] if not df_y.empty else df_y
        if not df_y.empty and df_w is None:
            yahoo_sin_wiki[t] = {
                "start": df_y.index.min().date().isoformat(),
                "end": df_y.index.max().date().isoformat(),
                "close_min": round(float(df_y["close"].min()), 4),
                "close_max": round(float(df_y["close"].max()), 4),
                "descartado": t in DESCARTADOS_YAHOO,
            }
        if df_y.empty or t in DESCARTADOS_YAHOO:
            df_y = None
        elif df_w is not None:
            coinc = _coincidencia(df_y, df_w)
            if coinc is not None and coinc < MIN_COINCIDENCIA:
                reuso[t], df_y = round(coinc, 4), None

        if df_y is not None and df_y.index.min() <= ini + TOLERANCIA and df_y.index.max() >= fin - TOLERANCIA:
            frames[t], fuentes[t] = df_y, "yahoo"
        elif df_w is not None:
            frames[t], fuentes[t] = df_w[df_w.index <= FIN], "wiki"
        elif df_y is not None:
            frames[t], fuentes[t] = df_y, "yahoo"
    return write_snapshot(
        nombre,
        frames,
        source="yahoo+wiki",
        base_dir=base_dir,
        fuentes=fuentes,
        extra={"reuso_cambiados_a_wiki": reuso, "yahoo_sin_wiki": yahoo_sin_wiki},
    )


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
        "reuso_cambiados_a_wiki": manifiesto.get("reuso_cambiados_a_wiki", {}),
        "yahoo_sin_wiki": manifiesto.get("yahoo_sin_wiki", {}),
        "por_ticker": filas,
    }
