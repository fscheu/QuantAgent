"""Screener semanal por sector y tabla candidatos_semanales (M5-32)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd

import quantagent.data.sp500 as sp500
from quantagent.data.fundamentales import metricas_fundamentales
from quantagent.data.snapshot import read_snapshot

ETF_SECTOR = {
    "XLB": "Materials", "XLC": "Communication Services", "XLE": "Energy", "XLF": "Financials",
    "XLI": "Industrials", "XLK": "Information Technology", "XLP": "Consumer Staples",
    "XLRE": "Real Estate", "XLU": "Utilities", "XLV": "Health Care", "XLY": "Consumer Discretionary",
}
ETF_DIRECTO_ETAPA3 = ("EEM", "EWZ", "SPY")
COLS_CANDIDATOS = ["fecha", "etf", "sector", "simbolo", "score", "calidad", "impulso", "fuerza_relativa", "rank"]


def _obtener_precios(precios, sym: str, es_etf: bool = False):
    if precios is not None and sym in precios:
        val = precios[sym]
        return val["close"] if isinstance(val, pd.DataFrame) and "close" in val.columns else val
    try:
        return read_snapshot("etf-m5-1d-2026-10" if es_etf else "acciones-sp500-2026-10b", sym)["close"]
    except Exception:
        return None


def _retorno(s, fecha, n: int) -> float:
    if s is None or len(s) == 0:
        return np.nan
    sub = s.loc[:fecha].dropna()
    return float(sub.iloc[-1] / sub.iloc[-n - 1] - 1.0) if len(sub) >= n + 1 else np.nan


def _metrica_fund(paneles, metrica: str, fecha, ticker: str) -> float:
    if paneles is not None and metrica in paneles and ticker in paneles[metrica].columns:
        df = paneles[metrica]
        for idx in (pd.Timestamp(fecha), pd.Timestamp(fecha).normalize()):
            if idx in df.index and pd.notna(val := df.loc[idx, ticker]):
                return float(val)
    return np.nan


def candidatos_semanales(
    fecha,
    etfs_habilitados: Iterable[str],
    precios: Optional[Mapping | pd.DataFrame] = None,
    paneles_fundamentales: Optional[dict[str, pd.DataFrame]] = None,
    composicion: str = "sp500-2026-10",
    top: int = 3,
) -> pd.DataFrame:
    """Screener semanal por sector: z-score intra sector (±3), excluye cuartil inferior de FR, top 3."""
    fecha_ts, resultado = pd.Timestamp(fecha), []

    for etf in etfs_habilitados:
        if etf in ETF_DIRECTO_ETAPA3:
            resultado.append({"fecha": fecha_ts, "etf": etf, "sector": "pais", "simbolo": etf,
                              "score": np.nan, "calidad": np.nan, "impulso": np.nan, "fuerza_relativa": np.nan, "rank": np.nan})
            continue
        if etf not in ETF_SECTOR:
            continue

        sec_name = ETF_SECTOR[etf]
        miembros = sp500.miembros(composicion, fecha_ts)
        universo = [t for t in miembros if sp500.sector(composicion, t, fecha_ts) == sec_name]
        if not universo:
            continue

        paneles = (paneles_fundamentales if paneles_fundamentales is not None
                   else metricas_fundamentales("fundamentales-sp500-2026-10", universo, pd.DatetimeIndex([fecha_ts])))
        s_etf = _obtener_precios(precios, etf, es_etf=True)
        r63_etf, r126_etf = _retorno(s_etf, fecha_ts, 63), _retorno(s_etf, fecha_ts, 126)
        m_q1 = "roe_4t" if sec_name == "Financials" else "roic_4t"
        m_q2 = "margen_pretax_yoy" if sec_name == "Financials" else "margen_op_yoy"

        filas = []
        for t in universo:
            s_t = _obtener_precios(precios, t, es_etf=False)
            r63_t, r126_t = _retorno(s_t, fecha_ts, 63), _retorno(s_t, fecha_ts, 126)
            rs3 = (r63_t - r63_etf) if not np.isnan(r63_t) and not np.isnan(r63_etf) else np.nan
            rs6 = (r126_t - r126_etf) if not np.isnan(r126_t) and not np.isnan(r126_etf) else np.nan
            crudos = [
                _metrica_fund(paneles, m_q1, fecha_ts, t), _metrica_fund(paneles, m_q2, fecha_ts, t),
                _metrica_fund(paneles, "aceleracion_ingresos", fecha_ts, t), _metrica_fund(paneles, "crecimiento_eps", fecha_ts, t),
                rs3, rs6,
            ]
            if sum(1 for v in crudos if np.isnan(v)) <= 1:
                filas.append({"simbolo": t, "crudos": crudos})
        if not filas:
            continue

        z_cols = []
        for k in range(6):
            vals = [r["crudos"][k] for r in filas if not np.isnan(r["crudos"][k])]
            mu, sigma = (float(np.mean(vals)), float(np.std(vals, ddof=0))) if vals else (0.0, 0.0)
            z_cols.append([0.0 if np.isnan(r["crudos"][k]) or sigma == 0
                           else float(np.clip((r["crudos"][k] - mu) / sigma, -3.0, 3.0)) for r in filas])

        for i, r in enumerate(filas):
            r["calidad"] = (z_cols[0][i] + z_cols[1][i]) / 2.0
            r["impulso"] = (z_cols[2][i] + z_cols[3][i]) / 2.0
            r["fuerza_relativa"] = (z_cols[4][i] + z_cols[5][i]) / 2.0
            r["score"] = 0.35 * r["calidad"] + 0.35 * r["impulso"] + 0.30 * r["fuerza_relativa"]

        q25 = float(pd.Series([r["fuerza_relativa"] for r in filas]).quantile(0.25))
        elegibles = [r for r in filas if r["fuerza_relativa"] >= q25]
        elegibles.sort(key=lambda r: (-r["score"], r["simbolo"]))
        for rank, r in enumerate(elegibles[:top], start=1):
            resultado.append({
                "fecha": fecha_ts, "etf": etf, "sector": sec_name, "simbolo": r["simbolo"],
                "score": r["score"], "calidad": r["calidad"], "impulso": r["impulso"],
                "fuerza_relativa": r["fuerza_relativa"], "rank": rank,
            })

    return pd.DataFrame(resultado, columns=COLS_CANDIDATOS)


def candidatos_semanales_panel(
    fechas: Iterable,
    etfs_habilitados: Union[Iterable[str], Mapping[Any, Iterable[str]]],
    precios: Optional[Mapping | pd.DataFrame] = None,
    paneles_fundamentales: Optional[dict[str, pd.DataFrame]] = None,
    composicion: str = "sp500-2026-10",
    top: int = 3,
) -> pd.DataFrame:
    """Concatena candidatos semanales para una lista de fechas."""
    frames = [
        candidatos_semanales(f, etfs_habilitados[f] if isinstance(etfs_habilitados, Mapping) else etfs_habilitados,
                             precios=precios, paneles_fundamentales=paneles_fundamentales,
                             composicion=composicion, top=top)
        for f in fechas
    ]
    no_vacios = [df for df in frames if not df.empty]
    return pd.concat(no_vacios, ignore_index=True) if no_vacios else pd.DataFrame(columns=COLS_CANDIDATOS)
