"""Métricas fundamentales punto en el tiempo desde SEC EDGAR (M5-31)."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from quantagent.data.snapshot import load_manifest, read_snapshot, write_snapshot

SEC_USER_AGENT = "QuantAgent Research admin@quantagent.dev"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

_TAX_BT = [
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
]
CONCEPTOS_FALLBACK = {
    "ingresos": [
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet", "RevenuesNetOfInterestExpense",
    ],
    # sin OperatingIncomeLoss (XOM, bancos) se usa el resultado antes de impuestos como aproximación
    "OperatingIncomeLoss": ["OperatingIncomeLoss", *_TAX_BT], "EarningsPerShareDiluted": ["EarningsPerShareDiluted"],
    "NetIncomeLoss": ["NetIncomeLoss", "ProfitLoss"], "StockholdersEquity": ["StockholdersEquity"], "deuda": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "IncomeTaxExpenseBenefit": ["IncomeTaxExpenseBenefit"], "IncomeLossFromContinuingOperationsBeforeIncomeTaxes": _TAX_BT,
}
# tickers cuyo CIK actual (company_tickers.json) es una holding nueva sin el historial anterior
CIK_PREVIOS = {"XOM": [34088], "BLK": [1364742], "BG": [1144519], "APO": [1411494]}
FORMULARIOS_VALIDOS = frozenset({"10-Q", "10-Q/A", "10-K", "10-K/A"})
CONCEPTOS_FLUJO = frozenset({"ingresos", "NetIncomeLoss", "OperatingIncomeLoss", "EarningsPerShareDiluted", "IncomeTaxExpenseBenefit", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"})


def mapa_tickers_cik() -> Dict[str, int]:
    """Obtiene el mapeo de ticker a CIK desde sec.gov."""
    req = urllib.request.Request(SEC_TICKERS_URL, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {item["ticker"].upper(): int(item["cik_str"]) for item in data.values()}


def descargar_facts_cik(cik: int) -> dict:
    """Descarga el JSON companyfacts de un CIK desde la API de la SEC."""
    req = urllib.request.Request(SEC_FACTS_URL_TEMPLATE.format(cik=cik), headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extraer_hechos(ticker: str, json_data: dict) -> pd.DataFrame:
    """Extrae hechos contables normalizados según fallbacks y formularios 10-Q/10-K."""
    us_gaap = json_data.get("facts", {}).get("us-gaap", {})
    filas = []
    for concepto, tags in CONCEPTOS_FALLBACK.items():
        cubiertos: Set[Tuple[str, str]] = set()
        for prioridad, tag in enumerate(tags):
            if tag not in us_gaap:
                continue
            for unit_facts in us_gaap[tag].get("units", {}).values():
                for f in unit_facts:
                    form, end = f.get("form", ""), str(f.get("end", "")).strip()
                    if form not in FORMULARIOS_VALIDOS or not end:
                        continue
                    start = str(f.get("start", "") or "").strip()
                    if (start, end) in cubiertos:
                        continue
                    try:
                        val = float(f.get("val"))
                    except (TypeError, ValueError):
                        continue
                    fy = f.get("fy")
                    filas.append({
                        "ticker": ticker, "concepto": concepto, "start": start, "end": end, "val": val,
                        "fy": int(fy) if fy is not None and not pd.isna(fy) else None,
                        "fp": str(f.get("fp", "")).strip(), "form": form,
                        "filed": str(f.get("filed", "")).strip(), "accn": str(f.get("accn", "")).strip(),
                        "prioridad": prioridad,
                    })
            for r in filas:
                if r["concepto"] == concepto and r["prioridad"] == prioridad:
                    cubiertos.add((r["start"], r["end"]))
    cols = ["ticker", "concepto", "start", "end", "val", "fy", "fp", "form", "filed", "accn"]
    return pd.DataFrame(filas).drop(columns=["prioridad"]) if filas else pd.DataFrame(columns=cols)


METRICAS = ["roic_4t", "margen_op_yoy", "aceleracion_ingresos", "crecimiento_eps", "roe_4t", "margen_pretax_yoy"]
MAX_ANTIGUEDAD_DIAS = 200


def preparar_ticker(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """Parsea fechas y ordena por presentación (filed, accn)."""
    df = df_ticker.reset_index(drop=True)
    df["filed"], df["end_dt"], df["start_dt"] = pd.to_datetime(df["filed"]), pd.to_datetime(df["end"]), pd.to_datetime(df["start"])
    df["days"] = (df["end_dt"] - df["start_dt"]).dt.days.fillna(0)
    return df.sort_values(["filed", "accn"], kind="stable")


def vista_punto_en_el_tiempo(df: pd.DataFrame, f: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Con lo presentado hasta f (última presentación por período) arma flujos trimestrales (Q4 = anual - 9M) y balances."""
    v = df[df["filed"] <= f].drop_duplicates(subset=["concepto", "start", "end"], keep="last")
    fl = v[v["concepto"].isin(CONCEPTOS_FLUJO)]
    q4 = fl[fl["days"].between(330, 395)].merge(fl[fl["days"].between(240, 305)], on=["concepto", "start"], suffixes=("_a", "_n"))
    q4 = q4[(q4["end_dt_a"] - q4["end_dt_n"]).dt.days.between(60, 115)]
    q4 = pd.DataFrame({"concepto": q4["concepto"], "end_dt": q4["end_dt_a"], "val": q4["val_a"] - q4["val_n"]})
    trim = pd.concat([fl[fl["days"].between(60, 115)][["concepto", "end_dt", "val"]], q4]).drop_duplicates(subset=["concepto", "end_dt"], keep="first")
    bal = v[v["concepto"].isin(["StockholdersEquity", "deuda"])]
    return trim.pivot(index="end_dt", columns="concepto", values="val").sort_index(), bal.pivot(index="end_dt", columns="concepto", values="val").sort_index()


def _ventana(piv: pd.DataFrame, cols: List[str], n: int) -> Optional[pd.DataFrame]:
    """Últimos n trimestres con todas las columnas, o None si faltan o hay huecos entre ends."""
    if piv.empty or any(c not in piv.columns for c in cols):
        return None
    w = piv[cols].dropna().iloc[-n:]
    if len(w) < n or not w.index.to_series().diff().dt.days.iloc[1:].between(75, 105).all():
        return None
    return w


def _balance(piv_b: pd.DataFrame, concepto: str, end: pd.Timestamp) -> float:
    if concepto not in piv_b.columns:
        return np.nan
    s = piv_b.loc[piv_b.index <= end, concepto].dropna()
    return float(s.iloc[-1]) if not s.empty and (end - s.index[-1]).days <= 120 else np.nan


def metricas_en_vista(piv_f: pd.DataFrame, piv_b: pd.DataFrame) -> Dict[str, Tuple[float, pd.Timestamp]]:
    """Las métricas con la vista dada: {métrica: (valor, end del último trimestre usado)}."""
    res = {}
    if (w := _ventana(piv_f, ["OperatingIncomeLoss"], 4)) is not None:
        fin = w.index[-1]
        imp = piv_f.reindex(w.index)[["IncomeTaxExpenseBenefit", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"]] if "IncomeTaxExpenseBenefit" in piv_f.columns and "IncomeLossFromContinuingOperationsBeforeIncomeTaxes" in piv_f.columns else None
        tasa = 0.21
        if imp is not None and not imp.isna().any().any() and imp.iloc[:, 1].sum() > 0:
            tasa = max(0.0, min(0.5, float(imp.iloc[:, 0].sum() / imp.iloc[:, 1].sum())))
        eq, deuda = _balance(piv_b, "StockholdersEquity", fin), _balance(piv_b, "deuda", fin)
        deuda = 0.0 if np.isnan(deuda) else deuda
        if not np.isnan(eq) and eq + deuda > 0:
            res["roic_4t"] = (float(w["OperatingIncomeLoss"].sum() * (1.0 - tasa) / (eq + deuda)), fin)
    for met, col in [("margen_op_yoy", "OperatingIncomeLoss"), ("margen_pretax_yoy", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes")]:
        if (w := _ventana(piv_f, ["ingresos", col], 8)) is not None:
            rh, oh, rp, op = w["ingresos"].iloc[4:].sum(), w[col].iloc[4:].sum(), w["ingresos"].iloc[:4].sum(), w[col].iloc[:4].sum()
            if rh > 0 and rp > 0:
                res[met] = (float(oh / rh - op / rp), w.index[-1])
    if (w := _ventana(piv_f, ["NetIncomeLoss"], 4)) is not None:
        eq = _balance(piv_b, "StockholdersEquity", w.index[-1])
        if eq > 0:
            res["roe_4t"] = (float(w["NetIncomeLoss"].sum() / eq), w.index[-1])
    if (w := _ventana(piv_f, ["ingresos"], 6)) is not None:
        r = w["ingresos"].to_numpy()
        if r[1] > 0 and r[0] > 0:
            res["aceleracion_ingresos"] = (float((r[5] - r[1]) / r[1] - (r[4] - r[0]) / r[0]), w.index[-1])
    if (w := _ventana(piv_f, ["EarningsPerShareDiluted"], 8)) is not None:
        ep = w["EarningsPerShareDiluted"].iloc[:4].sum()
        if ep > 0:
            res["crecimiento_eps"] = (float(w["EarningsPerShareDiluted"].iloc[4:].sum() / ep - 1.0), w.index[-1])
    return res


def calcular_metricas_ticker(df: pd.DataFrame, sesiones: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    """Métricas de un ticker en las sesiones: recalcula en cada fecha de presentación f y rige desde f+1 día."""
    puntos = {}
    for f in df["filed"].drop_duplicates():
        puntos[f.normalize() + pd.Timedelta(days=1)] = metricas_en_vista(*vista_punto_en_el_tiempo(df, f))
    res = {}
    for m in METRICAS:
        valor = pd.Series({t: p[m][0] if m in p else np.nan for t, p in puntos.items()}, index=list(puntos), dtype=float)
        fin = pd.Series({t: p[m][1] if m in p else pd.NaT for t, p in puntos.items()}, index=list(puntos), dtype="datetime64[ns]")
        v, e = valor.reindex(sesiones, method="ffill"), fin.reindex(sesiones, method="ffill")
        res[m] = v.where((v.index.to_series() - e).dt.days <= MAX_ANTIGUEDAD_DIAS)
    return res


def crear_snapshot_fundamentales(
    nombre: str, tickers: List[str], base_dir: Optional[Path] = None, delay: float = 0.2, downloader: Optional[Callable[[int], dict]] = None,
) -> dict:
    """Descarga facts de sec.gov para cada ticker con CIK y congela un snapshot versionado."""
    ciks, frames, sin_cik, con_cik, fallidos = mapa_tickers_cik(), {}, [], {}, []
    downloader_fn = downloader or descargar_facts_cik
    for t in sorted(set(tickers)):
        cik = ciks.get(t.upper()) or ciks.get(t.upper().replace(".", "-"))
        if cik is None:
            sin_cik.append(t)
        else:
            con_cik[t] = cik
    for i, (t, cik) in enumerate(sorted(con_cik.items())):
        try:
            df = pd.concat([extraer_hechos(t, downloader_fn(c)) for c in [cik, *CIK_PREVIOS.get(t, [])]], ignore_index=True)
            if not df.empty:
                df.index = pd.to_datetime(df["end"])
                frames[t] = df
        except Exception as exc:
            fallidos.append(f"{t}: {type(exc).__name__}")
        if delay > 0 and downloader is None and i < len(con_cik) - 1:
            time.sleep(delay)
    return write_snapshot(nombre, frames, source="sec-edgar", timeframe="quarterly", base_dir=base_dir, extra={"sin_cik": sin_cik, "fallidos": fallidos, "tickers_solicitados": len(tickers)})


def leer_fundamentales(nombre: str, ticker: str, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Lee el Parquet de fundamentales de un ticker sin ajuste de dividendos."""
    return read_snapshot(nombre, ticker, base_dir=base_dir, adjusted=False)


def metricas_fundamentales(nombre: str, tickers: List[str], sesiones: pd.DatetimeIndex, base_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Genera paneles (fecha × ticker) para cada métrica de METRICAS."""
    manifest = load_manifest(nombre, base_dir)
    simbolos = set(manifest["symbols"])
    res = {m: pd.DataFrame(np.nan, index=sesiones, columns=tickers, dtype=float) for m in METRICAS}
    for t in tickers:
        if t in simbolos:
            m_t = calcular_metricas_ticker(preparar_ticker(leer_fundamentales(nombre, t, base_dir=base_dir)), sesiones)
            for m_name in res:
                res[m_name][t] = m_t[m_name]
    return res


def reporte_cobertura_fundamentales(
    nombre: str, composicion: str = "sp500-2026-10", desde: str = "2010-01-01", hasta: Optional[str] = None, base_dir: Optional[Path] = None,
) -> dict:
    """Calcula la cobertura por métrica (% miembros-día desde 2010), tickers sin CIK y valores de muestra."""
    import pandas_market_calendars as mcal

    from quantagent.data.sp500 import miembros

    manifest = load_manifest(nombre, base_dir)
    sin_cik = manifest.get("sin_cik", [])
    nyse = mcal.get_calendar("NYSE")
    fin = pd.Timestamp(hasta) if hasta else pd.Timestamp.now()
    sesiones = nyse.schedule(pd.Timestamp(desde), fin).index.normalize()
    miembros_ses = [set(miembros(composicion, d, base_dir)) for d in sesiones]
    universo = sorted(set().union(*miembros_ses))
    paneles = metricas_fundamentales(nombre, universo, sesiones, base_dir=base_dir)
    total_md = sum(len(m) for m in miembros_ses)
    cobertura = {}
    for met, df_met in paneles.items():
        con_dato = sum(int((~df_met.loc[d, list(m_set)].isna()).sum()) for d, m_set in zip(sesiones, miembros_ses))
        cobertura[met] = {"miembro_dias_con_dato": con_dato, "total_miembro_dias": total_md, "pct": round(con_dato / total_md * 100.0, 2) if total_md > 0 else 0.0}
    muestra = {}
    for sym in ["AAPL", "JPM", "XOM"]:
        if sym in universo:
            muestra[sym] = {f: {m: (round(float(paneles[m].loc[pd.Timestamp(f), sym]), 4) if pd.Timestamp(f) in sesiones and not pd.isna(paneles[m].loc[pd.Timestamp(f), sym]) else None) for m in paneles} for f in ["2015-06-30", "2020-06-30"]}
    return {"cobertura": cobertura, "sin_cik": {"cantidad": len(sin_cik), "ejemplos": sin_cik[:10]}, "muestra": muestra}
