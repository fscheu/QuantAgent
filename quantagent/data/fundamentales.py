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
    "ingresos": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "OperatingIncomeLoss": ["OperatingIncomeLoss"], "EarningsPerShareDiluted": ["EarningsPerShareDiluted"],
    "StockholdersEquity": ["StockholdersEquity"], "deuda": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "IncomeTaxExpenseBenefit": ["IncomeTaxExpenseBenefit"], "IncomeLossFromContinuingOperationsBeforeIncomeTaxes": _TAX_BT,
}
FORMULARIOS_VALIDOS = frozenset({"10-Q", "10-Q/A", "10-K", "10-K/A"})
CONCEPTOS_FLUJO = frozenset({"ingresos", "OperatingIncomeLoss", "EarningsPerShareDiluted", "IncomeTaxExpenseBenefit", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"})


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


def procesar_ticker(df_ticker: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplica según primer publicado y deriva Q4 = FY - 9M."""
    if df_ticker.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = df_ticker.copy()
    df["filed"], df["end_dt"], df["start_dt"] = pd.to_datetime(df["filed"]), pd.to_datetime(df["end"]), pd.to_datetime(df["start"])
    df["days"] = (df["end_dt"] - df["start_dt"]).dt.days.fillna(0)
    df = df.sort_values(by=["filed", "accn"]).drop_duplicates(subset=["concepto", "start", "end"], keep="first").reset_index(drop=True)

    filas_q4 = []
    flujos = df[df["concepto"].isin(CONCEPTOS_FLUJO)]
    for (concepto, fy), grupo in flujos.groupby(["concepto", "fy"]):
        if pd.isna(fy):
            continue
        anuales = grupo[(grupo["fp"] == "FY") | (grupo["days"].between(330, 395))]
        nueve_m = grupo[(grupo["fp"] == "Q3") | (grupo["days"].between(240, 305))]
        if not anuales.empty and not nueve_m.empty:
            anual, nueve = anuales.iloc[0], nueve_m.iloc[0]
            filas_q4.append({
                "ticker": anual["ticker"], "concepto": concepto, "start": nueve["end"], "end": anual["end"],
                "val": anual["val"] - nueve["val"], "fy": fy, "fp": "Q4", "form": anual["form"],
                "filed": max(anual["filed"], nueve["filed"]), "accn": anual["accn"],
                "end_dt": anual["end_dt"], "start_dt": nueve["end_dt"], "days": (anual["end_dt"] - nueve["end_dt"]).days,
            })
    df = pd.concat([df, pd.DataFrame(filas_q4)], ignore_index=True) if filas_q4 else df
    flujos_trim = df[df["concepto"].isin(CONCEPTOS_FLUJO) & (df["days"].between(60, 115) | (df["fp"] == "Q4"))].sort_values("filed").drop_duplicates(subset=["concepto", "end"], keep="first")
    balances = df[df["concepto"].isin(["StockholdersEquity", "deuda"])].sort_values("filed").drop_duplicates(subset=["concepto", "end"], keep="first")
    return flujos_trim, balances


def calcular_metricas_ticker(flujos: pd.DataFrame, balances: pd.DataFrame, sesiones: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    """Calcula las 4 métricas para un ticker sobre las sesiones dadas con reindex ffill."""
    vacio = {m: pd.Series(np.nan, index=sesiones, dtype=float) for m in ["roic_4t", "margen_op_yoy", "aceleracion_ingresos", "crecimiento_eps"]}
    if flujos.empty:
        return vacio
    piv_f, filed_f = flujos.pivot(index="end_dt", columns="concepto", values="val").sort_index(), flujos.groupby("end_dt")["filed"].max()
    piv_b = balances.pivot(index="end_dt", columns="concepto", values="val").sort_index() if not balances.empty else pd.DataFrame()
    filed_b = balances.groupby("end_dt")["filed"].max() if not balances.empty else pd.Series(dtype="datetime64[ns]")

    trimestres, puntos = piv_f.index, []
    for i in range(len(trimestres)):
        max_filed = max([d for d in [filed_f.get(trimestres[i]), filed_b.get(trimestres[i])] if pd.notna(d)], default=pd.NaT)
        if pd.isna(max_filed):
            continue
        t_vis = max_filed.normalize() + pd.Timedelta(days=1)
        sub_f, n_trim = piv_f.iloc[:i + 1], i + 1
        roic, margen_yoy, acel, eps_crec = np.nan, np.nan, np.nan, np.nan

        if n_trim >= 4 and "OperatingIncomeLoss" in sub_f.columns:
            op_4t = sub_f["OperatingIncomeLoss"].iloc[-4:].sum()
            tax = sub_f["IncomeTaxExpenseBenefit"].iloc[-4:].sum() if "IncomeTaxExpenseBenefit" in sub_f.columns else np.nan
            ebt = sub_f["IncomeLossFromContinuingOperationsBeforeIncomeTaxes"].iloc[-4:].sum() if "IncomeLossFromContinuingOperationsBeforeIncomeTaxes" in sub_f.columns else np.nan
            tasa = 0.21 if (pd.isna(tax) or pd.isna(ebt) or ebt <= 0) else max(0.0, min(0.5, float(tax / ebt)))
            eq = piv_b.loc[piv_b.index <= trimestres[i], "StockholdersEquity"].dropna().iloc[-1] if not piv_b.empty and "StockholdersEquity" in piv_b.columns and not piv_b.loc[piv_b.index <= trimestres[i], "StockholdersEquity"].dropna().empty else np.nan
            deuda = piv_b.loc[piv_b.index <= trimestres[i], "deuda"].dropna().iloc[-1] if not piv_b.empty and "deuda" in piv_b.columns and not piv_b.loc[piv_b.index <= trimestres[i], "deuda"].dropna().empty else 0.0
            if not pd.isna(eq) and (eq + deuda) > 0:
                roic = float(op_4t * (1.0 - tasa) / (eq + deuda))

        if n_trim >= 8 and "OperatingIncomeLoss" in sub_f.columns and "ingresos" in sub_f.columns:
            rh, oh, rp, op = sub_f["ingresos"].iloc[-4:].sum(), sub_f["OperatingIncomeLoss"].iloc[-4:].sum(), sub_f["ingresos"].iloc[-8:-4].sum(), sub_f["OperatingIncomeLoss"].iloc[-8:-4].sum()
            if rh > 0 and rp > 0:
                margen_yoy = float((oh / rh) - (op / rp))

        if n_trim >= 6 and "ingresos" in sub_f.columns:
            r0, r4, r1, r5 = sub_f["ingresos"].iloc[-1], sub_f["ingresos"].iloc[-5], sub_f["ingresos"].iloc[-2], sub_f["ingresos"].iloc[-6]
            if r4 > 0 and r5 > 0:
                acel = float(((r0 - r4) / r4) - ((r1 - r5) / r5))

        if n_trim >= 8 and "EarningsPerShareDiluted" in sub_f.columns:
            eh, ep = sub_f["EarningsPerShareDiluted"].iloc[-4:].sum(), sub_f["EarningsPerShareDiluted"].iloc[-8:-4].sum()
            if ep > 0:
                eps_crec = float((eh / ep) - 1.0)

        puntos.append({"fecha": t_vis, "roic_4t": roic, "margen_op_yoy": margen_yoy, "aceleracion_ingresos": acel, "crecimiento_eps": eps_crec})

    if not puntos:
        return vacio
    df_p = pd.DataFrame(puntos).drop_duplicates(subset=["fecha"], keep="last").set_index("fecha").sort_index()
    res = {}
    for col in ["roic_4t", "margen_op_yoy", "aceleracion_ingresos", "crecimiento_eps"]:
        s = df_p[col].dropna()
        res[col] = pd.Series(np.nan, index=sesiones, dtype=float) if s.empty else df_p[col].reindex(df_p.index.union(sesiones)).ffill().reindex(sesiones)
    return res


def crear_snapshot_fundamentales(
    nombre: str, tickers: List[str], base_dir: Optional[Path] = None, delay: float = 0.2, downloader: Optional[Callable[[int], dict]] = None,
) -> dict:
    """Descarga facts de sec.gov para cada ticker con CIK y congela un snapshot versionado."""
    ciks, frames, sin_cik, con_cik = mapa_tickers_cik(), {}, [], {}
    downloader_fn = downloader or descargar_facts_cik
    for t in sorted(set(tickers)):
        cik = ciks.get(t.upper())
        if cik is None:
            sin_cik.append(t)
        else:
            con_cik[t] = cik
    for i, (t, cik) in enumerate(sorted(con_cik.items())):
        try:
            df = extraer_hechos(t, downloader_fn(cik))
            if not df.empty:
                df.index = pd.to_datetime(df["end"])
                frames[t] = df
        except Exception:
            pass
        if delay > 0 and downloader is None and i < len(con_cik) - 1:
            time.sleep(delay)
    return write_snapshot(nombre, frames, source="sec-edgar", timeframe="quarterly", base_dir=base_dir, extra={"sin_cik": sin_cik, "tickers_solicitados": len(tickers)})


def leer_fundamentales(nombre: str, ticker: str, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Lee el Parquet de fundamentales de un ticker sin ajuste de dividendos."""
    return read_snapshot(nombre, ticker, base_dir=base_dir, adjusted=False)


def metricas_fundamentales(nombre: str, tickers: List[str], sesiones: pd.DatetimeIndex, base_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Genera paneles (fecha × ticker) para roic_4t, margen_op_yoy, aceleracion_ingresos y crecimiento_eps."""
    manifest = load_manifest(nombre, base_dir)
    simbolos = set(manifest["symbols"])
    res = {m: pd.DataFrame(np.nan, index=sesiones, columns=tickers, dtype=float) for m in ["roic_4t", "margen_op_yoy", "aceleracion_ingresos", "crecimiento_eps"]}
    for t in tickers:
        if t in simbolos:
            flujos, balances = procesar_ticker(leer_fundamentales(nombre, t, base_dir=base_dir))
            m_t = calcular_metricas_ticker(flujos, balances, sesiones)
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
