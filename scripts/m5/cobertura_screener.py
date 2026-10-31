"""Cobertura del screener M5-32 por sector y fin de trimestre (QuantAgent-fnpr.4.8).

Lee snapshots locales. Las métricas fundamentales se calculan solo en las fechas pedidas (una vista
punto-en-el-tiempo por presentación vigente) y las funciones del screener (`_retorno`, `_metrica_fund`)
deciden qué es NaN, así que la clasificación sigue la regla de `candidatos_semanales`.

Uso: python scripts/m5/cobertura_screener.py [--csv RUTA] [--control]
"""

from __future__ import annotations

import argparse
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

import quantagent.data.sp500 as sp500
import quantagent.fabrica.screener as scr
from quantagent.data.fundamentales import (
    MAX_ANTIGUEDAD_DIAS,
    METRICAS,
    leer_fundamentales,
    metricas_en_vista,
    preparar_ticker,
    vista_punto_en_el_tiempo,
)
from quantagent.data.snapshot import load_manifest, read_snapshot

FUND, ACC, COMP = "fundamentales-sp500-2026-10", "acciones-sp500-2026-10b", "sp500-2026-10"
CSV_DEFECTO = "docs/06_implementation/QuantAgent-fnpr.4.8-IM-cobertura-screener.csv"


def fechas_trimestrales(desde="2010-03-01", hasta="2018-12-31") -> list[pd.Timestamp]:
    ses = mcal.get_calendar("NYSE").schedule(desde, hasta).index.normalize()
    return list(ses.to_series().groupby(ses.to_period("Q")).max())


def _metricas_ticker(args) -> tuple[str, dict]:
    """{(métrica, fecha): valor}; igual que calcular_metricas_ticker pero solo en `fechas`."""
    t, fechas = args
    df = preparar_ticker(leer_fundamentales(FUND, t))
    rige = df["filed"].dt.normalize() + pd.Timedelta(days=1)
    vistas, out = {}, {}
    for d in fechas:
        previas = df.loc[rige <= d, "filed"]
        if previas.empty:
            continue
        f = previas.max()
        if f not in vistas:
            vistas[f] = metricas_en_vista(*vista_punto_en_el_tiempo(df, f))
        for m, (val, fin) in vistas[f].items():
            if (d - fin).days <= MAX_ANTIGUEDAD_DIAS:
                out[(m, d)] = val
    return t, out


def paneles_en_fechas(tickers: list[str], fechas: list[pd.Timestamp]) -> dict[str, pd.DataFrame]:
    paneles = {m: pd.DataFrame(np.nan, index=pd.DatetimeIndex(fechas), columns=tickers, dtype=float) for m in METRICAS}
    with Pool(4) as pool:
        for t, out in pool.imap_unordered(_metricas_ticker, [(t, fechas) for t in tickers], chunksize=8):
            for (m, d), v in out.items():
                paneles[m].loc[d, t] = v
    return paneles


def clasificar(fechas, paneles, simbolos_fund: set[str], precios: dict) -> pd.DataFrame:
    """Una fila por (fecha, sector, miembro) con su causa de exclusión y las métricas que le faltan."""
    filas = []
    for d in fechas:
        miembros = sp500.miembros(COMP, d)
        for etf, sec in scr.ETF_SECTOR.items():
            s_etf = scr._obtener_precios(None, etf, es_etf=True)
            r63e, r126e = scr._retorno(s_etf, d, 63), scr._retorno(s_etf, d, 126)
            m_q1, m_q2 = ("roe_4t", "margen_pretax_yoy") if sec == "Financials" else ("roic_4t", "margen_op_yoy")
            for t in (x for x in miembros if sp500.sector(COMP, x, d) == sec):
                s_t = precios.get(t)
                crudos = {m_q1: scr._metrica_fund(paneles, m_q1, d, t), m_q2: scr._metrica_fund(paneles, m_q2, d, t),
                          "aceleracion_ingresos": scr._metrica_fund(paneles, "aceleracion_ingresos", d, t),
                          "crecimiento_eps": scr._metrica_fund(paneles, "crecimiento_eps", d, t),
                          "rs3": scr._retorno(s_t, d, 63) - r63e, "rs6": scr._retorno(s_t, d, 126) - r126e}
                faltan = [k for k, v in crudos.items() if np.isnan(v)]
                if t not in simbolos_fund:
                    causa = "sin_cik"
                elif t not in precios or np.isnan(crudos["rs3"]):
                    causa = "sin_precio"
                elif len(faltan) >= 2:
                    causa = "faltan_metricas"
                else:
                    causa = "elegible"
                filas.append({"fecha": d, "sector": sec, "simbolo": t, "causa": causa, "faltan": faltan})
    return pd.DataFrame(filas)


def resumen(det: pd.DataFrame) -> pd.DataFrame:
    g = det.groupby(["fecha", "sector"])
    out = pd.DataFrame({"miembros": g.size()})
    for c in ["elegible", "sin_cik", "sin_precio", "faltan_metricas"]:
        out["elegibles" if c == "elegible" else c] = g["causa"].apply(lambda s, c=c: int((s == c).sum()))
    out["cobertura"] = (out["elegibles"] / out["miembros"]).round(4)
    out = out.reset_index()
    out["fecha"] = out["fecha"].dt.date
    return out[["fecha", "sector", "miembros", "elegibles", "sin_cik", "sin_precio", "faltan_metricas", "cobertura"]]


def control_cruzado(fecha, paneles, precios) -> int:
    """Miembros de XLK+XLF que pasan el filtro de datos en `candidatos_semanales` (largo de la serie del cuantil de FR)."""
    largos = []

    class _Pd:
        def __getattr__(self, k):
            return getattr(pd, k)

        @staticmethod
        def Series(datos, *a, **k):
            largos.append(len(datos))
            return pd.Series(datos, *a, **k)

    original, scr.pd = scr.pd, _Pd()
    try:
        scr.candidatos_semanales(fecha, ["XLK", "XLF"], precios=precios, paneles_fundamentales=paneles)
    finally:
        scr.pd = original
    return sum(largos)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_DEFECTO)
    ap.add_argument("--control", action="store_true", help="control cruzado 2018-06-29 XLK+XLF")
    args = ap.parse_args()

    fechas = fechas_trimestrales()
    miembros_tot = sorted({t for d in fechas for t in sp500.miembros(COMP, d)})
    simbolos_fund = set(load_manifest(FUND)["symbols"])
    paneles = paneles_en_fechas([t for t in miembros_tot if t in simbolos_fund], fechas)
    simbolos_acc = set(load_manifest(ACC)["symbols"])
    px = {t: read_snapshot(ACC, t) for t in miembros_tot if t in simbolos_acc}
    precios = {t: df["close"] for t, df in px.items()}
    det = clasificar(fechas, paneles, simbolos_fund, precios)

    res = resumen(det)
    assert (res[["elegibles", "sin_cik", "sin_precio", "faltan_metricas"]].sum(axis=1) == res["miembros"]).all()
    res.to_csv(args.csv, index=False)
    res["anio"] = pd.to_datetime(res["fecha"]).dt.year
    print("== cobertura anual (promedio por fecha, %) ==")
    tabla = (res.groupby(["sector", "anio"])["cobertura"].mean().unstack() * 100).round(0)
    print(tabla.astype("Int64").to_markdown())
    print("== sector-años < 60% ==")
    print(", ".join(f"{s} {a} ({v:.0f}%)" for (s, a), v in tabla.stack().items() if v < 60))
    print("== causas por sector (miembro-fecha) ==")
    print(pd.crosstab(det["sector"], det["causa"]).to_markdown())
    print("== 3 métricas que más faltan por sector (causa faltan_metricas) ==")
    for sec, sub in det[det["causa"] == "faltan_metricas"].groupby("sector"):
        print(sec, Counter(m for fl in sub["faltan"] for m in fl).most_common(3))
    print("== grandes excluidas (volumen en dólares medio de 63 sesiones, top 10 por ticker) ==")
    exc = det[det["causa"] != "elegible"].drop_duplicates("simbolo", keep="last")
    usd = {t: float((px[t]["close"] * px[t]["volume"]).tail(63).mean()) for t in exc["simbolo"] if t in px}
    exc = exc[exc["simbolo"].isin(usd)].assign(usd=lambda x: x["simbolo"].map(usd)).nlargest(10, "usd")
    print(exc[["simbolo", "sector", "fecha", "causa", "faltan", "usd"]].to_string(index=False))
    if args.control:
        d = pd.Timestamp("2018-06-29")
        sub = det[(det["fecha"] == d) & det["sector"].isin(["Information Technology", "Financials"])]
        print("== control 2018-06-29 XLK+XLF ==")
        print("miembros", len(sub), sub["causa"].value_counts().to_dict())
        print("filas que pasan el filtro en candidatos_semanales:", control_cruzado(d, paneles, precios))


if __name__ == "__main__":
    main()
