"""H3 de M5 (QuantAgent-fnpr.4.4): H1 con las acciones del screener en lugar de cada ETF sectorial, 2011-2018.

Calcula los pesos semanales de H1 con las mismas piezas del embudo (`señales_transversales`, `pesos`), los pasa por
`pesos_h3` con `candidatos_semanales_panel` y mide H3 y H1 con la misma curva (llenado a la apertura de la sesión
siguiente a la decisión, costo `etf` sobre el volumen operado). Registra etapa 1 y 2 en hipotesis.csv e intentos.csv.

Uso: python scripts/m5/h3.py [--cache RUTA] [--sin-registro]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from cobertura_screener import ACC, COMP, FUND, clasificar

import quantagent.data.sp500 as sp500
import quantagent.fabrica.screener as scr
from quantagent.backtesting.vbt_filter import equity_metrics
from quantagent.backtesting.verdict import evaluar
from quantagent.cli.sweep import registrar_intento
from quantagent.data.fundamentales import metricas_fundamentales
from quantagent.data.snapshot import load_manifest, read_snapshot
from quantagent.fabrica.cartera import pesos, pesos_h3
from quantagent.fabrica.criterio import anios_positivos, error_estandar_delta_sharpe
from quantagent.fabrica.estado import registrar_estado
from quantagent.fabrica.transversal import MASCARAS, METRICAS, REGLAS, _ultima_sesion_de_semana, señales_transversales
from quantagent.settings import COSTOS_PERFILES

ETF_SNAP, DESDE, HASTA, ID = "etf-m5-1d-2026-10", pd.Timestamp("2011-01-01"), pd.Timestamp("2018-12-31"), "m5-h3"
ACTIVOS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SPY", "EEM", "EWZ"]


def pesos_h1():
    """Pesos de H1 por sesión (fecha × ETF) desde 2006, y fechas de rebalanceo con su sesión de decisión."""
    velas = {a: read_snapshot(ETF_SNAP, a).loc[:HASTA] for a in ACTIVOS}
    cierres = pd.DataFrame({a: v["close"] for a, v in velas.items()})
    panel = señales_transversales(cierres, METRICAS["sector-score-h1"], mascara=MASCARAS["sin-caida-m5"], **REGLAS["sector-score-h1"])
    w = pesos(cierres, {0: ACTIVOS}, seleccion={0: panel}, semanales={0}).fillna(0.0)
    w.columns = w.columns.droplevel(0)
    semana = cierres.index.to_series().apply(lambda d: d.isocalendar()[:2])
    primeras = cierres.index[(semana != semana.shift()).to_numpy()]
    previa = cierres.index.to_series().shift(1)
    assert all(_ultima_sesion_de_semana(previa[d]) for d in primeras[1:]), "decisión no es la última sesión de la semana"
    return velas, w, {d: previa[d] for d in primeras if DESDE <= d <= HASTA}


def candidatos(decisiones, etfs_por_fecha, cache):
    if cache and Path(cache).exists():
        return pickle.loads(Path(cache).read_bytes())
    fechas = sorted(set(decisiones))
    miembros = sorted({t for d in fechas for t in sp500.miembros(COMP, d)})
    en_fund, en_acc = set(load_manifest(FUND)["symbols"]), set(load_manifest(ACC)["symbols"])
    precios = {t: read_snapshot(ACC, t)["close"] for t in miembros if t in en_acc}
    precios |= {a: read_snapshot(ETF_SNAP, a)["close"] for a in ACTIVOS}
    paneles = metricas_fundamentales(FUND, [t for t in miembros if t in en_fund], pd.DatetimeIndex(fechas))
    cand = scr.candidatos_semanales_panel(fechas, etfs_por_fecha, precios=precios, paneles_fundamentales=paneles, composicion=COMP)
    if cache:
        Path(cache).write_bytes(pickle.dumps((cand, paneles, precios)))
    return cand, paneles, precios


def curva(w, opens, closes, costo):
    """Equity y volumen operado por sesión: a la apertura se pasa de los pesos de ayer a los de hoy."""
    w = w.reindex(closes.index).ffill().fillna(0.0)
    antes = w.shift(1).fillna(0.0)
    cols = list(w.columns)
    o, c = opens.reindex(columns=cols), closes.reindex(columns=cols)
    ret = (antes * (o / c.shift(1) - 1)).sum(axis=1) + (w * (c / o - 1)).sum(axis=1)
    operado = (w - antes).abs().sum(axis=1)
    return 100_000.0 * (1 + ret.fillna(0.0) - costo * operado).cumprod(), operado


def metricas(eq, operado):
    anios = (eq.index[-1] - eq.index[0]).days / 365.25
    return {**equity_metrics(eq, 0.0), "cagr": (eq.iloc[-1] / eq.iloc[0]) ** (1 / anios) - 1,
            "rotacion_anual": operado.sum() / anios, "anios_positivos": tuple(anios_positivos(eq))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", help="pickle con candidatos, paneles y precios")
    ap.add_argument("--sin-registro", action="store_true")
    args = ap.parse_args()

    velas, w1, decisiones = pesos_h1()
    etfs = {dec: [e for e in scr.ETF_SECTOR if w1.loc[d, e] > 0] for d, dec in decisiones.items()}
    cand, paneles, precios = candidatos(list(decisiones.values()), etfs, args.cache)
    w3 = pesos_h3(w1, cand, decisiones)

    sesiones = w1.index[(w1.index >= DESDE) & (w1.index <= HASTA)]
    acciones = sorted(set(w3.columns) - set(ACTIVOS))
    px = {t: read_snapshot(ACC, t).reindex(sesiones) for t in acciones}
    abre = pd.concat([pd.DataFrame({a: velas[a]["open"] for a in ACTIVOS}).reindex(sesiones)] + [pd.DataFrame({t: px[t]["open"] for t in acciones})], axis=1)
    cierra = pd.concat([pd.DataFrame({a: velas[a]["close"] for a in ACTIVOS}).reindex(sesiones)] + [pd.DataFrame({t: px[t]["close"] for t in acciones})], axis=1)
    costo = COSTOS_PERFILES["etf"]["slippage_pct"] + COSTOS_PERFILES["etf"]["commission_pct"]
    eq1, op1 = curva(w1.loc[decisiones.keys()], abre, cierra, costo)
    eq3, op3 = curva(w3, abre, cierra, costo)
    m1, m3 = metricas(eq1, op1), metricas(eq3, op3)
    delta, error = m3["sharpe"] - m1["sharpe"], error_estandar_delta_sharpe(eq3, eq1)
    trades = int((w3.diff().fillna(w3).abs() > 1e-12).sum().sum())
    pasa, motivos = evaluar(trades, float(eq3.iloc[-1] - 100_000.0), m3["sharpe"], m3["max_drawdown"])

    hay = w3.drop(columns=ACTIVOS, errors="ignore").gt(0)
    n_acc = hay.sum(axis=1)
    sectoriales = [e for e in scr.ETF_SECTOR if e in w3.columns]
    semanas_etf = (w3[sectoriales] > 0).any(axis=1)
    sectores_etf = int((w3[sectoriales] > 0).sum().sum())
    sectores_elegidos = sum(len(v) for v in etfs.values())
    sin_px = int(sum(px[t]["close"].isna().all() for t in acciones))
    print(f"H1  {m1}\nH3  {m3}\nΔSharpe {delta:+.3f} ± {error:.3f}  trades {trades}  V30 {'PASA' if pasa else motivos}")
    print(f"semanas {len(w3)}  acciones/semana {n_acc.mean():.2f} (solo semanas con alguna {n_acc[n_acc > 0].mean():.2f})  "
          f"semanas en efectivo {(w3.sum(axis=1) == 0).sum()}  semanas con algún ETF sectorial de respaldo {semanas_etf.sum()}  "
          f"sector-semanas de respaldo {sectores_etf} de {sectores_elegidos}  acciones elegidas sin precio {sin_px} de {len(acciones)}")

    if not args.sin_registro:
        sharpe = round(m3["sharpe"], 2)
        registrar_estado(ID, "1", "PASA", f"Sharpe de cartera {sharpe}")
        veredicto = "PASA" if pasa else "CORTA"
        registrar_estado(ID, "2", veredicto, f"cartera pasa V30; Sharpe {sharpe} contra H1 {m1['sharpe']:.2f} (Δ {delta:+.2f} ± {error:.2f}); sin control del motor" if pasa
                         else "; ".join(f"cartera: {m}" for m in motivos))
        registrar_intento("sector-score-h3", "cartera", ETF_SNAP, "etf", 1.0, {},
                          {"desde": DESDE.date().isoformat(), "hasta": HASTA.date().isoformat(), "trades": trades,
                           "pnl": round(float(eq3.iloc[-1] - 100_000.0), 2), "sharpe": sharpe},
                          hipotesis=ID, etapa="2", sharpe_base="tasa_0")

    simbolos_fund, previos = set(load_manifest(FUND)["symbols"]), {t: s for t, s in precios.items() if t not in ACTIVOS}
    todos, completo = [], dict(scr.ETF_SECTOR)
    for dec, es in etfs.items():  # `clasificar` recorre todos los sectores: se restringe a los elegidos esa semana
        scr.ETF_SECTOR.clear()
        scr.ETF_SECTOR.update({e: completo[e] for e in es})
        todos.append(clasificar([dec], paneles, simbolos_fund, previos))
    scr.ETF_SECTOR.update(completo)
    en_sector = pd.concat(todos, ignore_index=True)
    print("miembros-fecha de los sectores elegidos por causa (%):")
    print((en_sector["causa"].value_counts(normalize=True) * 100).round(1).to_string(), f"\nn = {len(en_sector)}")
    print((pd.crosstab(en_sector["fecha"].dt.year, en_sector["causa"], normalize="index") * 100).round(0).to_string())
    print((pd.crosstab(en_sector["sector"], en_sector["causa"], normalize="index") * 100).round(0).to_string())
    print("elegidos por sector (sector-semanas):", pd.Series([scr.ETF_SECTOR[e] for es in etfs.values() for e in es]).value_counts().to_dict())


if __name__ == "__main__":
    main()
