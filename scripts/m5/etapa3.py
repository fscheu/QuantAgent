"""M5-36: H1/H3 2019–2022, sin abrir 2023+. Ejecutar con PYTHONPATH=.:scripts/m5."""

import argparse
import csv
import pickle
from pathlib import Path

import h3
import pandas as pd
from cobertura_screener import paneles_en_fechas

from quantagent.backtesting.verdict import dsr_contra_intentos
from quantagent.data.snapshot import snapshot_root
from quantagent.fabrica.candado import abrir, fecha_apertura
from quantagent.fabrica.criterio import evaluar_etapa3_cartera, sharpe_benchmark

DESDE, HASTA = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31")
ACC = "acciones-sp500-2026-10c"


def contar_sesgo(detalle, pesos, decisiones):
    """Descartes por precio y cota de exposición: el top 3 contrafactual es desconocido.

    Solo cuenta miembros que pasarían el filtro de fundamentales (máximo un NaN).
    Si falta momentum, podrían ocupar hasta tres puestos; el ranking completo
    también cambia por los z-scores y el cuartil. No se imputan retornos.
    """
    filas = []
    for fecha, decision in decisiones.items():
        for etf, sector in h3.scr.ETF_SECTOR.items():
            peso = pesos.loc[fecha, etf]
            sub = detalle[(detalle.fecha == decision) & (detalle.sector == sector)]
            faltantes = sum(len(f) > 1 and len([m for m in f if m not in ("rs3", "rs6")]) <= 1
                            and any(m in f for m in ("rs3", "rs6")) for f in sub.faltan)
            filas.append(dict(fecha=fecha, anio=fecha.year, sector=sector, peso=peso,
                              descartes=faltantes if peso > 0 else 0,
                              puestos_max=min(3, faltantes) if peso > 0 else 0,
                              peso_max=peso if faltantes else 0.0))
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/tmp/ai-loop/h3-etapa3-2019-2022.pkl", help="Cache exclusivo de etapa 3")
    ap.add_argument("--sin-registro", action="store_true", help="Reproducir sin agregar intentos")
    args = ap.parse_args()
    if not args.sin_registro:
        for nombre in ("m5-h1", "m5-h3"):
            if fecha_apertura(nombre, "prueba") is None:
                abrir(nombre, "prueba")
    velas, w1, decisiones = h3.pesos_h1(DESDE, HASTA)
    etfs = {dec: [e for e in h3.scr.ETF_SECTOR if w1.loc[d, e] > 0] for d, dec in decisiones.items()}
    cache = Path(args.cache)
    if cache.exists():
        datos = pickle.loads(cache.read_bytes())
        assert datos[0] == (DESDE, HASTA, ACC), "Cache de otro período/snapshot"
        _, cand, detalle = datos
    else:
        fechas = sorted(decisiones.values())
        miembros = sorted({t for d in fechas for t in h3.sp500.miembros(h3.COMP, d)})
        fund = set(h3.load_manifest(h3.FUND)["symbols"])
        paneles = paneles_en_fechas([t for t in miembros if t in fund], fechas)
        precios = {t: None for t in miembros}  # Impide el fallback implícito al snapshot 10b.
        for snap in (h3.ACC, ACC):
            disponibles = set(h3.load_manifest(snap)["symbols"])
            for t in miembros:
                if t in disponibles:
                    s = h3.read_snapshot(snap, t)["close"].loc[:("2018-12-31" if snap == h3.ACC else HASTA)]
                    precios[t] = pd.concat([precios[t], s]) if precios[t] is not None else s
        precios.update({e: v["close"] for e, v in velas.items()})
        cand = h3.scr.candidatos_semanales_panel(fechas, etfs, precios, paneles, h3.COMP)
        # Clasificación con las mismas reglas, restringida a los sectores elegidos.
        partes, sectores = [], dict(h3.scr.ETF_SECTOR)
        for d in fechas:
            h3.scr.ETF_SECTOR.clear()
            h3.scr.ETF_SECTOR.update({e: sectores[e] for e in etfs[d]})
            partes.append(h3.clasificar([d], paneles, fund, {t: s for t, s in precios.items() if s is not None}))
        h3.scr.ETF_SECTOR.update(sectores)
        detalle = pd.concat(partes, ignore_index=True)
        cache.write_bytes(pickle.dumps(((DESDE, HASTA, ACC), cand, detalle)))
    w3 = h3.pesos_h3(w1, cand, decisiones)
    sesiones = w1.loc[DESDE:HASTA].index
    velas.update({t: h3.read_snapshot(ACC, t).loc[:HASTA] for t in set(w3.columns) - set(h3.ACTIVOS)})
    abre, cierre = [pd.DataFrame({t: v[c] for t, v in velas.items()}).reindex(sesiones) for c in ("open", "close")]
    diario = w3.reindex(sesiones, method="ffill").fillna(0)
    assert not ((diario > 0) & (abre[diario.columns].isna() | cierre[diario.columns].isna())).any().any(), "Posición sin precio"
    costo = sum(h3.COSTOS_PERFILES["etf"].values())
    eq1, op1 = h3.curva(w1.loc[list(decisiones)], abre, cierre, costo)
    eq3, op3 = h3.curva(w3, abre, cierre, costo)
    pd.DataFrame({"H1": eq1, "H3": eq3}).to_csv(cache.with_suffix(".equity.csv"))
    delta = h3.metricas(eq3, op3)["sharpe"] - h3.metricas(eq1, op1)["sharpe"]
    print(f"ΔSharpe H3-H1 {delta:+.6f} ± {h3.error_estandar_delta_sharpe(eq3, eq1):.6f}")
    bench2 = sharpe_benchmark(pd.DataFrame({e: velas[e]["close"] for e in h3.ACTIVOS}).loc[h3.DESDE:h3.HASTA])
    bench3 = sharpe_benchmark(cierre[h3.ACTIVOS])
    for nombre, eq, op, w, b3, s2, b2 in (("m5-h1", eq1, op1, w1.loc[list(decisiones)], bench3, .61, bench2),
                                          ("m5-h3", eq3, op3, w3, h3.metricas(eq1, op1)["sharpe"], .65, .61)):
        trades = int(w.diff().fillna(w).abs().gt(1e-12).sum().sum())
        pasa, motivos = evaluar_etapa3_cartera(eq, b3, s2, b2, trades=trades)
        veredicto = "PASA" if pasa else "CORTA"
        costos = float((eq.shift().fillna(100_000) * op * costo).sum())
        print(nombre, h3.metricas(eq, op), "trades", trades, "costos", costos, "bench2/3", b2, b3, veredicto, motivos)
        if not args.sin_registro:
            path = snapshot_root() / "intentos.csv"
            with path.open() as f:
                registrado = any(r["hipotesis"] == nombre and r["etapa"] == "3" for r in csv.DictReader(f))
            if not registrado:
                h3.registrar_estado(nombre, "3", veredicto, "; ".join(motivos) or "cumple criterio cartera etapa 3")
                h3.registrar_intento("sector-score-" + nombre[3:], "cartera", h3.ETF_SNAP, "etf", 1., {},
                                    dict(desde=str(DESDE.date()), hasta=str(HASTA.date()), trades=trades,
                                         pnl=float(eq.iloc[-1] - 100_000), sharpe=h3.metricas(eq, op)["sharpe"]),
                                    hipotesis=nombre, etapa="3", sharpe_base="tasa_0")
    path = snapshot_root() / "intentos.csv"
    if path.exists():
        print("DSR informativo", {n: dsr_contra_intentos(eq, path) for n, eq in (("H1", eq1), ("H3", eq3))})
    sesgo = contar_sesgo(detalle, w1, decisiones)
    sesgo.to_csv(cache.with_suffix(".sesgo.csv"), index=False)
    for clave in ("anio", "sector"):
        for k, sub in sesgo.groupby(clave):
            n = sub.fecha.nunique()
            print(k, "descartes", sub.descartes.sum(), "puestos_max", sub.puestos_max.sum(),
                  "semanas", sub.loc[sub.peso_max > 0, "fecha"].nunique(), "/", n,
                  "peso_max_promedio", sub.peso_max.sum() / n)
    n = len(decisiones)
    print("TOTAL", sesgo.descartes.sum(), sesgo.puestos_max.sum(), sesgo.loc[sesgo.peso_max > 0, "fecha"].nunique(),
          "/", n, "peso_max_promedio", sesgo.peso_max.sum() / n)
    print(sesgo.groupby(["sector", "anio"]).agg(descartes=("descartes", "sum"),
          puestos_max=("puestos_max", "sum"), semanas=("peso_max", lambda s: int(s.gt(0).sum())),
          peso_max=("peso_max", "mean")).to_string())


if __name__ == "__main__":
    main()
