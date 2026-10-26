"""Cálculo del criterio de paper de etapa 3 desde curvas guardadas, sin escrituras."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

import quantagent.data.snapshot as snap_mod
from quantagent.backtesting.vbt_filter import equity_metrics
from quantagent.backtesting.verdict import dsr_contra_intentos, n_efectivo
from quantagent.fabrica.cartera import curva_cartera
from quantagent.fabrica.criterio import anios_positivos, evaluar_etapa3_cartera, recargo_paper, sharpe_benchmark
from quantagent.fabrica.hipotesis import Hipotesis, cargar_hipotesis
from quantagent.fabrica.transversal import leer_seleccion


def reconstruir_curva_cartera(h: Hipotesis, carpeta: str | Path, cierres: pd.DataFrame) -> pd.Series:
    """Reconstruye la curva de cartera desde los CSV de equity guardados por pierna y activo."""
    sub = Path(carpeta) / h.id
    base = sub if sub.is_dir() else Path(carpeta)
    eq_dir = base / "equity" if (base / "equity").is_dir() else base
    piernas_map = {i: list(p.activos) for i, p in enumerate(h.piernas)}
    retornos_dict = {}
    for i, p in enumerate(h.piernas):
        for activo in p.activos:
            ruta = eq_dir / f"{p.clase}-{activo}.csv"
            ruta = ruta if ruta.exists() else eq_dir / f"{activo}.csv"
            if not ruta.exists():
                raise FileNotFoundError(f"Curva faltante: {p.clase}-{activo} en {eq_dir}")
            with ruta.open(newline="", encoding="utf-8") as f:
                filas = list(csv.DictReader(f))
            if len(filas) < 2 or "equity" not in filas[0]:
                raise ValueError(f"Curva inválida o vacía: {ruta}")
            col = "date" if "date" in filas[0] else "timestamp"
            eq = pd.Series([float(r["equity"]) for r in filas], index=pd.to_datetime([r[col] for r in filas]))
            retornos_dict[(i, activo)] = eq.pct_change().dropna()
    corrida = eq_dir.parent if eq_dir.name == "equity" else base
    seleccion = leer_seleccion(corrida, h.piernas, cierres.index)
    curva = curva_cartera(cierres, piernas_map, retornos_dict, seleccion=seleccion)
    if curva is None or curva.empty:
        raise ValueError(f"Curva de cartera vacía en {eq_dir}")
    return curva


def calcular_criterio_paper(
    cartera: Hipotesis | str | Path,
    carpeta_etapa2: str | Path,
    carpeta_etapa3: str | Path,
    snapshot: str,
    intentos_path: str | Path | None = None,
) -> dict:
    """Evalúa el criterio de paper sobre curvas guardadas de etapas 2 y 3 sin escribir registros."""
    h = cartera if isinstance(cartera, Hipotesis) else cargar_hipotesis(cartera)
    cierres = {a: snap_mod.read_snapshot(snapshot, a)["close"] for a in dict.fromkeys(a for p in h.piernas for a in p.activos)}
    c2 = pd.DataFrame({a: s.loc["2007-01-01":"2018-12-31"] for a, s in cierres.items()})
    c3 = pd.DataFrame({a: s.loc["2019-01-01":"2022-12-31"] for a, s in cierres.items()})
    eq2, eq3 = reconstruir_curva_cartera(h, carpeta_etapa2, c2), reconstruir_curva_cartera(h, carpeta_etapa3, c3)
    m2, m3 = equity_metrics(eq2, 0.0), equity_metrics(eq3, 0.0)
    sharpe2, sharpe3, bench2, bench3 = float(m2["sharpe"]), float(m3["sharpe"]), sharpe_benchmark(c2), sharpe_benchmark(c3)
    pasa, motivos = evaluar_etapa3_cartera(eq3, bench3, sharpe2, bench2)
    p_int = Path(intentos_path) if intentos_path else snap_mod.snapshot_root() / "intentos.csv"
    filas = list(csv.DictReader(p_int.open(newline="", encoding="utf-8"))) if p_int.exists() else []
    neff, ncrudo = (n_efectivo(filas), len(filas)) if filas else (0, 0)
    dsr_crudo = dsr_contra_intentos(eq3, p_int) if p_int.exists() else None
    dsr_eff = dsr_contra_intentos(eq3, p_int, neff) if p_int.exists() else None
    return {
        "id": h.id, "eq_etapa2": eq2, "eq_etapa3": eq3, "sharpe_etapa2": sharpe2, "benchmark_etapa2": bench2,
        "sharpe_etapa3": sharpe3, "benchmark_etapa3": bench3, "anios_positivos": anios_positivos(eq3),
        "max_drawdown": float(m3["max_drawdown"]), "pasa": pasa, "veredicto": "PASA" if pasa else "CORTA", "motivos": motivos,
        "n_crudo": ncrudo, "n_efectivo": neff, "dsr_crudo": dsr_crudo, "dsr_efectivo": dsr_eff, "recargo_paper": recargo_paper(dsr_eff),
    }


def formatear_tablas(res: dict) -> str:
    """Devuelve las tablas markdown con métricas, DSR y veredicto."""
    ap, rec = res["anios_positivos"], res["recargo_paper"]
    ap_str = f"{ap.positivos} de {ap.totales}" if hasattr(ap, "positivos") else str(ap)
    rec_str = f"{rec.tamano_pct} % del tamaño, {rec.semanas} semanas" if rec else "-"
    dsr_e = f"{res['dsr_efectivo']:.2f}" if res["dsr_efectivo"] is not None else "-"
    dsr_c = f"{res['dsr_crudo']:.2f}" if res["dsr_crudo"] is not None else "-"
    t1 = [[res["id"], f"{res['sharpe_etapa3']:.2f}", f"{res['benchmark_etapa3']:.2f}", f"{res['sharpe_etapa2']:.2f}",
           f"{res['benchmark_etapa2']:.2f}", ap_str, f"{res['max_drawdown'] * 100:.1f} %", res["veredicto"]]]
    h1 = ["Cartera", "Sharpe etapa 3", "Benchmark etapa 3", "Sharpe etapa 2", "Benchmark etapa 2", "Años positivos", "Drawdown", "Veredicto"]
    t2 = [[res["id"], dsr_e, dsr_c, rec_str]]
    h2 = ["Cartera", f"DSR con N efectivo ({res['n_efectivo']})", f"DSR con N crudo ({res['n_crudo']})", "Recargo de paper si pasara"]
    lineas = [tabulate(t1, headers=h1, tablefmt="github"), "", tabulate(t2, headers=h2, tablefmt="github"), f"\nVeredicto: {res['veredicto']}"]
    if res["motivos"]:
        lineas.append(f"Motivos: {'; '.join(res['motivos'])}")
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI para evaluar criterio de paper desde curvas guardadas."""
    parser = argparse.ArgumentParser(description="Evalúa el criterio de paper desde curvas guardadas.")
    parser.add_argument("cartera", help="YAML o id de la cartera")
    parser.add_argument("etapa2", help="Carpeta de curvas de etapa 2")
    parser.add_argument("etapa3", help="Carpeta de curvas de etapa 3")
    parser.add_argument("snapshot", help="Nombre del snapshot")
    parser.add_argument("--intentos", default=None, help="Ruta opcional a intentos.csv")
    args = parser.parse_args(argv)
    try:
        res = calcular_criterio_paper(args.cartera, args.etapa2, args.etapa3, args.snapshot, args.intentos)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(formatear_tablas(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
