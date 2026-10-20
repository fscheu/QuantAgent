"""Cálculo del criterio de paper de etapa 3 desde curvas guardadas, sin escrituras."""
from __future__ import annotations

import csv
from pathlib import Path
import pandas as pd

import quantagent.data.snapshot as snap_mod
from quantagent.backtesting.vbt_filter import equity_metrics
from quantagent.fabrica.cartera import curva_cartera
from quantagent.fabrica.criterio import anios_positivos, evaluar_etapa3_cartera, sharpe_benchmark
from quantagent.fabrica.hipotesis import Hipotesis, cargar_hipotesis


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
    curva = curva_cartera(cierres, piernas_map, retornos_dict)
    if curva is None or curva.empty:
        raise ValueError(f"Curva de cartera vacía en {eq_dir}")
    return curva


def calcular_criterio_paper(cartera: Hipotesis | str | Path, carpeta_etapa2: str | Path, carpeta_etapa3: str | Path, snapshot: str) -> dict:
    """Evalúa el criterio de paper sobre curvas guardadas de etapas 2 y 3 sin escribir registros."""
    h = cartera if isinstance(cartera, Hipotesis) else cargar_hipotesis(cartera)
    cierres = {a: snap_mod.read_snapshot(snapshot, a)["close"] for a in dict.fromkeys(a for p in h.piernas for a in p.activos)}
    c2 = pd.DataFrame({a: s.loc["2007-01-01":"2018-12-31"] for a, s in cierres.items()})
    c3 = pd.DataFrame({a: s.loc["2019-01-01":"2022-12-31"] for a, s in cierres.items()})
    eq2, eq3 = reconstruir_curva_cartera(h, carpeta_etapa2, c2), reconstruir_curva_cartera(h, carpeta_etapa3, c3)
    m2, m3 = equity_metrics(eq2, 0.0), equity_metrics(eq3, 0.0)
    sharpe2, sharpe3, bench2, bench3 = float(m2["sharpe"]), float(m3["sharpe"]), sharpe_benchmark(c2), sharpe_benchmark(c3)
    pasa, motivos = evaluar_etapa3_cartera(eq3, bench3, sharpe2, bench2)
    return {
        "id": h.id, "eq_etapa2": eq2, "eq_etapa3": eq3, "sharpe_etapa2": sharpe2, "benchmark_etapa2": bench2,
        "sharpe_etapa3": sharpe3, "benchmark_etapa3": bench3, "anios_positivos": anios_positivos(eq3),
        "max_drawdown": float(m3["max_drawdown"]), "pasa": pasa, "veredicto": "PASA" if pasa else "CORTA", "motivos": motivos,
    }
