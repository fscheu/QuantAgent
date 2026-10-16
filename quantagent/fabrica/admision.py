"""Etapa 0 del embudo: admisión de una hipótesis antes de gastar cómputo (QuantAgent-d4cq.10)."""

from __future__ import annotations

import pandas as pd

from quantagent.backtesting.signals import bar_signals, check_no_lookahead
from quantagent.cli.sweep import cli_aliases
from quantagent.fabrica.hipotesis import Hipotesis
from quantagent.strategy.registry import build_strategy

MAX_PARAMETROS_LIBRES = 6  # provisorio, decidido por Fede el 2026-10-15
MIN_JUSTIFICACION = 40
UNIVERSO_M2 = frozenset({"SPY", "IWM", "EFA", "EEM", "TLT", "SHY", "GLD", "DBC", "XLE", "XLF", "XLK", "XLU"})


def _senales(h: Hipotesis, combinacion: dict, velas: pd.DataFrame) -> pd.DataFrame:
    estrategia = build_strategy(cli_aliases().get(h.clase, h.clase), **combinacion)
    return bar_signals(estrategia, velas, h.activos[0], h.timeframe)


def admitir(h: Hipotesis, velas: pd.DataFrame) -> tuple[bool, list[str]]:
    """(True, []) si la hipótesis entra al embudo; si no, todos los motivos de rechazo."""
    motivos = []
    if len(h.justificacion.strip()) < MIN_JUSTIFICACION:
        motivos.append(f"justificación de {len(h.justificacion.strip())} caracteres; mínimo {MIN_JUSTIFICACION}")
    libres = sum(len(v) > 1 for v in h.parametros.values())
    if libres > MAX_PARAMETROS_LIBRES:
        motivos.append(f"{libres} parámetros libres; máximo {MAX_PARAMETROS_LIBRES}")
    ajenos = sorted(set(h.activos) - UNIVERSO_M2)
    if ajenos:
        motivos.append(f"activos fuera del universo M2: {', '.join(ajenos)}")
    if "timestamp" in velas.columns:
        velas = velas.set_index(pd.DatetimeIndex(velas["timestamp"]), drop=False)
    cortes = [velas.index[len(velas) // 2], velas.index[(3 * len(velas)) // 4]]
    for comb in h.combinaciones():
        difs = check_no_lookahead(lambda c, comb=comb: _senales(h, comb, c), velas, cortes)
        if difs:
            motivos.append(f"prueba de truncamiento fallida con {comb}: {difs[0]}")
    return not motivos, motivos
