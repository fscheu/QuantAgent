"""Etapa 0 del embudo: admisión de una hipótesis antes de gastar cómputo (QuantAgent-d4cq.10)."""

from __future__ import annotations

import pandas as pd

from quantagent.backtesting.signals import bar_signals, check_no_lookahead, check_no_lookahead_macro
from quantagent.cli.sweep import cli_aliases
from quantagent.data.snapshot import SnapshotError, snapshot_root
from quantagent.fabrica.critico import (
    UNIVERSOS_CSV,
    cargar_existentes,
    criticar,
    motivo_universo,
    registrar_universo,
)
from quantagent.fabrica.hipotesis import Hipotesis, Pierna
from quantagent.fabrica.transversal import MASCARAS, METRICAS, es_transversal, señales_transversales
from quantagent.strategy.registry import STRATEGY_REGISTRY, build_strategy

CLASE_INTERNA = "senal-externa"  # P05.4 (SenalExterna) lee una ruta arbitraria: solo la usa el motor
MAX_PARAMETROS_LIBRES = 6  # provisorio, decidido por Fede el 2026-10-15
MIN_JUSTIFICACION = 40
# Universo amplio de L12 (QuantAgent-d4cq.46/.47): los 29 ETFs del manifiesto
# tests/fixtures/snapshots/etf-1d-2026-10-amplio.json. Se eligió el manifiesto solo y no su unión con los 12
# de M2 porque los contiene a todos. Está copiado acá y no leído del archivo porque el manifiesto es un
# fixture de tests; test_admision.py verifica que no se desincronicen.
UNIVERSO_AMPLIO = frozenset(
    {
        "AGG", "DBC", "EEM", "EFA", "EWJ", "EWZ", "FXE", "FXI", "GLD", "IEF", "IWM", "LQD", "QQQ", "SHY", "SLV",
        "SPY", "TIP", "TLT", "USO", "VNQ", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    }
)


def _senales(p: Pierna | Hipotesis, combinacion: dict, velas: pd.DataFrame) -> pd.DataFrame:
    nombre = cli_aliases().get(p.clase, p.clase)
    estrategia = build_strategy(nombre, **combinacion)
    return bar_signals(estrategia, velas, p.activos[0], getattr(p, "timeframe", "1d"))


def admitir(h: Hipotesis, velas: pd.DataFrame, existentes: list[Hipotesis] | None = None,
            cierres: pd.DataFrame | None = None) -> tuple[bool, list[str]]:
    """(True, []) si la hipótesis entra al embudo; si no, todos los motivos de rechazo.

    `existentes` son las hipótesis ya conocidas; por defecto, las de `hipotesis/`.
    `cierres`: panel fecha × activo del universo, para la prueba de truncamiento de las piernas transversales.
    """
    motivos = []
    veredicto, motivo, _ = criticar(h, cargar_existentes() if existentes is None else existentes)
    if veredicto == "duplicada":
        motivos.append(motivo)
    try:
        universos = snapshot_root() / UNIVERSOS_CSV
    except SnapshotError:  # sin carpeta de datos no hay dónde congelar (solo tests sueltos)
        universos = None
    if universos and (congelado := motivo_universo(h, universos)):
        motivos.append(congelado)
    if len(h.justificacion.strip()) < MIN_JUSTIFICACION:
        motivos.append(f"justificación de {len(h.justificacion.strip())} caracteres; mínimo {MIN_JUSTIFICACION}")
    libres = sum(sum(len(v) > 1 for v in p.parametros.values()) for p in h.piernas)
    if libres > MAX_PARAMETROS_LIBRES:
        motivos.append(f"{libres} parámetros libres; máximo {MAX_PARAMETROS_LIBRES}")
    aliases = cli_aliases()
    for p in h.piernas:
        if p.clase in (CLASE_INTERNA, "SenalExterna"):
            motivos.append(f"clase interna, no válida en una hipótesis: {p.clase}")
        elif not es_transversal(p.clase) and aliases.get(p.clase, p.clase) not in STRATEGY_REGISTRY:
            motivos.append(f"clase no registrada: {p.clase}")
        if p.elegibilidad is not None and p.elegibilidad not in MASCARAS:
            motivos.append(f"elegibilidad no registrada: {p.elegibilidad}")
    ajenos = sorted({a for p in h.piernas for a in p.activos} - UNIVERSO_AMPLIO)
    if ajenos:
        motivos.append(f"activos fuera del universo amplio: {', '.join(ajenos)}")
    if "timestamp" in velas.columns:
        velas = velas.set_index(pd.DatetimeIndex(velas["timestamp"]), drop=False)
    cortes = [velas.index[len(velas) // 2], velas.index[(3 * len(velas)) // 4]]
    for p in h.piernas:
        if es_transversal(p.clase):
            if cierres is None:
                motivos.append(f"falta el panel de cierres para la prueba de truncamiento de {p.clase}")
                continue
            idx = cierres.index
            if p.elegibilidad is not None and p.elegibilidad not in MASCARAS:
                continue
            difs = check_no_lookahead(lambda c, p=p: señales_transversales(c, METRICAS[p.clase], mascara=MASCARAS.get(p.elegibilidad)), cierres,
                                      [idx[len(idx) // 2], idx[(3 * len(idx)) // 4]])
            if difs:
                motivos.append(f"prueba de truncamiento fallida en {p.clase}: {difs[0]}")
            continue
        if p.clase in (CLASE_INTERNA, "SenalExterna") or aliases.get(p.clase, p.clase) not in STRATEGY_REGISTRY:
            continue
        # Una clase con `requiere_macro = True` recibe `snapshot` (y `serie`) como parámetros de valor único.
        macro = getattr(STRATEGY_REGISTRY[aliases.get(p.clase, p.clase)]["cls"], "requiere_macro", False)
        for comb in p.combinaciones():
            senales = lambda c, comb=comb, p=p: _senales(p, comb, c)  # noqa: E731
            try:
                if not macro:
                    difs = check_no_lookahead(senales, velas, cortes)
                elif "snapshot" not in comb:
                    difs = ["la clase requiere_macro y falta el parámetro snapshot"]
                else:
                    difs = check_no_lookahead_macro(senales, velas, cortes, comb["snapshot"])
            except SnapshotError as e:
                difs = [str(e)]
            if difs:
                motivos.append(f"prueba de truncamiento fallida con {comb}: {difs[0]}")
    if not motivos and universos:
        registrar_universo(h, universos)
    return not motivos, motivos
