"""Crítico de la etapa 0: rechaza hipótesis duplicadas y reconoce variantes (QuantAgent-d4cq.41, P02)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from quantagent.fabrica.hipotesis import Hipotesis, cargar_hipotesis

HIPOTESIS_DIR = Path(__file__).resolve().parents[2] / "hipotesis"


def cargar_existentes(directorio: Path = HIPOTESIS_DIR) -> list[Hipotesis]:
    """Hipótesis del directorio, ordenadas por id: el orden es el de antigüedad que usa `criticar`."""
    return [cargar_hipotesis(p) for p in sorted(Path(directorio).glob("*.yaml"))]


def _piernas(h: Hipotesis) -> frozenset:
    return frozenset(
        (p.clase, json.dumps(p.parametros, sort_keys=True), tuple(sorted(p.activos))) for p in h.piernas
    )


def _clases(h: Hipotesis) -> frozenset:
    return frozenset(p.clase for p in h.piernas)


def criticar(h: Hipotesis, existentes: Iterable[Hipotesis]) -> tuple[str, str, str]:
    """(veredicto, motivo, familia). `existentes` va de la más antigua a la más nueva.

    Duplicada: mismo conjunto de piernas (clase, parámetros, activos), sin importar el orden.
    Variante: mismas clases con otros activos o grilla; hereda la familia de la más antigua.
    """
    otras = [e for e in existentes if e.id != h.id]
    for e in otras:
        if _piernas(e) == _piernas(h):
            return "duplicada", f"duplicada de {e.id}", e.id
    for e in otras:
        if _clases(e) == _clases(h):
            return "variante", f"variante de {e.id}: mismas clases, otros activos o grilla", e.id
    return "nueva", "", h.id
