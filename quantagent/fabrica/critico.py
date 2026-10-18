"""Crítico de la etapa 0: rechaza hipótesis duplicadas y reconoce variantes (QuantAgent-d4cq.41, P02)."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
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


UNIVERSOS_CSV = "hipotesis-universos.csv"


def hash_universo(h: Hipotesis) -> str:
    """sha256 de las piernas (clase + activos); la grilla puede cambiar, el universo no (D2)."""
    piernas = sorted((p.clase, sorted(p.activos)) for p in h.piernas)
    return hashlib.sha256(json.dumps(piernas).encode()).hexdigest()


def _declarado(id_: str, path: Path) -> tuple[str, str] | None:
    if path.exists():
        with path.open() as f:
            for fila in csv.reader(f):
                if fila[0] == id_:
                    return fila[1], fila[2]
    return None


def motivo_universo(h: Hipotesis, path: Path) -> str | None:
    """Motivo de rechazo si `h` ya declaró otro universo en `path`; None si es la primera vez o coincide."""
    previo = _declarado(h.id, path)
    if previo and previo[0] != hash_universo(h):
        return f"universo distinto del declarado el {previo[1]} (D2)"
    return None


def registrar_universo(h: Hipotesis, path: Path) -> None:
    """Guarda el hash de la primera admisión de `h.id`; el archivo es solo agregado."""
    if _declarado(h.id, path) is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as f:
            csv.writer(f).writerow([h.id, hash_universo(h), datetime.now(timezone.utc).date().isoformat()])
