"""Candado de apertura única por hipótesis (plan M3): prueba y reserva se abren una sola vez."""
import csv
from datetime import datetime, timezone

from quantagent.data.snapshot import snapshot_root

CANDADO_CSV = "hipotesis-candado.csv"
PERIODOS = ("prueba", "reserva")


class CandadoError(Exception):
    """El par (hipótesis, período) ya fue abierto."""


def _validar(periodo):
    if periodo not in PERIODOS:
        raise ValueError(f"periodo inválido: {periodo!r}; debe ser uno de {PERIODOS}")


def _fecha_apertura(hipotesis_id, periodo):
    path = snapshot_root() / CANDADO_CSV
    if not path.exists():
        return None
    with path.open() as f:
        for fila in csv.reader(f):
            if fila[:2] == [hipotesis_id, periodo]:
                return fila[2]
    return None


def abierto(hipotesis_id, periodo):
    _validar(periodo)
    return _fecha_apertura(hipotesis_id, periodo) is not None


def abrir(hipotesis_id, periodo):
    _validar(periodo)
    previa = _fecha_apertura(hipotesis_id, periodo)
    if previa is not None:
        raise CandadoError(
            f"{hipotesis_id}/{periodo} ya se abrió el {previa}; se abre una sola vez"
        )
    path = snapshot_root() / CANDADO_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        csv.writer(f).writerow(
            [hipotesis_id, periodo, datetime.now(timezone.utc).isoformat()]
        )
