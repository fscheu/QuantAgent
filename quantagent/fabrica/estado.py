"""Estado por hipótesis (plan M3): hasta qué etapa llegó y si pasó o cortó. Solo agregado."""
import csv
from datetime import datetime, timezone

from quantagent.data.snapshot import snapshot_root

ESTADO_CSV = "hipotesis.csv"
COLUMNAS = ("fecha_utc", "hipotesis", "etapa", "veredicto", "motivo")
ETAPAS = ("0", "1", "2", "3")
VEREDICTOS = ("PASA", "CORTA")


def registrar_estado(hipotesis_id, etapa, veredicto, motivo=""):
    if etapa not in ETAPAS:
        raise ValueError(f"etapa inválida: {etapa!r}; debe ser una de {ETAPAS}")
    if veredicto not in VEREDICTOS:
        raise ValueError(f"veredicto inválido: {veredicto!r}; debe ser uno de {VEREDICTOS}")
    path = snapshot_root() / ESTADO_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(COLUMNAS)
        w.writerow(
            [datetime.now(timezone.utc).isoformat(), hipotesis_id, etapa, veredicto, motivo]
        )


def ultimo_estado(hipotesis_id):
    path = snapshot_root() / ESTADO_CSV
    if not path.exists():
        return None
    ultimo = None
    with path.open(newline="") as f:
        for fila in csv.DictReader(f):
            if fila["hipotesis"] == hipotesis_id:
                ultimo = fila
    return ultimo
