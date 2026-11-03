"""Estado por hipótesis (plan M3): hasta qué etapa llegó y si pasó o cortó. Solo agregado."""
import csv
import logging
import re
from collections import Counter
from datetime import datetime, timezone

from quantagent.backtesting.verdict import filas_validas
from quantagent.data.snapshot import snapshot_root

ESTADO_CSV = "hipotesis.csv"
COLUMNAS = ("fecha_utc", "hipotesis", "etapa", "veredicto", "motivo")
ETAPAS = ("0", "1", "2", "3")
VEREDICTOS = ("PASA", "CORTA", "ALARMA", "ERROR")
logger = logging.getLogger(__name__)


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
    try:
        from quantagent.fabrica.registro import registrar_evento

        registrar_evento(hipotesis_id, "etapa", etapa, veredicto, motivo)
    except Exception:
        logger.warning("No se pudo registrar el evento de etapa para %s", hipotesis_id, exc_info=True)


def ultimo_estado(hipotesis_id):
    from quantagent.fabrica.registro import eventos_etapa

    try:
        etapas = eventos_etapa(hipotesis_id)
    except OSError:
        etapas = []
    if etapas:
        evento = etapas[-1]
        return {columna: evento.get(columna, "") for columna in COLUMNAS}
    path = snapshot_root() / ESTADO_CSV
    if not path.exists():
        return None
    ultimo = None
    with path.open(newline="") as f:
        for fila in csv.DictReader(f):
            if fila["hipotesis"] == hipotesis_id:
                ultimo = fila
    return ultimo


def resumen_tanda(desde=None):
    """Último estado por hipótesis e intentos dentro del período UTC indicado."""
    registros = {}
    for nombre in (ESTADO_CSV, "intentos.csv"):
        path = snapshot_root() / nombre
        filas = []
        if path.exists():
            with path.open(newline="") as f:
                filas = [r for r in csv.DictReader(f) if desde is None or
                         datetime.fromisoformat(r["fecha_utc"]) >= desde]
        registros[nombre] = filas
    from quantagent.fabrica.registro import eventos_etapa

    try:
        eventos_etapa_registro = eventos_etapa()
    except OSError:
        eventos_etapa_registro = []
    etapas = [e for e in eventos_etapa_registro
              if desde is None or datetime.fromisoformat(str(e.get("fecha_utc")).replace("Z", "+00:00")) >= desde]
    if etapas:
        ultimos = {}
        for evento in etapas:
            ultimos[evento.get("hipotesis")] = {columna: evento.get(columna, "") for columna in COLUMNAS}
    else:
        ultimos = {r["hipotesis"]: r for r in registros[ESTADO_CSV]}
    conteos = Counter((r["etapa"], r["veredicto"]) for r in ultimos.values())
    causas = Counter(causa
                     for r in ultimos.values() if r["veredicto"] != "PASA"
                     for causa in {re.sub(r"\b[A-Z][A-Z0-9.-]*:\s*", "", c).strip()
                                   for c in r["motivo"].split(";")
                                   if re.sub(r"\b[A-Z][A-Z0-9.-]*:\s*", "", c).strip()})
    intentos = sum(r.get("hipotesis") in ultimos for r in filas_validas(registros["intentos.csv"]))
    lineas = ["# Resumen de la tanda", "", "| Etapa | Veredicto | Hipótesis |",
              "| --- | --- | ---: |"]
    lineas += [f"| {e} | {v} | {n} |" for (e, v), n in sorted(conteos.items())]
    lineas += ["", "## Causas más frecuentes", ""]
    lineas += [f"- {causa}: {n}" for causa, n in causas.most_common(5)]
    return "\n".join([*lineas, "", f"Intentos: {intentos}", ""])
