"""Registro append-only de eventos de hipótesis."""

import json
import re
from datetime import datetime, timezone

from quantagent.data.snapshot import snapshot_root
from quantagent.fabrica.estado import ETAPAS, VEREDICTOS

EVENTOS = ("admision", "etapa", "intento", "ciclo")
REGISTRO_JSONL = "registro-hipotesis.jsonl"
_CICLO_RE = re.compile(r"\d{4}-(0[1-9]|1[0-2])\Z")


def _ruta_registro():
    return snapshot_root() / REGISTRO_JSONL


def registrar_evento(
    hipotesis_id,
    evento,
    etapa=None,
    veredicto=None,
    motivo="",
    ciclo=None,
    datos=None,
):
    if evento not in EVENTOS:
        raise ValueError(f"evento inválido: {evento!r}; debe ser uno de {EVENTOS}")
    if etapa is not None and etapa not in ETAPAS:
        raise ValueError(f"etapa inválida: {etapa!r}; debe ser una de {ETAPAS}")
    if veredicto is not None and veredicto not in VEREDICTOS:
        raise ValueError(f"veredicto inválido: {veredicto!r}; debe ser uno de {VEREDICTOS}")
    if evento == "etapa":
        if etapa is None:
            raise ValueError("etapa es obligatoria para evento 'etapa'")
        if veredicto is None:
            raise ValueError("veredicto es obligatorio para evento 'etapa'")
    if ciclo is not None and (not isinstance(ciclo, str) or not _CICLO_RE.fullmatch(ciclo)):
        raise ValueError(f"ciclo inválido: {ciclo!r}; debe tener formato YYYY-MM")

    registro = {
        "version_esquema": 1,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "hipotesis": hipotesis_id,
        "evento": evento,
        "etapa": etapa,
        "veredicto": veredicto,
        "motivo": motivo,
        "ciclo": ciclo,
        "datos": {} if datos is None else datos,
    }
    path = _ruta_registro()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def _leer_eventos():
    path = _ruta_registro()
    if not path.exists():
        return [], 0
    eventos = []
    corruptas = 0
    with path.open(encoding="utf-8") as f:
        for linea in f:
            try:
                evento = json.loads(linea)
            except json.JSONDecodeError:
                corruptas += 1
                continue
            if not isinstance(evento, dict):
                corruptas += 1
                continue
            eventos.append(evento)
    return eventos, corruptas


def lineas_corruptas():
    return _leer_eventos()[1]


def historial(hipotesis_id):
    eventos, _ = _leer_eventos()
    return [evento for evento in eventos if evento.get("hipotesis") == hipotesis_id]


def estado_actual(hipotesis_id):
    return next(
        (evento for evento in reversed(historial(hipotesis_id)) if evento.get("evento") == "etapa"),
        None,
    )


def estados_actuales():
    eventos, _ = _leer_eventos()
    estados = {}
    for evento in eventos:
        if evento.get("evento") == "etapa":
            estados[evento.get("hipotesis")] = evento
    return estados
