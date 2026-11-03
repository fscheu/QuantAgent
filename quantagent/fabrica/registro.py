"""Registro append-only de eventos de hipótesis."""

import json
import re
from datetime import datetime, timezone

from quantagent.data.snapshot import snapshot_manifest_hash, snapshot_root
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


def _eventos_ciclo(ciclo):
    if not isinstance(ciclo, str) or not _CICLO_RE.fullmatch(ciclo):
        raise ValueError(f"ciclo inválido: {ciclo!r}; debe tener formato YYYY-MM")
    return [e for e in _leer_eventos()[0] if e.get("ciclo") == ciclo]


def abrir_ciclo(ciclo, hipotesis_ids):
    if not isinstance(ciclo, str) or not _CICLO_RE.fullmatch(ciclo):
        raise ValueError(f"ciclo inválido: {ciclo!r}; debe tener formato YYYY-MM")
    hipotesis = list(hipotesis_ids)
    if len(hipotesis) > 5:
        raise ValueError("un ciclo admite como máximo 5 hipótesis")
    if len(set(hipotesis)) != len(hipotesis):
        raise ValueError("el ciclo no admite hipótesis duplicadas")
    if any(e.get("evento") == "ciclo" and e.get("datos", {}).get("accion") == "abrir"
           for e in _eventos_ciclo(ciclo)):
        raise ValueError(f"el ciclo {ciclo} ya fue abierto")
    registrar_evento(None, "ciclo", ciclo=ciclo, datos={"accion": "abrir", "hipotesis": hipotesis})


def cerrar_ciclo(ciclo):
    todos, _ = _leer_eventos()
    eventos = [e for e in todos if e.get("ciclo") == ciclo]
    apertura = next((e for e in eventos if e.get("evento") == "ciclo"
                     and e.get("datos", {}).get("accion") == "abrir"), None)
    if apertura is None:
        raise ValueError(f"el ciclo {ciclo} no está abierto")
    if any(e.get("evento") == "ciclo" and e.get("datos", {}).get("accion") == "cerrar" for e in eventos):
        raise ValueError(f"el ciclo {ciclo} ya fue cerrado")

    ids = apertura["datos"]["hipotesis"]
    inicio = todos.index(apertura)
    eventos = todos[inicio + 1:]
    etapas = {str(i): 0 for i in range(4)}
    promovidas = 0
    caidas, intentos = {}, {}
    snapshots = set()
    manifiestos = {}
    for hipotesis in ids:
        propios = [e for e in eventos if e.get("evento") in ("etapa", "intento")
                   and e.get("hipotesis") == hipotesis]
        vistas = {e.get("etapa") for e in propios if e.get("evento") == "etapa"}
        for etapa in vistas:
            if etapa in etapas:
                etapas[etapa] += 1
        estados = [e for e in propios if e.get("evento") == "etapa"]
        if any(e.get("etapa") == "3" and e.get("veredicto") == "PASA" for e in estados):
            promovidas += 1
        for e in estados:
            if e.get("veredicto") in ("CORTA", "ERROR", "ALARMA"):
                caidas[hipotesis] = {"etapa": e.get("etapa"), "veredicto": e.get("veredicto"), "motivo": e.get("motivo", "")}
        intentos[hipotesis] = sum(e.get("evento") == "intento" for e in propios)
        for e in propios:
            snapshot = e.get("datos", {}).get("snapshot") if e.get("evento") == "intento" else None
            if snapshot:
                snapshots.add(snapshot)
    for snapshot in sorted(snapshots):
        manifiestos[snapshot] = snapshot_manifest_hash(snapshot)
    recibo = {"ciclo": ciclo, "hipotesis": ids, "etapas": etapas, "promovidas": promovidas,
              "caidas": caidas, "intentos": intentos, "snapshots": sorted(snapshots),
              "hashes_manifiesto": manifiestos}
    motivo = "" if promovidas else "ninguna hipótesis promovida: ninguna llegó a etapa 3 con PASA"
    registrar_evento(None, "ciclo", ciclo=ciclo, motivo=motivo,
                     datos={"accion": "cerrar", "recibo": recibo})
    return recibo


def recibo_ciclo(ciclo):
    return next((e.get("datos", {}).get("recibo") for e in _eventos_ciclo(ciclo)
                 if e.get("evento") == "ciclo" and e.get("datos", {}).get("accion") == "cerrar"), None)


def ciclos_sin_etapa3():
    """Count the latest consecutive closed cycles without a stage 3 pass."""
    eventos, _ = _leer_eventos()
    cierres = [
        evento.get("datos", {}).get("recibo", {})
        for evento in eventos
        if evento.get("evento") == "ciclo"
        and evento.get("datos", {}).get("accion") == "cerrar"
    ]
    cuenta = 0
    for recibo in reversed(cierres):
        if recibo.get("promovidas", 0):
            break
        cuenta += 1
    return cuenta
