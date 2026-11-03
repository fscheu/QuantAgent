"""Registro append-only de eventos de hipótesis."""

import csv
import json
import re
from collections import Counter
from datetime import date, datetime, timezone

from quantagent.default_config import (
    CICLOS_PARA_REVISAR,
    DIA_INICIO_EMBUDO,
    DIA_PROPUESTA_HIPOTESIS,
    DIA_REVISION_PAPER,
)
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
    fecha_utc=None,
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
        "fecha_utc": fecha_utc if fecha_utc is not None else datetime.now(timezone.utc).isoformat(),
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


def importar_csv(dry_run=False):
    """Import hypothesis stages and attempts from the snapshot CSV files."""
    root = snapshot_root()
    existentes, _ = _leer_eventos()
    filas, omitidas = [], 0

    def leer(path, tipo):
        nonlocal omitidas
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hipotesis = row.get("hipotesis", "")
                if tipo == "intento" and not hipotesis:
                    continue
                try:
                    fecha_utc = str(row["fecha_utc"])
                    fecha = datetime.fromisoformat(fecha_utc.replace("Z", "+00:00"))
                    if tipo == "etapa":
                        etapa, veredicto = row["etapa"], row["veredicto"]
                        if etapa not in ETAPAS or veredicto not in VEREDICTOS:
                            raise ValueError
                        item = {"hipotesis": hipotesis, "evento": "etapa", "etapa": etapa,
                                "veredicto": veredicto, "motivo": row["motivo"], "ciclo": None,
                                "datos": {"importado": True}}
                    else:
                        params = json.loads(row["params"])
                        if not isinstance(params, dict):
                            raise ValueError
                        etapa = row.get("etapa")
                        datos = {"snapshot": row["snapshot"], "estrategia": row["estrategia"],
                                 "activo": row["activo"], "params": params,
                                 "sharpe": _decimal(row["sharpe"]), "trades": _numero_entero(row["trades"]),
                                 "pnl": _decimal(row["pnl"]), "importado": True}
                        item = {"hipotesis": hipotesis, "evento": "intento",
                                "etapa": etapa if etapa in ETAPAS else None, "veredicto": None,
                                "motivo": "", "ciclo": None, "datos": datos}
                    item["fecha_utc"] = fecha_utc
                    item["_orden"] = fecha.timestamp()
                    filas.append(item)
                except (AttributeError, KeyError, TypeError, ValueError):
                    omitidas += 1

    leer(root / "hipotesis.csv", "etapa")
    leer(root / "intentos.csv", "intento")
    filas.sort(key=lambda e: e["_orden"])

    def clave(e):
        if e["evento"] == "etapa":
            return ("etapa", e["hipotesis"], str(e["etapa"]), str(e["veredicto"]), str(e["motivo"]))
        d = e["datos"]
        return ("intento", e["hipotesis"], None if e["etapa"] is None else str(e["etapa"]), d["snapshot"],
                d["estrategia"], d["activo"], json.dumps(d["params"], sort_keys=True), d["sharpe"],
                _numero_entero(d["trades"]), d["pnl"])

    cuenta_existente = Counter()
    for evento in existentes:
        if evento.get("evento") not in ("etapa", "intento"):
            continue
        try:
            if evento["evento"] == "etapa":
                normal = {"evento": "etapa", "hipotesis": evento.get("hipotesis"),
                          "etapa": evento.get("etapa"), "veredicto": evento.get("veredicto"),
                          "motivo": evento.get("motivo", "")}
            else:
                d = evento.get("datos", {})
                normal = {"evento": "intento", "hipotesis": evento.get("hipotesis"),
                          "etapa": evento.get("etapa"), "datos": {
                              "snapshot": d.get("snapshot"), "estrategia": d.get("estrategia"),
                              "activo": d.get("activo"), "params": json.loads(d.get("params", "{}"))
                              if isinstance(d.get("params", {}), str) else d.get("params", {}),
                              "sharpe": _decimal(d.get("sharpe")), "trades": _numero_entero(d.get("trades")),
                              "pnl": _decimal(d.get("pnl"))}}
            cuenta_existente[clave(normal)] += 1
        except (TypeError, ValueError):
            continue
    vistos, agregar = Counter(), []
    for evento in filas:
        key = clave(evento)
        vistos[key] += 1
        if vistos[key] > cuenta_existente[key]:
            agregar.append(evento)
    agregadas = Counter(e["evento"] for e in agregar)
    if not dry_run:
        for e in agregar:
            e.pop("_orden")
            registrar_evento(e["hipotesis"], e["evento"], e["etapa"], e["veredicto"],
                             e["motivo"], datos=e["datos"], fecha_utc=e["fecha_utc"])
    return {"etapas": {"leidos": sum(e["evento"] == "etapa" for e in filas), "agregados": agregadas["etapa"]},
            "intentos": {"leidos": sum(e["evento"] == "intento" for e in filas), "agregados": agregadas["intento"]},
            "omitidas": omitidas}


def _decimal(valor):
    return None if valor in (None, "") else float(valor)


def _numero_entero(valor):
    try:
        return int(valor)
    except (TypeError, ValueError):
        return valor


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


def _fecha(evento):
    try:
        return datetime.fromisoformat(str(evento.get("fecha_utc")).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def estado_actual(hipotesis_id):
    etapas = [e for e in historial(hipotesis_id) if e.get("evento") == "etapa"]
    return max(etapas, key=_fecha, default=None)


def estados_actuales():
    eventos, _ = _leer_eventos()
    estados = {}
    for evento in eventos:
        if evento.get("evento") == "etapa":
            previo = estados.get(evento.get("hipotesis"))
            if previo is None or _fecha(evento) >= _fecha(previo):
                estados[evento.get("hipotesis")] = evento
    return estados


def eventos_etapa(hipotesis_id=None):
    """Return stage events ordered by UTC date, preserving file order for ties."""
    eventos, _ = _leer_eventos()
    etapas = [(i, e) for i, e in enumerate(eventos)
              if e.get("evento") == "etapa"
              and (hipotesis_id is None or e.get("hipotesis") == hipotesis_id)]
    return [e for _, e in sorted(etapas, key=lambda item: (_fecha(item[1]), item[0]))]


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


def calendario_ciclos(desde, cantidad=CICLOS_PARA_REVISAR):
    """Return scheduled dates and registry status for consecutive monthly cycles."""
    if not isinstance(desde, str) or not _CICLO_RE.fullmatch(desde):
        raise ValueError(f"ciclo inválido: {desde!r}; debe tener formato YYYY-MM")

    anio, mes = map(int, desde.split("-"))
    eventos, _ = _leer_eventos()
    calendario = []
    for _ in range(cantidad):
        ciclo = f"{anio:04d}-{mes:02d}"
        eventos_mes = [evento for evento in eventos if evento.get("ciclo") == ciclo]
        apertura = next((evento for evento in eventos_mes if evento.get("evento") == "ciclo"
                         and evento.get("datos", {}).get("accion") == "abrir"), None)
        cierre = next((evento for evento in eventos_mes if evento.get("evento") == "ciclo"
                       and evento.get("datos", {}).get("accion") == "cerrar"), None)
        estado = {"estado": "pendiente"}
        if apertura:
            estado = {"estado": "abierto"}
        if cierre:
            recibo = cierre.get("datos", {}).get("recibo", {})
            estado = {"estado": "cerrado", "promovidas": recibo.get("promovidas", 0)}
        calendario.append({
            "ciclo": ciclo,
            "propuesta": date(anio, mes, DIA_PROPUESTA_HIPOTESIS).isoformat(),
            "embudo": date(anio, mes, DIA_INICIO_EMBUDO).isoformat(),
            "revision_paper": date(anio, mes, DIA_REVISION_PAPER).isoformat(),
            **estado,
        })
        mes += 1
        if mes == 13:
            anio += 1
            mes = 1
    return calendario
