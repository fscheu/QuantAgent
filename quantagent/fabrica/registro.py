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


def importar_csv(dry_run=False, datos=None):
    """Importa CSV y JSONL a la DB (QuantAgent-1gi8.6); deduplica por claves naturales.

    Si una etapa del CSV coincide con una etapa JSONL por fecha, hipótesis, etapa,
    veredicto y motivo, se considera la misma fila aunque difieran sus datos.
    Las repeticiones idénticas dentro de cada fuente se conservan mediante contadores.
    """
    from quantagent.database import SessionLocal
    from sqlalchemy import select
    from quantagent.fabrica.campos_intentos import CAMPOS_INTENTOS
    from quantagent.models import Cycle, Hypothesis, HypothesisAttempt, HypothesisEvent

    # registro_db importa este módulo; la importación local evita el ciclo.
    from quantagent.fabrica import registro_db

    root = snapshot_root() if datos is None else snapshot_root(datos)
    omitidas = 0
    csv_stages, csv_attempts, json_events = [], [], []

    def parse_date(value):
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def read_csv(name, target, stage_file=False):
        nonlocal omitidas
        path = root / name
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ts = parse_date(row["fecha_utc"])
                    hid = row.get("hipotesis") or None
                    if stage_file:
                        stage, verdict = row["etapa"], row["veredicto"]
                        if stage not in ETAPAS or verdict not in VEREDICTOS:
                            raise ValueError
                        target.append(HypothesisEvent(ts=ts, hypothesis_id=hid, event="etapa",
                            stage=stage, verdict=verdict, reason=row["motivo"], cycle=None,
                            data={"importado": True}))
                    else:
                        if not set(CAMPOS_INTENTOS).issubset(row):
                            raise ValueError
                        params = json.loads(row["params"])
                        if not isinstance(params, dict):
                            raise ValueError
                        # La etapa de un intento es texto libre (p. ej. "comparar"): no se descarta.
                        stage = row["etapa"] or None
                        status = row["estado"] or None
                        if status not in (None, "", "valida", "invalida_bug"):
                            raise ValueError
                        sharpe, pnl = _decimal(row["sharpe"]), _decimal(row["pnl"])
                        trades = None if row["trades"] == "" else int(row["trades"])
                        extra = {k: row[k] for k in ("desde", "hasta", "costos", "tamano", "sharpe_base")}
                        extra["tamano"] = None if extra["tamano"] in (None, "") else float(extra["tamano"])
                        extra["costos"] = extra["costos"] or ""
                        extra["sharpe_base"] = extra["sharpe_base"] or ""
                        target.append(HypothesisAttempt(ts=ts, hypothesis_id=hid, snapshot=row["snapshot"],
                            strategy=row["estrategia"], asset=row["activo"], params=params,
                            sharpe=sharpe, trades=trades, pnl=pnl, stage=stage, status=status, extra=extra))
                except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    omitidas += 1

    read_csv("hipotesis.csv", csv_stages, True)
    read_csv("intentos.csv", csv_attempts)
    json_path = root / REGISTRO_JSONL
    if json_path.exists():
        with json_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    source = json.loads(line)
                    if not isinstance(source, dict):
                        raise ValueError
                    ts = parse_date(source["fecha_utc"])
                    event = source["evento"]
                    if event not in EVENTOS:
                        raise ValueError
                    stage, verdict = source.get("etapa"), source.get("veredicto")
                    if stage is not None and stage not in ETAPAS or verdict is not None and verdict not in VEREDICTOS:
                        raise ValueError
                    cycle, data = source.get("ciclo"), source.get("datos") or {}
                    if cycle is not None and not _CICLO_RE.fullmatch(cycle):
                        raise ValueError
                    if not isinstance(data, dict):
                        raise ValueError
                    hid = source.get("hipotesis") or None
                    row = HypothesisEvent(ts=ts, hypothesis_id=hid, event=event, stage=stage,
                        verdict=verdict, reason=source.get("motivo") or "", cycle=cycle, data=data)
                    attempt = None
                    if event == "intento":
                        params = data["params"]
                        if isinstance(params, str): params = json.loads(params)
                        if not isinstance(params, dict): raise ValueError
                        attempt = HypothesisAttempt(ts=ts, hypothesis_id=hid,
                            snapshot=data["snapshot"], strategy=data["estrategia"], asset=data["activo"],
                            params=params, sharpe=_decimal(data.get("sharpe")),
                            trades=None if data.get("trades") in (None, "") else int(data["trades"]),
                            pnl=_decimal(data.get("pnl")), stage=stage, status=None, extra={})
                    json_events.append((row, attempt, data))
                except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    omitidas += 1

    def json_key(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    def utc(ts):
        # SQLite devuelve fechas sin zona: se normaliza para comparar con las de las fuentes.
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)

    def event_key(row):
        return (utc(row.ts), row.hypothesis_id, row.event, row.stage, row.verdict, row.reason or "",
                row.cycle, json_key(row.data or {}))
    def attempt_key(row):
        return (utc(row.ts), row.hypothesis_id, row.snapshot, row.strategy, row.asset,
                json_key(row.params or {}), row.sharpe, row.trades, row.pnl, row.stage,
                row.status, json_key(row.extra or {}))

    with SessionLocal() as db:
        existing_events = db.scalars(select(HypothesisEvent)).all()
        existing_attempts = db.scalars(select(HypothesisAttempt)).all()
        existing_cycles = {c.cycle: c for c in db.scalars(select(Cycle)).all()}
        event_counts = Counter(event_key(r) for r in existing_events)
        attempt_counts = Counter(attempt_key(r) for r in existing_attempts)
        all_events = csv_stages + [r for r, _, _ in json_events]
        csv_overlaps = {(utc(r.ts), r.hypothesis_id, r.stage, r.verdict, r.reason or "")
                        for r in existing_events if r.event == "etapa"}
        csv_overlaps.update((utc(r.ts), r.hypothesis_id, r.stage, r.verdict, r.reason or "")
                            for r, _, _ in json_events if r.event == "etapa")
        # Deduplicate all sources as a multiset against the database.
        chosen_events = []
        source_event_counts = Counter()
        for row in all_events:
            if row in csv_stages and (utc(row.ts), row.hypothesis_id, row.stage, row.verdict,
                                      row.reason or "") in csv_overlaps:
                continue
            key = event_key(row)
            source_event_counts[key] += 1
            if source_event_counts[key] > event_counts[key]:
                chosen_events.append(row)
        source_attempts = csv_attempts + [a for _, a, _ in json_events if a is not None]
        attempt_counts_source = Counter()
        chosen_attempts = []
        for row in source_attempts:
            key = attempt_key(row)
            attempt_counts_source[key] += 1
            if attempt_counts_source[key] > attempt_counts[key]:
                chosen_attempts.append(row)

        # Cycle records are materialized from JSONL open/close events; the primary key is idempotent.
        cycle_changes = {}
        for row, _, data in json_events:
            if row.event != "ciclo" or not row.cycle:
                continue
            action = data.get("accion")
            if action not in ("abrir", "cerrar"):
                continue
            cycle = cycle_changes.get(row.cycle) or existing_cycles.get(row.cycle)
            if cycle is None:
                ids = data.get("hipotesis", []) if action == "abrir" else data.get("recibo", {}).get("hipotesis", [])
                cycle = Cycle(cycle=row.cycle, opened_at=row.ts, hypothesis_ids=ids)
                cycle_changes[row.cycle] = cycle
            if action == "cerrar":
                cycle.closed_at, cycle.receipt = row.ts, data.get("recibo")
            elif row.cycle not in existing_cycles:
                cycle.opened_at, cycle.hypothesis_ids = row.ts, data.get("hipotesis", [])

        source_rows = all_events + source_attempts
        hypotheses = {}
        for row in source_rows:
            if row.hypothesis_id:
                hypotheses[row.hypothesis_id] = min(hypotheses.get(row.hypothesis_id, row.ts), row.ts)
        for hid, earliest in hypotheses.items():
            if hid and db.get(Hypothesis, hid) is None:
                db.add(Hypothesis(id=hid, created_at=earliest))
        elegidos_e = {id(r) for r in chosen_events}
        elegidos_a = {id(r) for r in chosen_attempts}
        resultado = {
            "etapas": {"leidos": len(csv_stages), "agregados": sum(id(r) in elegidos_e for r in csv_stages)},
            "intentos": {"leidos": len(csv_attempts), "agregados": sum(id(r) in elegidos_a for r in csv_attempts)},
            "eventos": {"leidos": len(json_events), "agregados": sum(id(r) in elegidos_e for r, _, _ in json_events)},
            "ciclos": {"leidos": sum(r.event == "ciclo" for r, _, _ in json_events),
                       "agregados": sum(key not in existing_cycles for key in cycle_changes)},
            "omitidas": omitidas}
        if not dry_run:
            db.add_all(chosen_events)
            db.add_all(chosen_attempts)
            for cycle in cycle_changes.values():
                db.add(cycle)
            db.commit()
    return resultado


def _decimal(valor):
    return None if valor in (None, "") else float(valor)


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
