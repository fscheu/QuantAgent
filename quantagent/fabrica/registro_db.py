"""Persistencia SQL del registro de eventos e intentos de hipótesis."""

import hashlib
import json
from datetime import date, datetime, timezone

from sqlalchemy import or_, select

from quantagent.data.snapshot import snapshot_manifest_hash
from quantagent.database import SessionLocal
from quantagent.default_config import (
    CICLOS_PARA_REVISAR,
    DIA_INICIO_EMBUDO,
    DIA_PROPUESTA_HIPOTESIS,
    DIA_REVISION_PAPER,
)
from quantagent.fabrica.campos_intentos import CAMPOS_INTENTOS
from quantagent.fabrica.estado import ETAPAS, VEREDICTOS
from quantagent.fabrica.registro import _CICLO_RE, EVENTOS
from quantagent.models import Cycle, Hypothesis, HypothesisAttempt, HypothesisEvent


def _fecha(value):
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value):
    return _fecha(value).isoformat()


def _dict(row):
    return {"version_esquema": 1, "fecha_utc": _iso(row.ts), "hipotesis": row.hypothesis_id,
            "evento": row.event, "etapa": row.stage, "veredicto": row.verdict,
            "motivo": row.reason or "", "ciclo": row.cycle, "datos": row.data or {}}


def eventos(session=None):
    if session is not None:
        return [_dict(row) for row in session.scalars(select(HypothesisEvent).order_by(HypothesisEvent.id))]
    with SessionLocal() as db:
        return eventos(db)


def registrar_evento(hipotesis_id, evento, etapa=None, veredicto=None, motivo="", ciclo=None,
                     datos=None, fecha_utc=None):
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
    ts = _fecha(fecha_utc)
    with SessionLocal() as db:
        if hipotesis_id is not None and db.get(Hypothesis, hipotesis_id) is None:
            db.add(Hypothesis(id=hipotesis_id, created_at=ts))
        db.add(HypothesisEvent(ts=ts, hypothesis_id=hipotesis_id, event=evento, stage=etapa,
                               verdict=veredicto, reason=motivo, cycle=ciclo,
                               data={} if datos is None else datos))
        db.commit()


def registrar_intento(strategy_name, symbol, snapshot_name, costos, tamano_posicion, combo, row,
                      hipotesis=None, etapa=None, sharpe_base="tasa_libre", fecha_utc=None):
    ts = _fecha(fecha_utc)
    extra = {"desde": row["desde"], "hasta": row["hasta"], "costos": costos or "",
             "tamano": tamano_posicion, "sharpe_base": sharpe_base or ""}
    with SessionLocal() as db:
        if hipotesis is not None and db.get(Hypothesis, hipotesis) is None:
            db.add(Hypothesis(id=hipotesis, created_at=ts))
        db.add(HypothesisAttempt(ts=ts, hypothesis_id=hipotesis, snapshot=snapshot_name,
            strategy=strategy_name, asset=symbol, params=combo,
            sharpe=None if row["sharpe"] in (None, "") else float(row["sharpe"]),
            trades=None if row["trades"] in (None, "") else int(row["trades"]),
            pnl=None if row["pnl"] in (None, "") else float(row["pnl"]),
            stage=etapa, status=None, extra=extra))
        db.commit()


def intentos(valida_solo=True, session=None):
    if session is None:
        with SessionLocal() as db:
            rows = db.scalars(select(HypothesisAttempt).order_by(HypothesisAttempt.id)).all()
    else:
        rows = session.scalars(select(HypothesisAttempt).order_by(HypothesisAttempt.id)).all()
    out = []
    for r in rows:
        if valida_solo and (r.status == "invalida_bug" or r.snapshot == "sintetico"):
            continue
        values = {"fecha_utc": _iso(r.ts), "estrategia": r.strategy, "activo": r.asset,
                  "snapshot": r.snapshot, **(r.extra or {}),
                  "params": json.dumps(r.params, sort_keys=True),
                  "trades": r.trades, "pnl": r.pnl, "sharpe": r.sharpe,
                  "hipotesis": r.hypothesis_id, "etapa": r.stage, "estado": r.status or ""}
        out.append({key: "" if values.get(key) is None else values.get(key) for key in CAMPOS_INTENTOS})
    return out


def invalidar_intentos(filtro):
    hipotesis = filtro.get("hipotesis")
    desde = _fecha(filtro["desde"]) if filtro.get("desde") is not None else None
    hasta = _fecha(filtro["hasta"]) if filtro.get("hasta") is not None else None
    with SessionLocal() as db:
        rows = db.scalars(select(HypothesisAttempt).where(
            HypothesisAttempt.hypothesis_id == hipotesis,
            or_(HypothesisAttempt.status.is_(None), HypothesisAttempt.status != "invalida_bug"),
        )).all()
        selected = [r for r in rows if (desde is None or _fecha(r.ts) >= desde)
                    and (hasta is None or _fecha(r.ts) <= hasta)]
        for row in selected:
            row.status = "invalida_bug"
        db.commit()
        return len(selected)


def historial(hipotesis_id):
    return [e for e in eventos() if e["hipotesis"] == hipotesis_id]


def _fecha_evento(e):
    try:
        return _fecha(e.get("fecha_utc"))
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)


def estado_actual(hipotesis_id):
    return max((e for e in historial(hipotesis_id) if e["evento"] == "etapa"), key=_fecha_evento, default=None)


def estados_actuales():
    result = {}
    for e in eventos():
        if e["evento"] == "etapa" and (e["hipotesis"] not in result or _fecha_evento(e) >= _fecha_evento(result[e["hipotesis"]])):
            result[e["hipotesis"]] = e
    return result


def eventos_etapa(hipotesis_id=None):
    found = [(i, e) for i, e in enumerate(eventos()) if e["evento"] == "etapa" and
             (hipotesis_id is None or e["hipotesis"] == hipotesis_id)]
    return [e for _, e in sorted(found, key=lambda x: (_fecha_evento(x[1]), x[0]))]


def _ciclo_valido(ciclo):
    if not isinstance(ciclo, str) or not _CICLO_RE.fullmatch(ciclo):
        raise ValueError(f"ciclo inválido: {ciclo!r}; debe tener formato YYYY-MM")


def _eventos_ciclo(ciclo):
    _ciclo_valido(ciclo)
    return [e for e in eventos() if e["ciclo"] == ciclo]


def abrir_ciclo(ciclo, hipotesis_ids):
    _ciclo_valido(ciclo)
    ids = list(hipotesis_ids)
    if len(ids) > 5:
        raise ValueError("un ciclo admite como máximo 5 hipótesis")
    if len(set(ids)) != len(ids):
        raise ValueError("el ciclo no admite hipótesis duplicadas")
    if any(e["evento"] == "ciclo" and e["datos"].get("accion") == "abrir" for e in _eventos_ciclo(ciclo)):
        raise ValueError(f"el ciclo {ciclo} ya fue abierto")
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        db.add(Cycle(cycle=ciclo, opened_at=now, hypothesis_ids=ids))
        db.commit()
    registrar_evento(None, "ciclo", ciclo=ciclo, datos={"accion": "abrir", "hipotesis": ids})


def cerrar_ciclo(ciclo):
    all_events = eventos()
    cycle_events = [e for e in all_events if e["ciclo"] == ciclo]
    apertura = next((e for e in cycle_events if e["evento"] == "ciclo" and e["datos"].get("accion") == "abrir"), None)
    if apertura is None:
        raise ValueError(f"el ciclo {ciclo} no está abierto")
    if any(e["evento"] == "ciclo" and e["datos"].get("accion") == "cerrar" for e in cycle_events):
        raise ValueError(f"el ciclo {ciclo} ya fue cerrado")
    with SessionLocal() as db:
        cycle_row = db.get(Cycle, ciclo)
        ids = cycle_row.hypothesis_ids
        opened_at = _fecha(cycle_row.opened_at)
    own = all_events[all_events.index(apertura)+1:]
    etapas = {str(i): 0 for i in range(4)}
    promoted, caidas, intentos_por_h, snapshots = 0, {}, {}, set()
    por_etapa = {str(i): {k: 0 for k in ("arrived", "passed", "cut", "error", "alarm")} for i in range(4)}
    por_hipotesis = {}
    for h in ids:
        records = [e for e in own if e["evento"] in ("etapa", "intento") and e["hipotesis"] == h]
        for stage in {e["etapa"] for e in records if e["evento"] == "etapa"}:
            if stage in etapas:
                etapas[stage] += 1
        stages = sorted((e for e in records if e["evento"] == "etapa"), key=_fecha_evento)
        ultimo_por_etapa = {}
        for e in stages:
            if e["etapa"] in etapas:
                ultimo_por_etapa[e["etapa"]] = e
        for stage, e in ultimo_por_etapa.items():
            por_etapa[stage]["arrived"] += 1
            bucket = {"PASA": "passed", "CORTA": "cut", "ERROR": "error", "ALARMA": "alarm"}.get(e["veredicto"])
            if bucket:
                por_etapa[stage][bucket] += 1
        promoted += any(e["etapa"] == "3" and e["veredicto"] == "PASA" for e in stages)
        ultimo = stages[-1] if stages else None
        if ultimo and ultimo["veredicto"] in ("CORTA", "ERROR", "ALARMA"):
            caidas[h] = {"etapa": ultimo["etapa"], "veredicto": ultimo["veredicto"], "motivo": ultimo["motivo"]}
        status = ("promovida" if any(e["etapa"] == "3" and e["veredicto"] == "PASA" for e in stages)
                  else "cortada" if ultimo and ultimo["veredicto"] == "CORTA"
                  else "detenida" if ultimo and ultimo["veredicto"] in ("ERROR", "ALARMA") else "en_curso")
        por_hipotesis[h] = {"estado_final": status, "etapa": ultimo["etapa"] if ultimo else None,
                            "veredicto": ultimo["veredicto"] if ultimo else None,
                            "motivo": ultimo["motivo"] if ultimo else None}
        with SessionLocal() as db:
            attempts = db.scalars(select(HypothesisAttempt).where(
                HypothesisAttempt.hypothesis_id == h)).all()
        attempts = [a for a in attempts if _fecha(a.ts) >= opened_at]
        intentos_por_h[h] = len(attempts)
        snapshots.update(a.snapshot for a in attempts if a.snapshot)
    snapshots = sorted(snapshots)
    receipt = {"version": 2, "ciclo": ciclo, "hipotesis": ids, "etapas": etapas, "promovidas": promoted,
               "caidas": caidas, "intentos": intentos_por_h, "snapshots": snapshots,
               "hashes_manifiesto": {s: snapshot_manifest_hash(s) for s in snapshots},
               "por_etapa": por_etapa, "por_hipotesis": por_hipotesis}
    canonical = json.dumps(receipt, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    receipt["hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    reason = "" if promoted else "ninguna hipótesis promovida: ninguna llegó a etapa 3 con PASA"
    registrar_evento(None, "ciclo", ciclo=ciclo, motivo=reason, datos={"accion": "cerrar", "recibo": receipt})
    with SessionLocal() as db:
        row = db.get(Cycle, ciclo)
        row.closed_at = datetime.now(timezone.utc)
        row.receipt = receipt
        db.commit()
    return receipt


def recibo_ciclo(ciclo):
    with SessionLocal() as db:
        row = db.get(Cycle, ciclo)
        return leer_recibo(row) if row else None


def leer_recibo(ciclo_o_dict):
    """Normaliza recibos históricos sin mutar el JSON persistido."""
    receipt = ciclo_o_dict if isinstance(ciclo_o_dict, dict) else getattr(ciclo_o_dict, "receipt", None)
    if receipt is None:
        return None
    result = dict(receipt)
    if result.get("version") == 2:
        return result
    result["version"] = 1
    result["hash"] = None
    stages = result.get("etapas")
    result["por_etapa"] = ({s: {"arrived": stages.get(s), "passed": None, "cut": None,
                                 "error": None, "alarm": None} for s in ("0", "1", "2", "3")}
                           if isinstance(stages, dict) else None)
    drops = result.get("caidas")
    result["por_hipotesis"] = ({h: {"estado_final": "cortada" if d.get("veredicto") == "CORTA" else
                                     "detenida" if d.get("veredicto") in ("ERROR", "ALARMA") else None,
                                     "etapa": d.get("etapa"), "veredicto": d.get("veredicto"),
                                     "motivo": d.get("motivo")} for h, d in drops.items()}
                              if isinstance(drops, dict) else None)
    return result


def ciclos_sin_etapa3():
    with SessionLocal() as db:
        closes = [r.receipt or {} for r in db.scalars(select(Cycle).where(
            Cycle.closed_at.is_not(None)).order_by(Cycle.closed_at)).all()]
    count = 0
    for receipt in reversed(closes):
        if receipt.get("promovidas", 0):
            break
        count += 1
    return count


def calendario_ciclos(desde, cantidad=CICLOS_PARA_REVISAR):
    _ciclo_valido(desde)
    year, month = map(int, desde.split("-"))
    all_events = eventos()
    result = []
    for _ in range(cantidad):
        cycle = f"{year:04d}-{month:02d}"
        monthly = [e for e in all_events if e["ciclo"] == cycle]
        opened = next((e for e in monthly if e["evento"] == "ciclo" and e["datos"].get("accion") == "abrir"), None)
        closed = next((e for e in monthly if e["evento"] == "ciclo" and e["datos"].get("accion") == "cerrar"), None)
        status = {"estado": "abierto"} if opened else {"estado": "pendiente"}
        if closed:
            status = {"estado": "cerrado", "promovidas": closed["datos"].get("recibo", {}).get("promovidas", 0)}
        result.append({"ciclo": cycle, "propuesta": date(year, month, DIA_PROPUESTA_HIPOTESIS).isoformat(),
            "embudo": date(year, month, DIA_INICIO_EMBUDO).isoformat(),
            "revision_paper": date(year, month, DIA_REVISION_PAPER).isoformat(), **status})
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return result
