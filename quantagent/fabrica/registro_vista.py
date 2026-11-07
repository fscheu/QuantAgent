"""Vistas de lectura del registro de hipótesis."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from textwrap import fill

from quantagent.fabrica import estado
from quantagent.fabrica.hipotesis import nombre_e_idea
from quantagent.fabrica import registro_db as registro
from quantagent.models import Cycle


def _fecha(e):
    return datetime.fromisoformat(str(e.get("fecha_utc", "")).replace("Z", "+00:00"))

_CAIDAS = ("CORTA", "ERROR", "ALARMA")
_ETAPAS = ("0", "1", "2", "3")
_MIN_FOR_CHART = 2  # web/src/config.ts: funnelConfig.minForChart
_LIMITE_REVISION = 6  # config.ts: cycleConfig.reviewAfterCycles


def _aperturas(eventos):
    return {e.get("ciclo"): e for e in eventos
            if e.get("evento") == "ciclo" and e.get("datos", {}).get("accion") == "abrir"}


def _estados(eventos):
    actuales = {}
    for e in eventos:
        if e.get("evento") != "etapa":
            continue
        h = e.get("hipotesis")
        previo = actuales.get(h)
        if previo is None or _fecha(e) >= _fecha(previo):
            actuales[h] = e
    return actuales


def embudo(ciclo=None, periodo=None, *, ahora=None, session=None):
    """Resume el embudo; un ciclo concreto ignora el período y 'all' omite el límite temporal."""
    eventos = registro.eventos(session)
    aperturas = _aperturas(eventos)
    if ciclo == "all":
        ids_ciclo = {h for apertura in aperturas.values()
                     for h in apertura.get("datos", {}).get("hipotesis", [])}
        eventos = [e for e in eventos if e.get("hipotesis") in ids_ciclo or
                   (e.get("evento") == "ciclo" and e.get("ciclo") in aperturas)]
    if ciclo is not None:
        ids = aperturas.get(ciclo, {}).get("datos", {}).get("hipotesis", [])
        if ciclo != "all":
            eventos = [e for e in eventos if e.get("hipotesis") in ids or
                       (e.get("evento") == "ciclo" and e.get("ciclo") == ciclo)]
    if periodo not in (None, "all") and ciclo in (None, "all"):
        ahora = ahora or datetime.now(timezone.utc)
        dias = int(str(periodo).removesuffix("d"))
        desde = ahora - timedelta(days=dias)
        eventos = [e for e in eventos if desde <= _fecha(e) <= ahora]
    estados = _estados(eventos)
    etapas_evento = [e for e in eventos if e.get("evento") == "etapa"]
    etapas = []
    for etapa in _ETAPAS:
        por_hipotesis = {}
        for e in etapas_evento:
            if e.get("etapa") != etapa:
                continue
            h = e.get("hipotesis")
            previo = por_hipotesis.get(h)
            if previo is None or _fecha(e) >= _fecha(previo):
                por_hipotesis[h] = e
        en_etapa = list(por_hipotesis.values())
        actuales_en_etapa = [e for e in estados.values() if e.get("etapa") == etapa]
        etapas.append({"stage": f"e{etapa}",
                       "reached": len(por_hipotesis),
                       "here": sum(e.get("veredicto") == "PASA" for e in actuales_en_etapa),
                       "passed": sum(e.get("veredicto") == "PASA" for e in en_etapa),
                       "cut": sum(e.get("veredicto") == "CORTA" for e in en_etapa),
                       "error": sum(e.get("veredicto") == "ERROR" for e in en_etapa),
                       "alarm": sum(e.get("veredicto") == "ALARMA" for e in en_etapa)})
    causas = {}
    for e in estados.values():
        if e.get("veredicto") not in _CAIDAS:
            continue
        codigo, etiqueta = estado.causa_de_evento(e)
        codigo, etiqueta = codigo or None, etiqueta or None
        clave = (e.get("etapa"), e.get("veredicto"), codigo)
        item = causas.setdefault(clave, {"stage": f"e{e.get('etapa')}", "verdict": e.get("veredicto"),
                                        "cause_code": codigo, "cause_label": etiqueta,
                                        "count": 0, "example": (e.get("datos") or {}).get("measure")})
        item["count"] += 1
    return {"cycle": "all" if ciclo is None else ciclo,
            "period": "all" if periodo in (None, "all") else (f"{periodo}d" if str(periodo).isdigit() else periodo),
            "total": len(estados), "stages": etapas,
            # D3: paper se deriva de cuentas; el registro todavía no las expone.
            "later": [{"phase": "shadow", "reached": 0, "here": 0},
                      {"phase": "paper", "reached": 0, "here": 0},
                      {"phase": "candidate", "reached": 0, "here": 0}],
            "promoted": sum(e.get("etapa") == "3" and e.get("veredicto") == "PASA" for e in estados.values()),
            "sample_too_small": len(estados) < _MIN_FOR_CHART,
            "drop_reasons": list(causas.values())}


def ciclos(limite=6, *, session=None):
    eventos = registro.eventos(session)
    aperturas = _aperturas(eventos)
    resultado = []
    for ciclo, apertura in sorted(aperturas.items(), reverse=True)[:limite]:
        cierre = next((e for e in eventos if e.get("evento") == "ciclo" and e.get("ciclo") == ciclo
                       and e.get("datos", {}).get("accion") == "cerrar"), None)
        row = session.get(Cycle, ciclo) if session is not None else None
        recibo = registro.leer_recibo(row) if row else registro.recibo_ciclo(ciclo)
        ids = apertura.get("datos", {}).get("hipotesis", [])
        propios = [e for e in eventos if e.get("hipotesis") in ids]
        etapas = [e for e in propios if e.get("evento") == "etapa"]
        estados = _estados(propios)
        if recibo is not None:
            promovidas = recibo.get("promovidas", 0)
            caidas = recibo.get("caidas", {})
            alcanzadas = {f"e{e}": recibo.get("etapas", {}).get(e, 0) for e in _ETAPAS}
        else:
            caidas = {h: e for h, e in estados.items() if e.get("veredicto") in _CAIDAS}
            promovidas = sum(any(e.get("etapa") == "3" and e.get("veredicto") == "PASA"
                                 for e in etapas if e.get("hipotesis") == h) for h in ids)
            alcanzadas = {f"e{e}": len({x.get("hipotesis") for x in etapas if x.get("etapa") == e}) or None
                          for e in _ETAPAS}
        cut = sum(e.get("veredicto") == "CORTA" for e in caidas.values())
        held = sum(e.get("veredicto") in ("ERROR", "ALARMA") for e in caidas.values())
        inicio_embudo = min((e.get("fecha_utc") for e in etapas), default="")
        # CycleMilestone.date is non-nullable; an unknown date is represented by "".
        milestones = [{"key": "proposal", "day": 1, "date": apertura.get("fecha_utc", ""), "done": True},
                      {"key": "funnel_start", "day": None, "date": inicio_embudo, "done": bool(etapas)},
                      {"key": "close", "day": None, "date": cierre.get("fecha_utc", "") if cierre else "", "done": bool(cierre)}]
        resultado.append({"cycle": ciclo, "status": "closed" if cierre else "open",
                          "started_at": apertura.get("fecha_utc"),
                          "closed_at": cierre.get("fecha_utc") if cierre else None,
                          "entered": len(ids), "active": len(ids) - cut - held - promovidas,
                          "cut": cut, "held": held, "promoted": promovidas,
                          "reached_by_stage": alcanzadas, "milestones": milestones,
                          "headline": None, "receipt_id": ciclo if cierre else None})
    return resultado


def recibo_ciclo(ciclo, *, session=None):
    """Devuelve el recibo inmutable guardado o una vista del ciclo abierto."""
    resumen = next((c for c in ciclos(limite=None, session=session) if c["cycle"] == ciclo), None)
    if resumen is None:
        return None
    cycle_row = session.get(Cycle, ciclo) if session is not None else None
    recibo = (registro.leer_recibo(cycle_row) if cycle_row else registro.recibo_ciclo(ciclo)) if resumen["status"] == "closed" else None
    eventos = registro.eventos(session)
    ids = next((e.get("datos", {}).get("hipotesis", []) for e in eventos
                if e.get("evento") == "ciclo" and e.get("ciclo") == ciclo
                and e.get("datos", {}).get("accion") == "abrir"), [])
    propios = [e for e in eventos if e.get("hipotesis") in ids and e.get("evento") == "etapa"]
    estados = _estados(propios)
    por_etapa = recibo.get("por_etapa") if recibo else None
    stages = []
    for etapa in _ETAPAS:
        datos = (por_etapa or {}).get(etapa) or {}
        actual = [e for e in estados.values() if e.get("etapa") == etapa]
        stages.append({"stage": f"e{etapa}", "reached": datos.get("arrived") or 0 if recibo else len({e.get("hipotesis") for e in propios if e.get("etapa") == etapa}),
                       "here": sum(e.get("veredicto") == "PASA" for e in actual),
                       "passed": datos.get("passed") or 0 if recibo else sum(e.get("veredicto") == "PASA" for e in propios if e.get("etapa") == etapa),
                       "cut": datos.get("cut") or 0 if recibo else sum(e.get("veredicto") == "CORTA" for e in propios if e.get("etapa") == etapa),
                       "error": datos.get("error") or 0 if recibo else sum(e.get("veredicto") == "ERROR" for e in propios if e.get("etapa") == etapa),
                       "alarm": datos.get("alarm") or 0 if recibo else sum(e.get("veredicto") == "ALARMA" for e in propios if e.get("etapa") == etapa),
                       "pending": not recibo and not any(e.get("etapa") == etapa for e in propios)})
    por_h = recibo.get("por_hipotesis") if recibo else None
    drops = []
    for h, d in (por_h or {}).items() if recibo else ((h, e) for h, e in estados.items() if e.get("veredicto") in _CAIDAS):
        nombre, _ = nombre_e_idea(h)
        etapa, veredicto, motivo = d.get("etapa"), d.get("veredicto"), d.get("motivo")
        evento = next((e for e in reversed(propios) if e.get("hipotesis") == h and e.get("etapa") == etapa and e.get("veredicto") == veredicto), {})
        drops.append({"hypothesis_id": h, "name": nombre, "stage": f"e{etapa}", "verdict": veredicto,
                      "reason": motivo or "", "measure": (evento.get("datos") or {}).get("measure")})
    intentos = [i for i in registro.intentos(valida_solo=False, session=session) if i.get("hipotesis") in ids]
    nombres = {h: nombre_e_idea(h)[0] for h in ids}
    attempts = [{"hypothesis_id": h, "name": nombres[h],
                 "valid": sum(i.get("hipotesis") == h and i.get("estado") != "invalida_bug" for i in intentos),
                 "invalid": sum(i.get("hipotesis") == h and i.get("estado") == "invalida_bug" for i in intentos)} for h in ids]
    snapshots = [{"name": s, "as_of": "", "hash": (recibo or {}).get("hashes_manifiesto", {}).get(s, "")[:12]}
                 for s in (recibo or {}).get("snapshots", [])]
    if recibo and recibo.get("version") != 2:
        stages = [{**s, "passed": 0, "cut": 0, "error": 0, "alarm": 0} for s in stages]
    return {"id": ciclo if recibo else None, "hash": (recibo or {}).get("hash"), "cycle": ciclo,
            "partial": recibo is None, "generated_at": resumen["closed_at"] or datetime.now(timezone.utc).isoformat(),
            "counts": {k: resumen[k] for k in ("entered", "active", "cut", "held", "promoted")},
            "stages": stages, "drops": drops, "attempts": attempts, "snapshots": snapshots}


def ciclo_actual(*, session=None):
    ciclos_registro = ciclos(limite=None, session=session)
    return next((ciclo for ciclo in ciclos_registro if ciclo["status"] == "open"),
                ciclos_registro[0] if ciclos_registro else None)


def regla_revision(*, session=None):
    eventos = registro.eventos(session)
    cierres = {}
    for e in eventos:
        if e.get("evento") == "ciclo" and e.get("datos", {}).get("accion") == "cerrar":
            receipt = registro.leer_recibo(e.get("datos", {}).get("recibo", {}))
            cierres[e.get("ciclo")] = receipt.get("promovidas", 0)
    ordenados = sorted(cierres.items())
    racha = 0
    for _, promovidas in reversed(ordenados):
        if promovidas:
            break
        racha += 1
    ultimos = ordenados[-_LIMITE_REVISION:]
    return {"limit": _LIMITE_REVISION, "streak": racha, "reached": racha >= _LIMITE_REVISION,
            "last_cycles": [{"cycle": ciclo, "promoted": promovidas} for ciclo, promovidas in ultimos]}


def historial_resumen(*, session=None):
    eventos = [e for e in registro.eventos(session)
               if e.get("evento") == "etapa" and e.get("ciclo") is None]
    etapas = []
    for etapa in _ETAPAS:
        rows = [e for e in eventos if e.get("etapa") == etapa]
        etapas.append({"stage": f"e{etapa}", "total": len(rows),
                       "pass": sum(e.get("veredicto") == "PASA" for e in rows),
                       "cut": sum(e.get("veredicto") == "CORTA" for e in rows),
                       "error": sum(e.get("veredicto") == "ERROR" for e in rows),
                       "alarm": sum(e.get("veredicto") == "ALARMA" for e in rows)})
    intentos = registro.intentos(valida_solo=False, session=session)
    ids_de_ciclo = {h for apertura in _aperturas(registro.eventos(session)).values()
                    for h in apertura.get("datos", {}).get("hipotesis", [])}
    por_etiqueta = Counter(i.get("etapa") or None for i in intentos
                           if i.get("hipotesis") not in ids_de_ciclo)
    return {"stage_events": len(eventos), "attempts": sum(por_etiqueta.values()),
            "events_by_stage": etapas,
            "attempts_by_label": [{"label": label, "count": count}
                                  for label, count in sorted(por_etiqueta.items(), key=lambda item: item[0] or "")]}


def hipotesis(estado=None, etapa=None, ciclo=None, periodo=None, *, session=None, ahora=None):
    eventos = registro.eventos(session)
    aperturas = _aperturas(eventos)
    ciclo_por_h = {h: c for c, e in aperturas.items() for h in e.get("datos", {}).get("hipotesis", [])}
    actuales = _estados(eventos)
    intentos = registro.intentos(valida_solo=False, session=session)
    actividad = {}
    for evento in eventos:
        h = evento.get("hipotesis")
        if h is not None and (h not in actividad or _fecha(evento) > actividad[h]):
            actividad[h] = _fecha(evento)
    for intento in intentos:
        h = intento.get("hipotesis")
        if h is not None and (h not in actividad or _fecha(intento) > actividad[h]):
            actividad[h] = _fecha(intento)
    if periodo not in (None, "all") and ciclo is None:
        ahora = ahora or datetime.now(timezone.utc)
        desde = ahora - timedelta(days=int(str(periodo).removesuffix("d")))
    salida = []
    nombres = {}
    for h, ultimo in actuales.items():
        stage = f"e{ultimo.get('etapa')}"
        veredicto = ultimo.get("veredicto")
        promovida = ultimo.get("etapa") == "3" and veredicto == "PASA"
        status = ("promoted" if promovida else "discarded" if veredicto == "CORTA"
                  else "held" if veredicto in ("ERROR", "ALARMA") else "active")
        ciclo_h = ciclo_por_h.get(h)
        if periodo not in (None, "all") and ciclo is None and not (desde <= actividad.get(h, _fecha(ultimo)) <= ahora):
            continue
        if (estado is not None and estado != status) or (etapa is not None and etapa != stage) or (ciclo is not None and ciclo != ciclo_h):
            continue
        propios = [i for i in intentos if i.get("hipotesis") == h]
        if h not in nombres:
            nombres[h] = nombre_e_idea(h)
        nombre, idea = nombres[h]
        salida.append({"id": h, "name": nombre, "idea": idea, "cycle": ciclo_h, "stage": stage,
                       "verdict": veredicto, "status": status, "promoted": promovida,
                       "used_reserve": any(e.get("hipotesis") == h and e.get("evento") == "etapa" and e.get("etapa") == "3" for e in eventos), "attempts_valid": sum(i.get("estado") != "invalida_bug" for i in propios),
                       "attempts_invalid": sum(i.get("estado") == "invalida_bug" for i in propios),
                       "last_metric": None,
                       "discard_reason": ultimo.get("motivo") if veredicto in _CAIDAS else None,
                       "discard_measure": (ultimo.get("datos") or {}).get("measure") if veredicto in _CAIDAS else None,
                       "updated_at": ultimo.get("fecha_utc")})
    return sorted(salida, key=lambda h: (int(h["stage"][1:]), h["updated_at"] or ""), reverse=True)


def hipotesis_detalle(id, *, session=None):
    resumen = next((h for h in hipotesis(session=session) if h["id"] == id), None)
    if resumen is None:
        return None
    eventos = [e for e in registro.eventos(session) if e.get("hipotesis") == id and e.get("evento") == "etapa"]
    resumen["timeline"] = [{"stage": f"e{e.get('etapa')}", "verdict": e.get("veredicto"),
                            "decided_at": e.get("fecha_utc"), "reason": e.get("motivo", ""),
                            "measure": (e.get("datos") or {}).get("measure"),
                            "imported": bool((e.get("datos") or {}).get("importado"))}
                           for e in sorted(eventos, key=_fecha)]
    intentos = [i for i in registro.intentos(valida_solo=False, session=session) if i.get("hipotesis") == id]
    resumen["attempts"] = [{"id": str(n), "hypothesis_id": id, "stage": i.get("etapa"),
                            "ran_at": i.get("fecha_utc"), "subject": i.get("activo") or i.get("estrategia") or "",
                            "snapshot_hash": "", "sharpe": _numero(i.get("sharpe")),
                            "trades": _numero(i.get("trades")), "pnl_pct": _numero(i.get("pnl")),
                            "invalid": i.get("estado") == "invalida_bug",
                            "invalid_reason": None}
                           for n, i in enumerate(intentos, 1)]
    resumen.update(later_phases=[], backtests=[], paper_account=None)
    return resumen


def _numero(valor):
    return None if valor in (None, "") else float(valor)


def informe_ciclo(ciclo):
    registro._ciclo_valido(ciclo)
    eventos_ciclo = [e for e in registro.eventos() if e.get("ciclo") == ciclo]
    apertura = next((e for e in eventos_ciclo if e.get("evento") == "ciclo"
                     and e.get("datos", {}).get("accion") == "abrir"), None)
    if apertura is None:
        raise ValueError(f"el ciclo {ciclo} no existe")
    recibo = registro.recibo_ciclo(ciclo)
    parcial = recibo is None
    cierre = next((e for e in eventos_ciclo if e.get("evento") == "ciclo"
                   and e.get("datos", {}).get("accion") == "cerrar"), None)
    ids = apertura.get("datos", {}).get("hipotesis", [])
    if parcial:
        eventos = registro.eventos()
        inicio = eventos.index(apertura)
        propios = [e for e in eventos[inicio + 1:]
                   if e.get("hipotesis") in ids and e.get("evento") in ("etapa", "intento")]
        intentos_db = [i for i in registro.intentos() if i.get("hipotesis") in ids and
                       _fecha({"fecha_utc": i["fecha_utc"]}) >= _fecha(apertura)]
        recibo = {"ciclo": ciclo, "hipotesis": ids,
                  "etapas": {etapa: len({e.get("hipotesis") for e in propios
                                         if e.get("evento") == "etapa" and e.get("etapa") == etapa})
                             for etapa in _ETAPAS},
                  "promovidas": sum(any(e.get("evento") == "etapa" and e.get("etapa") == "3"
                                         and e.get("veredicto") == "PASA" and e.get("hipotesis") == h
                                         for e in propios) for h in ids),
                  "caidas": {}, "intentos": {}, "snapshots": [], "hashes_manifiesto": {}}
        for h in ids:
            propios_h = [e for e in propios if e.get("hipotesis") == h]
            caidas = [e for e in propios_h if e.get("evento") == "etapa" and e.get("veredicto") in _CAIDAS]
            if caidas:
                caida = max(caidas, key=_fecha)
                recibo["caidas"][h] = {k: caida.get(k, "") for k in ("etapa", "veredicto", "motivo")}
            recibo["intentos"][h] = sum(i.get("hipotesis") == h for i in intentos_db)
        recibo["snapshots"] = sorted({i.get("snapshot") for i in intentos_db if i.get("snapshot")})
        from quantagent.data.snapshot import snapshot_manifest_hash
        recibo["hashes_manifiesto"] = {s: snapshot_manifest_hash(s) for s in recibo["snapshots"]}

    lineas = [f"Ciclo {ciclo}: {len(recibo.get('hipotesis', ids))} hipótesis entraron;",
              f"{recibo.get('promovidas', 0)} promovieron.", ""]
    if parcial:
        lineas.append("AVISO: ciclo abierto, informe parcial.")
    lineas.append("Embudo:")
    for etapa in _ETAPAS:
        lineas.append(f"e{etapa}: {recibo.get('etapas', {}).get(etapa, 0)} alcanzaron esta etapa")
    lineas.extend(["", "Caídas:"])
    for h, caida in recibo.get("caidas", {}).items():
        motivo = caida.get("motivo", "") or "sin motivo registrado"
        causas = estado.causas_de_motivo(motivo)
        explicacion = "Indica que la evaluación detectó: " + ", ".join(sorted(causas)) if causas else "El motivo registrado describe por qué se detuvo."
        lineas.append(f"{h}: e{caida.get('etapa')} {caida.get('veredicto')}. Motivo: {motivo}")
        lineas.append(explicacion)
    lineas.extend(["", "Intentos:"])
    lineas.extend(f"{h}: {n}" for h, n in recibo.get("intentos", {}).items())
    lineas.extend(["", "Snapshots:"])
    for snapshot in recibo.get("snapshots", []):
        hash_manifiesto = recibo.get("hashes_manifiesto", {}).get(snapshot, "")
        lineas.append(f"{snapshot}: {(hash_manifiesto or '')[:12]}")
    if recibo.get("promovidas", 0) == 0 and cierre:
        lineas.extend(["", f"Cierre: {cierre.get('motivo') or 'sin promoción'}"])
    usados = {caida.get("veredicto") for caida in recibo.get("caidas", {}).values()}
    if recibo.get("promovidas", 0):
        usados.add("PASA")
    descripciones = {"PASA": "avanza", "CORTA": "se descarta",
                     "ERROR": "falló", "ALARMA": "requiere atención"}
    terminos = ["Glosario:", "e0–e3: etapas de evaluación."]
    terminos.extend(f"{veredicto}: {descripciones[veredicto]}." for veredicto in
                    ("PASA", "CORTA", "ERROR", "ALARMA") if veredicto in usados)
    terminos.append("snapshot: datos usados.")
    lineas.extend(["", *terminos])
    return "\n".join(fill(str(linea), width=60, break_long_words=True,
                            break_on_hyphens=False) if linea else "" for linea in lineas)
