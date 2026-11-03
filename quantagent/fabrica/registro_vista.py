"""Vistas de lectura del registro de hipótesis."""

from collections import Counter
from datetime import datetime, timedelta, timezone
from textwrap import fill

from quantagent.fabrica import estado, registro

_CAIDAS = ("CORTA", "ERROR", "ALARMA")
_ETAPAS = ("0", "1", "2", "3")


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
        if previo is None or registro._fecha(e) >= registro._fecha(previo):
            actuales[h] = e
    return actuales


def embudo(ciclo=None, periodo=None, *, ahora=None):
    eventos, _ = registro._leer_eventos()
    if ciclo is not None:
        aperturas = _aperturas(eventos)
        ids = aperturas.get(ciclo, {}).get("datos", {}).get("hipotesis", [])
        eventos = [e for e in eventos if e.get("hipotesis") in ids or
                   (e.get("evento") == "ciclo" and e.get("ciclo") == ciclo)]
    if periodo is not None:
        ahora = ahora or datetime.now(timezone.utc)
        desde = ahora - timedelta(days=periodo)
        eventos = [e for e in eventos if desde <= registro._fecha(e) <= ahora]
    estados = _estados(eventos)
    etapas_evento = [e for e in eventos if e.get("evento") == "etapa"]
    etapas = []
    for etapa in _ETAPAS:
        en_etapa = [e for e in estados.values() if e.get("etapa") == etapa]
        etapas.append({"stage": f"e{etapa}",
                       "reached": len({e.get("hipotesis") for e in etapas_evento if e.get("etapa") == etapa}),
                       "here": sum(e.get("veredicto") == "PASA" for e in en_etapa),
                       "dropped": sum(e.get("veredicto") in _CAIDAS for e in en_etapa)})
    causas = Counter()
    for e in estados.values():
        if e.get("veredicto") != "PASA":
            causas.update((e.get("etapa"), causa) for causa in estado.causas_de_motivo(e.get("motivo", "")))
    # El registro identifica ciclos con YYYY-MM, aunque api-types.ts declare number.
    return {"cycle": "all" if ciclo is None else ciclo, "period_days": periodo,
            "total": len(estados), "stages": etapas,
            "drop_reasons": [{"stage": f"e{etapa}", "reason": razon, "count": cantidad}
                             for (etapa, razon), cantidad in sorted(causas.items())]}


def ciclos(limite):
    eventos, _ = registro._leer_eventos()
    aperturas = _aperturas(eventos)
    resultado = []
    for ciclo, apertura in sorted(aperturas.items(), reverse=True)[:limite]:
        cierre = next((e for e in eventos if e.get("evento") == "ciclo" and e.get("ciclo") == ciclo
                       and e.get("datos", {}).get("accion") == "cerrar"), None)
        recibo = registro.recibo_ciclo(ciclo)
        ids = apertura.get("datos", {}).get("hipotesis", [])
        if recibo is not None:
            caidas = len(recibo.get("caidas", {}))
            alcanzadas = {f"e{e}": recibo.get("etapas", {}).get(e, 0) for e in _ETAPAS}
        else:
            estados = _estados(eventos)
            caidas = sum(h in estados and estados[h].get("veredicto") in _CAIDAS for h in ids)
            alcanzadas = {f"e{e}": len({x.get("hipotesis") for x in eventos
                                      if x.get("evento") == "etapa" and x.get("etapa") == e
                                      and x.get("hipotesis") in ids}) for e in _ETAPAS}
        resultado.append({"cycle": ciclo, "started_at": apertura.get("fecha_utc"),
                          "ended_at": cierre.get("fecha_utc") if cierre else None,
                          "entered": len(ids), "active": len(ids) - caidas, "dropped": caidas,
                          "reached_by_stage": alcanzadas, "headline": None})
    return resultado


def hipotesis(estado=None, etapa=None, ciclo=None):
    eventos, _ = registro._leer_eventos()
    aperturas = _aperturas(eventos)
    ciclo_por_h = {h: c for c, e in aperturas.items() for h in e.get("datos", {}).get("hipotesis", [])}
    actuales = _estados(eventos)
    salida = []
    for h, ultimo in actuales.items():
        stage = f"e{ultimo.get('etapa')}"
        descartada = ultimo.get("veredicto") in _CAIDAS
        status = "discarded" if descartada else "active"
        ciclo_h = ciclo_por_h.get(h)
        if (estado is not None and estado != status) or (etapa is not None and etapa != stage) or (ciclo is not None and ciclo != ciclo_h):
            continue
        salida.append({"id": h, "name": h, "idea": None, "cycle": ciclo_h, "stage": stage,
                       "status": status, "last_metric": None,
                       "discard_reason": ultimo.get("motivo") if descartada else None,
                       "discard_detail": None, "updated_at": ultimo.get("fecha_utc")})
    return sorted(salida, key=lambda h: (h["stage"], h["updated_at"] or ""), reverse=True)


def informe_ciclo(ciclo):
    eventos_ciclo = registro._eventos_ciclo(ciclo)
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
        eventos, _ = registro._leer_eventos()
        inicio = eventos.index(apertura)
        propios = [e for e in eventos[inicio + 1:]
                   if e.get("hipotesis") in ids and e.get("evento") in ("etapa", "intento")]
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
                caida = max(caidas, key=registro._fecha)
                recibo["caidas"][h] = {k: caida.get(k, "") for k in ("etapa", "veredicto", "motivo")}
            recibo["intentos"][h] = sum(e.get("evento") == "intento" for e in propios_h)
        recibo["snapshots"] = sorted({e.get("datos", {}).get("snapshot") for e in propios
                                      if e.get("evento") == "intento" and e.get("datos", {}).get("snapshot")})
        recibo["hashes_manifiesto"] = {s: registro.snapshot_manifest_hash(s) for s in recibo["snapshots"]}

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
