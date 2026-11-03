"""Vistas de lectura del registro de hipótesis."""

from collections import Counter
from datetime import datetime, timedelta, timezone

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
