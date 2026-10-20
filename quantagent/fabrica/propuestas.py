"""Chequeo de propuestas (QuantAgent-fqye): una entrada de `catalogo/propuestas/<id>.yaml` cumple las reglas del DC.

Formato de la entrada (el mismo estilo que `catalogo/primas.yaml`):
    id, idea (id de la idea de la que deriva), hipotesis_economica (texto),
    fuente: [<DOI>, ...], universo: {activos: [...], criterio: <cómo se definió>}, parametros_libres: 0

El chequeo es de forma: no verifica en la red que el DOI exista.
"""

import re

DOI = re.compile(r"^10\.\d{4,9}/\S+$")
MAX_VARIANTES = 2  # por idea y por lote (DC 2026-10-19, D3.7); la tercera se rechaza: es otra idea y va con su propia cita


def _idea(p):
    return str(p.get("idea") or "").strip().lower()


def validar_propuesta(p, previas=()):
    """Motivo de rechazo de la propuesta `p` (dict), o None si pasa. `previas`: las ya aceptadas en el lote."""
    if not str(p.get("id") or "").strip():
        return "falta id"
    if not _idea(p):
        return "falta idea: de qué idea deriva"
    if not str(p.get("hipotesis_economica") or "").strip():
        return "falta hipotesis_economica"
    fuente = p.get("fuente")
    if not fuente or not all(isinstance(d, str) and DOI.match(d) for d in fuente):
        return "fuente sin DOI válido (10.xxxx/...)"
    universo = p.get("universo") or {}
    if not universo.get("activos") or not str(universo.get("criterio") or "").strip():
        return "universo sin activos o sin criterio de definición"
    if p.get("parametros_libres") != 0:
        return "debe declarar parametros_libres: 0"
    if sum(_idea(q) == _idea(p) and q.get("id") != p["id"] for q in previas) >= MAX_VARIANTES:
        return f"tercera variante de la idea {p['idea']} en el lote (máximo {MAX_VARIANTES})"
    return None
