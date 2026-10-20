"""Chequeo de propuestas (QuantAgent-fqye): una entrada de `catalogo/propuestas/<id>.yaml` cumple las reglas del DC.

Formato de la entrada (el mismo estilo que `catalogo/primas.yaml`):
    id, idea (id de la idea de la que deriva), hipotesis_economica (texto),
    fuente: [<DOI>, ...], universo: {activos: [...], criterio: <cómo se definió>}, parametros_libres: 0,
    clase (clase de señal), ventana (opcional: los parámetros de la señal, como `parametros` de una pierna)

El chequeo es de forma: no verifica en la red que el DOI exista.
La originalidad (QuantAgent-xiya) es estructural: misma clase, mismos activos y misma ventana que una entrada existente.
"""

import re

DOI = re.compile(r"^10\.\d{4,9}/\S+$")
MAX_VARIANTES = 2  # por idea y por lote (DC 2026-10-19, D3.7); la tercera se rechaza: es otra idea y va con su propia cita


def _idea(p):
    return str(p.get("idea") or "").strip().lower()


def _estructura(e):
    return (e.get("clase"), frozenset((e.get("universo") or {}).get("activos") or ()), repr(sorted((e.get("ventana") or {}).items())))


def entradas_de_hipotesis(h):
    """Las piernas de una hipótesis ya probada, en el formato de una propuesta (para `existentes`)."""
    return [{"id": h.id, "clase": p.clase, "universo": {"activos": p.activos}, "ventana": p.parametros} for p in h.piernas]


def validar_propuesta(p, previas=(), existentes=()):
    """Motivo de rechazo de la propuesta `p` (dict), o None si pasa. `previas`: las ya aceptadas en el lote.

    `existentes`: entradas del catálogo e hipótesis probadas; una con la misma estructura la marca parecida y no se corre.
    """
    if not str(p.get("id") or "").strip():
        return "falta id"
    if not str(p.get("clase") or "").strip():
        return "falta clase de señal"
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
    parecida = next((e["id"] for e in existentes if _estructura(e) == _estructura(p)), None)
    return f"parecida a {parecida}: misma clase, activos y ventana" if parecida else None
