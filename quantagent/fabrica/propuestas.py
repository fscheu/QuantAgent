"""Chequeo de propuestas (QuantAgent-fqye): una entrada de `catalogo/propuestas/<id>.yaml` cumple las reglas del DC.

Formato de la entrada (el mismo estilo que `catalogo/primas.yaml`):
    id, idea (id de la idea de la que deriva), hipotesis_economica (texto),
    fuente: [<DOI>, ...], universo: {activos: [...], criterio: <cómo se definió>}, parametros_libres: 0,
    clase (clase de señal), ventana (opcional: los parámetros de la señal, como `parametros` de una pierna)

El chequeo contrasta el título de cada DOI con su cita mediante Crossref.
La originalidad (QuantAgent-xiya) es estructural: misma clase, mismos activos y misma ventana que una entrada existente.
"""

import re
import json
import urllib.error
import urllib.request
from urllib.parse import quote

DOI = re.compile(r"^10\.\d{4,9}/\S+$")
MAX_VARIANTES = 2  # por idea y por lote (DC 2026-10-19, D3.7); la tercera se rechaza: es otra idea y va con su propia cita
TEMAS = {"tendencia", "momentum-transversal", "baja-volatilidad", "valor", "carry", "calendario", "flujos-institucionales", "prima-de-volatilidad", "reversion-y-liquidez", "macro"}
FUENTES_DE_RETORNO = {"prima-de-riesgo", "provision-de-liquidez", "sesgo-de-comportamiento", "friccion"}
BENCHMARK = frozenset({"AGG", "DBC", "EEM", "EFA", "EWJ", "EWZ", "FXE", "FXI", "GLD", "IEF", "IWM", "LQD", "QQQ", "SHY", "SLV", "SPY", "TIP", "TLT", "USO", "VNQ", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"})


def advertencias_estructurales(p):
    """Devuelve códigos de advertencia de la propuesta, sin afectar su admisión."""
    advertencias = []
    fidelidad = str(p.get("fidelidad") or "").strip().lower()
    if fidelidad.startswith(("desvío:", "desvio:")):
        advertencias.append("desvio_de_la_cita")

    muestra = str(p.get("muestra_de_la_cita") or "")
    anios = re.findall(r"(?:19|20)\d{2}", muestra)
    if anios and 2007 <= int(anios[-1]) <= 2018:
        advertencias.append("muestra_solapada")

    piernas = p.get("piernas") or [p]
    if piernas and all(
        str(pierna.get("lado", "largo")).lower() == "largo"
        and (activos := (pierna["universo"].get("activos") if isinstance(pierna.get("universo"), dict)
                        else None) or pierna.get("activos"))
        and set(activos) <= BENCHMARK
        for pierna in piernas
    ):
        advertencias.append("overlay_solo_largo")
    return advertencias


def _titulo_crossref(doi):
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    request = urllib.request.Request(url, headers={"User-Agent": "QuantAgent/1.0"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())["message"]["title"][0]


def _coincide_titulo(titulo, cita):
    def _palabras(s):
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    palabras_titulo = _palabras(titulo)
    return bool(palabras_titulo) and len(palabras_titulo & _palabras(cita)) / len(palabras_titulo) >= 0.6


def _idea(p):
    return str(p.get("idea") or "").strip().lower()


def _estructura(e):
    return (e.get("clase"), frozenset((e.get("universo") or {}).get("activos") or ()), repr(sorted((e.get("ventana") or {}).items())))


def entradas_de_hipotesis(h):
    """Las piernas de una hipótesis ya probada, en el formato de una propuesta (para `existentes`)."""
    return [{"id": h.id, "clase": p.clase, "universo": {"activos": p.activos}, "ventana": p.parametros} for p in h.piernas]


def validar_propuesta(p, previas=(), existentes=(), crossref=None):
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
    if p.get("tema") not in TEMAS:
        return "tema ausente o fuera de lista"
    if p.get("fuente_de_retorno") not in FUENTES_DE_RETORNO:
        return "fuente_de_retorno ausente o fuera de lista"
    citas = p.get("cita_texto")
    if not isinstance(citas, list) or len(citas) != len(fuente):
        return "cita_texto debe tener una cita por DOI"
    crossref = crossref or _titulo_crossref
    for doi, cita in zip(fuente, citas):
        try:
            titulo = crossref(doi)
        except (OSError, TimeoutError, urllib.error.URLError, ValueError, KeyError, IndexError):
            return f"{doi}: sin verificar (Crossref no disponible)"
        if not _coincide_titulo(titulo, str(cita)):
            return f"{doi}: título Crossref no coincide con la cita"
    if sum(_idea(q) == _idea(p) and q.get("id") != p["id"] for q in previas) >= MAX_VARIANTES:
        return f"tercera variante de la idea {p['idea']} en el lote (máximo {MAX_VARIANTES})"
    parecida = next((e["id"] for e in existentes if _estructura(e) == _estructura(p)), None)
    return f"parecida a {parecida}: misma clase, activos y ventana" if parecida else None
