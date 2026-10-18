import re
from pathlib import Path

import yaml

from quantagent.fabrica.admision import UNIVERSO_AMPLIO
from quantagent.strategy.registry import cli_aliases

RAIZ = Path(__file__).parent.parent
DOI = re.compile(r"^10\.\d{4,9}/\S+$")
DOI_EN_TEXTO = re.compile(r"doi\.org/(\S+?)(?=[\s;,]|$)")
CLAVES = {"id", "prima", "idea", "fuente", "universo", "clase", "datos", "destino"}
DESTINOS = {"L12", "L13", "L14", "no-aplica"}

PRIMAS = yaml.safe_load((RAIZ / "catalogo" / "primas.yaml").read_text(encoding="utf-8"))["primas"]


def test_ids_unicos_y_claves_presentes():
    ids = [p["id"] for p in PRIMAS]
    assert len(ids) == len(set(ids))
    for p in PRIMAS:
        assert CLAVES <= p.keys(), f"{p['id']}: faltan {CLAVES - p.keys()}"
        assert {"activos", "criterio"} <= p["universo"].keys(), p["id"]
        assert p["datos"] in {"precios", "macro"}, p["id"]
        assert p["destino"] in DESTINOS, p["id"]
        assert p["fuente"], f"{p['id']}: sin fuente"


def test_no_aplica_exige_motivo():
    for p in PRIMAS:
        assert bool(p.get("motivo", "").strip()) == (p["destino"] == "no-aplica"), p["id"]


def _clases_validas() -> set[str]:
    clases = set(cli_aliases())
    try:  # P05.1 (piernas transversales) agrega quantagent/fabrica/transversal.py con su registro METRICAS
        from quantagent.fabrica.transversal import METRICAS
    except ImportError:
        return clases
    return clases | set(METRICAS)


def test_clase_es_null_o_alias_registrado():
    aliases = _clases_validas()
    for p in PRIMAS:
        assert p["clase"] is None or p["clase"] in aliases, f"{p['id']}: clase {p['clase']!r} no registrada"


def test_activos_dentro_del_universo_amplio():
    for p in PRIMAS:
        assert set(p["universo"]["activos"]) <= UNIVERSO_AMPLIO, p["id"]


def test_doi_con_forma_valida():
    for p in PRIMAS:
        for doi in p["fuente"]:
            assert DOI.match(doi), f"{p['id']}: DOI mal formado {doi!r}"


def test_hipotesis_citan_solo_doi_bien_formados():
    citados = 0
    for f in sorted((RAIZ / "hipotesis").glob("*.yaml")):
        fuente = str(yaml.safe_load(f.read_text(encoding="utf-8")).get("fuente", ""))
        for doi in DOI_EN_TEXTO.findall(fuente):
            citados += 1
            assert DOI.match(doi), f"{f.name}: DOI mal formado {doi!r}"
    assert citados, "ninguna hipótesis cita un DOI: el test no está verificando nada"
