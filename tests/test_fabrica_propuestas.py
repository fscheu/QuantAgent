"""QuantAgent-fqye: el chequeo de propuestas rechaza sin DOI, sin hipótesis y la tercera variante de una idea."""

import pytest
import yaml

from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica.propuestas import entradas_de_hipotesis, validar_propuesta


def _propuesta(**cambios):
    p = {"id": "p1", "idea": "momentum", "clase": "momentum-12m", "hipotesis_economica": "Subreacción a la información pública.",
         "fuente": ["10.1016/j.jfineco.2011.11.003"],
         "universo": {"activos": ["SPY", "TLT"], "criterio": "ETFs líquidos de la clase"}, "parametros_libres": 0}
    return {**p, **cambios}


def test_propuesta_completa_pasa():
    assert validar_propuesta(_propuesta()) is None


@pytest.mark.parametrize("cambios,motivo", [
    ({"fuente": []}, "DOI"),
    ({"fuente": ["Jegadeesh y Titman 1993"]}, "DOI"),
    ({"hipotesis_economica": "  "}, "hipotesis_economica"),
    ({"idea": ""}, "idea"),
    ({"id": ""}, "id"),
    ({"clase": ""}, "clase"),
    ({"universo": {"activos": ["SPY"]}}, "criterio"),
    ({"parametros_libres": 2}, "parametros_libres"),
])
def test_rechazos_con_motivo(cambios, motivo):
    assert motivo in validar_propuesta(_propuesta(**cambios))


def test_tercera_variante_de_la_misma_idea_se_rechaza():
    lote = []
    for i in (1, 2):
        p = _propuesta(id=f"v{i}")
        assert validar_propuesta(p, lote) is None
        lote.append(p)
    assert "tercera variante" in validar_propuesta(_propuesta(id="v3"), lote)
    assert "tercera variante" in validar_propuesta(_propuesta(id="v4", idea=" Momentum "), lote)
    assert validar_propuesta(_propuesta(id="otra", idea="carry"), lote) is None


def test_parecida_a_hipotesis_probada_solo_si_coinciden_clase_activos_y_ventana():
    h = cargar_hipotesis("hipotesis/momentum-12m.yaml")
    existentes = entradas_de_hipotesis(h)
    igual = _propuesta(universo={"activos": list(h.activos), "criterio": "los 12 ETFs"}, ventana=h.parametros)
    assert "parecida a momentum-12m" in validar_propuesta(igual, existentes=existentes)
    renta_fija = {"activos": ["AGG", "LQD", "IEF"], "criterio": "ETFs de renta fija"}
    assert validar_propuesta({**igual, "universo": renta_fija}, existentes=existentes) is None
    assert validar_propuesta({**igual, "ventana": {"lookback": [126]}}, existentes=existentes) is None
    assert validar_propuesta({**igual, "clase": "sma-cross"}, existentes=existentes) is None


def test_parecida_a_entrada_del_catalogo():
    primas = yaml.safe_load(open("catalogo/primas.yaml", encoding="utf-8"))["primas"]
    prima = next(e for e in primas if e["id"] == "momentum-series-de-tiempo")
    p = _propuesta(clase=prima["clase"], universo=prima["universo"])
    assert "parecida a momentum-series-de-tiempo" in validar_propuesta(p, existentes=primas)
