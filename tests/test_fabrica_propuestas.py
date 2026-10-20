"""QuantAgent-fqye: el chequeo de propuestas rechaza sin DOI, sin hipótesis y la tercera variante de una idea."""

import pytest

from quantagent.fabrica.propuestas import validar_propuesta


def _propuesta(**cambios):
    p = {"id": "p1", "idea": "momentum", "hipotesis_economica": "Subreacción a la información pública.",
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
