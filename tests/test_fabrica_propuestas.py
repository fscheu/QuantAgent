"""QuantAgent-fqye: el chequeo de propuestas rechaza sin DOI, sin hipótesis y la tercera variante de una idea."""

import pytest
import yaml

from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica import propuestas
from quantagent.fabrica.propuestas import advertencias_estructurales, entradas_de_hipotesis, validar_propuesta


@pytest.fixture(autouse=True)
def crossref_simulado(monkeypatch):
    titulos = {"10.1016/j.jfineco.2011.11.003": "Time Series Momentum", "10.3905/jwm.2007.674809": "A Century of Evidence on Trend Following Investing", "10.3905/jpm.2017.44.1.015": "A Century of Evidence on Trend Following Investing", "10.1111/jofi.12513": "Volatility Managed Portfolios", "10.1016/j.jfineco.2017.11.002": "Carry", "10.1093/rfs/hhr068": "Common Risk Factors in Currency Markets", "10.1111/jofi.12096": "An Anatomy of Commodity Futures Risk Premia"}
    monkeypatch.setattr(propuestas, "_titulo_crossref", lambda doi: titulos[doi])


def _propuesta(**cambios):
    p = {"id": "p1", "idea": "momentum", "clase": "momentum-12m", "hipotesis_economica": "Subreacción a la información pública.",
         "fuente": ["10.1016/j.jfineco.2011.11.003"],
         "universo": {"activos": ["SPY", "TLT"], "criterio": "ETFs líquidos de la clase"}, "parametros_libres": 0,
         "tema": "tendencia", "fuente_de_retorno": "prima-de-riesgo", "cita_texto": ["Time Series Momentum"]}
    return {**p, **cambios}


def test_propuesta_completa_pasa():
    assert validar_propuesta(_propuesta()) is None


def test_advertencias_estructurales_se_derivan_sin_bloquear():
    propuesta = _propuesta(fidelidad="desvío: la cita es largo-corto; acá solo largo",
                           muestra_de_la_cita="1985-2012")
    assert advertencias_estructurales(propuesta) == [
        "desvio_de_la_cita", "muestra_solapada", "overlay_solo_largo"
    ]
    assert validar_propuesta(propuesta) is None


def test_propuesta_fiel_sin_solapamiento_ni_overlay_no_advierte():
    propuesta = _propuesta(fidelidad="igual a la cita", muestra_de_la_cita="1985-2006",
                           universo={"activos": ["SPY", "IEF", "CL"], "criterio": "diversificado"})
    assert advertencias_estructurales(propuesta) == []


def test_overlay_se_emite_solo_si_todas_las_piernas_son_largas_y_del_benchmark():
    propuesta = {"piernas": [{"lado": "largo", "activos": ["SPY"]},
                              {"lado": "corto", "activos": ["IEF"]}]}
    assert "overlay_solo_largo" not in advertencias_estructurales(propuesta)


def test_doi_rechaza_cita_ajena_y_deja_error_de_red_sin_verificar():
    propuesta = _propuesta()
    assert "no coincide" in validar_propuesta(propuesta, crossref=lambda _: "Unrelated Economics Paper")
    assert "sin verificar" in validar_propuesta(propuesta, crossref=lambda _: (_ for _ in ()).throw(OSError("offline")))


def test_propuestas_existentes_completas_pasan():
    for nombre in ("carry-futuros", "tendencia-larga-corta", "volatilidad-gestionada"):
        propuesta = yaml.safe_load(open(f"catalogo/propuestas/{nombre}.yaml", encoding="utf-8"))
        assert validar_propuesta(propuesta) is None


def test_tema_flujos_institucionales_con_slug_pasa():
    assert validar_propuesta(_propuesta(tema="flujos-institucionales")) is None


@pytest.mark.parametrize("cambios,motivo", [
    ({"fuente": []}, "DOI"),
    ({"fuente": ["Jegadeesh y Titman 1993"]}, "DOI"),
    ({"hipotesis_economica": "  "}, "hipotesis_economica"),
    ({"idea": ""}, "idea"),
    ({"id": ""}, "id"),
    ({"clase": ""}, "clase"),
    ({"universo": {"activos": ["SPY"]}}, "criterio"),
    ({"parametros_libres": 2}, "parametros_libres"),
    ({"tema": "tema-inventado"}, "tema"),
    ({"fuente_de_retorno": "otra"}, "fuente_de_retorno"),
])
def test_rechazos_con_motivo(cambios, motivo):
    assert motivo in validar_propuesta(_propuesta(**cambios))


@pytest.mark.parametrize("campo", ["tema", "fuente_de_retorno"])
def test_rechaza_campos_nuevos_ausentes(campo):
    propuesta = _propuesta()
    del propuesta[campo]
    assert campo in validar_propuesta(propuesta)


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
