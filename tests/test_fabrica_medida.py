import pytest

from quantagent.fabrica.estado import causa_de_evento, medida, registrar_estado


def test_medida_causa_persisten_en_evento(db_temporal):
    from quantagent.fabrica.registro_db import eventos_etapa

    registro = medida("sharpe", 0.72, 0.8, ">=", "ratio")
    registrar_estado("H-medida", "1", "CORTA", "sharpe", registro, "sharpe_below")
    evento = eventos_etapa("H-medida")[0]
    assert evento["datos"]["measure"] == {
        "metric": "sharpe", "value": 0.72, "threshold": 0.8,
        "comparator": ">=", "unit": "ratio",
    }
    assert causa_de_evento(evento) == ("sharpe_below", "Sharpe por debajo del umbral")


def test_evento_historico_usa_motivo(db_temporal):
    from quantagent.fabrica.registro_db import eventos_etapa, registrar_evento

    registrar_evento("H-historico", "etapa", "0", "CORTA", "sharpe bajo; pocos trades")
    assert causa_de_evento(eventos_etapa("H-historico")[0]) == ("pocos trades", "pocos trades")


def test_medida_rechaza_comparator_invalido():
    with pytest.raises(ValueError, match="comparator inválido"):
        medida("sharpe", 0.7, 0.8, ">", "ratio")


def test_registro_viejo_no_guarda_claves_vacias(db_temporal):
    from quantagent.fabrica.registro_db import eventos_etapa

    registrar_estado("H-viejo", "0", "PASA")
    assert eventos_etapa("H-viejo")[0]["datos"] == {}
