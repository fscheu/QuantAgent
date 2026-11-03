import pytest

from quantagent.fabrica.estado import registrar_estado, ultimo_estado


def test_dos_registros_y_ultimo_estado(db_temporal):
    registrar_estado("H1", "0", "PASA")
    registrar_estado("H2", "1", "PASA")
    registrar_estado("H1", "1", "CORTA", "sin edge")
    u = ultimo_estado("H1")
    assert (u["etapa"], u["veredicto"], u["motivo"]) == ("1", "CORTA", "sin edge")


def test_sin_filas_none_y_motivo_roundtrip(db_temporal):
    assert ultimo_estado("H9") is None
    registrar_estado("H1", "2", "CORTA", 'sharpe < 0.5, "ruido"')
    assert ultimo_estado("H9") is None
    assert ultimo_estado("H1")["motivo"] == 'sharpe < 0.5, "ruido"'


@pytest.mark.parametrize("etapa,veredicto", [("4", "PASA"), ("0", "OK")])
def test_invalidos_levantan_y_no_escriben(db_temporal, etapa, veredicto):
    registrar_estado("H1", "0", "PASA")
    from quantagent.fabrica.registro_db import eventos
    antes = eventos()
    with pytest.raises(ValueError):
        registrar_estado("H1", etapa, veredicto)
    assert eventos() == antes


def test_ultimo_estado_usa_eventos_ordenados_por_fecha(db_temporal):
    from quantagent.fabrica.registro_db import registrar_evento
    registrar_evento("H1", "etapa", "0", "PASA", "viejo", fecha_utc="2026-01-01T00:00:00+00:00")
    registrar_evento("H1", "etapa", "2", "CORTA", "nuevo", fecha_utc="2099-02-01T00:00:00+00:00")
    assert ultimo_estado("H1")["etapa"] == "2"
