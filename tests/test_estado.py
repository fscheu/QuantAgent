import pytest

from quantagent.fabrica.estado import registrar_estado, ultimo_estado


@pytest.fixture(autouse=True)
def _raiz(tmp_path, monkeypatch):
    monkeypatch.setattr("quantagent.fabrica.estado.snapshot_root", lambda: tmp_path)
    return tmp_path / "hipotesis.csv"


def test_dos_registros_dejan_dos_filas_y_ultimo_es_el_segundo(_raiz):
    registrar_estado("H1", "0", "PASA")
    registrar_estado("H2", "1", "PASA")
    registrar_estado("H1", "1", "CORTA", "sin edge")
    # encabezado + 3 filas, un solo encabezado
    assert len(_raiz.read_text().splitlines()) == 4
    u = ultimo_estado("H1")
    assert (u["etapa"], u["veredicto"], u["motivo"]) == ("1", "CORTA", "sin edge")


def test_sin_filas_none_y_motivo_con_coma_y_comillas_roundtrip():
    assert ultimo_estado("H9") is None
    registrar_estado("H1", "2", "CORTA", 'sharpe < 0.5, "ruido"')
    assert ultimo_estado("H9") is None
    assert ultimo_estado("H1")["motivo"] == 'sharpe < 0.5, "ruido"'


@pytest.mark.parametrize("etapa,veredicto", [("4", "PASA"), ("0", "OK")])
def test_invalidos_levantan_y_no_escriben(_raiz, etapa, veredicto):
    registrar_estado("H1", "0", "PASA")
    antes = _raiz.read_text()
    with pytest.raises(ValueError):
        registrar_estado("H1", etapa, veredicto)
    assert _raiz.read_text() == antes


def test_invalido_no_crea_archivo(_raiz):
    with pytest.raises(ValueError):
        registrar_estado("H1", "4", "PASA")
    assert not _raiz.exists()
