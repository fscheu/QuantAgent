import pytest

from quantagent.fabrica.candado import CandadoError, abierto, abrir


@pytest.fixture(autouse=True)
def _raiz(tmp_path, monkeypatch):
    monkeypatch.setattr("quantagent.fabrica.candado.snapshot_root", lambda: tmp_path)
    return tmp_path


def test_primera_ok_segunda_falla_con_id_y_fecha(_raiz):
    assert not abierto("H1", "prueba")
    abrir("H1", "prueba")
    assert abierto("H1", "prueba")
    fecha = (_raiz / "hipotesis-candado.csv").read_text().strip().split(",")[2]
    with pytest.raises(CandadoError) as e:
        abrir("H1", "prueba")
    assert "H1" in str(e.value) and fecha in str(e.value)
    assert len((_raiz / "hipotesis-candado.csv").read_text().splitlines()) == 1


def test_pares_independientes():
    abrir("H1", "prueba")
    abrir("H1", "reserva")
    abrir("H2", "prueba")
    assert abierto("H1", "reserva") and abierto("H2", "prueba")
    assert not abierto("H2", "reserva")


def test_periodo_invalido():
    with pytest.raises(ValueError):
        abrir("H1", "train")
    with pytest.raises(ValueError):
        abierto("H1", "train")
