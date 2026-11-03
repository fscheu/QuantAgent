import json
import re
from pathlib import Path

import pytest

from quantagent.fabrica import registro


@pytest.fixture
def raiz(tmp_path, monkeypatch):
    monkeypatch.setattr("quantagent.fabrica.registro.snapshot_root", lambda: tmp_path)
    return tmp_path / "registro-hipotesis.jsonl"


def test_dos_llamadas_agregan_lineas_y_preservan_la_primera(raiz):
    registro.registrar_evento("H1", "admision", motivo="entra")
    primera = raiz.read_bytes().splitlines(keepends=True)[0]
    registro.registrar_evento("H1", "etapa", "0", "PASA")

    lineas = raiz.read_bytes().splitlines(keepends=True)
    assert len(lineas) == 2
    assert lineas[0] == primera
    assert json.loads(lineas[0])["version_esquema"] == 1


def test_estado_actual_es_ultima_etapa_e_ignora_otros_eventos(raiz):
    registro.registrar_evento("H1", "etapa", "0", "PASA")
    registro.registrar_evento("H1", "intento", datos={"snapshot": "s1"})
    registro.registrar_evento("H1", "etapa", "1", "CORTA", "sin edge")
    registro.registrar_evento("H1", "ciclo", ciclo="2026-11")

    assert registro.estado_actual("H1")["etapa"] == "1"
    assert registro.estado_actual("H1")["veredicto"] == "CORTA"


def test_sin_eventos_retorna_vacios(raiz):
    assert registro.historial("H1") == []
    assert registro.estado_actual("H1") is None
    assert registro.estados_actuales() == {}


@pytest.mark.parametrize(
    "kwargs, mensaje",
    [
        ({"etapa": "4", "veredicto": "PASA"}, "etapa inválida: '4'; debe ser una de ('0', '1', '2', '3')"),
        ({"etapa": "0", "veredicto": "OK"}, "veredicto inválido: 'OK'; debe ser uno de ('PASA', 'CORTA', 'ALARMA', 'ERROR')"),
    ],
)
def test_etapa_y_veredicto_invalidos(kwargs, mensaje, raiz):
    with pytest.raises(ValueError, match=re.escape(mensaje)):
        registro.registrar_evento("H1", "etapa", **kwargs)


def test_ciclo_invalido(raiz):
    with pytest.raises(ValueError, match="ciclo inválido"):
        registro.registrar_evento("H1", "ciclo", ciclo="2026-13")


def test_linea_corrupta_en_medio_se_ignora_y_cuenta(raiz):
    registro.registrar_evento("H1", "admision")
    with raiz.open("a", encoding="utf-8") as f:
        f.write("{mal json}\n[]\n")
    registro.registrar_evento("H2", "etapa", "0", "PASA")

    assert len(registro.historial("H1")) == 1
    assert registro.estado_actual("H2")["veredicto"] == "PASA"
    assert registro.lineas_corruptas() == 2


def test_estados_actuales_con_dos_hipotesis(raiz):
    registro.registrar_evento("H1", "etapa", "0", "PASA")
    registro.registrar_evento("H2", "etapa", "1", "CORTA")
    registro.registrar_evento("H1", "etapa", "2", "PASA")

    assert {k: v["etapa"] for k, v in registro.estados_actuales().items()} == {
        "H1": "2",
        "H2": "1",
    }


def test_abrir_ciclo_limita_a_cinco_hipotesis(raiz):
    with pytest.raises(ValueError, match="máximo 5"):
        registro.abrir_ciclo("2026-11", ["H1", "H2", "H3", "H4", "H5", "H6"])

    registro.abrir_ciclo("2026-11", ["H1", "H2", "H3", "H4", "H5"])
    assert json.loads(raiz.read_text().splitlines()[0])["datos"]["hipotesis"] == ["H1", "H2", "H3", "H4", "H5"]


def test_cerrar_sin_promovidas_guarda_recibo_y_es_inmutable(raiz):
    registro.abrir_ciclo("2026-11", ["H1"])
    registro.registrar_evento("H1", "etapa", "0", "CORTA", "filtro", ciclo="2026-11")
    recibo = registro.cerrar_ciclo("2026-11")

    assert recibo["promovidas"] == 0
    assert recibo["etapas"] == {"0": 1, "1": 0, "2": 0, "3": 0}
    assert json.loads(raiz.read_text().splitlines()[-1])["motivo"]
    with pytest.raises(ValueError, match="ya fue cerrado"):
        registro.cerrar_ciclo("2026-11")


def test_recibo_se_reconstruye_solo_desde_jsonl(raiz, tmp_path):
    registro.abrir_ciclo("2026-11", ["H1"])
    registro.registrar_evento("H1", "intento", ciclo="2026-11", datos={"snapshot": "ausente"})
    registro.registrar_evento("H1", "etapa", "3", "PASA", ciclo="2026-11")
    esperado = registro.cerrar_ciclo("2026-11")
    (tmp_path / "hipotesis.csv").unlink(missing_ok=True)
    (tmp_path / "intentos.csv").unlink(missing_ok=True)

    assert registro.recibo_ciclo("2026-11") == esperado
    assert esperado["intentos"] == {"H1": 1}
    assert esperado["promovidas"] == 1
    assert esperado["hashes_manifiesto"] == {"ausente": None}


def test_cli_ciclo_expone_help_epilog_y_referencia():
    from click.testing import CliRunner
    from quantagent.cli.__main__ import cli

    ayuda = CliRunner().invoke(cli, ["registro", "ciclo", "abrir", "--help"])
    assert ayuda.exit_code == 0
    assert "Open a cycle with up to five hypothesis IDs." in ayuda.output
    assert "Example: quantagent registro ciclo abrir 2026-11 H1 H2" in ayuda.output
    referencia = Path("docs/manual/reference/cli/registro/ciclo.md").read_text()
    assert "Example: quantagent registro ciclo cerrar 2026-11" in referencia
