import json
import re
import csv
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


def test_ciclos_sin_etapa3_cuenta_cierres_consecutivos_y_ignora_abiertos(raiz):
    for mes in range(1, 6):
        ciclo = f"2026-{mes:02d}"
        registro.abrir_ciclo(ciclo, [f"H{mes}"])
        registro.cerrar_ciclo(ciclo)
    registro.abrir_ciclo("2026-06", ["H6"])

    assert registro.ciclos_sin_etapa3() == 5

    registro.cerrar_ciclo("2026-06")
    assert registro.ciclos_sin_etapa3() == 6


def test_ciclo_con_pase_a_etapa3_reinicia_cuenta(raiz):
    for mes in range(1, 5):
        ciclo = f"2026-{mes:02d}"
        registro.abrir_ciclo(ciclo, [f"H{mes}"])
        registro.cerrar_ciclo(ciclo)
    registro.abrir_ciclo("2026-05", ["H5"])
    registro.registrar_evento("H5", "etapa", "3", "PASA", ciclo="2026-05")
    registro.cerrar_ciclo("2026-05")
    registro.abrir_ciclo("2026-06", ["H6"])
    registro.cerrar_ciclo("2026-06")

    assert registro.ciclos_sin_etapa3() == 1


def test_calendario_cruza_anio_y_usa_dias_configurados(raiz):
    from quantagent import default_config

    ciclos = registro.calendario_ciclos("2026-12", 2)
    assert [ciclo["ciclo"] for ciclo in ciclos] == ["2026-12", "2027-01"]
    assert ciclos[0]["propuesta"] == f"2026-12-{default_config.DIA_PROPUESTA_HIPOTESIS:02d}"
    assert ciclos[0]["embudo"] == f"2026-12-{default_config.DIA_INICIO_EMBUDO:02d}"
    assert ciclos[0]["revision_paper"] == f"2026-12-{default_config.DIA_REVISION_PAPER:02d}"
    assert ciclos[1]["propuesta"] == f"2027-01-{default_config.DIA_PROPUESTA_HIPOTESIS:02d}"


def test_calendario_informa_estados_del_registro(raiz):
    registro.abrir_ciclo("2026-01", ["H1"])
    registro.abrir_ciclo("2026-02", ["H2"])
    registro.registrar_evento("H2", "etapa", "3", "PASA", ciclo="2026-02")
    registro.cerrar_ciclo("2026-02")

    ciclos = registro.calendario_ciclos("2026-01", 3)
    assert ciclos[0]["estado"] == "abierto"
    assert ciclos[1]["estado"] == "cerrado"
    assert ciclos[1]["promovidas"] == 1
    assert ciclos[2]["estado"] == "pendiente"


def test_calendario_rechaza_desde_invalido(raiz):
    with pytest.raises(ValueError, match="ciclo inválido"):
        registro.calendario_ciclos("2026-13")


def test_cli_corte_informa_umbral_y_referencia():
    from click.testing import CliRunner
    from quantagent.cli.__main__ import cli

    ayuda = CliRunner().invoke(cli, ["registro", "corte", "--help"])
    assert ayuda.exit_code == 0
    assert "Show consecutive closed cycles without a stage 3 pass." in ayuda.output
    assert "Example: quantagent registro corte" in ayuda.output
    referencia = Path("docs/manual/reference/cli/registro.md").read_text()
    assert "quantagent registro corte" in referencia


def test_cli_ciclo_expone_help_epilog_y_referencia():
    from click.testing import CliRunner
    from quantagent.cli.__main__ import cli

    ayuda = CliRunner().invoke(cli, ["registro", "ciclo", "abrir", "--help"])
    assert ayuda.exit_code == 0
    assert "Open a cycle with up to five hypothesis IDs." in ayuda.output
    assert "Example: quantagent registro ciclo abrir 2026-11 H1 H2" in ayuda.output
    referencia = Path("docs/manual/reference/cli/registro/ciclo.md").read_text()
    assert "Example: quantagent registro ciclo cerrar 2026-11" in referencia


def _csv_importables(tmp_path):
    with (tmp_path / "hipotesis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fecha_utc", "hipotesis", "etapa", "veredicto", "motivo"])
        for i, h in enumerate(("H1", "H2", "H3"), 1):
            writer.writerow([f"2026-01-0{i}T00:00:00+00:00", h, "0", "PASA", "inicio"])
            writer.writerow([f"2026-01-0{i}T01:00:00+00:00", h, "1", "CORTA", "final"])
    with (tmp_path / "intentos.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fecha_utc", "estrategia", "activo", "snapshot", "desde", "hasta", "costos", "tamano", "params", "trades", "pnl", "sharpe", "hipotesis", "etapa", "sharpe_base", "estado"])
        writer.writerow(["2026-01-01T00:30:00+00:00", "s", "BTC", "snap", "", "", "", "", '{"x": 1}', "2", "3.5", "0.7", "H1", "0", "", ""])
        writer.writerow(["2026-01-02T00:30:00+00:00", "cartera", "cartera", "snap", "", "", "", "", "{}", "4", "", "0.9", "H2", "1", "", ""])


def test_importar_csv_preserva_estados_orden_e_idempotencia(raiz, tmp_path):
    _csv_importables(tmp_path)
    resultado = registro.importar_csv()

    assert resultado == {"etapas": {"leidos": 6, "agregados": 6},
                         "intentos": {"leidos": 2, "agregados": 2}, "omitidas": 0}
    assert {h: (e["etapa"], e["veredicto"]) for h, e in registro.estados_actuales().items()} == {
        "H1": ("1", "CORTA"), "H2": ("1", "CORTA"), "H3": ("1", "CORTA")}
    events = [json.loads(line) for line in raiz.read_text().splitlines()]
    assert [e["fecha_utc"] for e in events] == sorted(e["fecha_utc"] for e in events)
    assert events[0]["datos"]["importado"] is True
    assert registro.importar_csv()["etapas"]["agregados"] == 0
    assert registro.importar_csv()["intentos"]["agregados"] == 0


def test_importar_csv_dry_run_no_crea_registro(raiz, tmp_path):
    _csv_importables(tmp_path)
    from click.testing import CliRunner
    from quantagent.cli.__main__ import cli

    resultado = CliRunner().invoke(cli, ["registro", "importar", "--dry-run"])
    assert resultado.exit_code == 0
    assert "(dry-run) agregaría etapas 6 de 6" in resultado.output
    assert not raiz.exists()


def test_importar_csv_no_duplica_evento_vivo_y_omite_fila_mala(raiz, tmp_path):
    _csv_importables(tmp_path)
    registro.registrar_evento("H1", "etapa", "0", "PASA", "inicio",
                              fecha_utc="2026-02-01T00:00:00+00:00")
    with (tmp_path / "hipotesis.csv").open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["fecha-incorrecta", "H4", "9", "INVALIDO", ""])

    resultado = registro.importar_csv()
    assert resultado["omitidas"] == 1
    assert resultado["etapas"]["agregados"] == 5
    assert len(registro.historial("H1")) == 3
    assert registro.estado_actual("H1")["fecha_utc"] == "2026-02-01T00:00:00+00:00"
    assert registro.estados_actuales()["H1"]["etapa"] == "0"

def test_cli_calendario_muestra_ciclos_y_corte(raiz):
    from click.testing import CliRunner
    from quantagent.cli.__main__ import cli

    for mes in range(1, 7):
        ciclo = f"2026-{mes:02d}"
        registro.abrir_ciclo(ciclo, [f"H{mes}"])
        registro.cerrar_ciclo(ciclo)

    resultado = CliRunner().invoke(cli, ["registro", "calendario", "2026-12"])
    assert resultado.exit_code == 0
    assert "2026-12: propuesta 2026-12-01; embudo 2026-12-08; revisión paper 2026-12-25; pendiente" in resultado.output
    assert "Corte: 6/6 ciclos cerrados consecutivos sin pasar etapa 3." in resultado.output
    assert "Toca revisar el enfoque." in resultado.output
