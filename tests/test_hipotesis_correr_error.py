import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from quantagent.cli import hipotesis
from quantagent.cli.backtest import backtest_group
from quantagent.fabrica.hipotesis import cargar_hipotesis


@pytest.mark.parametrize("etapa", ["2", "3"])
@pytest.mark.parametrize("mensaje", ["falta un CSV", "x" * 250])
def test_error_registra_estado_y_propaga(tmp_path, monkeypatch, etapa, mensaje):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    archivo = Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml"
    h = cargar_hipotesis(archivo)
    monkeypatch.setattr(hipotesis, "admitir_hipotesis", lambda *a: (h, True, []))
    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", lambda *a: (
        [["SPY", 0.5, 30]], {}, "PASA", 0,
    ))
    monkeypatch.setattr(hipotesis, "validar_etapa2", lambda *a: ("PASA", [
        {"activo": "SPY", "sharpe_tasa_0": 0.5, "v30": "PASA", "motivos": []},
    ]))
    error = RuntimeError(mensaje)

    def fallar(*args):
        raise error

    def etapa3_prohibida(*args):
        pytest.fail("Un error en etapa 2 debe impedir la etapa 3")

    monkeypatch.setattr(hipotesis, "validar_etapa3", etapa3_prohibida)
    monkeypatch.setattr(hipotesis, f"validar_etapa{etapa}", fallar)
    result = CliRunner().invoke(backtest_group, ["hipotesis", "correr", str(archivo),
        "--snapshot", "sintetico", "--carpeta", str(tmp_path / "motor")])
    assert result.exit_code != 0
    assert result.exception is error
    with (tmp_path / "hipotesis.csv").open() as f:
        estados = list(csv.DictReader(f))
    assert [(r["etapa"], r["veredicto"]) for r in estados] == [
        ("0", "PASA"), ("1", "PASA"),
    ] + ([("2", "PASA")] if etapa == "3" else []) + [(etapa, "ERROR")]
    assert estados[-1]["hipotesis"] == h.id
    assert estados[-1]["motivo"] == f"RuntimeError: {mensaje}"[:200]
    candado = tmp_path / "hipotesis-candado.csv"
    assert candado.exists() == (etapa == "3")
    if etapa == "3":
        with candado.open() as f:
            assert [r[:2] for r in csv.reader(f)] == [[h.id, "prueba"]]


def test_error_motivo_traceback_5_lineas(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    archivo = Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml"
    h = cargar_hipotesis(archivo)
    monkeypatch.setattr(hipotesis, "admitir_hipotesis", lambda *a: (h, True, []))
    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", lambda *a: (
        [["SPY", 0.5, 30]], {}, "PASA", 0,
    ))
    tb_5_lineas = (
        "Traceback (most recent call last):\n"
        "  File 'motor.py', line 12, in validar_etapa2\n"
        "    correr_proceso()\n"
        "  File 'subproceso.py', line 45, in correr_proceso\n"
        "ValueError: argumento invalido en el motor\n"
    )
    error = RuntimeError(tb_5_lineas)

    def fallar(*args):
        raise error

    monkeypatch.setattr(hipotesis, "validar_etapa2", fallar)
    result = CliRunner().invoke(backtest_group, ["hipotesis", "correr", str(archivo),
        "--snapshot", "sintetico", "--carpeta", str(tmp_path / "motor")])
    assert result.exit_code != 0
    assert result.exception is error
    with (tmp_path / "hipotesis.csv").open() as f:
        estados = list(csv.DictReader(f))
    assert estados[-1]["veredicto"] == "ERROR"
    assert estados[-1]["motivo"] == "RuntimeError: ValueError: argumento invalido en el motor"


def test_motivo_error_ultima_linea_no_vacia():
    tb = (
        "Traceback (most recent call last):\n"
        "  File 'engine.py', line 10\n"
        "    calc()\n"
        "  File 'calc.py', line 20\n"
        "ZeroDivisionError: division by zero\n"
        "   \n"
    )
    assert hipotesis.motivo_error(RuntimeError(tb)) == "RuntimeError: ZeroDivisionError: division by zero"

