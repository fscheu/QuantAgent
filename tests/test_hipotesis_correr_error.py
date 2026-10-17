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
