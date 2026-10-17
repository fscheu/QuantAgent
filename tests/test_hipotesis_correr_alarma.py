import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from quantagent.cli import hipotesis
from quantagent.cli.backtest import backtest_group
from quantagent.fabrica.hipotesis import cargar_hipotesis


@pytest.mark.parametrize("sharpe_motor", [0.0, 0.5])
def test_alarma_corta_antes_del_candado(tmp_path, monkeypatch, sharpe_motor):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    archivo = Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml"
    h = cargar_hipotesis(archivo)
    monkeypatch.setattr(hipotesis, "admitir_hipotesis", lambda *a: (h, True, []))
    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", lambda *a: (
        [["SPY", 0.10, 30], ["TLT", 0.5, 30]], {}, "PASA", 0,
    ))
    monkeypatch.setattr(hipotesis, "validar_etapa2", lambda *a: ("PASA", [
        {"activo": "SPY", "sharpe_tasa_0": sharpe_motor, "v30": "NO PASA", "motivos": []},
        {"activo": "TLT", "sharpe_tasa_0": 1.0, "v30": "PASA", "motivos": []},
    ]))
    vistos = []

    def etapa3(*args):
        vistos.append(True)
        return "PASA", [{"activo": "TLT", "v30": "PASA", "dsr": 0.99, "motivos": []}]

    def compare_prohibido(*args, **kwargs):
        pytest.fail("No debe agregar llamadas a compare")

    monkeypatch.setattr(hipotesis, "validar_etapa3", etapa3)
    monkeypatch.setattr("subprocess.run", compare_prohibido)
    result = CliRunner().invoke(backtest_group, ["hipotesis", "correr", str(archivo),
        "--snapshot", "sintetico", "--carpeta", str(tmp_path / "motor")])
    alarma = sharpe_motor == 0
    assert result.exit_code == int(alarma), result.output
    with (tmp_path / "hipotesis.csv").open() as f:
        estados = list(csv.DictReader(f))
    assert [(r["etapa"], r["veredicto"]) for r in estados] == [
        ("0", "PASA"), ("1", "PASA"), ("2", "ALARMA" if alarma else "PASA"),
    ] + ([] if alarma else [("3", "PASA")])
    assert vistos == ([] if alarma else [True])
    candado = tmp_path / "hipotesis-candado.csv"
    assert candado.exists() == (not alarma)
    if alarma:
        assert estados[-1]["motivo"] == "filtro optimista en SPY"
        assert "Llegó a la etapa 2: ALARMA (filtro optimista en SPY)" in result.output
    else:
        with candado.open() as f:
            assert [r[:2] for r in csv.reader(f)] == [[h.id, "prueba"]]
