import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from quantagent.cli import hipotesis
from quantagent.cli.backtest import backtest_group
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica import registro_db
from quantagent.fabrica.estado import causa_de_evento
from quantagent.fabrica.filtro import MIN_SHARPE_ETAPA1, MIN_TRADES_ETAPA1


@pytest.mark.parametrize("sharpe_motor", [0.0, 0.5])
def test_alarma_corta_antes_del_candado(tmp_path, monkeypatch, db_temporal, sharpe_motor):
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
    estados = registro_db.eventos_etapa(h.id)
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


@pytest.mark.parametrize("sharpe,trades,metric,value,cause", [
    (0.5, 12, "trades", 12, "few_trades"),
    (0.04, 40, "sharpe", 0.04, "sharpe_below"),
])
def test_etapa1_guarda_medida_del_corte(tmp_path, monkeypatch, db_temporal,
                                        sharpe, trades, metric, value, cause):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    archivo = Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml"
    h = cargar_hipotesis(archivo)
    monkeypatch.setattr(hipotesis, "admitir_hipotesis", lambda *a: (h, True, []))
    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", lambda *a: (
        [], {}, "CORTA", "corte sintético", sharpe, trades,
    ))
    result = CliRunner().invoke(backtest_group, ["hipotesis", "correr", str(archivo),
        "--snapshot", "sintetico", "--carpeta", str(tmp_path / "motor"), "--hasta-etapa", "1"])
    assert result.exit_code == 1, result.output
    evento = registro_db.eventos_etapa(h.id)[-1]
    assert evento["datos"]["measure"]["metric"] == metric
    assert evento["datos"]["measure"]["value"] == value
    assert evento["datos"]["measure"]["threshold"] == (MIN_TRADES_ETAPA1 if metric == "trades" else MIN_SHARPE_ETAPA1)
    assert causa_de_evento(evento)[0] == cause


def test_etapa0_guarda_causa_admission(tmp_path, monkeypatch, db_temporal):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    archivo = Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml"
    h = cargar_hipotesis(archivo)
    monkeypatch.setattr(hipotesis, "admitir_hipotesis", lambda *a: (h, False, ["rechazo"]))
    result = CliRunner().invoke(backtest_group, ["hipotesis", "correr", str(archivo),
        "--snapshot", "sintetico", "--carpeta", str(tmp_path / "motor")])
    assert result.exit_code == 1, result.output
    evento = registro_db.eventos_etapa(h.id)[-1]
    assert evento["datos"]["cause_code"] == "admission"
    assert "measure" not in evento["datos"]


def test_error_etapa2_guarda_causa_runtime(tmp_path, monkeypatch, db_temporal):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    archivo = Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml"
    h = cargar_hipotesis(archivo)
    monkeypatch.setattr(hipotesis, "admitir_hipotesis", lambda *a: (h, True, []))
    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", lambda *a: ([], {}, "PASA", "ok", 0.5, 40))
    monkeypatch.setattr(hipotesis, "validar_etapa2", lambda *a: (_ for _ in ()).throw(RuntimeError("falla")))
    result = CliRunner().invoke(backtest_group, ["hipotesis", "correr", str(archivo),
        "--snapshot", "sintetico", "--carpeta", str(tmp_path / "motor")])
    evento = registro_db.eventos_etapa(h.id)[-1]
    assert evento["veredicto"] == "ERROR"
    assert evento["datos"]["cause_code"] == "runtime_error"
    assert "measure" not in evento["datos"]
