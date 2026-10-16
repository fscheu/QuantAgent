import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from quantagent.cli.backtest import backtest_group
from quantagent.cli import hipotesis
from quantagent.data.snapshot import write_snapshot


def test_admision_rango_y_registro(tmp_path, monkeypatch):
    rechazo = True
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.to_datetime(["2006-12-31", "2007-01-01", "2010-06-01", "2018-12-31", "2019-01-01"])
    velas = pd.DataFrame({c: 100.0 for c in ("open", "high", "low", "close", "adj_close", "volume")}, index=fechas)
    write_snapshot("sintetico", {"SPY": velas}, base_dir=tmp_path)
    data = yaml.safe_load((Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml").read_text())
    if rechazo:
        data.update(justificacion="corta", activos=["SPY", "AAPL"])
    archivo = tmp_path / "h.yaml"
    archivo.write_text(yaml.safe_dump(data))
    real = hipotesis.admitir
    vistos = []

    def observar(h, recibidas):
        vistos.extend(recibidas.index)
        return real(h, recibidas)

    monkeypatch.setattr(hipotesis, "admitir", observar)
    admision, admisiones = hipotesis.admitir_hipotesis, []

    def observar_admision(*args):
        r = admision(*args)
        admisiones.append(r)
        return r

    monkeypatch.setattr(hipotesis, "admitir_hipotesis", observar_admision)
    runner = CliRunner()
    out = tmp_path / "rechazada.csv"
    result = runner.invoke(backtest_group, ["hipotesis", "filtrar", str(archivo), "--snapshot", "sintetico", "--out", str(out)])
    assert not out.exists()
    assert result.exit_code == int(rechazo), result.output
    assert vistos == list(fechas[1:4])
    assert "PASA" not in result.output
    h, ok, motivos = admisiones[0]
    assert h.id == data["id"] and not ok
    assert motivos and all(motivo in result.output for motivo in motivos)
    if rechazo:
        assert "justificación" in result.output and "AAPL" in result.output
        with (tmp_path / "intentos.csv").open() as f:
            filas = list(csv.DictReader(f))
        assert len(filas) == 1 and filas[0]["hipotesis"] == data["id"] and filas[0]["etapa"] == "0"
    ayuda = runner.invoke(backtest_group, ["hipotesis", "filtrar", "--help"])
    assert ayuda.exit_code == 0 and "--snapshot" in ayuda.output and "ARCHIVO" in ayuda.output


@pytest.mark.parametrize("exportar", [False, True])
def test_filtro_real_dos_activos(tmp_path, monkeypatch, exportar):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    fechas = pd.date_range("2006-12-01", "2019-02-01", freq="7D")
    snapshots = {}
    for activo, periodo in [("SPY", 5), ("TLT", 9)]:
        close = 100 + 10 * np.sin(np.arange(len(fechas)) / periodo)
        snapshots[activo] = pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1,
             "close": close, "adj_close": close, "volume": 1000.0}, index=fechas,
        )
    write_snapshot("sintetico", snapshots, base_dir=tmp_path)
    data = yaml.safe_load((Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml").read_text())
    data.update(activos=["SPY", "TLT"], parametros={"rsi_period": [7], "oversold_threshold": [30.0]})
    archivo = tmp_path / "h.yaml"
    archivo.write_text(yaml.safe_dump(data))
    real, resultados = hipotesis.filtrar_activo, []

    def observar(h, velas, **kwargs):
        esperado = snapshots[h.activos[len(resultados)]].loc["2007":"2018"]
        assert list(velas.timestamp) == list(esperado.index)
        assert kwargs == {"costos": hipotesis.COSTOS_PERFILES["etf"], "capital": 100_000.0}
        r = real(h, velas, **kwargs)
        resultados.append(r)
        return r

    monkeypatch.setattr(hipotesis, "filtrar_activo", observar)
    filtro, filtros = hipotesis.filtrar_hipotesis, []

    def observar_filtro(*args):
        r = filtro(*args)
        filtros.append(r)
        return r

    monkeypatch.setattr(hipotesis, "filtrar_hipotesis", observar_filtro)
    out = tmp_path / "filtro.csv"
    args = ["--out", str(out)] if exportar else []
    result = CliRunner().invoke(backtest_group, ["hipotesis", "filtrar", str(archivo), "--snapshot", "sintetico", *args])
    assert result.exit_code == 0, result.output
    assert result.output.rstrip().endswith("CORTA (2 de 2 bajo 0,10 o con menos de 30 trades)")
    with (tmp_path / "intentos.csv").open() as f:
        filas = list(csv.DictReader(f))
    assert len(filas) == len(resultados) == 2
    for fila, activo, r in zip(filas, data["activos"], resultados):
        assert fila["activo"] == activo and fila["hipotesis"] == data["id"] and fila["etapa"] == "1"
        assert float(fila["sharpe"]) == r["sharpe_tasa_0"]
        assert int(fila["trades"]) == r["trades"] > 0
        celdas = next(line for line in result.output.splitlines() if activo in line).split("|")
        assert float(celdas[2]) == float(format(r["sharpe_tasa_0"], ".6g"))
        assert int(celdas[3]) == r["trades"]
    assert out.exists() == exportar
    metricas, parametros, veredicto, bajos = filtros[0]
    assert metricas == [[a, r["sharpe_tasa_0"], r["trades"]] for a, r in zip(data["activos"], resultados)]
    assert veredicto == "CORTA" and bajos == 2
    assert parametros == {a: hipotesis.parametros_elegidos(r["ganadores"]) for a, r in zip(data["activos"], resultados)}
    if exportar:
        with out.open() as f:
            filas = list(csv.DictReader(f))
        assert len(filas) == 2
        for fila, activo, r in zip(filas, data["activos"], resultados):
            assert list(fila) == ["activo", "sharpe_fuera_muestra", "trades", "params", "veredicto"]
            assert fila["activo"] == activo and fila["veredicto"] == "CORTA"
            assert float(fila["sharpe_fuera_muestra"]) == r["sharpe_tasa_0"]
            assert int(fila["trades"]) == r["trades"]
            assert fila["params"] == json.dumps(hipotesis.parametros_elegidos(r["ganadores"]), sort_keys=True)
            assert json.loads(fila["params"]) == parametros[activo]


@pytest.mark.parametrize("sharpes,final", [
    ([0.09] * 9 + [0.10] * 3, "CORTA (9 de 12 bajo 0,10)"),
    ([0.09] * 8 + [0.10] * 4, "PASA (8 de 12 bajo 0,10)"),
    ([0.09] * 8 + [float("nan")] + [0.10] * 3, "CORTA (9 de 12 bajo 0,10)"),
    ([0.09] * 8 + [None] + [0.10] * 3, "CORTA (9 de 12 bajo 0,10)"),
])
def test_bordes_veredicto_etapa1(tmp_path, monkeypatch, sharpes, final):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    activos = ["SPY", "IWM", "EFA", "EEM", "TLT", "SHY", "GLD", "DBC", "XLE", "XLF", "XLK", "XLU"]
    fechas = pd.date_range("2007-01-01", periods=40)
    velas = pd.DataFrame(
        {c: 100.0 for c in ("open", "high", "low", "close", "adj_close", "volume")}, index=fechas,
    )
    write_snapshot("sintetico", dict.fromkeys(activos, velas), base_dir=tmp_path)
    data = yaml.safe_load((Path(__file__).parent / "fixtures/hipotesis/rsi-valida.yaml").read_text())
    data.update(activos=activos, parametros={"rsi_period": [7], "oversold_threshold": [30.0]})
    archivo = tmp_path / "h.yaml"
    archivo.write_text(yaml.safe_dump(data))
    resultados = iter({"sharpe_tasa_0": s, "trades": 30, "ganadores": [{"params": {}, "prueba_desde": fechas[0]}]} for s in sharpes)
    monkeypatch.setattr(hipotesis, "filtrar_activo", lambda *a, **kw: next(resultados))
    result = CliRunner().invoke(backtest_group, ["hipotesis", "filtrar", str(archivo), "--snapshot", "sintetico"])
    assert result.exit_code == 0, result.output
    assert result.output.rstrip().endswith(final.replace(")", " o con menos de 30 trades)"))
    with (tmp_path / "intentos.csv").open() as f:
        filas = list(csv.DictReader(f))
    tabla = [line.split("|") for line in result.output.splitlines()[2:-1]]
    assert len(filas) == len(tabla) == 12
    for i, (fila, celdas, activo, sharpe) in enumerate(zip(filas, tabla, activos, sharpes)):
        assert fila["activo"] == celdas[1].strip() == activo
        assert fila["etapa"] == "1" and fila["hipotesis"] == data["id"]
        assert int(fila["trades"]) == int(celdas[3]) == 30
        if sharpe is None:
            assert fila["sharpe"] == celdas[2].strip() == ""
        elif np.isnan(sharpe):
            assert np.isnan(float(fila["sharpe"])) and np.isnan(float(celdas[2]))
        else:
            assert float(fila["sharpe"]) == float(celdas[2]) == sharpe
