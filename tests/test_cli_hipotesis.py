import csv
from pathlib import Path

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from quantagent.cli.backtest import backtest_group
from quantagent.cli import hipotesis
from quantagent.data.snapshot import write_snapshot


@pytest.mark.parametrize("rechazo", [False, True])
def test_admision_rango_y_registro(tmp_path, monkeypatch, rechazo):
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
    runner = CliRunner()
    result = runner.invoke(backtest_group, ["hipotesis", "filtrar", str(archivo), "--snapshot", "sintetico"])
    assert result.exit_code == int(rechazo), result.output
    assert vistos == list(fechas[1:4])
    assert "PASA" not in result.output
    if rechazo:
        assert "justificación" in result.output and "AAPL" in result.output
        with (tmp_path / "intentos.csv").open() as f:
            filas = list(csv.DictReader(f))
        assert len(filas) == 1 and filas[0]["hipotesis"] == data["id"] and filas[0]["etapa"] == "0"
    else:
        assert "Admisión aprobada" in result.output
        assert not (tmp_path / "intentos.csv").exists()
    ayuda = runner.invoke(backtest_group, ["hipotesis", "filtrar", "--help"])
    assert ayuda.exit_code == 0 and "--snapshot" in ayuda.output and "ARCHIVO" in ayuda.output
