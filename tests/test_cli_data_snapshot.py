"""`data snapshot create` y `verify` (V02). La descarga de Yahoo se reemplaza por un DataFrame fijo; sin red."""

import pandas as pd
import pytest
from click.testing import CliRunner

from quantagent.cli import data as cli_data
from quantagent.cli.data import data_group


def _fixed(symbol: str, start: str) -> pd.DataFrame:
    base = 100.0 + len(symbol)
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"], name="timestamp")
    return pd.DataFrame(
        {
            "open": [base, base + 1, base + 2],
            "high": [base + 1, base + 2, base + 3],
            "low": [base - 1, base, base + 1],
            "close": [base + 0.5, base + 1.5, base + 2.5],
            "adj_close": [base + 0.4, base + 1.4, base + 2.4],
            "volume": [10, 20, 30],
        },
        index=idx,
    )


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setattr(cli_data, "download_daily", _fixed)
    return CliRunner()


def _create(runner, name="t"):
    return runner.invoke(data_group, ["snapshot", "create", "--name", name, "--symbols", "SPY,TLT", "--start", "2020-01-01"])


def test_crea_y_verify_da_ok(runner):
    result = _create(runner)
    assert result.exit_code == 0, result.output
    verify = runner.invoke(data_group, ["snapshot", "verify", "--name", "t"])
    assert verify.exit_code == 0, verify.output
    assert verify.output.strip() == "OK 2 símbolos, 6 filas"


def test_segundo_create_con_el_mismo_nombre_sale_con_1_sin_tocar_nada(runner, tmp_path):
    assert _create(runner).exit_code == 0
    before = {p.name: p.read_bytes() for p in (tmp_path / "t").iterdir()}
    result = _create(runner)
    assert result.exit_code == 1
    assert "ya existe" in result.output
    assert {p.name: p.read_bytes() for p in (tmp_path / "t").iterdir()} == before


def test_verify_nombra_el_archivo_alterado(runner, tmp_path):
    assert _create(runner).exit_code == 0
    path = tmp_path / "t" / "SPY.parquet"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    result = runner.invoke(data_group, ["snapshot", "verify", "--name", "t"])
    assert result.exit_code == 1
    assert "SPY.parquet" in result.output


def test_simbolo_sin_filas_no_crea_nada(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(cli_data, "download_daily", lambda s, start: _fixed(s, start).iloc[0:0])
    result = _create(runner)
    assert result.exit_code == 1
    assert not (tmp_path / "t").exists()
