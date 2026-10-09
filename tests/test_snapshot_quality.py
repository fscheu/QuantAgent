"""`data snapshot check` (V04): reporte de calidad contra el calendario NYSE sobre snapshots armados a mano. Sin red."""

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from quantagent.cli.data import data_group
from quantagent.data.snapshot import check_snapshot, write_snapshot

# Sesiones NYSE de enero 2024 (el 1 es feriado): 2, 3, 4, 5, 8, 9, 10, 11, 12.
DIAS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]


def _frame(dias=DIAS) -> pd.DataFrame:
    n = len(dias)
    close = np.arange(100.0, 100.0 + n)
    return pd.DataFrame(
        {"open": close - 0.2, "high": close + 1, "low": close - 1, "close": close, "adj_close": close, "volume": [1000] * n},
        index=pd.DatetimeIndex(dias, name="timestamp"),
    )


@pytest.fixture
def snap(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    return tmp_path


def test_reporta_la_fecha_borrada_y_sale_con_1(snap):
    write_snapshot("t", {"SPY": _frame(), "TLT": _frame([d for d in DIAS if d != "2024-01-09"])}, base_dir=snap)

    result = CliRunner().invoke(data_group, ["snapshot", "check", "--name", "t"])

    assert result.exit_code == 1
    assert "SPY: faltan 0" in result.output
    assert "TLT: faltan 1" in result.output
    assert "faltan: 2024-01-09" in result.output


def test_cuenta_dias_de_mas_ohlc_volumen_y_saltos_sin_fallar(snap):
    df = _frame(DIAS + ["2024-01-13"])  # un sábado
    df.loc["2024-01-04", "high"] = df.loc["2024-01-04", "close"] - 5  # high por debajo del cierre
    df.loc["2024-01-05", "volume"] = 0
    df.loc["2024-01-10":, ["open", "high", "low", "close", "adj_close"]] *= 1.5  # nivel +50% desde el 10, sin split
    write_snapshot("t", {"SPY": df}, base_dir=snap)

    r = check_snapshot("t", snap)["SPY"]
    assert r["faltan"] == []
    assert r["de_mas"] == ["2024-01-13"]
    assert r["ohlc_incoherente"] == 1
    assert r["volumen_cero"] == 1
    assert r["saltos"] == ["2024-01-10"]

    result = CliRunner().invoke(data_group, ["snapshot", "check", "--name", "t"])
    assert result.exit_code == 0, result.output
