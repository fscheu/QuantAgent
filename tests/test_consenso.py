import pandas as pd
import pytest
from click.testing import CliRunner

from quantagent.cli.data import data_group
from quantagent.data.consenso import guardar_foto


class ClienteFalso:
    def __init__(self, datos=None, fallos=None):
        self.datos = datos or {
            "AAPL": [
                {"symbol": "AAPL", "date": "2024-09-30", "epsAvg": 6.7, "epsHigh": 6.8, "epsLow": 6.5, "revenueAvg": 3.9e11},
                {"symbol": "AAPL", "date": "2025-09-30", "epsAvg": 7.3, "epsHigh": 7.5, "epsLow": 7.1, "revenueAvg": 4.1e11},
            ],
            "MSFT": [
                {"symbol": "MSFT", "date": "2024-06-30", "epsAvg": 11.8, "epsHigh": 12.0, "epsLow": 11.5, "revenueAvg": 2.4e11},
            ],
        }
        self.fallos = fallos or set()

    def estimaciones_analistas(self, symbol, period="annual"):
        if symbol in self.fallos:
            raise RuntimeError(f"falla de red simulada para {symbol}")
        return self.datos.get(symbol, [])


def test_dos_fotos_dias_distintos_no_se_pisan(tmp_path):
    cliente = ClienteFalso()
    f1 = guardar_foto(["AAPL", "MSFT"], cliente, fecha="2026-10-27", base_dir=tmp_path)
    contenido1_antes = f1.read_bytes()

    f2 = guardar_foto(["AAPL"], cliente, fecha="2026-10-28", base_dir=tmp_path)

    assert f1.exists() and f2.exists()
    assert f1 != f2
    assert f1.name == "2026-10-27.parquet"
    assert f2.name == "2026-10-28.parquet"
    assert f1.read_bytes() == contenido1_antes

    df1 = pd.read_parquet(f1)
    df2 = pd.read_parquet(f2)
    assert len(df1) == 3
    assert set(df1["symbol"]) == {"AAPL", "MSFT"}
    assert (df1["fecha_foto"] == "2026-10-27").all()

    assert len(df2) == 2
    assert set(df2["symbol"]) == {"AAPL"}
    assert (df2["fecha_foto"] == "2026-10-28").all()


def test_misma_fecha_dos_veces_falla(tmp_path):
    cliente = ClienteFalso()
    guardar_foto(["AAPL"], cliente, fecha="2026-10-27", base_dir=tmp_path)
    with pytest.raises(FileExistsError, match="ya existe"):
        guardar_foto(["AAPL"], cliente, fecha="2026-10-27", base_dir=tmp_path)


def test_simbolo_sin_datos_o_error_sigue_con_los_demas(tmp_path, capsys):
    cliente = ClienteFalso(fallos={"ROTO"})
    archivo = guardar_foto(["ROTO", "VACIO", "AAPL"], cliente, fecha="2026-10-27", base_dir=tmp_path)

    salida = capsys.readouterr().out
    assert "ROTO: error" in salida
    assert "VACIO: sin datos" in salida

    df = pd.read_parquet(archivo)
    assert set(df["symbol"]) == {"AAPL"}
    assert len(df) == 2


def test_cli_foto_consenso(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("FMP_API_KEY", "dummy-key")
    monkeypatch.setattr("quantagent.data.sources.fmp.FmpClient", lambda: ClienteFalso())

    runner = CliRunner()
    res = runner.invoke(data_group, ["consenso", "foto", "--simbolos", "AAPL,MSFT", "--fecha", "2026-10-27"])
    assert res.exit_code == 0, res.output
    assert "Foto de consenso guardada" in res.output
    assert (tmp_path / "consenso-eps" / "2026-10-27.parquet").exists()

    # Repetir fecha debe fallar
    res_rep = runner.invoke(data_group, ["consenso", "foto", "--simbolos", "AAPL", "--fecha", "2026-10-27"])
    assert res_rep.exit_code != 0
    assert "ya existe" in res_rep.output

    # Desde archivo
    archivo_simbolos = tmp_path / "simbolos.txt"
    archivo_simbolos.write_text("AAPL\n# comentario\nMSFT\n\n", encoding="utf-8")
    res_arch = runner.invoke(data_group, ["consenso", "foto", "--archivo", str(archivo_simbolos), "--fecha", "2026-10-28"])
    assert res_arch.exit_code == 0, res_arch.output
    assert (tmp_path / "consenso-eps" / "2026-10-28.parquet").exists()


def test_cli_foto_sin_api_key_falla(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.delenv("FMP_API_KEY", raising=False)

    runner = CliRunner()
    res = runner.invoke(data_group, ["consenso", "foto", "--simbolos", "AAPL", "--fecha", "2026-10-27"])
    assert res.exit_code != 0
    assert "falta FMP_API_KEY" in res.output


def test_sin_datos_no_guarda_archivo(tmp_path):
    with pytest.raises(RuntimeError, match="ningún símbolo"):
        guardar_foto(["ROTO"], ClienteFalso(fallos={"ROTO"}), fecha="2026-10-27", base_dir=tmp_path)
    assert not (tmp_path / "consenso-eps" / "2026-10-27.parquet").exists()
