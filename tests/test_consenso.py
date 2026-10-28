import pandas as pd
import pytest
from click.testing import CliRunner

from quantagent.cli.data import data_group
from quantagent.data.consenso import guardar_foto
from quantagent.data.sources.yahoo_eps import YahooEpsClient


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
    res = runner.invoke(data_group, ["consenso", "foto", "--fuente", "fmp", "--simbolos", "AAPL,MSFT", "--fecha", "2026-10-27"])
    assert res.exit_code == 0, res.output
    assert "Foto de consenso guardada" in res.output
    assert (tmp_path / "consenso-eps" / "2026-10-27.parquet").exists()

    # Repetir fecha debe fallar
    res_rep = runner.invoke(data_group, ["consenso", "foto", "--fuente", "fmp", "--simbolos", "AAPL", "--fecha", "2026-10-27"])
    assert res_rep.exit_code != 0
    assert "ya existe" in res_rep.output

    # Desde archivo
    archivo_simbolos = tmp_path / "simbolos.txt"
    archivo_simbolos.write_text("AAPL\n# comentario\nMSFT\n\n", encoding="utf-8")
    res_arch = runner.invoke(data_group, ["consenso", "foto", "--fuente", "fmp", "--archivo", str(archivo_simbolos), "--fecha", "2026-10-28"])
    assert res_arch.exit_code == 0, res_arch.output
    assert (tmp_path / "consenso-eps" / "2026-10-28.parquet").exists()


def test_cli_foto_sin_api_key_falla(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.delenv("FMP_API_KEY", raising=False)

    runner = CliRunner()
    res = runner.invoke(data_group, ["consenso", "foto", "--fuente", "fmp", "--simbolos", "AAPL", "--fecha", "2026-10-27"])
    assert res.exit_code != 0
    assert "falta FMP_API_KEY" in res.output


def test_cli_fuente_yahoo_y_fmp(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setattr("quantagent.data.sources.yahoo_eps.YahooEpsClient", lambda: ClienteFalso())
    monkeypatch.setattr("quantagent.data.sources.fmp.FmpClient", lambda: ClienteFalso())

    runner = CliRunner()
    # Default es yahoo -> carpeta consenso-eps-yahoo
    res = runner.invoke(data_group, ["consenso", "foto", "--simbolos", "AAPL", "--fecha", "2026-10-27"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "consenso-eps-yahoo" / "2026-10-27.parquet").exists()

    # Repetir fecha en yahoo falla sin pisar
    res_rep = runner.invoke(data_group, ["consenso", "foto", "--simbolos", "AAPL", "--fecha", "2026-10-27"])
    assert res_rep.exit_code != 0
    assert "ya existe" in res_rep.output

    # Fuente fmp explícita -> carpeta consenso-eps
    res_fmp = runner.invoke(data_group, ["consenso", "foto", "--fuente", "fmp", "--simbolos", "AAPL", "--fecha", "2026-10-27"])
    assert res_fmp.exit_code == 0, res_fmp.output
    assert (tmp_path / "consenso-eps" / "2026-10-27.parquet").exists()


class DummyTicker:
    def __init__(self, sym: str):
        self.sym = sym
        if sym == "VACIO":
            self.eps_trend = pd.DataFrame()
            self.eps_revisions = pd.DataFrame()
        elif sym == "ERROR":
            raise RuntimeError("error simulado")
        else:
            self.eps_trend = pd.DataFrame(
                {
                    "current": [1.5, 2.0],
                    "7daysAgo": [1.4, 1.9],
                    "30daysAgo": [1.3, 1.8],
                    "60daysAgo": [1.2, 1.7],
                    "90daysAgo": [1.1, 1.6],
                    "currency": ["USD", "USD"],
                },
                index=["0q", "+1q"],
            )
            self.eps_revisions = pd.DataFrame(
                {
                    "upLast7days": [1, 2],
                    "upLast30days": [3, 4],
                    "downLast30days": [0, 1],
                    "downLast7Days": [0, 0],
                    "currency": ["USD", "USD"],
                },
                index=["0q", "+1q"],
            )


def test_yahoo_eps_client(monkeypatch):
    consultados = []

    def mock_ticker(sym, session=None):
        consultados.append(sym)
        return DummyTicker(sym)

    monkeypatch.setattr("yfinance.Ticker", mock_ticker)

    client = YahooEpsClient()
    # Símbolo con punto se consulta con guión y se guarda con punto
    filas = client.estimaciones_analistas("BRK.B")
    assert consultados == ["BRK-B"]
    assert len(filas) == 2
    por_periodo = {f["period"]: f for f in filas}
    assert "0q" in por_periodo and "+1q" in por_periodo
    f0 = por_periodo["0q"]
    assert f0["symbol"] == "BRK.B"
    assert f0["period"] == "0q"
    assert f0["current"] == 1.5
    assert f0["d7"] == 1.4
    assert f0["d30"] == 1.3
    assert f0["d60"] == 1.2
    assert f0["d90"] == 1.1
    assert f0["up_7d"] == 1
    assert f0["up_30d"] == 3
    assert f0["down_30d"] == 0
    assert f0["down_7d"] == 0
    assert f0["currency"] == "USD"

    # Sin datos -> lista vacía
    assert client.estimaciones_analistas("VACIO") == []
    # Error -> lista vacía
    assert client.estimaciones_analistas("ERROR") == []


def test_sin_datos_no_guarda_archivo(tmp_path):
    with pytest.raises(RuntimeError, match="ningún símbolo"):
        guardar_foto(["ROTO"], ClienteFalso(fallos={"ROTO"}), fecha="2026-10-27", base_dir=tmp_path)
    assert not (tmp_path / "consenso-eps" / "2026-10-27.parquet").exists()
