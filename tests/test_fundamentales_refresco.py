"""Refresco diario real sobre snapshots temporales, sin descargas de red (M5-55a)."""

from datetime import date

import pandas as pd
import pytest
from click.testing import CliRunner

from quantagent.cli.data import data_group
from quantagent.data.fundamentales import crear_snapshot_fundamentales, refrescar_fundamentales
from quantagent.data.snapshot import load_manifest, verify_snapshot
from quantagent.data.sp500 import CURRENT_CSV, HISTORICAL_CSV, crear_snapshot_sp500

HOY = date(2026, 10, 31)
NUEVO = "fundamentales-sp500-2026-10-31"
HISTORICO = "fundamentales-sp500-2026-10"


def facts(cik):
    tags = {}
    for tag, valor in {"Revenues": 100, "OperatingIncomeLoss": 20, "NetIncomeLoss": 10,
                       "EarningsPerShareDiluted": 2, "StockholdersEquity": 200,
                       "IncomeLossFromContinuingOperationsBeforeIncomeTaxes": 15}.items():
        hechos = []
        for i, fin in enumerate(pd.date_range("2024-09-30", periods=8, freq="QE")):
            hechos.append({"form": "10-Q", "start": str((fin - pd.offsets.QuarterEnd() + pd.Timedelta(days=1)).date()),
                           "end": str(fin.date()), "filed": str((fin + pd.Timedelta(days=30)).date()),
                           "val": valor + i, "fp": "Q1", "fy": fin.year, "accn": str(i)})
        tags[tag] = {"units": {"USD": hechos}}
    return {"facts": {"us-gaap": tags}}


@pytest.fixture
def datos(tmp_path, monkeypatch):
    csv = tmp_path / "csv"
    csv.mkdir()
    pd.DataFrame([{"date": "2020-01-02", "tickers": "OLD"},
                  {"date": "2026-08-18", "tickers": "AAPL,MSFT,JPM"}]).to_csv(csv / HISTORICAL_CSV, index=False)
    pd.DataFrame([{"Symbol": t, "GICS Sector": s} for t, s in
                  [("AAPL", "Information Technology"), ("MSFT", "Information Technology"), ("JPM", "Financials")]]).to_csv(csv / CURRENT_CSV, index=False)
    crear_snapshot_sp500("comp", base_dir=tmp_path, csv_dir=csv)
    monkeypatch.setattr("quantagent.data.snapshot.VERSIONED_DIR", tmp_path / "versionados")
    monkeypatch.setattr("quantagent.data.fundamentales.mapa_tickers_cik", lambda: {"AAPL": 1, "MSFT": 2, "JPM": 3, "OLD": 4})
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(tmp_path))
    return tmp_path


def test_crea_verifica_universo_actual_y_no_pisa(datos):
    pedidos = []

    def bajar(cik):
        pedidos.append(cik)
        return facts(cik)

    res = refrescar_fundamentales(HOY, "comp", datos, downloader=bajar)
    assert res["snapshot_vigente"] == NUEVO and res["nuevo"]
    manifest = verify_snapshot(NUEVO, datos)
    assert set(manifest["symbols"]) == {"AAPL", "MSFT", "JPM"}
    assert sorted(pedidos) == [1, 2, 3]
    original = (datos / NUEVO / "manifest.json").read_bytes()
    pedidos.clear()
    segunda = refrescar_fundamentales(HOY, "comp", datos, downloader=bajar)
    assert segunda["snapshot_vigente"] == NUEVO and not segunda["nuevo"]
    assert pedidos == []
    assert (datos / NUEVO / "manifest.json").read_bytes() == original
    assert verify_snapshot(NUEVO, datos) == manifest


@pytest.mark.parametrize("fallo", ["descarga", "parcial", "cobertura"])
def test_rechazo_conserva_historico_y_limpia_parcial(datos, fallo):
    crear_snapshot_fundamentales(HISTORICO, ["AAPL"], datos, downloader=facts)
    previo = (datos / HISTORICO / "manifest.json").read_bytes()

    def bajar(cik):
        if fallo == "descarga" or (fallo == "parcial" and cik == 2):
            raise OSError("SEC no disponible")
        return {} if fallo == "cobertura" and cik == 2 else facts(cik)

    res = refrescar_fundamentales(HOY, "comp", datos, downloader=bajar)
    assert res["snapshot_vigente"] == HISTORICO and not res["nuevo"]
    motivo = {"descarga": "no hay símbolos", "parcial": "descarga fallida", "cobertura": "cobertura bajo el umbral 60.0%"}
    assert motivo[fallo] in res["motivo"]
    assert not (datos / NUEVO).exists()
    assert not list(datos.glob(".refresco-*"))
    assert (datos / HISTORICO / "manifest.json").read_bytes() == previo


def test_fallback_mas_reciente_omite_corrupto_y_futuro(datos):
    primero = refrescar_fundamentales(date(2026, 10, 28), "comp", datos, downloader=facts)["snapshot_vigente"]
    ultimo = refrescar_fundamentales(date(2026, 10, 29), "comp", datos, downloader=facts)["snapshot_vigente"]
    corrupto = refrescar_fundamentales(date(2026, 10, 30), "comp", datos, downloader=facts)["snapshot_vigente"]
    (datos / corrupto / "AAPL.parquet").write_bytes(b"corrupto")
    refrescar_fundamentales(date(2026, 11, 2), "comp", datos, downloader=facts)
    res = refrescar_fundamentales(HOY, "comp", datos, downloader=lambda cik: {})
    assert res["snapshot_vigente"] == ultimo and ultimo != primero
    assert verify_snapshot(ultimo, datos) == load_manifest(ultimo, datos)
    assert (datos / corrupto / "AAPL.parquet").read_bytes() == b"corrupto"


@pytest.mark.parametrize("faltan", [1, 2])
def test_elegibilidad_tolera_una_metrica_pero_no_dos(datos, faltan):
    def bajar(cik):
        resultado = facts(cik)
        tags = resultado["facts"]["us-gaap"]
        if cik == 3:
            del tags["OperatingIncomeLoss"]  # Financials usa ROE y margen pretax.
        del tags["EarningsPerShareDiluted"]
        if faltan == 2:
            del tags["Revenues"]
        return resultado

    res = refrescar_fundamentales(HOY, "comp", datos, downloader=bajar)
    assert res["snapshot_vigente"] == (NUEVO if faltan == 1 else None)
    assert (datos / NUEVO).exists() == (faltan == 1)


def test_cli_fallback_valido_sale_cero(datos):
    crear_snapshot_fundamentales(HISTORICO, ["AAPL"], datos, downloader=facts)
    salida = CliRunner().invoke(data_group, ["fundamentales", "refrescar", "--composicion", "inexistente"])
    assert salida.exit_code == 0
    assert HISTORICO in salida.output and "refresco rechazado" in salida.output


def test_existente_invalido_no_se_borra_y_cli_sin_vigente_sale_uno(datos):
    crear_snapshot_fundamentales(NUEVO, ["AAPL"], datos, downloader=facts)
    archivo = datos / NUEVO / "AAPL.parquet"
    archivo.write_bytes(b"corrupto")
    res = refrescar_fundamentales(HOY, "comp", datos, downloader=facts)
    assert res["snapshot_vigente"] is None and "sha256" in res["motivo"]
    assert archivo.read_bytes() == b"corrupto"
    salida = CliRunner().invoke(data_group, ["fundamentales", "refrescar", "--composicion", "inexistente"])
    assert salida.exit_code == 1 and "ninguno" in salida.output


def test_cli_existente_valido_sale_cero(datos, monkeypatch):
    refrescar_fundamentales(HOY, "comp", datos, downloader=facts)

    class Reloj:
        today = staticmethod(lambda: HOY)

    monkeypatch.setattr("quantagent.cli.data.date", Reloj)
    salida = CliRunner().invoke(data_group, ["fundamentales", "refrescar", "--composicion", "comp"])
    assert salida.exit_code == 0
    assert NUEVO in salida.output and "ya existe" in salida.output
