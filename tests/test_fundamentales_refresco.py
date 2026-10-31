"""Refresco diario real sobre snapshots temporales, sin descargas de red (M5-55a)."""

from datetime import date
from http.client import IncompleteRead
from io import BytesIO
from threading import Event, current_thread
from time import monotonic
from urllib.error import HTTPError

import pandas as pd
import pytest
from click.testing import CliRunner

from quantagent.cli.data import data_group
from quantagent.data.fundamentales import MAX_FALLIDOS_PCT, crear_snapshot_fundamentales, descargar_facts_cik, refrescar_fundamentales
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


@pytest.fixture
def universo_amplio(datos, monkeypatch):
    tickers = [f"T{i:02}" for i in range(50)]
    csv = datos / "csv-amplio"
    csv.mkdir()
    pd.DataFrame([{"date": "2026-08-18", "tickers": ",".join(tickers)}]).to_csv(csv / HISTORICAL_CSV, index=False)
    pd.DataFrame([{"Symbol": t, "GICS Sector": "Information Technology"} for t in tickers]).to_csv(csv / CURRENT_CSV, index=False)
    crear_snapshot_sp500("comp-amplia", base_dir=datos, csv_dir=csv)
    monkeypatch.setattr("quantagent.data.fundamentales.mapa_tickers_cik", lambda: dict(zip(tickers, range(1, 51))))
    return datos


@pytest.mark.parametrize("cantidad", [1, 2])
def test_limite_fallidos_y_progreso(universo_amplio, monkeypatch, cantidad):
    datos = universo_amplio
    crear_snapshot_fundamentales(HISTORICO, ["T01"], datos, downloader=facts)
    previo = (datos / HISTORICO / "manifest.json").read_bytes()
    fallidos = [f"T{i:02}: ConnectionError" for i in range(cantidad)]

    def bajar(cik):
        if 1 <= cik <= cantidad:
            raise ConnectionError("SEC interrumpió la conexión")
        return facts(cik)

    progreso = []
    res = refrescar_fundamentales(HOY, "comp-amplia", datos, downloader=bajar, progreso=lambda n, total: progreso.append((n, total)))
    assert progreso == [(50, 50)]
    assert (datos / HISTORICO / "manifest.json").read_bytes() == previo
    assert not list(datos.glob(".refresco-*"))
    if cantidad * 100 / 50 <= MAX_FALLIDOS_PCT:
        assert res["nuevo"] and res["snapshot_vigente"] == NUEVO
        assert res["fallidos"] == fallidos
        assert "fallidos dentro del límite" in res["motivo"]
        assert verify_snapshot(NUEVO, datos)["fallidos"] == fallidos
        segunda = refrescar_fundamentales(HOY, "comp-amplia", datos, downloader=bajar)
        assert segunda["fallidos"] == fallidos and not segunda["nuevo"]
    else:
        assert not res["nuevo"] and res["snapshot_vigente"] == HISTORICO
        assert f"supera {MAX_FALLIDOS_PCT}%" in res["motivo"]
        assert not (datos / NUEVO).exists()


def test_cli_progreso_stderr_cada_50(universo_amplio, monkeypatch):
    monkeypatch.setattr("quantagent.data.fundamentales.descargar_facts_cik", facts)
    monkeypatch.setattr("quantagent.data.fundamentales.time.sleep", lambda _: None)
    monkeypatch.setattr("quantagent.cli.data.date", type("Reloj", (), {"today": staticmethod(lambda: HOY)}))
    salida = CliRunner().invoke(data_group, ["fundamentales", "refrescar", "--composicion", "comp-amplia"])
    assert salida.exit_code == 0, salida.output
    assert salida.stderr == "Fundamentales: 50/50 tickers procesados\n"
    assert "tickers procesados" not in salida.stdout


@pytest.mark.parametrize("error", [IncompleteRead(b"parcial", 100), ConnectionError("conexión cortada")])
def test_descarga_reintenta_lectura_incompleta_y_conexion(monkeypatch, error):
    pedidos, esperas = [], []

    class Respuesta(BytesIO):
        def read(self):
            if len(pedidos) < 3:
                raise error
            return super().read()

    def abrir(req, timeout):
        pedidos.append((req.full_url, timeout))
        return Respuesta(b'{"cik": 1}')

    monkeypatch.setattr("urllib.request.urlopen", abrir)
    monkeypatch.setattr("quantagent.data.fundamentales.time.sleep", esperas.append)
    assert descargar_facts_cik(1) == {"cik": 1}
    assert len(pedidos) == 3 and esperas == [2, 4]


@pytest.mark.parametrize("fase", ["apertura", "lectura"])
def test_descarga_tope_total_aunque_socket_no_agote_timeout(monkeypatch, fase):
    liberar, hilos, esperas = Event(), [], []

    class Respuesta(BytesIO):
        def read(self):
            liberar.wait(2)
            return super().read()

    def abrir(req, timeout):
        hilos.append(current_thread())
        if fase == "apertura":
            liberar.wait(2)
        return Respuesta(b"{}")

    monkeypatch.setattr("urllib.request.urlopen", abrir)
    monkeypatch.setattr("quantagent.data.fundamentales.FACTS_TIMEOUT_TOTAL", 0.02)
    monkeypatch.setattr("quantagent.data.fundamentales.time.sleep", esperas.append)
    inicio = monotonic()
    try:
        with pytest.raises(TimeoutError):
            descargar_facts_cik(1)
        assert monotonic() - inicio < 1
        assert len(hilos) == 3 and esperas == [2, 4]
    finally:
        liberar.set()
        for hilo in hilos:
            hilo.join(timeout=1)


def test_descarga_http_error_no_reintenta(monkeypatch):
    pedidos = []

    def abrir(req, timeout):
        pedidos.append(req)
        raise HTTPError(req.full_url, 503, "SEC no disponible", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", abrir)
    with pytest.raises(HTTPError):
        descargar_facts_cik(1)
    assert len(pedidos) == 1
