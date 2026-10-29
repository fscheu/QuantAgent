"""Snapshot de acciones del S&P 500 (QuantAgent-fnpr.1.3): fuente por ticker, mapeo WIKI, alias y cobertura. Sin red."""

import pandas as pd
import pytest
from click.testing import CliRunner

from quantagent.cli.data import data_group
from quantagent.data import acciones
from quantagent.data.acciones import crear_snapshot_acciones, reporte_cobertura
from quantagent.data.snapshot import read_snapshot, verify_snapshot, write_snapshot
from quantagent.data.sp500 import CURRENT_CSV, HISTORICAL_CSV, crear_snapshot_sp500

SESIONES = acciones._sesiones()


def _frame(fechas, precios=None, volumen=10):
    idx = pd.DatetimeIndex(pd.to_datetime(fechas), name="timestamp")
    p = pd.Series(precios if precios is not None else 100.0, index=idx, dtype=float)
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "adj_close": p, "volume": volumen}, index=idx)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Composición a mano, tabla de alias propia y snapshot S&P 500 en tmp_path."""
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    pd.DataFrame(
        [
            {"date": "2007-01-03", "tickers": "AAA,BBB,CCC,LEHMQ"},
            {"date": "2008-09-16", "tickers": "AAA,BBB,CCC"},
            {"date": "2010-01-04", "tickers": "AAA,CCC,DDD"},
        ]
    ).to_csv(csv_dir / HISTORICAL_CSV, index=False)
    pd.DataFrame([{"Symbol": "AAA", "Security": "A", "GICS Sector": "Financials"}]).to_csv(
        csv_dir / CURRENT_CSV, index=False
    )
    crear_snapshot_sp500("sp500-2026-10", base_dir=tmp_path, csv_dir=csv_dir)

    alias = tmp_path / "alias.csv"
    alias.write_text("ticker_sp500,ticker_fuente,fuente,nota\nLEHMQ,LEH,wiki,Lehman\nCCC,CCC.X,yahoo,sufijo Yahoo\n")
    monkeypatch.setattr(acciones, "ALIAS_CSV", alias)
    return tmp_path


def _yahoo_falso(symbol, start):
    frames = {
        "AAA": _frame(["2006-12-29", "2018-12-31", "2019-01-03"], [10.0, 20.0, 99.0]),
        "CCC.X": _frame(["2012-01-03", "2018-12-31"]),  # no cubre desde 2007: se usa igual si WIKI no lo tiene
        "DDD": _frame(["2010-01-04", "2018-12-31"], [7.0, 8.0]),
    }
    return frames.get(symbol, pd.DataFrame())


def _wiki(tmp_path):
    path = tmp_path / "WIKI_PRICES.csv"
    pd.DataFrame(
        [
            ("BBB", "2007-01-03", 1.0, 99.0, 98.0, 99.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            ("BBB", "2009-12-31", 2.0, 3.0, 1.0, 2.5, 500.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            ("LEH", "2007-01-03", 60.0, 61.0, 59.0, 60.5, 1000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            ("LEH", "2008-09-15", 4.0, 4.5, 0.1, 0.2, 7e6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            ("DDD", "2010-01-04", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        ],
        columns=[
            "ticker",
            "date",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
            "adj_volume",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ex-dividend",
            "split_ratio",
        ],
    ).to_csv(path, index=False)
    return path


def test_fuente_por_ticker_mapeo_alias_y_deslistada(entorno):
    manifest = crear_snapshot_acciones("acc", _wiki(entorno), base_dir=entorno, yahoo=_yahoo_falso)

    assert manifest["fuentes"] == {"AAA": "yahoo", "BBB": "wiki", "CCC": "yahoo", "DDD": "yahoo", "LEHMQ": "wiki"}
    assert manifest["source"] == "yahoo+wiki"
    assert manifest["symbols"]["AAA"]["end"] == "2018-12-31"  # Yahoo se recorta a 2018-12-31
    assert manifest["symbols"]["DDD"]["rows"] == 2 and read_snapshot("acc", "DDD", entorno)["close"].iloc[0] == 7.0

    bbb = read_snapshot("acc", "BBB", entorno)  # WIKI: OHLC y volumen salen de las columnas adj_*, factor 1
    assert bbb.iloc[1][["open", "high", "low", "close"]].tolist() == [2.0, 3.0, 1.0, 2.5]
    assert bbb["volume"].tolist() == [9, 500]

    leh = manifest["symbols"]["LEHMQ"]  # la deslistada conserva sus precios hasta el último día
    assert (leh["start"], leh["end"]) == ("2007-01-03", "2008-09-15")
    assert "CCC" in manifest["yahoo_sin_wiki"]
    assert verify_snapshot("acc", entorno)["name"] == "acc"


def test_reuso_ticker_coincidencia_baja_usa_wiki(entorno):
    fechas = pd.date_range("2007-01-03", periods=15, freq="B")
    wiki_path = entorno / "WIKI_REUSO.csv"
    pd.DataFrame(
        [("AAA", f.strftime("%Y-%m-%d"), 10.0 + i, 10.0 + i, 10.0 + i, 10.0 + i, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
         for i, f in enumerate(fechas)],
        columns=["ticker", "date", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume", "open", "high", "low", "close", "volume", "ex-dividend", "split_ratio"],
    ).to_csv(wiki_path, index=False)

    # Yahoo para AAA con serie completamente distinta (precios que caen en vez de subir -> dif retorno > 0.5%)
    def yahoo_divergente(symbol, start):
        if symbol == "AAA":
            p = [100.0 - i * 5 for i in range(len(fechas))]
            return _frame(fechas, p)
        return pd.DataFrame()

    manifest = crear_snapshot_acciones("reuso_snap", wiki_path, base_dir=entorno, yahoo=yahoo_divergente)
    assert manifest["fuentes"]["AAA"] == "wiki"
    assert "AAA" in manifest["reuso_cambiados_a_wiki"]
    assert manifest["reuso_cambiados_a_wiki"]["AAA"] < 0.9


def test_reporte_cobertura_con_datos_a_mano(entorno):
    sesiones = pd.DatetimeIndex(SESIONES)
    d1 = sesiones[sesiones >= "2010-01-04"]
    bbb_hasta = sesiones[sesiones <= "2007-12-31"]
    saltos = _frame(d1, [10.0] * 4 + [20.0] * (len(d1) - 4))  # +100% el quinto día
    write_snapshot(
        "cob",
        {
            "AAA": _frame(sesiones),
            "BBB": _frame(bbb_hasta),
            "DDD": saltos,
            "LEHMQ": _frame(sesiones[sesiones <= "2008-09-12"]),
        },
        base_dir=entorno,
        fuentes={"AAA": "yahoo", "BBB": "wiki", "DDD": "yahoo", "LEHMQ": "wiki"},
    )

    r = reporte_cobertura("cob", base_dir=entorno)

    miembro = {
        "AAA": len(sesiones),
        "BBB": int((sesiones < "2010-01-04").sum()),
        "CCC": len(sesiones),
        "DDD": len(d1),
        "LEHMQ": int((sesiones < "2008-09-16").sum()),
    }
    con_precio = {
        "AAA": len(sesiones),
        "BBB": len(bbb_hasta),
        "CCC": 0,
        "DDD": len(d1),
        "LEHMQ": int((sesiones <= "2008-09-12").sum()),
    }
    assert r["miembro_dias"] == sum(miembro.values())
    assert r["miembro_dias_con_precio"] == sum(con_precio.values())
    assert r["pct_global"] == pytest.approx(sum(con_precio.values()) / sum(miembro.values()) * 100)
    assert r["sin_precio"] == ["CCC"]
    assert r["terminan_antes"] == {
        "BBB": miembro["BBB"] - len(bbb_hasta),
        "LEHMQ": int((sesiones[sesiones < "2008-09-16"] > "2008-09-12").sum()),
    }
    assert [(s["ticker"], s["fecha"]) for s in r["saltos"]] == [("DDD", d1[4].date().isoformat())]
    assert r["saltos"][0]["retorno"] == pytest.approx(1.0)
    assert r["pct_por_anio"][2007] == pytest.approx(75.0)  # 4 miembros todo 2007; CCC sin precio
    assert r["pct_por_anio"][2011] == pytest.approx(2 / 3 * 100)  # AAA y DDD con precio, CCC no


def test_cli_cobertura_guarda_json_y_csv(entorno, monkeypatch):
    monkeypatch.setenv("QUANTAGENT_SNAPSHOT_DIR", str(entorno))
    sesiones = pd.DatetimeIndex(SESIONES)
    write_snapshot("cob", {"AAA": _frame(sesiones)}, base_dir=entorno)

    res = CliRunner().invoke(data_group, ["acciones", "cobertura", "--name", "cob"])

    assert res.exit_code == 0, res.output
    assert "Sin precio (4): BBB, CCC, DDD, LEHMQ" in res.output
    assert (entorno / "cob" / "cobertura.json").exists() and (entorno / "cob" / "cobertura.csv").exists()
