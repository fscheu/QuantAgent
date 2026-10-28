"""Tests de composición histórica y sector GICS del S&P 500 (QuantAgent-fnpr.1.4)."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from quantagent.data.snapshot import SnapshotError
from quantagent.data.sp500 import (
    CURRENT_CSV,
    HISTORICAL_CSV,
    RECLASIFICACIONES_CSV,
    SALIDOS_CSV,
    crear_snapshot_sp500,
    miembros,
    sector,
    verificar_snapshot_sp500,
)


def _armar_csvs_chicos(csv_dir):
    csv_dir.mkdir(parents=True, exist_ok=True)
    hist_df = pd.DataFrame(
        [
            {"date": "2008-09-12", "tickers": "AAPL,GOOGL,LEH,LEHMQ,SPG,V"},
            {"date": "2008-09-15", "tickers": "AAPL,GOOGL,LEH,LEHMQ,MSFT,SPG,V"},
            {"date": "2008-10-01", "tickers": "AAPL,GOOGL,MSFT,SPG,V"},
            {"date": "2012-01-03", "tickers": "AAPL,DIS,GOOGL,MSFT,SPG,V,WIN"},
            {"date": "2018-12-24", "tickers": "AAPL,DIS,FB,GOOGL,MSFT,SPG,V,VNO"},
            {"date": "2019-01-02", "tickers": "AAPL,DIS,FB,GOOGL,MSFT,SPG,TSLA,V"},
        ]
    )
    actual_df = pd.DataFrame(
        [
            {"Symbol": "AAPL", "Security": "Apple Inc.", "GICS Sector": "Information Technology"},
            {"Symbol": "DIS", "Security": "Walt Disney", "GICS Sector": "Communication Services"},
            {"Symbol": "GOOGL", "Security": "Alphabet Inc.", "GICS Sector": "Communication Services"},
            {"Symbol": "MSFT", "Security": "Microsoft", "GICS Sector": "Information Technology"},
            {"Symbol": "SPG", "Security": "Simon Property Group", "GICS Sector": "Real Estate"},
            {"Symbol": "TSLA", "Security": "Tesla, Inc.", "GICS Sector": "Consumer Discretionary"},
            {"Symbol": "V", "Security": "Visa Inc.", "GICS Sector": "Financials"},
        ]
    )
    hist_df.to_csv(csv_dir / HISTORICAL_CSV, index=False)
    actual_df.to_csv(csv_dir / CURRENT_CSV, index=False)


def test_crear_verificar_y_no_pisar_snapshot_sp500(tmp_path):
    csv_dir = tmp_path / "csv"
    datos_dir = tmp_path / "datos"
    _armar_csvs_chicos(csv_dir)

    manifiesto = crear_snapshot_sp500("sp500-test", base_dir=datos_dir, csv_dir=csv_dir)
    assert manifiesto["source"] == "fja05680/sp500"
    assert manifiesto["sha"] == "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
    assert manifiesto["rows"] == 6
    assert manifiesto["start"] == "2008-09-12"
    assert manifiesto["end"] == "2019-01-02"
    assert verificar_snapshot_sp500("sp500-test", base_dir=datos_dir) == manifiesto

    with pytest.raises(SnapshotError, match="ya existe"):
        crear_snapshot_sp500("sp500-test", base_dir=datos_dir, csv_dir=csv_dir)

    versionado_dir = tmp_path / "versionado"
    versionado_dir.mkdir()
    (versionado_dir / "sp500-test.json").write_text(json.dumps(manifiesto))
    assert verificar_snapshot_sp500("sp500-test", base_dir=datos_dir, versioned_dir=versionado_dir) == manifiesto

    ruta_parquet = datos_dir / "sp500-test" / "composicion.parquet"
    ruta_parquet.write_bytes(ruta_parquet.read_bytes() + b"corrupto")
    with pytest.raises(SnapshotError, match="composicion"):
        verificar_snapshot_sp500("sp500-test", base_dir=datos_dir)


def test_miembros_en_tres_fechas_leh_y_error_antes_de_primera(tmp_path):
    csv_dir = tmp_path / "csv"
    datos_dir = tmp_path / "datos"
    _armar_csvs_chicos(csv_dir)
    crear_snapshot_sp500("sp500-test", base_dir=datos_dir, csv_dir=csv_dir)

    assert miembros("sp500-test", "2008-09-15", base_dir=datos_dir) == [
        "AAPL", "GOOGL", "LEH", "LEHMQ", "MSFT", "SPG", "V"
    ]
    assert miembros("sp500-test", "2012-01-03", base_dir=datos_dir) == [
        "AAPL", "DIS", "GOOGL", "MSFT", "SPG", "V", "WIN"
    ]
    # 2018-12-31 usa la fila anterior más cercana (2018-12-24) y no ve 2019-01-02 (TSLA)
    assert miembros("sp500-test", "2018-12-31", base_dir=datos_dir) == [
        "AAPL", "DIS", "FB", "GOOGL", "MSFT", "SPG", "V", "VNO"
    ]

    m_sep12 = miembros("sp500-test", "2008-09-12", base_dir=datos_dir)
    m_oct01 = miembros("sp500-test", "2008-10-01", base_dir=datos_dir)
    assert "LEH" in m_sep12 and "LEHMQ" in m_sep12
    assert "LEH" not in m_oct01 and "LEHMQ" not in m_oct01

    with pytest.raises(SnapshotError, match="anterior a la primera fecha"):
        miembros("sp500-test", "2008-09-01", base_dir=datos_dir)


def test_sector_reclasificaciones_googl_reit_v_y_salidos(tmp_path, monkeypatch):
    csv_dir = tmp_path / "csv"
    datos_dir = tmp_path / "datos"
    _armar_csvs_chicos(csv_dir)
    crear_snapshot_sp500("sp500-test", base_dir=datos_dir, csv_dir=csv_dir)

    # Criterio 1: GOOGL es Information Technology en 2017-06-30 y Communication Services en 2019-06-28
    assert sector("sp500-test", "GOOGL", "2017-06-30", base_dir=datos_dir) == "Information Technology"
    assert sector("sp500-test", "GOOGL", "2019-06-28", base_dir=datos_dir) == "Communication Services"

    # Criterio 3: SPG (REIT) es Financials en 2015 y Real Estate en 2017; V es IT en 2022 y Financials en 2023-06
    assert sector("sp500-test", "SPG", "2015-06-15", base_dir=datos_dir) == "Financials"
    assert sector("sp500-test", "SPG", "2017-06-15", base_dir=datos_dir) == "Real Estate"
    assert sector("sp500-test", "V", "2022-06-15", base_dir=datos_dir) == "Information Technology"
    assert sector("sp500-test", "V", "2023-06-15", base_dir=datos_dir) == "Financials"

    # Miembros salidos en gics_salidos.csv (incluyendo reclasificaciones sobre salidos)
    assert sector("sp500-test", "WIN", "2015-01-15", base_dir=datos_dir) == "Telecommunication Services"
    assert sector("sp500-test", "LEHMQ", "2008-09-12", base_dir=datos_dir) == "Financials"
    assert sector("sp500-test", "FB", "2017-06-30", base_dir=datos_dir) == "Information Technology"
    assert sector("sp500-test", "FB", "2019-06-28", base_dir=datos_dir) == "Communication Services"
    assert sector("sp500-test", "VNO", "2015-06-15", base_dir=datos_dir) == "Financials"
    assert sector("sp500-test", "VNO", "2017-06-15", base_dir=datos_dir) == "Real Estate"
    assert sector("sp500-test", "TICKER_INEXISTENTE", "2020-01-01", base_dir=datos_dir) is None

    salidos_tmp = tmp_path / "salidos_desc.csv"
    salidos_tmp.write_text("ticker,sector,fuente\nXYZ,,desconocido\n")
    monkeypatch.setattr("quantagent.data.sp500.SALIDOS_CSV", salidos_tmp)
    assert sector("sp500-test", "XYZ", "2015-01-01", base_dir=datos_dir) is None


def test_tablas_versionadas_existen_y_tienen_formato_esperado():
    df_salidos = pd.read_csv(SALIDOS_CSV, keep_default_na=False)
    assert list(df_salidos.columns) == ["ticker", "sector", "fuente"]
    assert len(df_salidos) == 429
    assert df_salidos["ticker"].is_unique

    df_recl = pd.read_csv(RECLASIFICACIONES_CSV, keep_default_na=False)
    assert list(df_recl.columns) == ["ticker", "desde", "sector_anterior", "sector_nuevo"]
    assert {"2016-09-01", "2018-09-24", "2023-03-17"}.issubset(set(df_recl["desde"]))


def test_miembros_2019_hoy_tienen_sector_en_cada_fin_de_mes(tmp_path):
    snap_env = os.environ.get("QUANTAGENT_SNAPSHOT_DIR")
    candidatos = [
        Path(snap_env) if snap_env else None,
        Path("local-data"),
        Path("/home/baiscf/repos/QuantAgent/local-data"),
    ]
    base_dir = next((p for p in candidatos if p and (p / "sp500-2026-10").exists()), None)
    if base_dir is not None:
        snap = "sp500-2026-10"
    else:
        csv_dir, base_dir = tmp_path / "csv", tmp_path / "datos"
        _armar_csvs_chicos(csv_dir)
        hist_path = csv_dir / HISTORICAL_CSV
        df_hist = pd.read_csv(hist_path)
        salidos_2019 = (
            "AAPL,AMTM,BBWI,BIO,CDAY,CTLT,CTRA,CZR,DAY,ENPH,EPAM,ETSY,FI,FRC,LUMN,"
            "MKTX,MOH,MTCH,NLOK,OGN,PARA,PAYC,PEAK,PENN,POOL,SATS,SBNY,SEDG,SOLS,TFX,VIAC,VNT"
        )
        df_hist = pd.concat(
            [df_hist, pd.DataFrame([{"date": "2020-06-22", "tickers": salidos_2019}])],
            ignore_index=True,
        )
        df_hist.to_csv(hist_path, index=False)
        snap = "sp500-test"
        crear_snapshot_sp500(snap, base_dir=base_dir, csv_dir=csv_dir)

    df_salidos = pd.read_csv(SALIDOS_CSV, keep_default_na=False)
    desconocidos = set(
        df_salidos[df_salidos["fuente"].astype(str).str.strip() == "desconocido"]["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    hoy = pd.Timestamp.today().normalize()
    fines_de_mes = [
        pd.Timestamp(anio, mes, 1) + pd.offsets.MonthEnd(0)
        for anio in range(2019, hoy.year + 1)
        for mes in range(1, 13)
        if (anio, mes) <= (hoy.year, hoy.month)
    ]
    sector_por_ticker = {}
    for fin_mes in fines_de_mes:
        for t in miembros(snap, fin_mes, base_dir=base_dir):
            if t not in sector_por_ticker:
                sector_por_ticker[t] = sector(snap, t, fin_mes, base_dir=base_dir)
        sin_sector = [
            t
            for t in miembros(snap, fin_mes, base_dir=base_dir)
            if t not in desconocidos and sector_por_ticker[t] is None
        ]
        assert sin_sector == [], f"{fin_mes.date()}: {sin_sector}"

