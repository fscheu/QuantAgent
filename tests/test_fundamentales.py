"""Tests de métricas fundamentales punto en el tiempo (M5-31)."""

from unittest.mock import patch

import pandas as pd
import pytest

from quantagent.data.fundamentales import (
    calcular_metricas_ticker,
    crear_snapshot_fundamentales,
    metricas_fundamentales,
    procesar_ticker,
)


def _fila(ticker, concepto, start, end, val, filed, fy=2020, fp="Q1", form="10-Q", accn="accn-0"):
    return {"ticker": ticker, "concepto": concepto, "start": start, "end": end, "val": float(val), "fy": fy, "fp": fp, "form": form, "filed": filed, "accn": accn}


def test_punto_en_el_tiempo_visibilidad():
    """(a) Un balance filed el día d no se ve en d ni en d-1, sí en d+1."""
    df = pd.DataFrame([
        _fila("ABC", "OperatingIncomeLoss", "2019-10-01", "2019-12-31", 100, "2020-01-10", fp="Q1"),
        _fila("ABC", "StockholdersEquity", "", "2019-12-31", 500, "2020-01-10"),
        _fila("ABC", "StockholdersEquity", "", "2020-03-31", 1000, "2020-04-15"),
    ])
    _, balances = procesar_ticker(df)
    assert balances[balances["filed"] < "2020-04-14"]["val"].iloc[-1] == 500
    assert balances[balances["filed"] < "2020-04-15"]["val"].iloc[-1] == 500
    assert balances[balances["filed"] < "2020-04-16"]["val"].iloc[-1] == 1000


def test_restatacion_primer_publicado():
    """(b) Ante restatación, se usa el original (primer publicado) para ese período."""
    df = pd.DataFrame([
        _fila("XYZ", "OperatingIncomeLoss", "2020-01-01", "2020-03-31", 100, "2020-04-15", accn="orig-1"),
        _fila("XYZ", "OperatingIncomeLoss", "2020-01-01", "2020-03-31", 250, "2020-10-15", accn="restat-2", form="10-Q/A"),
    ])
    flujos, _ = procesar_ticker(df)
    assert len(flujos) == 1 and flujos.iloc[0]["val"] == 100 and flujos.iloc[0]["accn"] == "orig-1"


def test_q4_derivado():
    """(c) Q4 = anual (FY) - 9M acumulado cuando ambos existen."""
    df = pd.DataFrame([
        _fila("DER", "ingresos", "2020-01-01", "2020-09-30", 300, "2020-10-20", fy=2020, fp="Q3"),
        _fila("DER", "ingresos", "2020-01-01", "2020-12-31", 420, "2021-02-15", fy=2020, fp="FY", form="10-K"),
    ])
    q4 = procesar_ticker(df)[0]
    assert len(q4) == 1 and q4.iloc[0]["val"] == 120.0 and q4.iloc[0]["end"] == "2020-12-31"


def test_metricas_caso_a_mano():
    """(d) Caso a mano para cada una de las 4 métricas."""
    trimestres = [
        ("2018-01-01", "2018-03-31", 100, 20, 1.0, 5, 20, "2018-04-15"),
        ("2018-04-01", "2018-06-30", 100, 20, 1.0, 5, 20, "2018-07-15"),
        ("2018-07-01", "2018-09-30", 100, 20, 1.0, 5, 20, "2018-10-15"),
        ("2018-10-01", "2018-12-31", 100, 20, 1.0, 5, 20, "2019-02-15"),
        ("2019-01-01", "2019-03-31", 110, 30, 1.5, 6, 25, "2019-04-15"),
        ("2019-04-01", "2019-06-30", 120, 30, 1.5, 6, 25, "2019-07-15"),
        ("2019-07-01", "2019-09-30", 125, 30, 1.5, 6, 25, "2019-10-15"),
        ("2019-10-01", "2019-12-31", 140, 30, 1.5, 7, 25, "2020-02-15"),
    ]
    filas = []
    for s, e, ing, op, eps, tax, ebt, fld in trimestres:
        filas.extend([
            _fila("M", "ingresos", s, e, ing, fld), _fila("M", "OperatingIncomeLoss", s, e, op, fld),
            _fila("M", "EarningsPerShareDiluted", s, e, eps, fld), _fila("M", "IncomeTaxExpenseBenefit", s, e, tax, fld),
            _fila("M", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes", s, e, ebt, fld),
            _fila("M", "StockholdersEquity", "", e, 300, fld), _fila("M", "deuda", "", e, 100, fld),
        ])
    flujos, balances = procesar_ticker(pd.DataFrame(filas))
    m = calcular_metricas_ticker(flujos, balances, pd.to_datetime(["2020-02-16"]))
    f = pd.Timestamp("2020-02-16")
    assert m["roic_4t"].loc[f] == pytest.approx(120 * (1 - 0.25) / 400)
    assert m["margen_op_yoy"].loc[f] == pytest.approx((120 / 495) - 0.20)
    assert m["aceleracion_ingresos"].loc[f] == pytest.approx(0.15)
    assert m["crecimiento_eps"].loc[f] == pytest.approx(0.50)


def test_snapshot_e2e(tmp_path):
    """(e) Integración: crear_snapshot_fundamentales y metricas_fundamentales."""
    facts = {
        "facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [{"form": "10-Q", "start": "2020-01-01", "end": "2020-03-31", "val": 100, "filed": "2020-04-15", "fp": "Q1", "fy": 2020}]}},
            "StockholdersEquity": {"units": {"USD": [{"form": "10-Q", "end": "2020-03-31", "val": 500, "filed": "2020-04-15"}]}},
        }}
    }
    with patch("quantagent.data.fundamentales.mapa_tickers_cik", return_value={"FAKE": 123}):
        manifest = crear_snapshot_fundamentales("test-snap", ["FAKE", "SIN_CIK"], base_dir=tmp_path, downloader=lambda cik: facts)
    assert "FAKE" in manifest["symbols"] and "SIN_CIK" in manifest["sin_cik"]
    paneles = metricas_fundamentales("test-snap", ["FAKE", "SIN_CIK"], pd.to_datetime(["2020-04-16"]), base_dir=tmp_path)
    assert paneles["roic_4t"]["SIN_CIK"].isna().all()
