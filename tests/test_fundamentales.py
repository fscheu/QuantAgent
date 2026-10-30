"""Tests de métricas fundamentales punto en el tiempo (M5-31)."""

from unittest.mock import patch

import pandas as pd
import pytest

from quantagent.data.fundamentales import (
    calcular_metricas_ticker,
    crear_snapshot_fundamentales,
    extraer_hechos,
    metricas_fundamentales,
    preparar_ticker,
    vista_punto_en_el_tiempo,
)

# (start, end, ingresos, operating, eps, impuesto, antes de impuestos, filed)
TRIMESTRES = [
    ("2018-01-01", "2018-03-31", 100, 20, 1.0, 5, 20, "2018-04-15"),
    ("2018-04-01", "2018-06-30", 100, 20, 1.0, 5, 20, "2018-07-15"),
    ("2018-07-01", "2018-09-30", 100, 20, 1.0, 5, 20, "2018-10-15"),
    ("2018-10-01", "2018-12-31", 100, 20, 1.0, 5, 20, "2019-02-15"),
    ("2019-01-01", "2019-03-31", 110, 30, 1.5, 6, 25, "2019-04-15"),
    ("2019-04-01", "2019-06-30", 120, 30, 1.5, 6, 25, "2019-07-15"),
    ("2019-07-01", "2019-09-30", 125, 30, 1.5, 6, 25, "2019-10-15"),
    ("2019-10-01", "2019-12-31", 140, 30, 1.5, 7, 25, "2020-02-15"),
]


def _fila(concepto, start, end, val, filed, fy=2020, fp="Q1", form="10-Q", accn="accn-0"):
    return {"ticker": "T", "concepto": concepto, "start": start, "end": end, "val": float(val), "fy": fy, "fp": fp, "form": form, "filed": filed, "accn": accn}


def _filas_trimestres(trimestres=TRIMESTRES):
    filas = []
    for s, e, ing, op, eps, tax, ebt, fld in trimestres:
        filas.extend([
            _fila("ingresos", s, e, ing, fld), _fila("OperatingIncomeLoss", s, e, op, fld),
            _fila("EarningsPerShareDiluted", s, e, eps, fld), _fila("IncomeTaxExpenseBenefit", s, e, tax, fld),
            _fila("IncomeLossFromContinuingOperationsBeforeIncomeTaxes", s, e, ebt, fld),
            _fila("StockholdersEquity", "", e, 300, fld), _fila("deuda", "", e, 100, fld),
        ])
    return filas


def _metricas(filas, fechas):
    sesiones = pd.to_datetime(fechas)
    return calcular_metricas_ticker(preparar_ticker(pd.DataFrame(filas)), sesiones)


def test_punto_en_el_tiempo_visibilidad():
    """(a) Lo presentado el día d no se ve en d-1 ni en d; sí en d+1."""
    m = _metricas(_filas_trimestres(), ["2020-02-14", "2020-02-15", "2020-02-16"])
    eps = m["crecimiento_eps"]
    assert eps.iloc[:2].isna().all() and eps.iloc[2] == pytest.approx(0.5)


def test_restatacion_ultima_presentacion_visible():
    """(b) Original filed d1, restatado filed d2: entre d1 y d2 se ve el original, desde d2+1 el restatado."""
    filas = _filas_trimestres() + [_fila("EarningsPerShareDiluted", "2019-10-01", "2019-12-31", 2.5, "2020-03-10", form="10-Q/A", accn="restat")]
    eps = _metricas(filas, ["2020-02-16", "2020-03-10", "2020-03-11"])["crecimiento_eps"]
    assert list(eps) == pytest.approx([0.5, 0.5, 0.75])


def test_split_comparativo_restatado_en_presentacion_posterior():
    """(c) El EPS del año previo, restatado por un split en las presentaciones siguientes, no distorsiona el crecimiento."""
    filas = [f for f in _filas_trimestres() if not (f["concepto"] == "EarningsPerShareDiluted" and f["end"] < "2019-01-01")]
    for s, e, *_, fld in TRIMESTRES[:4]:
        filas.append(_fila("EarningsPerShareDiluted", s, e, 4.0, fld))
    for (s, e, *_, fld), nuevo in zip(TRIMESTRES[:4], TRIMESTRES[4:]):
        filas.append(_fila("EarningsPerShareDiluted", s, e, 0.5, nuevo[-1]))
    assert _metricas(filas, ["2020-02-16"])["crecimiento_eps"].iloc[0] == pytest.approx(6.0 / 2.0 - 1.0)


def test_dato_viejo_es_nan():
    """(d) Si el último trimestre usado terminó hace más de 200 días respecto de t, la métrica es NaN."""
    eps = _metricas(_filas_trimestres(), ["2020-07-17", "2020-07-19"])["crecimiento_eps"]
    assert eps.iloc[0] == pytest.approx(0.5) and pd.isna(eps.iloc[1])


def test_hueco_de_trimestre_es_nan():
    """(e) Con un trimestre faltante en la ventana no se calcula (no se toman 8 filas no consecutivas)."""
    sin_q = [f for f in _filas_trimestres() if not (f["concepto"] == "EarningsPerShareDiluted" and f["end"] == "2018-09-30")]
    m = _metricas(sin_q, ["2020-02-16"])
    assert pd.isna(m["crecimiento_eps"].iloc[0]) and m["aceleracion_ingresos"].iloc[0] == pytest.approx(0.15)


def test_q4_derivado_con_q3_de_tres_y_nueve_meses():
    """(f) Q4 = anual - 9M aunque el mismo filing traiga también el Q3 de 3 meses."""
    df = preparar_ticker(pd.DataFrame([
        _fila("ingresos", "2020-01-01", "2020-09-30", 300, "2020-10-20", fp="Q3"),
        _fila("ingresos", "2020-07-01", "2020-09-30", 120, "2020-10-20", fp="Q3"),
        _fila("ingresos", "2020-01-01", "2020-12-31", 420, "2021-02-15", fp="FY", form="10-K"),
    ]))
    piv, _ = vista_punto_en_el_tiempo(df, pd.Timestamp("2021-02-15"))
    assert piv.loc["2020-12-31", "ingresos"] == 120.0 and piv.loc["2020-09-30", "ingresos"] == 120.0
    assert pd.Timestamp("2020-12-31") not in vista_punto_en_el_tiempo(df, pd.Timestamp("2021-02-14"))[0].index


def test_metricas_caso_a_mano():
    """(g) Caso a mano para cada una de las 4 métricas."""
    m = _metricas(_filas_trimestres(), ["2020-02-16"])
    assert m["roic_4t"].iloc[0] == pytest.approx(120 * (1 - 0.25) / 400)
    assert m["margen_op_yoy"].iloc[0] == pytest.approx((120 / 495) - 0.20)
    assert m["aceleracion_ingresos"].iloc[0] == pytest.approx(0.15)
    assert m["crecimiento_eps"].iloc[0] == pytest.approx(0.50)


def _filas_roe(equity=400, filed_ultimo="2020-02-15"):
    ni = [(10, "2019-01-01", "2019-03-31", "2019-04-15"), (12, "2019-04-01", "2019-06-30", "2019-07-15"),
          (14, "2019-07-01", "2019-09-30", "2019-10-15"), (16, "2019-10-01", "2019-12-31", filed_ultimo)]
    return [f for f in _filas_trimestres() if f["concepto"] != "StockholdersEquity"] + [
        _fila("NetIncomeLoss", s, e, v, fld) for v, s, e, fld in ni] + [_fila("StockholdersEquity", "", "2019-12-31", equity, filed_ultimo)]


def test_roe_4t_y_margen_pretax_a_mano():
    """(i) ROE = (10+12+14+16) / 400; margen pretax = 100/495 - 80/400."""
    m = _metricas(_filas_roe(), ["2020-02-16"])
    assert m["roe_4t"].iloc[0] == pytest.approx(52 / 400)
    assert m["margen_pretax_yoy"].iloc[0] == pytest.approx(100 / 495 - 80 / 400)


def test_roe_dato_posterior_a_t_no_se_ve():
    """(j) Un trimestre presentado el 2020-03-01 no entra al ROE hasta 2020-03-02."""
    roe = _metricas(_filas_roe(filed_ultimo="2020-03-01"), ["2020-02-16", "2020-03-01", "2020-03-02"])["roe_4t"]
    assert roe.iloc[:2].isna().all() and roe.iloc[2] == pytest.approx(52 / 400)


def test_roe_equity_no_positivo_es_nan():
    """(k) Con equity <= 0 el ROE es NaN."""
    assert all(pd.isna(_metricas(_filas_roe(equity=eq), ["2020-02-16"])["roe_4t"].iloc[0]) for eq in (0, -50))


def test_snapshot_ticker_con_punto_usa_cik_con_guion(tmp_path):
    """BRK.B se resuelve vía BRK-B (formato SEC) y se guarda con el ticker original."""
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"form": "10-Q", "start": "2020-01-01", "end": "2020-03-31", "val": 100, "filed": "2020-04-15"}]}}}}}
    with patch("quantagent.data.fundamentales.mapa_tickers_cik", return_value={"BRK-B": 1067983}):
        manifest = crear_snapshot_fundamentales("test-snap", ["BRK.B"], base_dir=tmp_path, downloader=lambda cik: facts)
    assert "BRK.B" in manifest["symbols"] and "BRK.B" not in manifest["sin_cik"]


def test_snapshot_ticker_fuera_de_sec_usa_cik_historico(tmp_path):
    """ANTM no está en company_tickers.json: se resuelve con el CSV de CIK históricos; uno desconocido sigue en sin_cik."""
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"form": "10-Q", "start": "2020-01-01", "end": "2020-03-31", "val": 100, "filed": "2020-04-15"}]}}}}}
    pedidos = []
    with patch("quantagent.data.fundamentales.mapa_tickers_cik", return_value={}):
        manifest = crear_snapshot_fundamentales("test-snap", ["ANTM", "XXXX"], base_dir=tmp_path, downloader=lambda cik: pedidos.append(cik) or facts)
    assert pedidos == [1156039] and list(manifest["symbols"]) == ["ANTM"] and manifest["sin_cik"] == ["XXXX"]


def test_snapshot_e2e(tmp_path):
    """(h) Integración: crear_snapshot_fundamentales (con un ticker que falla) y metricas_fundamentales."""
    facts = {
        "facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [{"form": "10-Q", "start": "2020-01-01", "end": "2020-03-31", "val": 100, "filed": "2020-04-15", "fp": "Q1", "fy": 2020}]}},
            "StockholdersEquity": {"units": {"USD": [{"form": "10-Q", "end": "2020-03-31", "val": 500, "filed": "2020-04-15"}]}},
        }}
    }

    def bajar(cik):
        if cik == 456:
            raise OSError("boom")
        return facts

    with patch("quantagent.data.fundamentales.mapa_tickers_cik", return_value={"FAKE": 123, "BAD": 456}):
        manifest = crear_snapshot_fundamentales("test-snap", ["FAKE", "BAD", "SIN_CIK"], base_dir=tmp_path, downloader=bajar)
    assert "FAKE" in manifest["symbols"] and "SIN_CIK" in manifest["sin_cik"] and manifest["fallidos"] == ["BAD: OSError"]
    paneles = metricas_fundamentales("test-snap", ["FAKE", "SIN_CIK"], pd.to_datetime(["2020-04-16"]), base_dir=tmp_path)
    assert paneles["roic_4t"]["SIN_CIK"].isna().all()


def _hecho(val, filed, form="10-Q", start="2017-04-01", end="2017-06-30"):
    return {"form": form, "start": start, "end": end, "val": val, "filed": filed, "fp": "Q2", "fy": 2017, "accn": f"a-{filed}"}


def _facts(**tags):
    return {"facts": {"us-gaap": {t: {"units": {"USD": h}} for t, h in tags.items()}}}


def test_extraer_hechos_tag_menor_prioridad_presentado_antes_se_conserva():
    """El 10-Q 2017 (SalesRevenueNet) no lo tapa el 10-K 2018 (Revenues) del mismo período."""
    df = preparar_ticker(extraer_hechos("T", _facts(SalesRevenueNet=[_hecho(90, "2017-08-05")], Revenues=[_hecho(95, "2018-11-05", "10-K")])))
    ingresos = {f: vista_punto_en_el_tiempo(df, pd.Timestamp(f))[0]["ingresos"].iloc[0] for f in ("2017-12-31", "2019-01-31")}
    assert ingresos == {"2017-12-31": 90, "2019-01-31": 95}


def test_extraer_hechos_misma_presentacion_queda_el_prioritario():
    df = extraer_hechos("T", _facts(SalesRevenueNet=[_hecho(90, "2017-08-05")], Revenues=[_hecho(95, "2017-08-05")]))
    assert df["val"].tolist() == [95.0]


def test_roe_con_equity_incluyendo_participacion_no_controlante():
    nuevo = "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    ni = [_hecho(v, f, start=s, end=e) for v, s, e, f in [(10, "2019-01-01", "2019-03-31", "2019-04-15"), (12, "2019-04-01", "2019-06-30", "2019-07-15"),
          (14, "2019-07-01", "2019-09-30", "2019-10-15"), (16, "2019-10-01", "2019-12-31", "2020-02-15")]]
    df = extraer_hechos("T", _facts(NetIncomeLoss=ni, **{nuevo: [{"form": "10-K", "end": "2019-12-31", "val": 400, "filed": "2020-02-15"}]}))
    roe = calcular_metricas_ticker(preparar_ticker(df), pd.to_datetime(["2020-02-16"]))["roe_4t"].iloc[0]
    assert roe == pytest.approx(52 / 400)
