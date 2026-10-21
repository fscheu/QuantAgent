import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica.criterio import (
    AniosPositivos,
    RecargoPaper,
    anios_positivos,
    curva_benchmark,
    evaluar_etapa3_cartera,
    recargo_paper,
    sharpe_benchmark,
    sharpe_por_anio,
    sharpes_en_exceso,
)


@pytest.fixture
def tasa_macro(tmp_path):
    from quantagent.data.macro import _escribir

    datos = pd.DataFrame({
        "fecha": pd.to_datetime(["2020-01-02", "2020-01-06", "2020-01-06"]),
        "realtime_start": pd.to_datetime(["2020-01-02", "2020-01-07", "2020-01-09"]),
        "realtime_end": pd.to_datetime(["2099-01-01", "2020-01-08", "2099-01-01"]),
        "valor": [3.0, 6.0, 12.0],
    })
    _escribir(tmp_path / "macro-2026-10", "macro-2026-10", {"DGS3MO": datos})
    return tmp_path


def test_sharpes_exceso_tasa_compartida_sin_anticipacion(tasa_macro):
    fechas = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"])
    # El lunes aún se conoce 3%; el martes 6%; la revisión a 12% recién el jueves.
    libre = np.array([.03 * 3, .03, .06, .06, .12]) / 365.25
    curva = pd.Series(100 * np.r_[1, np.cumprod(1 + libre)], index=fechas)
    assert sharpes_en_exceso(curva, curva * 2, base_dir=tasa_macro) == pytest.approx((0, 0), abs=1e-10)
    excesos = np.array([.001, -.002, .003, -.001, .002])
    cartera = pd.Series(100 * np.r_[1, np.cumprod(1 + libre + excesos)], index=fechas)
    benchmark = pd.Series(100 * np.r_[1, np.cumprod(1 + libre - excesos)], index=fechas)
    esperado = excesos.mean() / excesos.std(ddof=1) * np.sqrt(5 * 365.25 / 7)
    assert sharpes_en_exceso(cartera, benchmark, base_dir=tasa_macro) == pytest.approx((esperado, -esperado))


def test_sharpes_exceso_falta_cobertura_explicita(tasa_macro):
    curva = pd.Series([100., 101., 99.], index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match="DGS3MO sin cobertura"):
        sharpes_en_exceso(curva, curva, base_dir=tasa_macro)
    with pytest.raises(ValueError, match="mismas fechas"):
        sharpes_en_exceso(curva, curva.shift(1, freq="D"), base_dir=tasa_macro)


def _generar_curva(anios_suben=(2019, 2020, 2022), anios_bajan=(2021,), sesiones=250):
    fechas, valores, val = [], [], 100.0
    for anio in sorted(list(anios_suben) + list(anios_bajan)):
        dias = pd.bdate_range(f"{anio}-01-02", periods=sesiones)
        paso = 0.05 if anio in anios_suben else -0.05
        for d in dias:
            val += paso + np.sin(len(fechas)) * 0.01
            fechas.append(d)
            valores.append(val)
    return pd.Series(valores, index=pd.DatetimeIndex(fechas))


def test_anios_positivos_3_de_4():
    eq = _generar_curva()
    ap = anios_positivos(eq)
    assert ap == (3, 4)
    assert ap.positivos == 3 and ap.totales == 4
    pos, tot = ap
    assert (pos, tot) == (3, 4)


def test_anio_menos_30_sesiones_se_descarta_e_informa():
    eq = _generar_curva()
    eq_corto = pd.concat([pd.Series(90.0, index=pd.bdate_range("2018-12-01", periods=15)), eq])
    with pytest.warns(UserWarning, match="Año 2018 descartado.*15 sesiones"):
        sharpes = sharpe_por_anio(eq_corto)
    assert 2018 not in sharpes and len(sharpes) == 4
    ap = anios_positivos(sharpes)
    assert ap == (3, 4)
    assert ap.positivos == 3 and ap.totales == 4


def test_curva_ya_calculada_sin_lectura_datos():
    dias = pd.bdate_range("2019-01-02", periods=40)
    vals = [100.0 + i * 0.1 for i in range(40)]
    ap = anios_positivos(vals, fechas=dias)
    assert ap == (1, 1)
    assert ap.positivos == 1 and ap.totales == 1


def test_recargo_paper_tramos_y_bordes():
    # Tramo 1: DSR < 0.50
    assert recargo_paper(0.0) == (50, 12)
    assert recargo_paper(0.49) == (50, 12)

    # Borde 0.50 y Tramo 2: 0.50 <= DSR < 0.95
    assert recargo_paper(0.50) == (75, 6)
    assert recargo_paper(0.70) == (75, 6)
    assert recargo_paper(0.949) == (75, 6)

    # Borde 0.95 y Tramo 3: DSR >= 0.95
    assert recargo_paper(0.95) == (100, 0)
    assert recargo_paper(1.0) == (100, 0)

    # Atributos NamedTuple y desempaque
    res = recargo_paper(0.50)
    assert res.tamano_pct == 75 and res.semanas == 6
    tamano, semanas = res
    assert (tamano, semanas) == (75, 6)


def test_recargo_paper_none_y_nan():
    assert recargo_paper(None) == (50, 12)
    assert recargo_paper(float("nan")) == (50, 12)


def test_curva_y_sharpe_benchmark():
    fechas = pd.bdate_range("2020-01-01", "2020-03-31")
    df = pd.DataFrame({
        "SPY": 100.0 * np.cumprod(1 + np.full(len(fechas), 0.001)),
        "TLT": 100.0 * np.cumprod(1 + np.full(len(fechas), 0.002)),
    }, index=fechas)
    eq = curva_benchmark(df)
    assert len(eq) == len(fechas)
    assert eq.iloc[0] == 100_000.0
    assert eq.iloc[-1] > eq.iloc[0]
    sb = sharpe_benchmark(df)
    assert sb > 0

    # DataFrame vacío o None
    assert curva_benchmark(None).empty
    assert curva_benchmark(pd.DataFrame()).empty
    assert sharpe_benchmark(None) == 0.0


def test_evaluar_etapa3_cartera_casos():
    fechas = pd.bdate_range("2019-01-01", "2022-12-31")
    eq_ok = pd.Series(100_000.0 * np.cumprod(1 + np.full(len(fechas), 0.001)), index=fechas)

    # Caso base pasa
    pasa, motivos = evaluar_etapa3_cartera(eq_ok, sharpe_bench3=0.0, sharpe_cartera_etapa2=1.0, sharpe_bench2=0.5, trades=10)
    assert pasa is True
    assert motivos == []

    # Curva vacía o None
    assert evaluar_etapa3_cartera(None, 0.0) == (False, ["curva de cartera vacía"])
    assert evaluar_etapa3_cartera(pd.Series(dtype=float), 0.0) == (False, ["curva de cartera vacía"])

    # Trades < 2
    pasa, motivos = evaluar_etapa3_cartera(eq_ok, sharpe_bench3=0.0, sharpe_cartera_etapa2=1.0, sharpe_bench2=0.5, trades=1)
    assert pasa is False
    assert "trades < 2" in motivos

    # Sharpe <= 0
    eq_neg = pd.Series(100_000.0 * np.cumprod(1 + np.full(len(fechas), -0.001)), index=fechas)
    pasa, motivos = evaluar_etapa3_cartera(eq_neg, sharpe_bench3=-1.0, sharpe_cartera_etapa2=1.0, sharpe_bench2=0.5)
    assert pasa is False
    assert "Sharpe de cartera <= 0" in motivos

    # Sharpe cartera etapa 3 <= benchmark
    pasa, motivos = evaluar_etapa3_cartera(eq_ok, sharpe_bench3=1e20, sharpe_cartera_etapa2=1.0, sharpe_bench2=0.5)
    assert pasa is False
    assert any("Sharpe de cartera" in m and "benchmark" in m for m in motivos)

    # Sharpe etapa 2 <= benchmark etapa 2
    pasa, motivos = evaluar_etapa3_cartera(eq_ok, sharpe_bench3=0.0, sharpe_cartera_etapa2=0.2, sharpe_bench2=0.5)
    assert pasa is False
    assert any("Sharpe etapa 2" in m and "benchmark" in m for m in motivos)

    # Menos de 3 años positivos
    ret_2_anios = np.where(fechas.year >= 2021, -0.001, 0.001)
    eq_2 = pd.Series(100_000.0 * np.cumprod(1 + ret_2_anios), index=fechas)
    pasa, motivos = evaluar_etapa3_cartera(eq_2, sharpe_bench3=-10.0, sharpe_cartera_etapa2=1.0, sharpe_bench2=0.5)
    assert pasa is False
    assert any("2 de 4" in m or "(< 3)" in m for m in motivos)


def test_evaluar_etapa3_cartera_sin_etapa2_o_sin_benchmark():
    fechas = pd.bdate_range("2019-01-01", "2022-12-31")
    eq_ok = pd.Series(100_000.0 * np.cumprod(1 + np.full(len(fechas), 0.001)), index=fechas)

    # Falta Sharpe de etapa 2
    pasa, motivos = evaluar_etapa3_cartera(eq_ok, sharpe_bench3=0.0, sharpe_cartera_etapa2=None, sharpe_bench2=0.5, trades=10)
    assert pasa is False
    assert "sin Sharpe de etapa 2" in motivos

    # Falta benchmark de etapa 2
    pasa, motivos = evaluar_etapa3_cartera(eq_ok, sharpe_bench3=0.0, sharpe_cartera_etapa2=1.0, sharpe_bench2=None, trades=10)
    assert pasa is False
    assert "sin benchmark de etapa 2" in motivos

    # Faltan ambos
    pasa, motivos = evaluar_etapa3_cartera(eq_ok, sharpe_bench3=0.0, sharpe_cartera_etapa2=None, sharpe_bench2=None, trades=10)
    assert pasa is False
    assert "sin Sharpe de etapa 2" in motivos
    assert "sin benchmark de etapa 2" in motivos

