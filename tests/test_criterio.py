import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica.criterio import (
    AniosPositivos,
    RecargoPaper,
    anios_positivos,
    recargo_paper,
    sharpe_por_anio,
)


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
