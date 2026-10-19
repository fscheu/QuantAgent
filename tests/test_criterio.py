import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica.criterio import anios_positivos, sharpe_por_anio


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
    assert ap == "3 de 4" and ap == (3, 4) and ap >= 3
    assert ap.positivos == 3 and ap.totales == 4
    pos, tot = ap
    assert (pos, tot) == (3, 4)


def test_anio_menos_30_sesiones_se_descarta_e_informa():
    eq = _generar_curva()
    eq_corto = pd.concat([pd.Series(90.0, index=pd.bdate_range("2018-12-01", periods=15)), eq])
    with pytest.warns(UserWarning, match="Año 2018 descartado.*15 sesiones"):
        sharpes = sharpe_por_anio(eq_corto)
    assert 2018 not in sharpes and len(sharpes) == 4
    assert anios_positivos(sharpes) == "3 de 4"


def test_curva_ya_calculada_sin_lectura_datos():
    dias = pd.bdate_range("2019-01-02", periods=40)
    vals = [100.0 + i * 0.1 for i in range(40)]
    assert anios_positivos(vals, fechas=dias) == "1 de 1"
