import numpy as np
import pandas as pd
from quantagent.fabrica.regimen import (
    clasificar_regimen,
    permutar_regimenes_bloques,
    sharpe_por_regimen,
)


def test_clasificar_regimen_sin_mirar_futuro():
    fechas = pd.date_range("2020-01-01", periods=300, freq="D")
    rng = np.random.default_rng(42)
    precios = pd.Series(100.0 * np.exp(rng.normal(0, 0.01, size=300).cumsum()), index=fechas)
    reg1 = clasificar_regimen(precios, ventana_vol=5, mediana_vol_dias=20, ventana_media=15)
    t = fechas[250]
    precios_alterados = precios.copy()
    precios_alterados.loc[t:] = precios_alterados.loc[t:] * 3.0
    reg2 = clasificar_regimen(precios_alterados, ventana_vol=5, mediana_vol_dias=20, ventana_media=15)
    pd.testing.assert_series_equal(reg1.loc[:t], reg2.loc[:t])
    assert not reg1.loc[fechas[251]:].equals(reg2.loc[fechas[251]:])


def test_sharpe_por_regimen_series_sinteticas():
    r = pd.Series([0.01, 0.02, 0.03, -0.01, -0.02, -0.03])
    reg = pd.Series(["A", "A", "A", "B", "B", "B"])
    res = sharpe_por_regimen(r, reg, anualizar=False)
    assert np.isclose(res["A"], 0.02 / pd.Series([0.01, 0.02, 0.03]).std())
    assert np.isclose(res["B"], -0.02 / pd.Series([-0.01, -0.02, -0.03]).std())


def test_permutar_regimenes_bloques_conserva_frecuencias():
    fechas = pd.date_range("2020-01-01", "2020-04-30", freq="D")
    regs = pd.Series(np.where(np.arange(len(fechas)) % 2 == 0, "A", "B"), index=fechas)
    perm = permutar_regimenes_bloques(regs, seed=42)
    assert len(perm) == len(regs)
    assert perm.value_counts().to_dict() == regs.value_counts().to_dict()
    assert not perm.equals(regs)
