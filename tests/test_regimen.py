from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quantagent.fabrica import regimen
from quantagent.fabrica.regimen import (
    clasificar_regimen,
    filtro_caida,
    macro_theme_score,
    permutar_regimenes_bloques,
    price_momentum_score,
    sector_score,
    sharpe_por_regimen,
    z_temas,
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


def test_price_momentum_score_calculo_a_mano():
    precios = pd.DataFrame({"ACT": [100.0, 110.0, 121.0, 108.9]})
    score = price_momentum_score(precios, ventanas=(1, 2), ventana_vol=3)
    assert score.iloc[:3, 0].isna().all()
    esperado = -0.275 * np.sqrt(3)
    assert np.isclose(score.iloc[3, 0], esperado)


def test_filtro_caida_tabla_cuatro_casos():
    fechas = pd.date_range("2020-01-01", periods=270, freq="B")
    precios = pd.DataFrame(
        {
            "ACT_ALCISTA": np.linspace(50, 200, 270),
            "ACT_BAJISTA": np.linspace(200, 50, 270),
        },
        index=fechas,
    )
    score = price_momentum_score(precios, ventanas=(21, 63, 126), ventana_vol=60)
    assert score.loc[fechas[220], "ACT_ALCISTA"] > 0
    assert score.loc[fechas[220], "ACT_BAJISTA"] < 0

    spy_precios = [100.0] * 200 + [80.0] * 21 + [150.0] * 49
    spy = pd.Series(spy_precios, index=fechas)
    filtro = filtro_caida(score, spy, ventana_media=200, pendiente=20)

    # 1. Caso score < 0 y SPY en caída -> Bloquea (True)
    t_caida = fechas[220]
    assert bool(filtro.loc[t_caida, "ACT_BAJISTA"]) is True
    # 2. Caso score > 0 y SPY en caída -> No bloquea (False)
    assert bool(filtro.loc[t_caida, "ACT_ALCISTA"]) is False

    # 3. Caso score < 0 y SPY sobre media (no en caída) -> No bloquea (False)
    t_no_caida = fechas[230]
    assert bool(filtro.loc[t_no_caida, "ACT_BAJISTA"]) is False
    # 4. Caso score > 0 y SPY sobre media (no en caída) -> No bloquea (False)
    assert bool(filtro.loc[t_no_caida, "ACT_ALCISTA"]) is False

    # SPY bajo la media pero con pendiente positiva (no bloquea)
    spy_pend_pos = pd.Series([50.0] * 30 + [100.0] * 190 + [85.0] * 50, index=fechas)
    filtro_pend_pos = filtro_caida(score, spy_pend_pos, ventana_media=200, pendiente=20)
    t_pend_pos = fechas[240]
    assert spy_pend_pos.loc[t_pend_pos] < spy_pend_pos.rolling(200).mean().loc[t_pend_pos]
    assert (
        spy_pend_pos.rolling(200).mean().loc[t_pend_pos]
        > spy_pend_pos.rolling(200).mean().shift(20).loc[t_pend_pos]
    )
    assert bool(filtro_pend_pos.loc[t_pend_pos, "ACT_BAJISTA"]) is False
    assert bool(filtro_pend_pos.loc[t_pend_pos, "ACT_ALCISTA"]) is False


def test_score_y_filtro_sin_mirar_futuro():
    fechas = pd.date_range("2020-01-01", periods=280, freq="B")
    rng = np.random.default_rng(42)
    p_activos = pd.DataFrame(
        {
            "A": 100.0 * np.exp(rng.normal(0, 0.01, size=280).cumsum()),
            "B": 100.0 * np.exp(rng.normal(0, 0.01, size=280).cumsum()),
        },
        index=fechas,
    )
    p_spy = pd.Series(100.0 * np.exp(rng.normal(0, 0.01, size=280).cumsum()), index=fechas)

    t = fechas[249]
    score_t = price_momentum_score(p_activos.loc[:t])
    filtro_t = filtro_caida(score_t, p_spy.loc[:t])

    # Recalculamos con 30 sesiones extra después de t (hasta 279)
    score_completo = price_momentum_score(p_activos)
    filtro_completo = filtro_caida(score_completo, p_spy)

    pd.testing.assert_frame_equal(score_t, score_completo.loc[:t])
    pd.testing.assert_frame_equal(filtro_t, filtro_completo.loc[:t])


def test_falta_historia_retorna_nan_o_false():
    fechas = pd.date_range("2020-01-01", periods=100, freq="B")
    precios = pd.DataFrame({"A": np.linspace(100, 150, 100)}, index=fechas)
    score = price_momentum_score(precios, ventanas=(21, 63, 126), ventana_vol=60)
    assert score["A"].isna().all()

    spy = pd.Series(np.linspace(100, 50, 100), index=fechas)
    filtro = filtro_caida(score, spy, ventana_media=200, pendiente=20)
    assert (filtro["A"] == False).all()
    assert filtro["A"].dtype == bool




def test_macro_theme_score_calculado_a_mano():
    priors = {
        "universo": ["X", "Y", "Z"],
        "temas": {
            "a": {"series": [{"serie": "s1"}, {"serie": "s2", "invertir": True}],
                  "signos": {"positivo": {"activos": ["X"]}, "negativo": {"activos": ["Y"]}}},
            "b": {"series": [{"serie": "s3"}], "signos": {"positivo": {"activos": ["X"]}}},
        },
    }
    z_series = pd.DataFrame({"s1": [1.0, 1.0], "s2": [0.5, 0.5], "s3": [2.0, np.nan]},
                            index=pd.to_datetime(["2020-01-03", "2020-01-10"]))
    score = macro_theme_score(z_temas(z_series, priors), priors)
    # fila 1: z_a = (1 - 0,5) / 2 = 0,25; z_b = 2 -> X = (0,25 + 2) / 2, Y = -0,25, Z sin signos = 0
    assert score.iloc[0].tolist() == [1.125, -0.25, 0.0]
    # fila 2: b sin dato no cuenta en el denominador de X
    assert score.iloc[1].tolist() == [0.25, -0.25, 0.0]


def test_priors_m5_versionados_completos():
    priors = yaml.safe_load((Path(__file__).parents[1] / "catalogo" / "m5-priors.yaml").read_text(encoding="utf-8"))
    assert "ARGT" not in priors["universo"]
    for tema, d in priors["temas"].items():
        for celda in d["signos"].values():
            assert set(celda["activos"]) <= set(priors["universo"]), tema
            assert celda["fuentes"] and all(f["cita"] and f["link"] for f in celda["fuentes"]), tema


def test_sector_score_calculado_a_mano():
    idx = pd.to_datetime(["2020-01-03", "2020-01-10"])
    momentum = pd.DataFrame({"A": [1.0, 1.0], "B": [2.0, 3.0], "C": [3.0, np.nan]}, index=idx)
    macro = pd.DataFrame({"A": [3.0, 1.0], "B": [1.0, 5.0], "C": [2.0, 9.0]}, index=idx)
    # fila 1: z(mom) = [-1, 0, 1], z(macro) = [1, -1, 0]
    mixto = sector_score(macro, momentum, 0.5)
    assert mixto.iloc[0].tolist() == [0.0, -0.5, 0.5]
    assert sector_score(macro, momentum, 0.0).iloc[0].tolist() == [-1.0, 0.0, 1.0]
    # fila 2: C sin momentum queda NaN y no entra en el z transversal de los otros dos
    r = 0.5 ** 0.5
    assert mixto.iloc[1, :2].tolist() == pytest.approx([-r, r])
    assert np.isnan(mixto.iloc[1, 2])


PRIORS_SINTETICOS = {
    "version": 7,
    "universo": ["SPY", "A", "B"],
    "temas": {
        "crecimiento": {"series": [{"serie": "s1"}], "signos": {"positivo": {"activos": ["A"]}}},
        "estres": {"series": [{"serie": "s2"}], "signos": {"negativo": {"activos": ["B"]}}},
    },
}


@pytest.fixture
def cierres_sinteticos(monkeypatch):
    monkeypatch.setattr(
        regimen, "z_series_macro", lambda priors, sesiones, base_dir=None: pd.DataFrame({"s1": 0.5, "s2": 0.0}, index=sesiones)
    )
    idx = pd.bdate_range("2005-01-03", "2008-12-31")
    rng = np.random.default_rng(1)
    c = pd.DataFrame({a: 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(idx)))) for a in ("SPY", "A", "B")}, index=idx)
    c.loc[:"2007-06-01", "B"] = np.nan  # B sin historia hasta mediados de 2007
    return c


def test_regimen_semanal_una_fila_por_semana_y_activo_sin_historia(cierres_sinteticos):
    tabla = regimen.regimen_semanal(cierres_sinteticos, PRIORS_SINTETICOS, 0.5)
    sesiones = cierres_sinteticos.loc["2006-01-01":].index
    assert len(tabla) == sesiones.to_series().groupby(sesiones.to_period("W")).ngroups
    assert tabla["fecha"].is_unique and tabla["fecha"].min() >= pd.Timestamp("2006-01-01")
    assert (tabla["version"] == 7).all() and set(tabla["estado"]) <= {"risk-on", "neutral", "risk-off"}
    # B tiene momentum recién ~126 sesiones después de su primer cierre
    sin_historia = tabla[tabla["fecha"] < "2007-11-01"]
    assert sin_historia["score_B"].isna().all()
    assert not sin_historia["habilitados"].str.contains("B").any()
    assert tabla["score_B"].notna().any() and tabla["habilitados"].str.contains("A").any()


def test_regimen_semanal_no_mira_el_futuro(cierres_sinteticos):
    completa = regimen.regimen_semanal(cierres_sinteticos, PRIORS_SINTETICOS, 0.5)
    t = completa["fecha"].iloc[100]
    truncada = regimen.regimen_semanal(cierres_sinteticos.loc[:t], PRIORS_SINTETICOS, 0.5)
    pd.testing.assert_frame_equal(truncada.iloc[[-1]].reset_index(drop=True), completa.iloc[[100]].reset_index(drop=True))
