import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica.cartera import combinar, pesos

PIERNAS = {"p1": ["A", "B"], "p2": ["C"]}


def _cierres(n=120, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    ret = rng.standard_normal((n, 3)) * [0.01, 0.02, 0.015]
    return pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=["A", "B", "C"])


def test_igual_riesgo_dentro_de_pierna_y_suma_uno():
    w = pesos(_cierres(), PIERNAS, ventana=60).iloc[-1]
    # σ(B) es ~2× σ(A): A pesa ~2/3 de la pierna (cada pierna 1/2), C sola vale 1/2
    assert w[("p1", "A")] / w[("p1", "B")] == pytest.approx(2, rel=0.25)
    assert w[("p1", "A")] + w[("p1", "B")] == pytest.approx(0.5)
    assert w[("p2", "C")] == pytest.approx(0.5)
    assert w.sum() == pytest.approx(1)


def test_proporcion_exacta_2_a_1():
    # retornos alternando ±1% y ±2%: σ exacta en proporción 1:2
    n = 80
    idx = pd.bdate_range("2020-01-01", periods=n)
    signo = np.where(np.arange(n) % 2 == 0, 1, -1)
    c = pd.DataFrame({"A": 100 * np.cumprod(1 + 0.01 * signo), "B": 100 * np.cumprod(1 + 0.02 * signo)}, index=idx)
    w = pesos(c, {"p": ["A", "B"]}, ventana=60).iloc[-1]
    assert w[("p", "A")] == pytest.approx(2 / 3, abs=0.02)
    assert w[("p", "B")] == pytest.approx(1 / 3, abs=0.02)


def test_futuro_no_cambia_pesos_vigentes():
    c = _cierres()
    t = c.index[90]
    c2 = c.copy()
    c2.loc[c.index > t] *= 1.5
    c2.loc[c.index > t, "A"] *= np.linspace(1, 3, (c.index > t).sum())
    pd.testing.assert_frame_equal(pesos(c, PIERNAS).loc[:t], pesos(c2, PIERNAS).loc[:t])


def test_pesos_solo_cambian_primera_sesion_del_mes():
    c = _cierres()
    w = pesos(c, PIERNAS)
    cambia = w.diff().abs().sum(axis=1) > 1e-12
    primera = c.index.to_series().dt.to_period("M").ne(c.index.to_series().dt.to_period("M").shift())
    assert cambia.any()
    assert not (cambia & ~primera).any()


def test_antes_de_ventana_pesan_por_igual():
    w = pesos(_cierres(), PIERNAS, ventana=60).iloc[0]
    assert w[("p1", "A")] == w[("p1", "B")] == pytest.approx(0.25)


def _seleccion(cierres, elegidos_por_mes):
    """Panel 0/1 donde cada mes (en orden) tiene elegidos fijos, como lo deja la señal transversal."""
    meses = cierres.index.to_period("M")
    sel = pd.DataFrame(0, index=cierres.index, columns=cierres.columns)
    for i, mes in enumerate(meses.unique()):
        sel.loc[meses == mes, elegidos_por_mes[i % len(elegidos_por_mes)]] = 1
    return sel


def test_pierna_transversal_reparte_solo_entre_elegidos():
    c = _cierres(n=200)
    sel = _seleccion(c, [["A"], ["B", "C"], ["A", "B"], ["C"], ["A", "B", "C"], ["A", "B"], ["A"], ["C"]])
    w = pesos(c, {"x": ["A", "B", "C"]}, ventana=60, seleccion={"x": sel})
    # mes de índice 5 (ya con 60 sesiones de σ): elegidos A y B; C pesa 0 y la pierna entera suma 1
    f = w.loc[c.index.to_period("M") == c.index.to_period("M").unique()[5]].iloc[0]
    assert f[("x", "C")] == 0
    assert f.sum() == pytest.approx(1)
    # σ(B) ≈ 2×σ(A): A pesa ~2/3 entre los dos elegidos
    assert f[("x", "A")] / f[("x", "B")] == pytest.approx(2, rel=0.3)
    # mes con un solo elegido: ese activo pesa toda la pierna
    f0 = w.loc[c.index.to_period("M") == c.index.to_period("M").unique()[3]].iloc[0]
    assert f0[("x", "C")] == pytest.approx(1)
    assert f0[("x", "A")] == f0[("x", "B")] == 0


def test_transversal_cuenta_en_S_y_no_altera_piernas_por_activo():
    c = _cierres(n=200)
    sel = _seleccion(c, [["A", "C"]] * 8)
    con = pesos(c, {"p1": ["A", "B"], "t": ["A", "B", "C"]}, seleccion={"t": sel}).iloc[-1]
    solo = pesos(c, {"p1": ["A", "B"], "p2": ["A", "B", "C"]}).iloc[-1]
    assert con[("t", "B")] == 0
    assert con[("t", "A")] + con[("t", "C")] == pytest.approx(0.5)
    assert con.sum() == pytest.approx(1)
    # la pierna por activo pesa igual que si la otra fuera por activo: 1/S = 1/2 en ambos casos
    assert con[("p1", "A")] == pytest.approx(solo[("p1", "A")])
    assert con[("p1", "B")] == pytest.approx(solo[("p1", "B")])


def test_transversal_sin_elegidos_queda_en_efectivo():
    c = _cierres(n=200)
    sel = pd.DataFrame(0, index=c.index, columns=c.columns)
    w = pesos(c, {"p1": ["A"], "t": ["A", "B", "C"]}, seleccion={"t": sel}).iloc[-1]
    assert w[("p1", "A")] == pytest.approx(0.5)
    assert w.xs("t").sum() == 0


def test_futuro_no_cambia_pesos_con_seleccion():
    c = _cierres(n=200)
    sel = _seleccion(c, [["A"], ["B", "C"], ["A", "B"], ["C"], ["A", "B", "C"], ["B"], ["A"], ["C"]])
    t = c.index[120]
    c2 = c.copy()
    c2.loc[c.index > t] *= 1.5
    c2.loc[c.index > t, "A"] *= np.linspace(1, 3, (c.index > t).sum())
    pd.testing.assert_frame_equal(
        pesos(c, PIERNAS, seleccion={"p1": sel[["A", "B"]]}).loc[:t],
        pesos(c2, PIERNAS, seleccion={"p1": sel[["A", "B"]]}).loc[:t])


def test_combinar_a_mano():
    idx = pd.bdate_range("2020-01-01", periods=2)
    cols = pd.MultiIndex.from_tuples([("p1", "A"), ("p1", "B"), ("p2", "C")])
    w = pd.DataFrame([[0.2, 0.3, 0.5]] * 2, index=idx, columns=cols)
    r = pd.DataFrame([[0.01, -0.02, 0.0], [0.0, 0.01, 0.04]], index=idx, columns=cols)
    out = combinar(r, w)
    assert out.iloc[0] == pytest.approx(0.2 * 0.01 + 0.3 * -0.02)
    assert out.iloc[1] == pytest.approx(0.3 * 0.01 + 0.5 * 0.04)
