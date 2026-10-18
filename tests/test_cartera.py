import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica.cartera import combinar, evaluar_cartera, pesos

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


def test_combinar_a_mano():
    idx = pd.bdate_range("2020-01-01", periods=2)
    cols = pd.MultiIndex.from_tuples([("p1", "A"), ("p1", "B"), ("p2", "C")])
    w = pd.DataFrame([[0.2, 0.3, 0.5]] * 2, index=idx, columns=cols)
    r = pd.DataFrame([[0.01, -0.02, 0.0], [0.0, 0.01, 0.04]], index=idx, columns=cols)
    out = combinar(r, w)
    assert out.iloc[0] == pytest.approx(0.2 * 0.01 + 0.3 * -0.02)
    assert out.iloc[1] == pytest.approx(0.3 * 0.01 + 0.5 * 0.04)


def test_evaluar_cartera_una_pierna_un_activo_coincide():
    from quantagent.backtesting.vbt_filter import equity_metrics

    c = _cierres()
    idx = c.index[60:]
    ret_a = pd.Series(0.001, index=idx)
    s = evaluar_cartera(c, {"p1": ["A"]}, {("p1", "A"): ret_a})
    eq = 100_000.0 * (1 + ret_a).cumprod()
    assert s == pytest.approx(equity_metrics(eq, 0.0)["sharpe"])
