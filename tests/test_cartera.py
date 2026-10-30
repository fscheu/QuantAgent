import numpy as np
import pandas as pd
import pytest

from quantagent.fabrica.cartera import (
    combinar,
    curva_cartera,
    evaluar_cartera,
    pesos,
    pesos_f1,
    pesos_h3,
    pesos_objetivo,
)

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


def test_pesos_objetivo_fecha_coincide_con_pesos_de_curva():
    cierres = _cierres()
    piernas = {"p1": ["A", "B"], "p2": ["C"]}
    idx = cierres.index[60:]
    retornos = {(p, a): cierres[a].pct_change().loc[idx] for p, activos in piernas.items() for a in activos}
    curva = curva_cartera(cierres, piernas, retornos)
    fecha = cierres.index[80]
    fecha_curva = curva.index[curva.index <= fecha][-1]
    esperados = pesos(cierres, piernas).loc[fecha_curva].T.groupby(level=1).sum().to_dict()
    assert fecha_curva in curva.index
    assert pesos_objetivo(cierres, piernas, fecha) == pytest.approx(esperados)


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


def test_f1_usa_desvio_60_40_en_los_ultimos_cinco_cierres():
    idx = pd.bdate_range("2024-01-31", periods=7)
    cierres = pd.DataFrame({"SPY": [100, 90, 110, 100, 100, 100, 100], "IEF": [100] * 7}, index=idx)
    w = pesos_f1(cierres)
    assert tuple(w.loc[idx[1]]) == (0.5, 0.5)  # fuera de los últimos cinco
    assert tuple(w.loc[idx[2]]) == (1.0, 0.0)  # SPY bajo 60 % al cierre previo
    cierres.loc[idx[2], "SPY"] = 110
    assert tuple(pesos_f1(cierres).loc[idx[3]]) == (0.0, 1.0)
    cierres.loc[idx[2], "SPY"] = 100
    assert tuple(pesos_f1(cierres).loc[idx[3]]) == (0.5, 0.5)  # exactamente 60 %


def test_f1_no_consulta_cierres_posteriores_a_la_decision():
    cierres = pd.DataFrame({"SPY": [100, 90, 110, 100, 100, 100, 100], "IEF": [100] * 7}, index=pd.bdate_range("2024-01-31", periods=7))
    fecha = cierres.index[3]
    alterados = cierres.copy()
    alterados.loc[alterados.index > fecha] *= 3
    pd.testing.assert_series_equal(pesos_f1(cierres).loc[fecha], pesos_f1(alterados).loc[fecha])


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


def test_evaluar_cartera_una_pierna_un_activo_coincide():
    from quantagent.backtesting.vbt_filter import equity_metrics

    c = _cierres()
    idx = c.index[60:]
    ret_a = pd.Series(0.001, index=idx)
    s = evaluar_cartera(c, {"p1": ["A"]}, {("p1", "A"): ret_a})
    eq = 100_000.0 * (1 + ret_a).cumprod()
    assert s == pytest.approx(equity_metrics(eq, 0.0)["sharpe"])


def test_curva_y_evaluacion_transversal_usan_seleccion():
    from quantagent.backtesting.vbt_filter import equity_metrics

    idx = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"])
    cierres = pd.DataFrame({"A": [100, 101, 102], "B": [100, 102, 104], "C": [100, 103, 106]}, index=idx)
    seleccion = {"t": pd.DataFrame({"A": [1, 1, 0], "B": [0, 0, 1], "C": [0, 0, 1]}, index=idx)}
    retornos = {("t", "A"): pd.Series([0.0, 0.1, 0.0], index=idx),
                ("t", "B"): pd.Series([0.0, 0.0, 0.02], index=idx),
                ("t", "C"): pd.Series([0.0, 0.0, 0.02], index=idx)}
    piernas = {"t": ["A", "B", "C"]}

    solo_a = {"t": pd.DataFrame({"A": 1, "B": 0, "C": 0}, index=idx)}
    assert curva_cartera(cierres, piernas, retornos, seleccion=solo_a).iloc[-1] == pytest.approx(110_000)
    curva = curva_cartera(cierres, piernas, retornos, seleccion=seleccion)
    assert curva.iloc[-1] == pytest.approx(112_200)
    assert curva_cartera(cierres, piernas, retornos).iloc[-1] == pytest.approx(104_711.1111)
    sharpe_t4 = evaluar_cartera(cierres, piernas, retornos, seleccion=seleccion)
    assert sharpe_t4 == pytest.approx(equity_metrics(curva, 0.0)["sharpe"])
    assert sharpe_t4 != pytest.approx(evaluar_cartera(cierres, piernas, retornos))


def test_pesos_rebalancea_cada_semana_solo_la_pierna_semanal():
    c = _cierres()
    sel = {"p1": pd.DataFrame(1, index=c.index, columns=["A", "B"])}
    w = pesos(c, PIERNAS, seleccion=sel, semanales={"p1"})
    cambia = w.diff().abs() > 1e-12
    semana = c.index.to_series().apply(lambda d: d.isocalendar()[:2])
    mes = c.index.to_series().dt.to_period("M")
    primera_semana = semana.ne(semana.shift())
    primero_mes = mes.ne(mes.shift())
    assert (cambia[("p1", "A")] & ~primera_semana & ~primero_mes).sum() == 0
    assert (cambia[("p1", "A")] & primera_semana & ~primero_mes).any()
    assert not (cambia[("p2", "C")] & ~primero_mes).any()
    assert pesos(c, PIERNAS, seleccion=sel).equals(pesos(c, PIERNAS, seleccion=sel, semanales=()))


def test_pesos_h3_reemplaza_cada_etf_sectorial_por_sus_top3():
    fecha, decision = pd.Timestamp("2015-01-05"), pd.Timestamp("2015-01-02")
    h1 = pd.DataFrame({"XLK": [0.4], "XLE": [0.2], "XLF": [0.1], "SPY": [0.3], "XLU": [0.0]}, index=[fecha])
    cand = pd.DataFrame({
        "fecha": [decision] * 6, "etf": ["XLK"] * 3 + ["XLF"] * 2 + ["SPY"], "simbolo": ["A", "B", "C", "D", "E", "SPY"],
        "rank": [1, 2, 3, 1, 2, float("nan")],
    })
    w = pesos_h3(h1, cand, {fecha: decision}).loc[fecha]
    assert w[["A", "B", "C"]].tolist() == pytest.approx([0.4 / 3] * 3)
    assert w[["D", "E"]].tolist() == pytest.approx([0.05, 0.05])
    assert w["XLE"] == 0.2 and w["SPY"] == 0.3  # sin candidatos cae al ETF; SPY sigue como ETF
    assert "XLK" not in w.index and w.get("XLU", 0.0) == 0.0 and w.sum() == pytest.approx(1.0)
    cero = h1.mul(0.0)
    assert pesos_h3(cero, cand, {fecha: decision}).loc[fecha].sum() == 0 and len(pesos_h3(cero, cand, {fecha: decision})) == 1
