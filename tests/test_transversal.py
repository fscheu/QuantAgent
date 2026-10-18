import numpy as np
import pandas as pd

from quantagent.backtesting.signals import check_no_lookahead
from quantagent.fabrica.transversal import METRICAS, señales_transversales
from quantagent.strategy.momentum_12m_strategy import is_last_session_of_month


def _retorno_5(cierres):
    return cierres.pct_change(5, fill_method=None)


def _doce_menos_uno(cierres):
    """cierre[t-21] / cierre[t-252] - 1, la métrica de momentum que usará P06 (acá solo sirve de ejemplo)."""
    return cierres.shift(21) / cierres.shift(252) - 1


def _paneles_dos_regimenes():
    """6 activos con crecimiento diario constante; el 2023-02-15 el orden se invierte."""
    idx = pd.bdate_range("2022-10-03", "2023-04-28")
    tasas = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
    retornos = np.where(np.asarray(idx < pd.Timestamp("2023-02-15"))[:, None], tasas, tasas[::-1])
    return pd.DataFrame(100 * np.cumprod(1 + retornos, axis=0), index=idx, columns=list("ABCDEF"))


def _aleatorio(n=700, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    c = pd.DataFrame(100 * np.cumprod(1 + rng.standard_normal((n, 8)) * 0.01, axis=0), index=idx, columns=list("HGFEDCBA"))
    c.loc[:idx[400], ["B", "C"]] = np.nan  # ETFs que nacieron después: sin historia al principio
    return c


def test_elige_el_tercio_superior_y_cambia_cuando_cambia_el_orden():
    s = señales_transversales(_paneles_dos_regimenes(), _retorno_5)
    # antes de la primera decisión (2022-10-31) no hay posición
    assert s.loc[:"2022-10-28"].sum().sum() == 0
    # 6 activos -> k = 2: los de mayor tasa son E y F
    assert list(s.columns[s.loc["2023-01-31"] == 1]) == ["E", "F"]
    assert list(s.columns[s.loc["2023-02-27"] == 1]) == ["E", "F"]
    # el 2023-02-28 el ranking ya se invirtió (ventana de 5 sesiones toda en el régimen nuevo)
    assert list(s.columns[s.loc["2023-02-28"] == 1]) == ["A", "B"]
    assert list(s.columns[s.loc["2023-03-30"] == 1]) == ["A", "B"]


def test_empate_se_resuelve_por_ticker_alfabetico():
    idx = pd.bdate_range("2022-12-01", "2023-02-28")
    plano = pd.DataFrame(100.0, index=idx, columns=list("FEDCBA"))
    s = señales_transversales(plano, _retorno_5)
    assert sorted(s.columns[s.loc["2023-01-31"] == 1]) == ["A", "B"]


def test_k_sale_de_los_disponibles_y_con_menos_de_tres_no_hay_pierna():
    c = _paneles_dos_regimenes()
    # 9 activos pero 3 sin historia -> 6 disponibles -> k = 2, no 3
    for extra in "GHI":
        c[extra] = np.nan
    s = señales_transversales(c, _retorno_5)
    assert s.loc["2023-01-31"].sum() == 2
    # solo 2 con historia: no hay pierna
    s2 = señales_transversales(c[["A", "B"]], _retorno_5)
    assert s2.sum().sum() == 0


def test_activo_sin_historia_nunca_entra():
    c = _aleatorio()
    s = señales_transversales(c, _doce_menos_uno)
    assert s.loc[:c.index[400], ["B", "C"]].sum().sum() == 0


def test_prueba_de_truncamiento_sobre_todo_el_panel():
    c = _aleatorio()
    cortes = [c.index[i] for i in (300, 420, 480, 560, 640)]
    assert check_no_lookahead(lambda p: señales_transversales(p, _doce_menos_uno), c, cortes) == []


def test_la_prueba_detecta_una_metrica_que_mira_el_futuro():
    c = _aleatorio()
    cortes = [c.index[i] for i in (420, 480, 560)]
    futuro = lambda p: p.shift(-5) / p - 1  # noqa: E731  (usa el precio de dentro de 5 sesiones)
    assert check_no_lookahead(lambda p: señales_transversales(p, futuro), c, cortes) != []


def test_igual_a_una_implementacion_independiente_por_bucle():
    c = _aleatorio()
    esperado = pd.DataFrame(0, index=c.index, columns=c.columns)
    vigente = {a: 0 for a in c.columns}
    for i, fecha in enumerate(c.index):
        if is_last_session_of_month(fecha) and i >= 252:
            punt = {a: c[a].iloc[i - 21] / c[a].iloc[i - 252] - 1 for a in c.columns}
            punt = {a: v for a, v in punt.items() if not np.isnan(v)}
            vigente = {a: 0 for a in c.columns}
            if len(punt) >= 3:
                for a in sorted(punt, key=lambda a: (-punt[a], a))[: max(1, len(punt) // 3)]:
                    vigente[a] = 1
        esperado.loc[fecha] = pd.Series(vigente)
    pd.testing.assert_frame_equal(señales_transversales(c, _doce_menos_uno), esperado)
    assert esperado.to_numpy().sum() > 0  # el panel sí produce posiciones; no es una comparación de ceros


def test_registro_de_metricas_arranca_vacio():
    assert METRICAS == {}
