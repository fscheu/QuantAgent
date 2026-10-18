from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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


def _panel_volatilidades():
    """8 activos con retornos alternados +a/-a (σ ≈ a): el activo 'Vk' tiene σ creciente con k."""
    idx = pd.bdate_range("2021-01-01", periods=600)
    signo = np.where(np.arange(len(idx)) % 2 == 0, 1.0, -1.0)[:, None]
    amplitudes = np.array([0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016])
    return pd.DataFrame(100 * np.cumprod(1 + signo * amplitudes, axis=0), index=idx, columns=[f"V{k}" for k in range(8)])


def test_baja_volatilidad_puntua_menos_sigma_y_exige_ventana_completa():
    c = _panel_volatilidades()
    m = METRICAS["baja-volatilidad"](c)
    # los primeros 252 retornos no alcanzan hasta la sesión 253: antes, NaN
    assert m.iloc[:252].isna().all().all()
    ultimo = m.iloc[-1]
    assert (ultimo < 0).all()
    assert ultimo.is_monotonic_decreasing  # V0 (más calmo) puntúa más alto
    # valor conocido contra el desvío calculado a mano sobre la ventana
    ventana = c["V3"].pct_change().iloc[-252:]
    assert ultimo["V3"] == -ventana.std()


def test_baja_volatilidad_elige_el_tercio_mas_calmo():
    s = señales_transversales(_panel_volatilidades(), METRICAS["baja-volatilidad"])
    # 8 activos -> k = 2: V0 y V1; antes de tener 252 retornos no hay pierna
    assert list(s.columns[s.iloc[-1] == 1]) == ["V0", "V1"]
    assert s.iloc[:252].sum().sum() == 0


def test_baja_volatilidad_no_mira_el_futuro():
    c = _aleatorio(n=800)
    cortes = [c.index[i] for i in (420, 520, 620, 700)]
    assert check_no_lookahead(lambda p: señales_transversales(p, METRICAS["baja-volatilidad"]), c, cortes) == []


def test_baja_volatilidad_igual_a_un_bucle_independiente():
    c = _aleatorio(n=800)
    esperado = pd.DataFrame(0, index=c.index, columns=c.columns)
    vigente = {a: 0 for a in c.columns}
    for i, fecha in enumerate(c.index):
        if is_last_session_of_month(fecha) and i >= 252:
            sigma = {}
            for a in c.columns:
                r = [c[a].iloc[j] / c[a].iloc[j - 1] - 1 for j in range(i - 251, i + 1)]
                if not np.isnan(r).any():
                    sigma[a] = np.std(r, ddof=1)
            vigente = {a: 0 for a in c.columns}
            if len(sigma) >= 3:
                for a in sorted(sigma, key=lambda a: (sigma[a], a))[: max(1, len(sigma) // 3)]:
                    vigente[a] = 1
        esperado.loc[fecha] = pd.Series(vigente)
    pd.testing.assert_frame_equal(señales_transversales(c, METRICAS["baja-volatilidad"]), esperado)
    assert esperado.to_numpy().sum() > 0


def test_universo_de_baja_volatilidad_es_solo_acciones_y_tiene_al_menos_6():
    """T5: con menos de 6 ETFs el tercio es 1 solo. El catálogo declara 17 ETFs de acciones antes de correr."""
    ruta = Path(__file__).resolve().parent.parent / "catalogo" / "primas.yaml"
    entrada = next(e for e in yaml.safe_load(ruta.read_text())["primas"] if e["clase"] == "baja-volatilidad")
    activos = set(entrada["universo"]["activos"])
    assert len(activos) >= 6
    assert activos.isdisjoint({"AGG", "IEF", "LQD", "SHY", "TIP", "TLT", "GLD", "SLV", "DBC", "USO", "FXE", "VNQ"})
