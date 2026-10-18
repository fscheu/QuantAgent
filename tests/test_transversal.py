from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quantagent.backtesting.signals import check_no_lookahead
from quantagent.fabrica.admision import UNIVERSO_AMPLIO
from quantagent.fabrica.transversal import (
    METRICAS,
    MIN_ACTIVOS_PIERNA,
    UNIVERSO_ACCIONES,
    baja_volatilidad,
    señales_transversales,
)
from quantagent.strategy.momentum_12m_strategy import is_last_session_of_month

RAIZ = Path(__file__).parent.parent


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


def _paneles_volatilidad(n=400):
    """6 activos con retornos alternados +a/-a de amplitud creciente: A el más calmo, F el más volátil."""
    idx = pd.bdate_range("2021-01-04", periods=n)
    amplitudes = np.array([0.002, 0.004, 0.006, 0.008, 0.010, 0.012])
    signo = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)[:, None]
    return pd.DataFrame(100 * np.cumprod(1 + signo * amplitudes, axis=0), index=idx, columns=list("ABCDEF"))


def test_baja_volatilidad_devuelve_menos_sigma_y_exige_252_retornos():
    c = _paneles_volatilidad()
    m = baja_volatilidad(c)
    # 252 retornos recién completos en la fila 252 (el primer retorno es la fila 1)
    assert m.iloc[:252].isna().all().all()
    assert m.iloc[252].notna().all()
    esperado = -c.pct_change().iloc[1:253].std()
    pd.testing.assert_series_equal(m.iloc[252], esperado, check_names=False)
    # mayor es mejor: el más calmo tiene el puntaje más alto
    assert list(m.iloc[300].sort_values(ascending=False).index) == list("ABCDEF")


def test_baja_volatilidad_elige_el_tercio_mas_calmo():
    s = señales_transversales(_paneles_volatilidad(), METRICAS["baja-volatilidad"])
    ultima = s.index[-1]
    # 6 activos -> k = 2: entran A y B (los de menor volatilidad)
    assert list(s.columns[s.loc[ultima] == 1]) == ["A", "B"]
    assert s.loc[:"2021-11-30"].sum().sum() == 0  # antes de la fila 252 no hay ranking


def test_baja_volatilidad_no_mira_el_futuro():
    c = _aleatorio()
    cortes = [c.index[i] for i in (300, 420, 480, 560, 640)]
    assert check_no_lookahead(lambda p: señales_transversales(p, METRICAS["baja-volatilidad"]), c, cortes) == []
    assert (señales_transversales(c, baja_volatilidad).to_numpy().sum()) > 0


def test_baja_volatilidad_ignora_un_precio_futuro_de_otro_activo():
    c = _aleatorio()
    corte = c.index[500]
    alterado = c.copy()
    alterado.loc[alterado.index > corte, "A"] *= 50  # el futuro de A no puede cambiar el ranking de antes
    antes = señales_transversales(c, baja_volatilidad).loc[:corte]
    pd.testing.assert_frame_equal(señales_transversales(alterado, baja_volatilidad).loc[:corte], antes)


def test_universo_de_acciones_declarado_por_definicion():
    assert len(UNIVERSO_ACCIONES) >= MIN_ACTIVOS_PIERNA  # T5: con menos de 6 el tercio es 1 solo
    assert UNIVERSO_ACCIONES <= UNIVERSO_AMPLIO
    # ningún bono, materia prima, divisa ni inmobiliario
    assert not UNIVERSO_ACCIONES & {"AGG", "IEF", "LQD", "SHY", "TIP", "TLT", "GLD", "SLV", "DBC", "USO", "FXE", "VNQ"}
    # coincide con la lista `*acciones` del catálogo (P01)
    catalogo = yaml.safe_load((RAIZ / "catalogo" / "primas.yaml").read_text(encoding="utf-8"))["primas"]
    por_id = {p["id"]: p for p in catalogo}
    assert set(por_id["baja-volatilidad-transversal"]["universo"]["activos"]) == UNIVERSO_ACCIONES
    assert por_id["baja-volatilidad-transversal"]["clase"] == "baja-volatilidad"


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
