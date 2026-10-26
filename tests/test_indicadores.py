"""Tests para indicadores técnicos compartidos (ATR, Chandelier, SafeZone)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.strategy.indicadores import atr, chandelier_long, safezone_long

FIXTURE_SPY = Path(__file__).parent / "fixtures" / "spy-2y-1d.csv"


# ---------------------------------------------------------------------------
# Cálculos independientes de referencia en Python puro (sin importar quantagent)
# ---------------------------------------------------------------------------


def _ref_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"].tolist()
    low = df["low"].tolist()
    close = df["close"].tolist()
    m = len(df)
    tr = [0.0] * m
    for i in range(m):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
    res = [float("nan")] * m
    if m >= n and n > 0:
        res[n - 1] = sum(tr[:n]) / n
        for i in range(n, m):
            res[i] = (res[i - 1] * (n - 1) + tr[i]) / n
    return pd.Series(res, index=df.index, dtype=float)


def _ref_chandelier_long(df: pd.DataFrame, n: int = 22, mult: float = 3.0) -> pd.Series:
    atr_vals = _ref_atr(df, n=n).tolist()
    high = df["high"].tolist()
    m = len(df)
    res = [float("nan")] * m
    if m >= n and n > 0:
        for i in range(n - 1, m):
            hh = max(high[i - n + 1 : i + 1])
            res[i] = hh - mult * atr_vals[i]
    return pd.Series(res, index=df.index, dtype=float)


def _ref_safezone_long(df: pd.DataFrame, n: int = 10, coef: float = 2.0) -> pd.Series:
    low = df["low"].tolist()
    m = len(df)
    pen = [0.0] * m
    for i in range(1, m):
        if low[i] < low[i - 1]:
            pen[i] = low[i - 1] - low[i]
    res = [float("nan")] * m
    if m >= n and n > 0:
        for i in range(n - 1, m):
            window_pen = pen[i - n + 1 : i + 1]
            pos = [p for p in window_pen if p > 0.0]
            avg = sum(pos) / len(pos) if pos else 0.0
            res[i] = low[i] - coef * avg
    return pd.Series(res, index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def df_spy():
    return pd.read_csv(FIXTURE_SPY)


def test_atr_coincide_con_calculo_independiente_spy(df_spy):
    act = atr(df_spy, n=14)
    exp = _ref_atr(df_spy, n=14)
    assert act.isna().sum() == 13
    np.testing.assert_allclose(act.dropna().to_numpy(), exp.dropna().to_numpy(), atol=1e-9)


def test_chandelier_long_coincide_con_calculo_independiente_spy(df_spy):
    act = chandelier_long(df_spy, n=22, mult=3.0)
    exp = _ref_chandelier_long(df_spy, n=22, mult=3.0)
    assert act.isna().sum() == 21
    np.testing.assert_allclose(act.dropna().to_numpy(), exp.dropna().to_numpy(), atol=1e-9)


def test_safezone_long_coincide_con_calculo_independiente_spy(df_spy):
    act = safezone_long(df_spy, n=10, coef=2.0)
    exp = _ref_safezone_long(df_spy, n=10, coef=2.0)
    assert act.isna().sum() == 9
    np.testing.assert_allclose(act.dropna().to_numpy(), exp.dropna().to_numpy(), atol=1e-9)


@pytest.mark.parametrize("t", [15, 21, 30, 50, 100, 250, 519])
def test_ninguna_funcion_mira_al_futuro(df_spy, t):
    df_trunc = df_spy.iloc[: t + 1].copy()

    # ATR
    a_full = atr(df_spy, n=14).iloc[t]
    a_trunc = atr(df_trunc, n=14).iloc[t]
    assert np.isclose(a_full, a_trunc, atol=1e-12, equal_nan=True)

    # Chandelier
    c_full = chandelier_long(df_spy, n=22, mult=3.0).iloc[t]
    c_trunc = chandelier_long(df_trunc, n=22, mult=3.0).iloc[t]
    assert np.isclose(c_full, c_trunc, atol=1e-12, equal_nan=True)

    # SafeZone
    s_full = safezone_long(df_spy, n=10, coef=2.0).iloc[t]
    s_trunc = safezone_long(df_trunc, n=10, coef=2.0).iloc[t]
    assert np.isclose(s_full, s_trunc, atol=1e-12, equal_nan=True)


def test_casos_borde():
    vacio = pd.DataFrame(columns=["high", "low", "close"])
    assert atr(vacio, n=14).empty
    assert safezone_long(vacio, n=10).empty

    corto = pd.DataFrame({"high": [10.0, 11.0], "low": [9.0, 9.5], "close": [9.8, 10.5]})
    assert atr(corto, n=14).isna().all()
    assert chandelier_long(corto, n=22).isna().all()
    assert safezone_long(corto, n=10).isna().all()

    with pytest.raises(ValueError):
        atr(corto, n=0)
    with pytest.raises(ValueError):
        chandelier_long(corto, n=-1)
    with pytest.raises(ValueError):
        safezone_long(corto, n=0)
