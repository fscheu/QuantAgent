"""Tests para indicadores técnicos compartidos (ATR, Chandelier, SafeZone)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.strategy.indicadores import (
    atr,
    bollinger_width,
    chandelier_long,
    donchian_high,
    force_index,
    safezone_long,
    volumen_relativo,
)

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


def _ref_force_index(df: pd.DataFrame, n: int = 2) -> pd.Series:
    close = df["close"].tolist()
    vol = df["volume"].tolist()
    m = len(df)
    res = [float("nan")] * m
    if m >= 2 and n > 0:
        alpha = 2.0 / (n + 1.0)
        res[1] = (close[1] - close[0]) * vol[1]
        for i in range(2, m):
            res[i] = (1.0 - alpha) * res[i - 1] + alpha * (close[i] - close[i - 1]) * vol[i]
    return pd.Series(res, index=df.index, dtype=float)


def _ref_donchian_high(df: pd.DataFrame, n: int = 20) -> pd.Series:
    high = df["high"].tolist()
    m = len(df)
    res = [float("nan")] * m
    if m >= n and n > 0:
        for i in range(n - 1, m):
            res[i] = max(high[i - n + 1 : i + 1])
    return pd.Series(res, index=df.index, dtype=float)


def _ref_bollinger_width(close, n: int = 20, k: float = 2.0) -> pd.Series:
    s = close["close"] if isinstance(close, pd.DataFrame) else close
    vals = s.tolist()
    m = len(vals)
    res = [float("nan")] * m
    if m >= n and n > 1:
        for i in range(n - 1, m):
            win = vals[i - n + 1 : i + 1]
            mean = sum(win) / n
            var = sum((x - mean) ** 2 for x in win) / (n - 1)
            std = var ** 0.5
            res[i] = (2.0 * k * std) / mean
    return pd.Series(res, index=s.index, dtype=float)


def _ref_volumen_relativo(df, n: int = 20) -> pd.Series:
    v = df["volume"] if isinstance(df, pd.DataFrame) else df
    vol = v.tolist()
    m = len(vol)
    res = [float("nan")] * m
    if m >= n and n > 0:
        for i in range(n - 1, m):
            win = vol[i - n + 1 : i + 1]
            avg = sum(win) / n
            res[i] = vol[i] / avg if avg != 0.0 else float("nan")
    return pd.Series(res, index=v.index, dtype=float)


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


def test_force_index_coincide_con_calculo_independiente_spy(df_spy):
    act = force_index(df_spy, n=2)
    exp = _ref_force_index(df_spy, n=2)
    assert act.isna().sum() == 1
    np.testing.assert_allclose(act.dropna().to_numpy(), exp.dropna().to_numpy(), atol=1e-9)


def test_donchian_high_coincide_con_calculo_independiente_spy(df_spy):
    act = donchian_high(df_spy, n=20)
    exp = _ref_donchian_high(df_spy, n=20)
    assert act.isna().sum() == 19
    np.testing.assert_allclose(act.dropna().to_numpy(), exp.dropna().to_numpy(), atol=1e-9)


def test_bollinger_width_coincide_con_calculo_independiente_spy(df_spy):
    act = bollinger_width(df_spy["close"], n=20, k=2.0)
    exp = _ref_bollinger_width(df_spy["close"], n=20, k=2.0)
    assert act.isna().sum() == 19
    np.testing.assert_allclose(act.dropna().to_numpy(), exp.dropna().to_numpy(), atol=1e-9)
    act_df = bollinger_width(df_spy, n=20, k=2.0)
    np.testing.assert_allclose(act_df.dropna().to_numpy(), exp.dropna().to_numpy(), atol=1e-9)


def test_volumen_relativo_coincide_con_calculo_independiente_spy(df_spy):
    act = volumen_relativo(df_spy, n=20)
    exp = _ref_volumen_relativo(df_spy, n=20)
    assert act.isna().sum() == 19
    np.testing.assert_allclose(act.dropna().to_numpy(), exp.dropna().to_numpy(), atol=1e-9)
    act_s = volumen_relativo(df_spy["volume"], n=20)
    np.testing.assert_allclose(act_s.dropna().to_numpy(), exp.dropna().to_numpy(), atol=1e-9)


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

    # Force Index
    f_full = force_index(df_spy, n=2).iloc[t]
    f_trunc = force_index(df_trunc, n=2).iloc[t]
    assert np.isclose(f_full, f_trunc, atol=1e-12, equal_nan=True)

    # Donchian High
    d_full = donchian_high(df_spy, n=20).iloc[t]
    d_trunc = donchian_high(df_trunc, n=20).iloc[t]
    assert np.isclose(d_full, d_trunc, atol=1e-12, equal_nan=True)

    # Bollinger Width
    b_full = bollinger_width(df_spy, n=20, k=2.0).iloc[t]
    b_trunc = bollinger_width(df_trunc, n=20, k=2.0).iloc[t]
    assert np.isclose(b_full, b_trunc, atol=1e-12, equal_nan=True)

    # Volumen Relativo
    v_full = volumen_relativo(df_spy, n=20).iloc[t]
    v_trunc = volumen_relativo(df_trunc, n=20).iloc[t]
    assert np.isclose(v_full, v_trunc, atol=1e-12, equal_nan=True)


def test_casos_borde():
    vacio = pd.DataFrame(columns=["high", "low", "close", "volume"])
    assert atr(vacio, n=14).empty
    assert safezone_long(vacio, n=10).empty
    assert force_index(vacio, n=2).empty
    assert donchian_high(vacio, n=20).empty
    assert bollinger_width(vacio, n=20).empty
    assert volumen_relativo(vacio, n=20).empty

    corto = pd.DataFrame({"high": [10.0, 11.0], "low": [9.0, 9.5], "close": [9.8, 10.5], "volume": [100, 200]})
    assert atr(corto, n=14).isna().all()
    assert chandelier_long(corto, n=22).isna().all()
    assert safezone_long(corto, n=10).isna().all()
    assert donchian_high(corto, n=20).isna().all()
    assert bollinger_width(corto, n=20).isna().all()
    assert volumen_relativo(corto, n=20).isna().all()

    with pytest.raises(ValueError):
        atr(corto, n=0)
    with pytest.raises(ValueError):
        chandelier_long(corto, n=-1)
    with pytest.raises(ValueError):
        safezone_long(corto, n=0)
    with pytest.raises(ValueError):
        force_index(corto, n=0)
    with pytest.raises(ValueError):
        donchian_high(corto, n=0)
    with pytest.raises(ValueError):
        bollinger_width(corto, n=0)
    with pytest.raises(ValueError):
        volumen_relativo(corto, n=0)
