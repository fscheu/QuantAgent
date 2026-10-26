"""Indicadores técnicos compartidos sobre series y DataFrames OHLCV."""

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Average True Range de Wilder (J. Welles Wilder Jr., 1978).

    Cálculo:
      TR_0 = High_0 - Low_0
      TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
      ATR_{n-1} = promedio simple(TR_0 ... TR_{n-1})
      ATR_t = (ATR_{t-1} * (n - 1) + TR_t) / n  para t >= n
    Devuelve NaN para t < n - 1.
    """
    if n <= 0:
        raise ValueError("n debe ser mayor a 0")
    m = len(df)
    if m == 0:
        return pd.Series(dtype=float, index=df.index)

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    tr = np.empty(m, dtype=float)
    tr[0] = high[0] - low[0]
    if m > 1:
        hl = high[1:] - low[1:]
        hc = np.abs(high[1:] - close[:-1])
        lc = np.abs(low[1:] - close[:-1])
        tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    res = np.full(m, np.nan, dtype=float)
    if m >= n:
        res[n - 1] = np.mean(tr[:n])
        for i in range(n, m):
            res[i] = (res[i - 1] * (n - 1) + tr[i]) / n

    return pd.Series(res, index=df.index, dtype=float)


def chandelier_long(df: pd.DataFrame, n: int = 22, mult: float = 3.0) -> pd.Series:
    """
    Chandelier Exit para posiciones long (Chuck LeBeau).

    Cálculo:
      max(high de n velas) - mult * ATR(n)
    Devuelve NaN para t < n - 1.
    """
    if n <= 0:
        raise ValueError("n debe ser mayor a 0")
    atr_val = atr(df, n=n)
    highest_high = df["high"].rolling(window=n, min_periods=n).max()
    return highest_high - mult * atr_val


def safezone_long(df: pd.DataFrame, n: int = 10, coef: float = 2.0) -> pd.Series:
    """
    SafeZone Stop para posiciones long (Alexander Elder).

    Cálculo:
      Al cierre de la vela t:
      safezone = low[t] - coef * promedio_penetraciones
      donde una penetración a la baja en la vela i ocurre cuando low[i] < low[i-1]
      (magnitud: low[i-1] - low[i]). El promedio se calcula sobre las penetraciones
      estrictamente positivas de las últimas n velas terminando en t (0.0 si ninguna).
    Devuelve NaN para t < n - 1.
    """
    if n <= 0:
        raise ValueError("n debe ser mayor a 0")
    m = len(df)
    if m == 0:
        return pd.Series(dtype=float, index=df.index)

    low = df["low"].to_numpy(dtype=float)
    pen = np.zeros(m, dtype=float)
    if m > 1:
        diff = low[:-1] - low[1:]
        pen[1:] = np.maximum(0.0, diff)

    res = np.full(m, np.nan, dtype=float)
    if m >= n:
        pen_series = pd.Series(pen)
        sum_pen = pen_series.rolling(window=n, min_periods=n).sum().to_numpy()
        count_pen = (pen_series > 0.0).astype(float).rolling(window=n, min_periods=n).sum().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_pen = np.where(count_pen > 0, sum_pen / count_pen, 0.0)
        res[n - 1:] = low[n - 1:] - coef * avg_pen[n - 1:]

    return pd.Series(res, index=df.index, dtype=float)
