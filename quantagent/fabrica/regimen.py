"""Clasificación de régimen de mercado y análisis de rendimiento por régimen."""
import numpy as np
import pandas as pd


def clasificar_regimen(
    cierres_spy: pd.Series,
    ventana_vol: int = 20,
    mediana_vol_dias: int = 252,
    ventana_media: int = 200,
) -> pd.Series:
    """Clasifica régimen de SPY con volatilidad 20d y SMA 200d con datos hasta t-1 (shift(1))."""
    ret = cierres_spy.pct_change()
    vol = ret.rolling(ventana_vol).std()
    med_vol = vol.rolling(mediana_vol_dias).median()
    sma = cierres_spy.rolling(ventana_media).mean()

    valido = vol.notna() & med_vol.notna() & sma.notna()
    alta_vol = vol > med_vol
    alcista = cierres_spy >= sma

    reg = pd.Series(index=cierres_spy.index, dtype=object)
    reg[valido & alcista & ~alta_vol] = "alcista_baja_vol"
    reg[valido & alcista & alta_vol] = "alcista_alta_vol"
    reg[valido & ~alcista & ~alta_vol] = "bajista_baja_vol"
    reg[valido & ~alcista & alta_vol] = "bajista_alta_vol"
    return reg.shift(1)


def sharpe_por_regimen(
    retornos: pd.Series,
    regimenes: pd.Series,
    risk_free_rate: float = 0.0,
    anualizar: bool = True,
) -> dict[str, float]:
    """Calcula el ratio de Sharpe diario por cada categoría de régimen."""
    df = pd.DataFrame({"ret": retornos, "reg": regimenes}).dropna()
    ppy = 252.0 if anualizar else 1.0
    factor = np.sqrt(ppy)
    resultado = {}
    for cat, grupo in df.groupby("reg"):
        ret = grupo["ret"]
        if len(ret) < 2 or ret.std() == 0:
            resultado[str(cat)] = 0.0
        else:
            sr = (ret.mean() - risk_free_rate / ppy) / ret.std() * factor
            resultado[str(cat)] = float(sr)
    return resultado


def permutar_regimenes_bloques(regimenes: pd.Series, seed: int | None = None) -> pd.Series:
    """Permuta bloques mensuales de etiquetas para contraste contra ruido conservando frecuencias."""
    if regimenes.empty:
        return regimenes.copy()
    idx = pd.to_datetime(regimenes.index)
    idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
    grupos = [g.to_numpy() for _, g in regimenes.groupby(idx_naive.to_period("M"), sort=False)]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(grupos))
    valores = np.concatenate([grupos[i] for i in perm])
    return pd.Series(valores, index=regimenes.index, name=regimenes.name)
