"""Regla del Juego 2 para una celda estrategia × activo."""

from math import e, sqrt

import numpy as np
from scipy.stats import norm, ttest_1samp

# Umbrales fijados por Fede en el PR #140 (V30).
MIN_TRADES = 2
MIN_SHARPE_TASA_0 = 0.10
MAX_DRAWDOWN = 0.35


def p_valor_retornos(equity) -> float | None:
    """T-test a una cola de retornos simples diarios, con tasa de referencia 0."""
    curva = np.asarray(equity, dtype=float)
    retornos = curva[1:] / curva[:-1] - 1
    if len(retornos) < 3 or np.std(retornos, ddof=1) == 0:
        return None
    return float(ttest_1samp(retornos, 0, alternative="greater").pvalue)


def evaluar(trades, pnl, sharpe_tasa_0, max_drawdown) -> tuple[bool, list[str]]:
    motivos = []
    criterios = (
        ("trades", trades, lambda v: v >= MIN_TRADES, "trades < 2"),
        ("pnl", pnl, lambda v: v > 0, "pnl <= 0"),
        ("sharpe_tasa_0", sharpe_tasa_0, lambda v: v >= MIN_SHARPE_TASA_0, "sharpe_tasa_0 < 0.10"),
        ("max_drawdown", max_drawdown, lambda v: v <= MAX_DRAWDOWN, "max_drawdown > 0.35"),
    )
    for campo, valor, cumple, motivo in criterios:
        if valor is None or valor == "":
            motivos.append(f"sin {campo}")
        elif not cumple(valor):
            motivos.append(motivo)
    return not motivos, motivos


EULER_MASCHERONI = 0.5772156649015329


def sharpe_deflactado(sr, n_intentos, var_sr_intentos, n_obs, asimetria, curtosis) -> float:
    """Probabilidad de que `sr` supere al máximo Sharpe esperado por azar entre N intentos.

    Bailey y López de Prado (2014), "The Deflated Sharpe Ratio". `sr` es el Sharpe por
    vela, sin anualizar; `var_sr_intentos` es la varianza de esos Sharpe por vela entre
    los intentos y `n_obs` es T, la cantidad de velas. `curtosis` es la curtosis de los
    retornos sin restar 3 (normal = 3). Con un solo intento SR0 = 0.
    """
    if n_intentos == 1:
        sr0 = 0.0
    else:
        sr0 = sqrt(var_sr_intentos) * (
            (1 - EULER_MASCHERONI) * norm.ppf(1 - 1 / n_intentos)
            + EULER_MASCHERONI * norm.ppf(1 - 1 / (n_intentos * e))
        )
    denominador = sqrt(1 - asimetria * sr + (curtosis - 1) / 4 * sr**2)
    return float(norm.cdf((sr - sr0) * sqrt(n_obs - 1) / denominador))
