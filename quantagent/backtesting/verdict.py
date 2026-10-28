"""Regla del Juego 2 para una celda estrategia × activo."""

import csv
from math import e, isfinite, sqrt

import numpy as np
from scipy.stats import kurtosis, norm, skew, ttest_1samp

# Umbrales fijados por Fede en el PR #140 (V30).
MIN_TRADES = 2
MIN_SHARPE_TASA_0 = 0.10
MAX_DRAWDOWN = 0.35


def montecarlo_bloques(equity, n=1000, bloque=20, semilla=0) -> tuple[float, float] | None:
    """Bootstrap de bloques completos solapados; DD como fracción desde el máximo."""
    curva = np.asarray(equity, dtype=float)
    retornos = curva[1:] / curva[:-1] - 1
    largo = len(retornos)
    if largo < 2 * bloque:
        return None
    rng = np.random.default_rng(semilla)
    inicios = rng.integers(0, largo - bloque + 1, size=(n, (largo + bloque - 1) // bloque))
    indices = (inicios[..., None] + np.arange(bloque)).reshape(n, -1)[:, :largo]
    simuladas = np.column_stack((np.ones(n), np.cumprod(1 + retornos[indices], axis=1)))
    finales, real = simuladas[:, -1], curva[-1] / curva[0]
    # Los empates incluyen las diferencias de redondeo del producto de retornos.
    percentil = np.mean((finales <= real) | np.isclose(finales, real, rtol=1e-12, atol=1e-12))
    drawdowns = np.max(1 - simuladas / np.maximum.accumulate(simuladas, axis=1), axis=1)
    return float(percentil), float(np.percentile(drawdowns, 95))


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


def filas_validas(filas):
    """Excluye intentos invalidados por bug y corridas sintéticas."""
    return [f for f in filas if f.get("estado") != "invalida_bug" and f.get("snapshot") != "sintetico"]


def n_efectivo(filas) -> int:
    """Cuenta intentos distintos por (estrategia, params, activo, desde, hasta)."""
    distintos = {
        (
            f.get("estrategia"),
            f.get("params"),
            f.get("activo"),
            f.get("desde"),
            f.get("hasta"),
        )
        for f in filas_validas(filas)
    }
    return len(distintos)


def dsr_contra_intentos(equity, intentos_path, n_intentos: int | None = None) -> float | None:
    """Sharpe deflactado de una curva contra el registro `intentos.csv` (N = filas válidas o n_intentos)."""
    curva = np.asarray(equity, dtype=float)
    retornos = curva[1:] / curva[:-1] - 1
    if len(retornos) < 3 or np.std(retornos, ddof=1) == 0:
        return None
    with open(intentos_path, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    filas_con_sharpe = []
    sharpes, sharpes_tasa_0 = [], []
    for fila in filas_validas(filas):
        raw = fila.get("sharpe")
        if raw is not None and raw != "":
            try:
                valor = float(raw)
            except (TypeError, ValueError):
                continue
            if not isfinite(valor) or abs(valor) >= 1e6:
                continue
            s = valor / sqrt(252)
            sharpes.append(s)
            if fila.get("sharpe_base") == "tasa_0":
                sharpes_tasa_0.append(s)
        filas_con_sharpe.append(fila)
    sharpes_var = sharpes_tasa_0 if len(sharpes_tasa_0) >= 2 else sharpes
    var_sr = float(np.var(sharpes_var, ddof=1)) if len(sharpes_var) >= 2 else 0.0
    sr = float(np.mean(retornos) / np.std(retornos, ddof=1))
    n = max(n_intentos if n_intentos is not None else len(filas_con_sharpe), 1)
    return sharpe_deflactado(
        sr, n, var_sr, len(retornos),
        float(skew(retornos)), float(kurtosis(retornos, fisher=False)),
    )

