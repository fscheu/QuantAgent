"""Tests de RsiRegimenStrategy (QuantAgent-d4cq.38, M02): series armadas a mano, sin red."""

import numpy as np
import pandas as pd

from quantagent.backtesting.signals import bar_signals, check_no_lookahead
from quantagent.strategy.registry import build_strategy, cli_aliases
from quantagent.strategy.rsi_regimen_strategy import RsiRegimenStrategy

# 30 sesiones de caída casi constante (-1% +/- 0,2%): RSI en sobreventa y volatilidad propia de ~0,2%.
_CAIDA = [-0.008, -0.012] * 15


def _velas(historia_sigma: float, retornos_finales=_CAIDA, n_historia: int = 320) -> list[dict]:
    rng = np.random.default_rng(3)
    rets = np.concatenate([rng.normal(0.0, historia_sigma, n_historia), retornos_finales])
    close = 100 * np.cumprod(1 + rets)
    idx = pd.bdate_range("2020-01-02", periods=len(close))
    return [
        {"timestamp": t, "open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1e6}
        for t, c in zip(idx, close)
    ]


def _senal(velas):
    est = RsiRegimenStrategy()
    ventana = velas[-est.required_history_bars:]
    return est.generate_signal(ventana, "SPY", "1d", ventana[-1]["close"])


def test_sobreventa_con_volatilidad_baja_no_entra():
    velas = _velas(historia_sigma=0.015)  # la historia es mucho mas volatil que la caida final
    rsi = RsiRegimenStrategy()._calculate_rsi(pd.Series([v["close"] for v in velas]), 14).iloc[-1]
    assert rsi < 30
    assert _senal(velas) is None


def test_sobreventa_con_volatilidad_alta_entra_largo():
    velas = _velas(historia_sigma=0.0005)  # la caida final es mucho mas volatil que la historia
    s = _senal(velas)
    assert s is not None and s.decision == "LONG"


def test_volatilidad_alta_sin_sobreventa_no_entra():
    subida = [0.012, 0.008] * 15  # misma volatilidad que la caida, RSI alto
    assert _senal(_velas(0.0005, subida)) is None


def test_no_vende_en_sobrecompra():
    # Con volatilidad alta y RSI > 70 la estrategia base iria corta; esta es solo larga.
    assert _senal(_velas(0.0005, [0.012, 0.008] * 15)) is None


def test_historia_insuficiente_no_opera():
    velas = _velas(0.0005)
    assert RsiRegimenStrategy().generate_signal(velas[-100:], "SPY", "1d", velas[-1]["close"]) is None


def test_registrada_con_alias_y_parametros():
    assert cli_aliases()["rsi-regimen"] == "RsiRegimenStrategy"
    assert isinstance(build_strategy("RsiRegimenStrategy", vol_window=10), RsiRegimenStrategy)


def test_la_senal_en_t_no_depende_de_velas_posteriores():
    # Calma, caida con volatilidad y rebote, dos veces: hay entradas reales que comparar.
    rng = np.random.default_rng(5)
    ciclo = np.concatenate([_CAIDA, rng.normal(0.004, 0.002, 30)])
    rets = np.concatenate([rng.normal(0, 0.0005, 300), ciclo, rng.normal(0, 0.0005, 100), ciclo])
    close = 100 * np.cumprod(1 + rets)
    idx = pd.bdate_range("2020-01-02", periods=len(close))
    velas = pd.DataFrame(
        {"timestamp": idx, "open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=idx
    )
    est = build_strategy("RsiRegimenStrategy")
    completa = bar_signals(est, velas, "SPY", "1d")
    assert (completa["entry"] == 1).any()
    cortes = [idx[len(idx) // 2], idx[(3 * len(idx)) // 4], idx[-1]]
    assert check_no_lookahead(lambda c: bar_signals(est, c, "SPY", "1d"), velas, cortes) == []
