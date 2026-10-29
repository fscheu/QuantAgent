"""Tests para Arquetipo B (ruptura después de compresión) — QuantAgent-fnpr.3.6 (M5-23)."""

import numpy as np
import pandas as pd
import pytest

from quantagent.models import ActivePosition, ExitPolicy, OrderSide
from quantagent.strategy.arquetipo_b_strategy import ArquetipoBStrategy
from quantagent.strategy.base import TradingStrategy
from quantagent.strategy.indicadores import atr, bollinger_width, donchian_high, volumen_relativo


def _make_bars(n: int = 210, base: float = 100.0) -> pd.DataFrame:
    """Genera serie sintética con tendencia alcista sostenida."""
    closes = base + np.linspace(0, 50, n)
    return pd.DataFrame(
        [
            {
                "open": c - 0.2,
                "high": c + 0.5,
                "low": c - 0.5,
                "close": c,
                "volume": 1000.0,
            }
            for c in closes
        ]
    )


def test_arquetipo_b_metadata_and_defaults():
    strategy = ArquetipoBStrategy()
    assert issubclass(ArquetipoBStrategy, TradingStrategy)
    assert strategy.describe()["name"] == "ArquetipoBStrategy"
    assert strategy.describe()["display_name"] == "Arquetipo B (Ruptura)"
    assert strategy.get_default_exit_policy() == ExitPolicy.TRAILING_STOP
    assert strategy.required_history_bars >= 201
    assert strategy._min_candles == strategy.required_history_bars


def test_arquetipo_b_trend_and_compression_conditions():
    strategy = ArquetipoBStrategy()
    df = _make_bars(210)

    # Si la historia es insuficiente
    assert strategy.generate_signal(df.iloc[:100].to_dict(orient="records"), "TEST", "1d", 100.0) is None

    # En df regular, la compresión de Bollinger y el gatillo de volumen/Donchian no se dan simultáneamente
    sig = strategy.generate_signal(df.to_dict(orient="records"), "TEST", "1d", float(df["close"].iloc[-1]))
    assert sig is None


def _make_breakout_df() -> pd.DataFrame:
    """Construye serie con tendencia alcista, compresión de volatilidad y ruptura Donchian con volumen."""
    n = 220
    trend = np.linspace(50, 100, n)
    v1 = [trend[i] + 3.0 * np.sin(i * 0.5) for i in range(160)]
    v2 = [trend[160 + i] + (3.0 - 2.8 * (i / 40)) * np.sin(i * 0.5) for i in range(40)]
    v3 = [trend[200 + i] + 0.1 * np.sin(i * 0.5) for i in range(19)]
    v4 = [trend[-1] + 1.0]
    closes = v1 + v2 + v3 + v4
    rows = [
        {
            "open": c - 0.2 if i == len(closes) - 1 else c,
            "high": c + 0.5,
            "low": c - 0.5,
            "close": c,
            "volume": 2500.0 if i == len(closes) - 1 else 1000.0,
        }
        for i, c in enumerate(closes)
    ]
    return pd.DataFrame(rows)


def test_arquetipo_b_signal_generation_and_atr_stop():
    strategy = ArquetipoBStrategy(
        bollinger_period=20,
        bollinger_lookback=60,
        bollinger_quantile=0.20,
        ema_fast=50,
        ema_slow=200,
        donchian_period=20,
        volume_period=20,
        volume_factor=1.5,
        atr_period=14,
        stop_atr_mult=2.5,
    )
    df = _make_breakout_df()
    kline = df.to_dict(orient="records")
    current_price = float(df["close"].iloc[-1])
    sig = strategy.generate_signal(kline, "TEST", "1d", current_price)

    assert sig is not None
    assert sig.decision == "LONG"
    assert sig.entry_order == "market"
    assert sig.exit_policy == ExitPolicy.TRAILING_STOP
    assert sig.sin_take_profit is True
    assert sig.take_profit is None

    # Verificación del stop inicial en múltiplo de ATR (2.5 ATR)
    atr_val = float(atr(df, n=14).iloc[-1])
    expected_stop = current_price - 2.5 * atr_val
    assert sig.stop_loss == pytest.approx(expected_stop, rel=1e-5)


def test_arquetipo_b_trigger_requires_donchian_breakout_and_volume():
    strategy = ArquetipoBStrategy()
    df = _make_breakout_df()

    # Si el volumen no supera 1.5x promedio
    df_low_vol = df.copy()
    df_low_vol.loc[df_low_vol.index[-1], "volume"] = 1000.0
    assert strategy.generate_signal(df_low_vol.to_dict(orient="records"), "TEST", "1d", float(df["close"].iloc[-1])) is None

    # Si el cierre no supera el Donchian high previo
    df_no_breakout = df.copy()
    df_no_breakout.loc[df_no_breakout.index[-1], "close"] = float(df["close"].iloc[-2])
    assert strategy.generate_signal(df_no_breakout.to_dict(orient="records"), "TEST", "1d", float(df["close"].iloc[-2])) is None


def test_chandelier_exit_in_should_exit():
    strategy = ArquetipoBStrategy(chandelier_mult=3.0, atr_period=14)
    df = _make_bars(30, base=100.0)
    atr_val = float(atr(df, n=14).iloc[-1])
    pos = ActivePosition(
        id=1,
        symbol="TEST",
        side=OrderSide.BUY,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=None,
        quantity=10.0,
        candles_since_entry=5,
        exit_policy=ExitPolicy.TRAILING_STOP,
    )
    highest = float(df.iloc[-(pos.candles_since_entry + 1):]["high"].max())
    chand_stop = highest - 3.0 * atr_val

    # Si el precio actual está por encima del stop Chandelier, no sale y actualiza stop_loss
    should_exit, reason = strategy.should_exit(pos, chand_stop + 1.0, df)
    assert should_exit is False
    assert float(pos.stop_loss) == pytest.approx(max(90.0, chand_stop))

    # Si el precio actual cae por debajo del stop Chandelier, sale por TRAILING_STOP
    should_exit, reason = strategy.should_exit(pos, chand_stop - 0.1, df)
    assert should_exit is True
    assert reason == "TRAILING_STOP"
