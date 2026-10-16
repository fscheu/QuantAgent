"""Puente estrategia -> columnas de señales por vela (QuantAgent-d4cq.3)."""

from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd

from ..models import ActivePosition, ExitPolicy, OrderSide
from ..strategy.base import TradingStrategy

_DECISION = {"LONG": 1, "SHORT": -1}


def bar_signals(strategy: TradingStrategy, candles: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    """Señales de `strategy` por vela, sin que vea nunca velas posteriores a la que evalúa.

    `candles`: columnas timestamp, open, high, low, close, volume (las que recibe la estrategia).
    Devuelve, con el índice de `candles`: entry (+1 LONG, -1 SHORT, 0), exit (bool, solo la regla propia
    de should_exit), stop_loss_pct, take_profit_pct, trailing_stop_pct (NaN sin señal).
    """
    out = pd.DataFrame(
        {"entry": 0, "exit": False, "stop_loss_pct": np.nan, "take_profit_pct": np.nan, "trailing_stop_pct": np.nan},
        index=candles.index,
    )
    n = strategy.required_history_bars
    if n <= 0:
        n = 30
    for i in range(len(candles)):
        window = candles.iloc[max(0, i + 1 - n): i + 1]
        price = float(window["close"].iloc[-1])
        signal = strategy.generate_signal(window.to_dict(orient="records"), symbol, timeframe, price)
        if signal is not None and signal.decision in _DECISION:
            ref = signal.entry_price or price
            out.iloc[i, out.columns.get_loc("entry")] = _DECISION[signal.decision]
            if signal.stop_loss is not None:
                out.iloc[i, out.columns.get_loc("stop_loss_pct")] = abs(ref - signal.stop_loss) / ref
            if signal.take_profit is not None:
                out.iloc[i, out.columns.get_loc("take_profit_pct")] = abs(signal.take_profit - ref) / ref
            if signal.trailing_stop_pct is not None:
                out.iloc[i, out.columns.get_loc("trailing_stop_pct")] = signal.trailing_stop_pct
        # Posición de prueba larga con stop y take profit inalcanzables: solo dispara la regla de la estrategia.
        probe = ActivePosition(
            symbol=symbol, side=OrderSide.BUY, entry_price=price, stop_loss=0.0, take_profit=1e12, quantity=1.0,
            decision_timestamp=datetime.min, candles_since_entry=0, exit_policy=ExitPolicy.SL_TP_ONLY,
            prediction_horizon=3, candles_direction=[],
        )
        out.iloc[i, out.columns.get_loc("exit")] = bool(strategy.should_exit(probe, price, window)[0])
    return out


def check_no_lookahead(signal_fn: Callable[[pd.DataFrame], pd.DataFrame], candles: pd.DataFrame, cuts) -> list[str]:
    """Prueba de truncamiento (QuantAgent-d4cq.4): lista vacía = `signal_fn` no mira el futuro.

    Para cada T de `cuts` compara `signal_fn(candles.loc[:T])` con `signal_fn(candles).loc[:T]`; `candles` debe
    tener índice de fechas. Devuelve un texto por cada (corte, columna) que difiere, con la primera fecha distinta.
    """
    full = signal_fn(candles)
    diffs = []
    for cut in cuts:
        truncated = signal_fn(candles.loc[:cut])
        expected = full.loc[:cut]
        for col in expected.columns:
            if col not in truncated.columns:
                diffs.append(f"{col}: falta en la señal truncada en {cut}")
                continue
            a, b = truncated[col], expected[col]
            same = (a == b) | (a.isna() & b.isna())
            if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
                same |= np.isclose(a, b, equal_nan=True)
            if not same.all():
                diffs.append(f"{col}: difiere con corte en {cut}, primera fecha {same.index[~same][0]}")
    return diffs
