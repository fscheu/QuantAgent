"""Puente estrategia -> columnas de señales por vela (QuantAgent-d4cq.3)."""

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
    for i in range(len(candles)):
        window = candles.iloc[: i + 1]
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
