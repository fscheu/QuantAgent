"""RSI de sobreventa que solo entra con volatilidad propia alta (QuantAgent-d4cq.38, M02 de L14)."""

from typing import Dict, List, Optional

import pandas as pd

from .base import TradingSignal
from .rsi_strategy import RSIMeanReversionStrategy

MEDIAN_WINDOW = 252  # sesiones de la mediana de volatilidad; fijo, no es parámetro libre


class RsiRegimenStrategy(RSIMeanReversionStrategy):
    """
    Reversión en sobreventa, solo largos, condicionada al régimen de volatilidad (Nagel 2012).

    Entra como `RSIMeanReversionStrategy` (RSI < oversold_threshold, mismos stop, take profit y
    trailing) solo si la volatilidad realizada de las últimas `vol_window` sesiones del propio
    activo supera la mediana de esa misma volatilidad en las últimas 252 sesiones. Solo largos:
    la prima que se mide es la de proveer liquidez comprando sobreventa.
    """

    PARAMS = {
        "rsi_period": {"type": int, "default": 14, "description": "RSI calculation period."},
        "oversold_threshold": {"type": float, "default": 30.0, "description": "RSI level that triggers a long setup."},
        "vol_window": {"type": int, "default": 20, "description": "Sessions of the realized volatility."},
    }

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {
            "name": cls.__name__,
            "display_name": "RSI with Volatility Regime",
            "type": "deterministic",
            "description": "Long-only RSI oversold entry gated by own realized volatility above its 252-session median.",
        }

    def __init__(self, rsi_period: int = 14, oversold_threshold: float = 30.0, vol_window: int = 20):
        if vol_window < 2:
            raise ValueError("vol_window tiene que ser al menos 2")
        super().__init__(rsi_period=rsi_period, oversold_threshold=oversold_threshold)
        self.vol_window = vol_window

    @property
    def required_history_bars(self) -> int:
        # MEDIAN_WINDOW volatilidades, cada una con vol_window retornos (el primero necesita una vela previa).
        return max(self.rsi_period + 1, self.vol_window + MEDIAN_WINDOW)

    def _high_volatility(self, close: pd.Series) -> bool:
        vol = close.pct_change().rolling(self.vol_window).std().dropna().iloc[-MEDIAN_WINDOW:]
        return len(vol) == MEDIAN_WINDOW and bool(vol.iloc[-1] > vol.median())

    def generate_signal(
        self,
        kline_data: List[Dict],
        symbol: str,
        timeframe: str,
        current_price: float,
    ) -> Optional[TradingSignal]:
        if len(kline_data) < self.required_history_bars:
            return None
        if not self._high_volatility(pd.DataFrame(kline_data)["close"].astype(float)):
            return None
        signal = super().generate_signal(kline_data, symbol, timeframe, current_price)
        return signal if signal is not None and signal.decision == "LONG" else None
