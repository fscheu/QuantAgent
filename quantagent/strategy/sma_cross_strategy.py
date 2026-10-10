"""Cruce de medias móviles simples 50/200 sobre velas diarias (QuantAgent-cok)."""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..models import ActivePosition, ExitPolicy
from .base import TradingSignal, TradingStrategy

# El modelo de posición exige stop loss y take profit, y el motor reemplaza un valor
# vacío por 2% / 3%. Estos dos niveles no se pueden tocar: equivalen a "apagado".
UNREACHABLE_STOP_LOSS = 0.01
UNREACHABLE_TAKE_PROFIT = 1_000_000_000.0


class SmaCrossStrategy(TradingStrategy):
    """
    Seguimiento de tendencia, solo largos, sobre el cierre diario.

    Parámetros:
        fast (default 50): sesiones de la media móvil simple rápida.
        slow (default 200): sesiones de la media móvil simple lenta.

    Entrada: al cierre de la sesión en que la media rápida pasa a estar por encima
    de la lenta (la sesión anterior estaba por debajo o igual). No entra a mitad de
    una tendencia en curso: espera el próximo cruce.
    Salida: al cierre de la primera sesión en que la rápida está por debajo de la lenta.
    Sin take profit y sin stop loss de protección (niveles inalcanzables).
    """

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {
            "name": cls.__name__,
            "display_name": "SMA Cross 50/200",
            "type": "deterministic",
            "description": "Long-only daily trend following: fast SMA crossing the slow SMA.",
        }

    def __init__(self, fast: int = 50, slow: int = 200):
        if not 0 < fast < slow:
            raise ValueError("fast tiene que ser mayor que 0 y menor que slow")
        self.fast = fast
        self.slow = slow

    @property
    def required_history_bars(self) -> int:
        # slow velas para la media de hoy + 1 para la de la sesión anterior.
        return self.slow + 1

    def _smas(self, close: pd.Series, offset: int = 0) -> Tuple[float, float]:
        """Medias rápida y lenta al cierre de la vela `offset` sesiones atrás."""
        end = len(close) - offset
        window = close.iloc[:end].astype(float)
        return float(window.iloc[-self.fast:].mean()), float(window.iloc[-self.slow:].mean())

    def generate_signal(
        self,
        kline_data: List[Dict],
        symbol: str,
        timeframe: str,
        current_price: float,
    ) -> Optional[TradingSignal]:
        if len(kline_data) < self.required_history_bars:
            return None

        close = pd.DataFrame(kline_data)["close"]
        fast_now, slow_now = self._smas(close)
        fast_prev, slow_prev = self._smas(close, offset=1)
        if not (fast_prev <= slow_prev and fast_now > slow_now):
            return None

        return TradingSignal(
            decision="LONG",
            confidence=1.0,
            entry_price=current_price,
            stop_loss=UNREACHABLE_STOP_LOSS,
            take_profit=UNREACHABLE_TAKE_PROFIT,
            reasoning=(
                f"SMA{self.fast}={fast_now:.4f} cruzó arriba de SMA{self.slow}={slow_now:.4f}"
            ),
            exit_policy=ExitPolicy.SL_TP_ONLY,
        )

    def should_exit(
        self,
        position: ActivePosition,
        current_price: float,
        ohlc_data: pd.DataFrame,
    ) -> Tuple[bool, Optional[str]]:
        if len(ohlc_data) < self.slow:
            return (False, None)
        fast_now, slow_now = self._smas(ohlc_data["close"])
        if fast_now < slow_now:
            return (True, "SMA_CROSS_DOWN")
        return (False, None)

    def should_reevaluate(self, position: ActivePosition, current_price: float) -> bool:
        return False

    def get_default_exit_policy(self) -> ExitPolicy:
        return ExitPolicy.SL_TP_ONLY
