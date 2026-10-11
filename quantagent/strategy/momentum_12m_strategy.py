"""Momentum absoluto de 12 meses con decisión mensual, velas diarias (QuantAgent-jgn)."""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..models import ActivePosition, ExitPolicy
from .base import TradingSignal, TradingStrategy
from .sma_cross_strategy import UNREACHABLE_STOP_LOSS, UNREACHABLE_TAKE_PROFIT


@lru_cache(maxsize=None)
def is_last_session_of_month(day: pd.Timestamp) -> bool:
    """True si `day` es la última sesión de su mes según el calendario de NYSE.

    Se resuelve solo con la fecha de la vela y el calendario del mercado (que se
    publica por adelantado): no hace falta ver la vela siguiente.
    """
    import pandas_market_calendars as mcal

    day = pd.Timestamp(day).tz_localize(None).normalize()
    sessions = mcal.get_calendar("NYSE").valid_days(day, day + pd.Timedelta(days=10))
    following = [s for s in sessions.tz_localize(None) if s > day]
    return bool(following) and following[0].month != day.month


class Momentum12mStrategy(TradingStrategy):
    """
    Momentum absoluto, solo largos, sobre el cierre diario.

    Parámetros:
        lookback (default 252): sesiones del retorno de referencia (12 meses).

    La decisión se toma solo en la última sesión de cada mes calendario:
    Entrada: cierre mayor que el cierre de `lookback` sesiones antes, sin posición.
    Salida: retorno de `lookback` sesiones menor o igual a cero.
    El resto de los días no hace nada. Sin take profit y sin stop loss de
    protección (niveles inalcanzables).
    """

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {
            "name": cls.__name__,
            "display_name": "Absolute Momentum 12m",
            "type": "deterministic",
            "description": "Long-only monthly decision on the sign of the 12-month return.",
        }

    def __init__(self, lookback: int = 252):
        if lookback <= 0:
            raise ValueError("lookback tiene que ser mayor que 0")
        self.lookback = lookback

    @property
    def required_history_bars(self) -> int:
        # La vela de decisión + la de `lookback` sesiones antes.
        return self.lookback + 1

    def _monthly_return(self, candles: pd.DataFrame) -> Optional[float]:
        """Retorno de `lookback` sesiones si la última vela es de decisión; si no, None."""
        if len(candles) < self.required_history_bars:
            return None
        if not is_last_session_of_month(pd.Timestamp(candles["timestamp"].iloc[-1])):
            return None
        close = candles["close"].astype(float)
        return float(close.iloc[-1] / close.iloc[-1 - self.lookback] - 1.0)

    def generate_signal(
        self,
        kline_data: List[Dict],
        symbol: str,
        timeframe: str,
        current_price: float,
    ) -> Optional[TradingSignal]:
        if len(kline_data) < self.required_history_bars:
            return None
        ret = self._monthly_return(pd.DataFrame(kline_data))
        if ret is None or ret <= 0:
            return None

        return TradingSignal(
            decision="LONG",
            confidence=1.0,
            entry_price=current_price,
            stop_loss=UNREACHABLE_STOP_LOSS,
            take_profit=UNREACHABLE_TAKE_PROFIT,
            reasoning=f"Retorno de {self.lookback} sesiones = {ret:+.2%} a fin de mes",
            exit_policy=ExitPolicy.SL_TP_ONLY,
        )

    def should_exit(
        self,
        position: ActivePosition,
        current_price: float,
        ohlc_data: pd.DataFrame,
    ) -> Tuple[bool, Optional[str]]:
        ret = self._monthly_return(ohlc_data)
        if ret is not None and ret <= 0:
            return (True, "MOMENTUM_NON_POSITIVE")
        return (False, None)

    def should_reevaluate(self, position: ActivePosition, current_price: float) -> bool:
        return False

    def get_default_exit_policy(self) -> ExitPolicy:
        return ExitPolicy.SL_TP_ONLY
