"""Estacionalidad de Halloween: largo de la última sesión de octubre a la de abril (QuantAgent-d4cq.P04)."""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..models import ActivePosition, ExitPolicy
from .base import TradingSignal, TradingStrategy
from .momentum_12m_strategy import is_last_session_of_month
from .sma_cross_strategy import UNREACHABLE_STOP_LOSS, UNREACHABLE_TAKE_PROFIT

ENTRY_MONTH = 10  # Bouman y Jacobsen (2002): noviembre a abril rinde más que mayo a octubre
EXIT_MONTH = 4


class HalloweenStrategy(TradingStrategy):
    """
    Estacionalidad, solo largos, sobre el cierre diario. Sin parámetros libres.

    Entrada: cierre de la última sesión de octubre (según el calendario de NYSE).
    Salida: cierre de la última sesión de abril. El resto del año no hace nada.
    Depende solo de la fecha de la vela. Sin take profit y sin stop loss de
    protección (niveles inalcanzables).
    """

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {
            "name": cls.__name__,
            "display_name": "Halloween Seasonality",
            "type": "deterministic",
            "description": "Long from the last October session to the last April session.",
        }

    @property
    def required_history_bars(self) -> int:
        return 1

    @staticmethod
    def _is_month_end(day, month: int) -> bool:
        day = pd.Timestamp(day)
        return day.month == month and is_last_session_of_month(day)

    def generate_signal(
        self, kline_data: List[Dict], symbol: str, timeframe: str, current_price: float
    ) -> Optional[TradingSignal]:
        if not kline_data or not self._is_month_end(kline_data[-1]["timestamp"], ENTRY_MONTH):
            return None
        return TradingSignal(
            decision="LONG",
            confidence=1.0,
            entry_price=current_price,
            stop_loss=UNREACHABLE_STOP_LOSS,
            take_profit=UNREACHABLE_TAKE_PROFIT,
            reasoning="Última sesión de octubre: entra el tramo noviembre-abril",
            exit_policy=ExitPolicy.SL_TP_ONLY,
        )

    def should_exit(
        self, position: ActivePosition, current_price: float, ohlc_data: pd.DataFrame
    ) -> Tuple[bool, Optional[str]]:
        if len(ohlc_data) and self._is_month_end(ohlc_data["timestamp"].iloc[-1], EXIT_MONTH):
            return (True, "SEASON_END")
        return (False, None)

    def should_reevaluate(self, position: ActivePosition, current_price: float) -> bool:
        return False

    def get_default_exit_policy(self) -> ExitPolicy:
        return ExitPolicy.SL_TP_ONLY
