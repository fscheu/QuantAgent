"""Efecto de cambio de mes (McConnell y Xu 2008, Lakonishok y Smidt 1988)."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from ..models import ActivePosition, ExitPolicy
from .base import TradingSignal, TradingStrategy
from .sma_cross_strategy import UNREACHABLE_STOP_LOSS, UNREACHABLE_TAKE_PROFIT


class TurnOfTheMonthStrategy(TradingStrategy):
    """Efecto de cambio de mes: largos del último día hábil del mes al 3.er día hábil siguiente."""

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {"name": cls.__name__, "display_name": "Turn of the Month", "type": "deterministic", "description": "Long from the last business day of the month to the 3rd business day of the next month."}

    @property
    def required_history_bars(self) -> int:
        return 5

    @staticmethod
    def _is_entry_day(day: pd.Timestamp) -> bool:
        day = pd.Timestamp(day).tz_localize(None).normalize()
        return day.weekday() < 5 and (day.day == day.days_in_month or (day.weekday() == 4 and day.days_in_month - day.day <= 2))

    @staticmethod
    def _is_exit_day(ohlc_data: pd.DataFrame) -> bool:
        if len(ohlc_data) == 0:
            return False
        ts_col = ohlc_data["timestamp"] if "timestamp" in ohlc_data.columns else ohlc_data.index
        day = pd.Timestamp(ts_col.iloc[-1]).tz_localize(None).normalize()
        if day.weekday() >= 5 or day.day > 10:
            return False
        ts = pd.to_datetime(ts_col).dt.tz_localize(None).dt.normalize()
        return int(((ts.dt.year == day.year) & (ts.dt.month == day.month)).sum()) == 3

    def generate_signal(self, kline_data: List[Dict], symbol: str, timeframe: str, current_price: float) -> Optional[TradingSignal]:
        if not kline_data or not self._is_entry_day(kline_data[-1]["timestamp"]):
            return None
        return TradingSignal(
            decision="LONG", confidence=1.0, entry_price=current_price,
            stop_loss=UNREACHABLE_STOP_LOSS, take_profit=UNREACHABLE_TAKE_PROFIT,
            reasoning="Último día hábil del mes: entra turn-of-the-month",
            exit_policy=ExitPolicy.SL_TP_ONLY,
        )

    def should_exit(self, position: ActivePosition, current_price: float, ohlc_data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        return (True, "TURN_OF_THE_MONTH_EXIT") if self._is_exit_day(ohlc_data) else (False, None)

    def should_reevaluate(self, position: ActivePosition, current_price: float) -> bool:
        return False

    def get_default_exit_policy(self) -> ExitPolicy:
        return ExitPolicy.SL_TP_ONLY
