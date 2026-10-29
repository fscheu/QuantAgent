"""Estrategia F1 de rebalanceo de fin de mes."""

from typing import Dict, List, Optional

import pandas as pd

from ..fabrica.cartera import pesos_f1
from ..models import ActivePosition
from .base import TradingSignal, TradingStrategy


class RebalanceoFinDeMesStrategy(TradingStrategy):
    """Provee los pesos objetivo F1 para el panel diario SPY/IEF."""

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {"name": cls.__name__, "display_name": "Rebalanceo de fin de mes (F1)",
                "type": "deterministic", "description": "Pesos F1 SPY/IEF en los últimos cinco cierres del mes."}

    @property
    def required_history_bars(self) -> int:
        return 1

    @staticmethod
    def posiciones_objetivo(cierres: pd.DataFrame) -> pd.DataFrame:
        return pesos_f1(cierres)

    def generate_signal(self, kline_data: List[Dict], symbol: str, timeframe: str,
                        current_price: float) -> Optional[TradingSignal]:
        return None

    def should_reevaluate(self, position: ActivePosition, current_price: float) -> bool:
        return False
