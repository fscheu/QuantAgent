from typing import Dict, List, Optional

import pandas as pd

from ..data.macro import valor_conocido
from ..models import ExitPolicy
from .base import TradingSignal, TradingStrategy
from .momentum_12m_strategy import is_last_session_of_month
from .sma_cross_strategy import UNREACHABLE_STOP_LOSS, UNREACHABLE_TAKE_PROFIT

class TsmomLargoCortoStrategy(TradingStrategy):
    requiere_macro = True

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {"name": cls.__name__, "display_name": "TSMOM largo/corto", "type": "deterministic",
                "description": "Monthly 12-month momentum versus DGS3MO."}

    def __init__(self, snapshot: str):
        self.snapshot = snapshot

    @property
    def required_history_bars(self) -> int:
        return 253

    def generate_signal(self, kline_data: List[Dict], symbol: str, timeframe: str, current_price: float) -> Optional[TradingSignal]:
        if len(kline_data) < self.required_history_bars:
            return None
        candles = pd.DataFrame(kline_data)
        fecha = pd.Timestamp(candles["timestamp"].iloc[-1])
        if not is_last_session_of_month(fecha):
            return None
        inicio = pd.Timestamp(candles["timestamp"].iloc[-253])
        tasa = valor_conocido(self.snapshot, "DGS3MO", inicio)
        if tasa is None:
            return None
        retorno = float(candles["close"].iloc[-1] / candles["close"].iloc[-253] - 1)
        decision = "LONG" if retorno > tasa[1] / 100 else "SHORT"
        return TradingSignal(decision=decision, confidence=1.0, entry_price=current_price,
                            stop_loss=UNREACHABLE_STOP_LOSS, take_profit=UNREACHABLE_TAKE_PROFIT,
                            reasoning=f"Retorno 252 sesiones {retorno:+.2%}; DGS3MO {tasa[1]:.2f}%",
                            exit_policy=ExitPolicy.SL_TP_ONLY)

    def should_reevaluate(self, position, current_price: float) -> bool:
        return False

    def get_default_exit_policy(self) -> ExitPolicy:
        return ExitPolicy.SL_TP_ONLY
