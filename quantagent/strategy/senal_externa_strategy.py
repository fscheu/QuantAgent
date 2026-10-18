"""Estrategia interna que replica una serie 0/1 externa (QuantAgent-d4cq.63, P05.4)."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from ..models import ActivePosition, ExitPolicy
from .base import TradingSignal, TradingStrategy
from .sma_cross_strategy import UNREACHABLE_STOP_LOSS, UNREACHABLE_TAKE_PROFIT


class SenalExterna(TradingStrategy):
    """Estrategia interna que replica una serie 0/1 externa; no valida el ranking (eso lo cubren tests de transversal)."""

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {"name": cls.__name__, "display_name": "Señal Externa", "type": "deterministic", "description": "Serie 0/1 externa"}

    def __init__(self, ruta: str = ""):
        self.ruta = ruta
        self._entry: set = set()
        self._exit: set = set()
        if ruta and Path(ruta).exists():
            s = pd.read_csv(ruta).set_index("timestamp")["posicion"].astype(int)
            s.index = pd.to_datetime(s.index)
            prev = s.shift(1, fill_value=0)
            self._entry = set(s.index[(s == 1) & (prev == 0)])
            self._exit = set(s.index[(s == 0) & (prev == 1)])

    @property
    def required_history_bars(self) -> int:
        return 1

    def generate_signal(self, kline_data: List[Dict], symbol: str, timeframe: str, current_price: float) -> Optional[TradingSignal]:
        if kline_data and pd.to_datetime(kline_data[-1]["timestamp"]) in self._entry:
            return TradingSignal(
                decision="LONG", confidence=1.0, entry_price=current_price,
                stop_loss=UNREACHABLE_STOP_LOSS, take_profit=UNREACHABLE_TAKE_PROFIT,
                reasoning="Paso de 0 a 1 en serie externa", exit_policy=ExitPolicy.SL_TP_ONLY,
            )
        return None

    def should_exit(self, position: ActivePosition, current_price: float, ohlc_data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        if len(ohlc_data):
            raw = ohlc_data["timestamp"].iloc[-1] if "timestamp" in ohlc_data.columns else ohlc_data.index[-1]
            if pd.to_datetime(raw) in self._exit:
                return True, "SENAL_EXTERNA_EXIT"
        return False, None

    def should_reevaluate(self, position: ActivePosition, current_price: float) -> bool:
        return False
