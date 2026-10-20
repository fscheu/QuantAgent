"""Regla de pendiente de curva sobre bonos (QuantAgent-d4cq.72, M04 de L14)."""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..data.macro import valor_conocido
from ..models import ActivePosition, ExitPolicy
from .base import TradingSignal, TradingStrategy
from .sma_cross_strategy import UNREACHABLE_STOP_LOSS, UNREACHABLE_TAKE_PROFIT


class CurvaPendienteStrategy(TradingStrategy):
    """
    Largo en bonos mientras la pendiente de la curva, conocida en la fecha, es positiva.

    Parámetros (de valor único; los lee `admitir` para la prueba de vintage):
        snapshot: nombre del snapshot macro, resuelto desde `$QUANTAGENT_SNAPSHOT_DIR`.
        serie (default T10Y2Y): diferencia de rendimientos 10 años menos 2 años.

    Entrada: pendiente conocida > 0 y sin posición. Salida: pendiente conocida <= 0. El umbral 0 es
    fijo (signo, no ajuste). Sin pendiente conocida todavía, no hace nada. Lee la serie solo con
    `valor_conocido` (lo publicado en la fecha, no la última revisión). Sin take profit ni stop loss
    de protección. Universo por definición: AGG, IEF, LQD, TIP, TLT; SHY queda afuera por duración corta.
    """

    requiere_macro = True

    PARAMS = {
        "snapshot": {"type": str, "default": None, "description": "Macro snapshot name."},
        "serie": {"type": str, "default": "T10Y2Y", "description": "Curve slope series (10y minus 2y)."},
    }

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {
            "name": cls.__name__,
            "display_name": "Yield Curve Slope",
            "type": "deterministic",
            "description": "Long-only bond ETFs while the publication-date yield curve slope is positive.",
        }

    def __init__(self, snapshot: str, serie: str = "T10Y2Y"):
        self.snapshot = snapshot
        self.serie = serie

    @property
    def required_history_bars(self) -> int:
        return 1

    def _pendiente(self, fecha) -> Optional[float]:
        conocido = valor_conocido(self.snapshot, self.serie, pd.Timestamp(fecha))
        return None if conocido is None else conocido[1]

    def generate_signal(
        self,
        kline_data: List[Dict],
        symbol: str,
        timeframe: str,
        current_price: float,
    ) -> Optional[TradingSignal]:
        pendiente = self._pendiente(kline_data[-1]["timestamp"])
        if pendiente is None or pendiente <= 0:
            return None
        return TradingSignal(
            decision="LONG",
            confidence=1.0,
            entry_price=current_price,
            stop_loss=UNREACHABLE_STOP_LOSS,
            take_profit=UNREACHABLE_TAKE_PROFIT,
            reasoning=f"Pendiente {self.serie} = {pendiente:+.2f} puntos",
            exit_policy=ExitPolicy.SL_TP_ONLY,
        )

    def should_exit(
        self,
        position: ActivePosition,
        current_price: float,
        ohlc_data: pd.DataFrame,
    ) -> Tuple[bool, Optional[str]]:
        pendiente = self._pendiente(ohlc_data["timestamp"].iloc[-1])
        if pendiente is not None and pendiente <= 0:
            return (True, "CURVE_SLOPE_NON_POSITIVE")
        return (False, None)

    def should_reevaluate(self, position: ActivePosition, current_price: float) -> bool:
        return False

    def get_default_exit_policy(self) -> ExitPolicy:
        return ExitPolicy.SL_TP_ONLY
