"""Arquetipo B: Ruptura después de compresión — QuantAgent-M5."""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..models import ActivePosition, ExitPolicy
from .base import TradingSignal, TradingStrategy
from .indicadores import atr, bollinger_width, donchian_high, volumen_relativo


class ArquetipoBStrategy(TradingStrategy):
    """
    Ruptura después de compresión (Arquetipo B).

    Condición:
      - Ancho de Bollinger 20 en el 20% inferior de las últimas 60 sesiones.
      - Cierre > EMA 50 > EMA 200.
    Gatillo:
      - Cierre sobre el máximo de Donchian 20 de las velas previas con volumen > 1,5 x promedio 20.
    Entrada:
      - Apertura siguiente (default del motor).
    Stop inicial:
      - Entrada - 2,5 ATR.
    Salida:
      - Chandelier (máximo desde la entrada - 3 ATR).
    """

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {
            "name": cls.__name__,
            "display_name": "Arquetipo B (Ruptura)",
            "type": "deterministic",
            "description": "Ruptura de compresión con Bandas de Bollinger, Donchian y Chandelier.",
        }

    def __init__(
        self,
        bollinger_period: int = 20,
        bollinger_k: float = 2.0,
        bollinger_lookback: int = 60,
        bollinger_quantile: float = 0.20,
        ema_fast: int = 50,
        ema_slow: int = 200,
        donchian_period: int = 20,
        volume_period: int = 20,
        volume_factor: float = 1.5,
        atr_period: int = 14,
        stop_atr_mult: float = 2.5,
        chandelier_mult: float = 3.0,
    ):
        self.bollinger_period = bollinger_period
        self.bollinger_k = bollinger_k
        self.bollinger_lookback = bollinger_lookback
        self.bollinger_quantile = bollinger_quantile
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.donchian_period = donchian_period
        self.volume_period = volume_period
        self.volume_factor = volume_factor
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.chandelier_mult = chandelier_mult

    @property
    def required_history_bars(self) -> int:
        return (
            max(
                self.ema_slow,
                self.bollinger_lookback + self.bollinger_period,
                self.donchian_period,
                self.volume_period,
                self.atr_period,
            )
            + 1
        )

    @property
    def _min_candles(self) -> int:
        return self.required_history_bars

    def generate_signal(
        self, kline_data: List[Dict], symbol: str, timeframe: str, current_price: float
    ) -> Optional[TradingSignal]:
        if len(kline_data) < self.required_history_bars:
            return None

        df = pd.DataFrame(kline_data)
        close = df["close"]
        c = float(close.iloc[-1])

        ema_f = float(close.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1])
        ema_s = float(close.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1])
        if not (c > ema_f > ema_s):
            return None

        bw = bollinger_width(df, n=self.bollinger_period, k=self.bollinger_k)
        if len(bw) < self.bollinger_lookback:
            return None
        bw_win = bw.iloc[-self.bollinger_lookback:]
        if bw_win.isna().any():
            return None
        if float(bw.iloc[-1]) > float(bw_win.quantile(self.bollinger_quantile)):
            return None

        dh = donchian_high(df, n=self.donchian_period).shift(1)
        dh_val = float(dh.iloc[-1])
        if pd.isna(dh_val) or c <= dh_val:
            return None

        vr = volumen_relativo(df, n=self.volume_period)
        vr_val = float(vr.iloc[-1])
        if pd.isna(vr_val) or vr_val <= self.volume_factor:
            return None

        atr_series = atr(df, n=self.atr_period)
        atr_val = float(atr_series.iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return None

        stop_loss = current_price - self.stop_atr_mult * atr_val
        return TradingSignal(
            decision="LONG",
            confidence=1.0,
            entry_order="market",
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=None,
            sin_take_profit=True,
            exit_policy=ExitPolicy.TRAILING_STOP,
            reasoning=f"Arquetipo B: Breakout @ {c:.2f}, Stop @ {stop_loss:.2f}",
        )

    def should_exit(
        self, position: ActivePosition, current_price: float, ohlc_data: pd.DataFrame
    ) -> Tuple[bool, Optional[str]]:
        if len(ohlc_data) >= self.atr_period:
            atr_val = float(atr(ohlc_data, n=self.atr_period).iloc[-1])
            if not pd.isna(atr_val) and atr_val > 0:
                bars = max(1, (position.candles_since_entry or 0) + 1)
                highest = float(ohlc_data.iloc[-bars:]["high"].max())
                chand_stop = highest - self.chandelier_mult * atr_val
                if current_price <= chand_stop:
                    return True, "TRAILING_STOP"
                if position.stop_loss is not None and chand_stop > float(position.stop_loss):
                    position.stop_loss = chand_stop
        if position.stop_loss is not None and current_price <= float(position.stop_loss):
            return True, "STOP_LOSS"
        return False, None

    def should_reevaluate(self, position: ActivePosition, current_price: float) -> bool:
        return False

    def get_default_exit_policy(self) -> ExitPolicy:
        return ExitPolicy.TRAILING_STOP
