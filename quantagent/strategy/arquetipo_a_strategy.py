"""Arquetipo A: Pullback en tendencia (variante de Triple Screen) — QuantAgent-M5."""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..models import ActivePosition, ExitPolicy
from .base import TradingSignal, TradingStrategy
from .indicadores import atr, force_index, safezone_long


class ArquetipoAStrategy(TradingStrategy):
    """
    Alexander Elder's Triple Screen variant: Pullback en tendencia (Arquetipo A).

    Pantalla 1 semanal: pendiente EMA 26 semanal > 0 e histograma MACD semanal subiendo.
    Pantalla 2 diaria: Force Index 2 < 0 o RSI 3 < 30, con cierre >= EMA 22.
    Entrada: stop de compra en máximo de la vela anterior + 0,1 ATR, vence a 3 velas.
    Stop inicial: max(mínimo del swing - 2 ATR, SafeZone).
    Salida: Chandelier (máximo desde la entrada - 3 ATR).
    """

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {
            "name": cls.__name__,
            "display_name": "Arquetipo A (Pullback)",
            "type": "deterministic",
            "description": "Pullback en tendencia con Triple Screen, SafeZone y Chandelier.",
        }

    def __init__(
        self,
        weekly_bars: int = 5,
        trend_ema_period: int = 26,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        force_index_period: int = 2,
        rsi_period: int = 3,
        rsi_oversold: float = 30.0,
        ema_trend_period: int = 22,
        entry_atr_mult: float = 0.1,
        entry_expiry_bars: int = 3,
        atr_period: int = 14,
        swing_window: int = 5,
        swing_atr_mult: float = 2.0,
        safezone_period: int = 10,
        safezone_coef: float = 2.0,
        chandelier_mult: float = 3.0,
    ):
        self.weekly_bars = weekly_bars
        self.trend_ema_period = trend_ema_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.force_index_period = force_index_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.ema_trend_period = ema_trend_period
        self.entry_atr_mult = entry_atr_mult
        self.entry_expiry_bars = entry_expiry_bars
        self.atr_period = atr_period
        self.swing_window = swing_window
        self.swing_atr_mult = swing_atr_mult
        self.safezone_period = safezone_period
        self.safezone_coef = safezone_coef
        self.chandelier_mult = chandelier_mult

    @property
    def required_history_bars(self) -> int:
        weekly = self.weekly_bars * (max(self.trend_ema_period, self.macd_slow + self.macd_signal) + 2)
        daily = max(self.ema_trend_period, self.atr_period, self.safezone_period, self.swing_window) + 2
        return max(weekly, daily)

    @property
    def _min_candles(self) -> int:
        return self.required_history_bars

    def generate_signal(
        self, kline_data: List[Dict], symbol: str, timeframe: str, current_price: float
    ) -> Optional[TradingSignal]:
        if len(kline_data) < self.required_history_bars:
            return None

        df = pd.DataFrame(kline_data)
        if not self._screen1_weekly(self._aggregate_weekly_bars(df)) or not self._screen2_daily(df):
            return None

        atr_series = atr(df, n=self.atr_period)
        atr_val = float(atr_series.iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return None

        entry_price = float(df["high"].iloc[-1]) + self.entry_atr_mult * atr_val
        swing_low = float(df["low"].iloc[-self.swing_window:].min())
        stop_swing = swing_low - self.swing_atr_mult * atr_val
        sz = safezone_long(df, n=self.safezone_period, coef=self.safezone_coef)
        sz_val = float(sz.iloc[-1]) if len(sz) > 0 and not pd.isna(sz.iloc[-1]) else stop_swing
        stop_loss = max(stop_swing, sz_val)
        if stop_loss >= entry_price:
            stop_loss = entry_price - self.swing_atr_mult * atr_val

        return TradingSignal(
            decision="LONG",
            confidence=1.0,
            entry_order="stop",
            entry_price=entry_price,
            entry_expiry_bars=self.entry_expiry_bars,
            stop_loss=stop_loss,
            take_profit=None,
            exit_policy=ExitPolicy.TRAILING_STOP,
            reasoning=f"Arquetipo A: Buy stop @ {entry_price:.2f}, Stop @ {stop_loss:.2f}",
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

    def _aggregate_weekly_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        rem = len(df) % self.weekly_bars
        t = df.iloc[rem:] if rem > 0 else df
        blocks = [t.iloc[i : i + self.weekly_bars] for i in range(0, len(t), self.weekly_bars)]
        return pd.DataFrame({
            "open": [float(b["open"].iloc[0]) for b in blocks],
            "high": [float(b["high"].max()) for b in blocks],
            "low": [float(b["low"].min()) for b in blocks],
            "close": [float(b["close"].iloc[-1]) for b in blocks],
            "volume": [float(b["volume"].sum()) for b in blocks],
        })

    def _screen1_weekly(self, weekly_df: pd.DataFrame) -> bool:
        min_bars = max(self.trend_ema_period, self.macd_slow + self.macd_signal) + 2
        if len(weekly_df) < min_bars:
            return False
        c = weekly_df["close"]
        ema = c.ewm(span=self.trend_ema_period, adjust=False).mean()
        if float(ema.iloc[-1] - ema.iloc[-2]) <= 0:
            return False
        m = c.ewm(span=self.macd_fast, adjust=False).mean() - c.ewm(span=self.macd_slow, adjust=False).mean()
        h = m - m.ewm(span=self.macd_signal, adjust=False).mean()
        return float(h.iloc[-1] - h.iloc[-2]) > 0

    def _screen2_daily(self, df: pd.DataFrame) -> bool:
        if len(df) < max(self.ema_trend_period, self.rsi_period, self.force_index_period) + 1:
            return False
        c = df["close"]
        if float(c.iloc[-1]) < float(c.ewm(span=self.ema_trend_period, adjust=False).mean().iloc[-1]):
            return False
        fi = force_index(df, n=self.force_index_period)
        if len(fi) > 0 and not pd.isna(fi.iloc[-1]) and float(fi.iloc[-1]) < 0:
            return True
        d = c.diff()
        gain = d.where(d > 0, 0.0).ewm(alpha=1.0 / self.rsi_period, adjust=False).mean()
        loss = (-d.where(d < 0, 0.0)).ewm(alpha=1.0 / self.rsi_period, adjust=False).mean()
        rsi = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-10))))
        return float(rsi.iloc[-1]) < self.rsi_oversold
