"""52-Week High Momentum / Breakout Strategy for US equities (M1)."""

from typing import Dict, List, Optional

import pandas as pd

from ..models import ActivePosition, ExitPolicy
from .base import TradingSignal, TradingStrategy


class FiftyTwoWeekHighStrategy(TradingStrategy):
    """
    Long-only 52-week high momentum / breakout strategy for US equities.

    Based on George & Hwang (2004): generates LONG signals when price breaks
    above the rolling 52-week high with trend and volume confirmation.

    All parameters are for daily OHLCV data.
    """

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {
            "name": cls.__name__,
            "display_name": "52-Week High Momentum",
            "type": "deterministic",
            "description": "Long-only breakout strategy with trend and volume confirmation.",
        }

    def __init__(
        self,
        lookback_days: int = 252,
        proximity_threshold: float = 0.98,
        trend_ma_period: int = 50,
        volume_ma_period: int = 20,
        volume_factor: float = 1.5,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        trailing_stop_pct: float = 0.08,
    ):
        self.lookback_days = lookback_days
        self.proximity_threshold = proximity_threshold
        self.trend_ma_period = trend_ma_period
        self.volume_ma_period = volume_ma_period
        self.volume_factor = volume_factor
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct

    def generate_signal(
        self,
        kline_data: List[Dict],
        symbol: str,
        timeframe: str,
        current_price: float,
    ) -> Optional[TradingSignal]:
        min_candles = self.lookback_days + max(self.trend_ma_period, self.volume_ma_period) + 1
        if len(kline_data) < min_candles:
            return None

        df = pd.DataFrame(kline_data)

        high_52w = self._compute_52w_high(df["high"])

        sma_close = self._compute_sma(df["close"], self.trend_ma_period)
        sma_val = sma_close.iloc[-1]
        if pd.isna(sma_val):
            return None
        trend_ok = current_price > sma_val

        vol_ma = df["volume"].rolling(self.volume_ma_period).mean()
        vol_ok = df["volume"].iloc[-1] > self.volume_factor * (vol_ma.iloc[-1] + 1e-10)

        breakout = current_price > high_52w

        if not (breakout and trend_ok and vol_ok):
            return None

        raw = (current_price - high_52w) / (high_52w + 1e-10)
        confidence = max(0.1, min(1.0, raw * 10))

        return TradingSignal(
            decision="LONG",
            confidence=confidence,
            entry_price=current_price,
            stop_loss=current_price * (1 - self.stop_loss_pct),
            take_profit=current_price * (1 + self.take_profit_pct),
            reasoning=(
                f"52w-high breakout: price={current_price:.2f} > high_52w={high_52w:.2f} "
                f"(+{raw*100:.1f}%), trend_ok={trend_ok}, vol_ok={vol_ok}"
            ),
            exit_policy=ExitPolicy.TRAILING_STOP,
            trailing_stop_pct=self.trailing_stop_pct,
        )

    def should_reevaluate(
        self,
        position: ActivePosition,
        current_price: float,
    ) -> bool:
        return False

    def get_default_exit_policy(self) -> ExitPolicy:
        return ExitPolicy.TRAILING_STOP

    def _compute_52w_high(self, highs: pd.Series) -> float:
        # Exclude current (last) candle; use completed candles only
        return float(highs.iloc[-(self.lookback_days + 1):-1].max())

    def _compute_sma(self, series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean()

    def _confidence(self, current_price: float, high_52w: float) -> float:
        raw = (current_price - high_52w) / (high_52w + 1e-10)
        return max(0.1, min(1.0, raw * 10))
