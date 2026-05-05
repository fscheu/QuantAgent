"""Triple Screen Strategy (Alexander Elder) — QuantAgent-vna M1 implementation."""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..models import ActivePosition, ExitPolicy
from .base import TradingSignal, TradingStrategy

_EPS = 1e-10


class TripleScreenStrategy(TradingStrategy):
    """
    Alexander Elder's Triple Screen trading system.

    Three successive filters before any entry:
      Screen 1 — weekly EMA slope identifies dominant trend direction.
      Screen 2 — stochastic oscillator detects pullback within trend.
      Screen 3 — breakout above/below prior candle's high/low triggers entry.
    """

    def __init__(
        self,
        weekly_bars: int = 5,
        trend_ema_period: int = 13,
        stoch_k_period: int = 5,
        stoch_d_period: int = 3,
        stoch_oversold: float = 20.0,
        stoch_overbought: float = 80.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        trailing_stop_pct: float = 0.05,
    ):
        self.weekly_bars = weekly_bars
        self.trend_ema_period = trend_ema_period
        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct

    @property
    def _min_candles(self) -> int:
        return (
            self.weekly_bars * (self.trend_ema_period + 1)
            + self.stoch_k_period
            + self.stoch_d_period
        )

    def generate_signal(
        self,
        kline_data: List[Dict],
        symbol: str,
        timeframe: str,
        current_price: float,
    ) -> Optional[TradingSignal]:
        if len(kline_data) < self._min_candles:
            return None

        df = pd.DataFrame(kline_data)

        weekly_df = self._aggregate_weekly_bars(df)
        trend = self._screen1_trend(weekly_df)
        if trend is None:
            return None

        screen2_active = self._screen2_oscillator(df, trend)
        if not screen2_active:
            return None

        screen3_active = self._screen3_trigger(kline_data, trend, current_price)
        if not screen3_active:
            return None

        stoch_k, _ = self._stochastic(df)
        confidence = self._confidence(stoch_k, trend)

        if trend == "UP":
            decision = "LONG"
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        else:
            decision = "SHORT"
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)

        reasoning = (
            f"Screen 1: trend={trend} (EMA slope on {self.weekly_bars}-bar aggregates); "
            f"Screen 2: stoch_k={stoch_k:.1f} ({'oversold' if trend == 'UP' else 'overbought'}); "
            f"Screen 3: breakout {'above prior high' if trend == 'UP' else 'below prior low'}"
        )

        return TradingSignal(
            decision=decision,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            exit_policy=ExitPolicy.TRAILING_STOP,
            trailing_stop_pct=self.trailing_stop_pct,
        )

    def should_reevaluate(self, position: ActivePosition, current_price: float) -> bool:
        return False

    def get_default_exit_policy(self) -> ExitPolicy:
        return ExitPolicy.TRAILING_STOP

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate_weekly_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reshape kline_data into synthetic higher-TF bars of `weekly_bars` candles."""
        n = len(df)
        complete = (n // self.weekly_bars) * self.weekly_bars
        trimmed = df.iloc[:complete].copy()

        groups = []
        for start in range(0, complete, self.weekly_bars):
            block = trimmed.iloc[start : start + self.weekly_bars]
            groups.append(
                {
                    "open": block["open"].iloc[0],
                    "high": block["high"].max(),
                    "low": block["low"].min(),
                    "close": block["close"].iloc[-1],
                    "volume": block["volume"].sum(),
                }
            )
        return pd.DataFrame(groups)

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        """Exponential moving average with alpha = 2/(period+1)."""
        return series.ewm(span=period, adjust=False).mean()

    def _stochastic(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Compute last completed Stochastic %K and %D values."""
        lowest_low = df["low"].rolling(window=self.stoch_k_period, min_periods=self.stoch_k_period).min()
        highest_high = df["high"].rolling(window=self.stoch_k_period, min_periods=self.stoch_k_period).max()
        pct_k = 100.0 * (df["close"] - lowest_low) / (highest_high - lowest_low + _EPS)
        pct_d = pct_k.rolling(window=self.stoch_d_period, min_periods=self.stoch_d_period).mean()
        return float(pct_k.iloc[-1]), float(pct_d.iloc[-1])

    def _screen1_trend(self, weekly_df: pd.DataFrame) -> Optional[str]:
        """Return 'UP', 'DOWN', or None when insufficient bars."""
        if len(weekly_df) < self.trend_ema_period + 1:
            return None
        ema = self._ema(weekly_df["close"], self.trend_ema_period)
        slope = float(ema.iloc[-1]) - float(ema.iloc[-2])
        return "UP" if slope > 0 else "DOWN"

    def _screen2_oscillator(self, df: pd.DataFrame, trend: str) -> bool:
        """Return True when stochastic confirms a pullback within the trend."""
        stoch_k, _ = self._stochastic(df)
        if trend == "UP":
            return stoch_k <= self.stoch_oversold
        else:
            return stoch_k >= self.stoch_overbought

    def _screen3_trigger(self, kline_data: List[Dict], trend: str, current_price: float) -> bool:
        """Return True when price breaks out of the prior completed candle."""
        if len(kline_data) < 2:
            return False
        prior = kline_data[-2]
        if trend == "UP":
            return current_price > float(prior["high"])
        else:
            return current_price < float(prior["low"])

    def _confidence(self, stoch_k: float, trend: str) -> float:
        """Map stochastic depth to a [0.1, 1.0] confidence score."""
        if trend == "UP":
            raw = (self.stoch_oversold - stoch_k) / (self.stoch_oversold + _EPS)
        else:
            raw = (stoch_k - self.stoch_overbought) / (100.0 - self.stoch_overbought + _EPS)
        return max(0.1, min(1.0, raw))
