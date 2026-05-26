"""RSI Mean Reversion Strategy - example non-LLM strategy."""

from typing import Dict, List, Optional

import pandas as pd

from ..models import ActivePosition, ExitPolicy
from .base import TradingSignal, TradingStrategy


class RSIMeanReversionStrategy(TradingStrategy):
    """
    Simple RSI-based mean reversion strategy (no LLM).

    Rules:
    - BUY (LONG) when RSI < 30 (oversold)
    - SELL (SHORT) when RSI > 70 (overbought)
    - HOLD otherwise
    """

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {
            "name": cls.__name__,
            "display_name": "RSI Mean Reversion",
            "type": "deterministic",
            "description": "Buys oversold RSI and sells overbought RSI extremes.",
        }

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
        trailing_stop_pct: float = 0.05,
    ):
        """
        Initialize RSI Mean Reversion Strategy.

        Args:
            rsi_period: RSI calculation period (default: 14)
            oversold_threshold: RSI threshold for buy signal (default: 30)
            overbought_threshold: RSI threshold for sell signal (default: 70)
            stop_loss_pct: Stop loss percentage (default: 2%)
            take_profit_pct: Take profit percentage (default: 3%)
            trailing_stop_pct: Trailing stop percentage (default: 5%)
        """
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
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
        """
        Generate trading signal based on RSI.

        Args:
            kline_data: List of OHLCV candles
            symbol: Asset symbol
            timeframe: Timeframe
            current_price: Current market price

        Returns:
            TradingSignal or None if HOLD
        """
        # Need at least rsi_period + 1 candles
        if len(kline_data) < self.rsi_period + 1:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(kline_data)

        # Calculate RSI
        rsi = self._calculate_rsi(df["close"], self.rsi_period)
        current_rsi = rsi.iloc[-1]

        # Generate signal
        decision = "HOLD"
        confidence = 0.5

        if current_rsi < self.oversold_threshold:
            decision = "LONG"
            # Confidence increases as RSI gets lower
            confidence = 1.0 - (current_rsi / self.oversold_threshold)
        elif current_rsi > self.overbought_threshold:
            decision = "SHORT"
            # Confidence increases as RSI gets higher
            confidence = (current_rsi - self.overbought_threshold) / (
                100 - self.overbought_threshold
            )

        # If HOLD, return None
        if decision == "HOLD":
            return None

        # Calculate SL/TP
        if decision == "LONG":
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        else:  # SHORT
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)

        return TradingSignal(
            decision=decision,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=f"RSI {current_rsi:.2f} - {'oversold' if decision == 'LONG' else 'overbought'}",
            exit_policy=ExitPolicy.TRAILING_STOP,
            trailing_stop_pct=self.trailing_stop_pct,
        )

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate RSI indicator.

        Args:
            prices: Series of closing prices
            period: RSI period

        Returns:
            Series of RSI values
        """
        delta = prices.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def should_reevaluate(
        self,
        position: ActivePosition,
        current_price: float,
    ) -> bool:
        """
        RSI strategy does not re-evaluate once position is opened.

        Returns:
            False always
        """
        return False

    def get_default_exit_policy(self) -> ExitPolicy:
        """Get default exit policy (trailing stop)."""
        return ExitPolicy.TRAILING_STOP
