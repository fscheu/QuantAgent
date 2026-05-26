"""Base classes for trading strategies with Template Method Pattern."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from ..models import ActivePosition, ExitPolicy, OrderSide


class TradingSignal(BaseModel):
    """Standardized trading signal output from any strategy."""

    decision: str = Field(description="Trading decision: 'LONG', 'SHORT', or 'HOLD'")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Decision confidence (0.0-1.0)"
    )
    entry_price: Optional[float] = Field(
        default=None, description="Suggested entry price"
    )
    stop_loss: Optional[float] = Field(
        default=None, description="Suggested stop loss price"
    )
    take_profit: Optional[float] = Field(
        default=None, description="Suggested take profit price"
    )
    reasoning: str = Field(default="", description="Reasoning for the decision")
    exit_policy: ExitPolicy = Field(
        default=ExitPolicy.TRAILING_STOP, description="Exit policy for position"
    )
    trailing_stop_pct: Optional[float] = Field(
        default=None, description="Trailing stop percentage (0.0-1.0)"
    )
    max_hold_candles: Optional[int] = Field(
        default=None, description="Maximum candles to hold position"
    )


class TradingStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Implements Template Method Pattern for exit logic:
    - should_exit() has DEFAULT implementation (fixed % SL/TP + trailing)
    - Strategies can override for custom logic (ATR-based, time-based, etc.)
    """

    @abstractmethod
    def generate_signal(
        self,
        kline_data: List[Dict],
        symbol: str,
        timeframe: str,
        current_price: float,
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal based on market data.

        Args:
            kline_data: List of OHLCV candles (dicts with keys: open, high, low, close, volume, timestamp)
            symbol: Asset symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "4h")
            current_price: Current market price

        Returns:
            TradingSignal if action needed, None otherwise
        """
        pass

    @classmethod
    def describe(cls) -> Dict[str, str]:
        """Return basic strategy metadata for registries and UIs."""
        return {
            "name": cls.__name__,
            "display_name": cls.__name__,
            "type": "unknown",
            "description": "",
        }

    def should_exit(
        self,
        position: ActivePosition,
        current_price: float,
        ohlc_data: pd.DataFrame,
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide if position should be closed (Template Method).

        DEFAULT IMPLEMENTATION: Fixed % SL/TP + trailing stop check.
        Strategies can override for custom logic (ATR, volatility-based, etc.).

        Args:
            position: Active position to evaluate
            current_price: Current market price
            ohlc_data: OHLC DataFrame for calculations (columns: open, high, low, close, volume, timestamp)

        Returns:
            (should_exit, reason):
                - should_exit: True if position should close
                - reason: "STOP_LOSS", "TAKE_PROFIT", "TRAILING_STOP", "TIME_EXPIRED", or None
        """
        # Check stop loss
        if position.side == OrderSide.BUY:
            if current_price <= float(position.stop_loss):
                return (True, "STOP_LOSS")
        else:  # SHORT
            if current_price >= float(position.stop_loss):
                return (True, "STOP_LOSS")

        # Check take profit
        if position.side == OrderSide.BUY:
            if current_price >= float(position.take_profit):
                return (True, "TAKE_PROFIT")
        else:  # SHORT
            if current_price <= float(position.take_profit):
                return (True, "TAKE_PROFIT")

        # Check trailing stop
        if position.exit_policy == ExitPolicy.TRAILING_STOP:
            if self._check_trailing_stop(position, current_price):
                return (True, "TRAILING_STOP")

        # Check time-based exit
        if (
            position.max_hold_candles
            and position.candles_since_entry >= position.max_hold_candles
        ):
            if position.exit_policy == ExitPolicy.TIME_BASED:
                return (True, "TIME_EXPIRED")

        return (False, None)

    def _check_trailing_stop(
        self, position: ActivePosition, current_price: float
    ) -> bool:
        """
        Check trailing stop condition (default: % fixed).

        Strategies can override to use ATR or other metrics.

        Args:
            position: Active position
            current_price: Current market price

        Returns:
            True if trailing stop triggered
        """
        if not position.trailing_stop_pct:
            return False

        # Update highest/lowest seen prices
        if position.side == OrderSide.BUY:
            # LONG position
            if position.highest_price_seen is None or current_price > float(
                position.highest_price_seen
            ):
                position.highest_price_seen = current_price

            # Trailing stop from highest
            trailing_sl = float(position.highest_price_seen) * (
                1 - position.trailing_stop_pct
            )
            return current_price < trailing_sl

        else:
            # SHORT position
            if position.lowest_price_seen is None or current_price < float(
                position.lowest_price_seen
            ):
                position.lowest_price_seen = current_price

            # Trailing stop from lowest
            trailing_sl = float(position.lowest_price_seen) * (
                1 + position.trailing_stop_pct
            )
            return current_price > trailing_sl

    @abstractmethod
    def should_reevaluate(
        self,
        position: ActivePosition,
        current_price: float,
    ) -> bool:
        """
        Determine if position should be re-evaluated by agent.

        Only relevant for ExitPolicy.REEVALUATE.

        Args:
            position: Active position
            current_price: Current market price

        Returns:
            True if strategy should be invoked again
        """
        pass

    def get_default_exit_policy(self) -> ExitPolicy:
        """
        Get default exit policy for this strategy.

        Can be overridden by subclasses.

        Returns:
            Default ExitPolicy enum value
        """
        return ExitPolicy.TRAILING_STOP

    @property
    def required_history_bars(self) -> int:
        """Minimum OHLCV bars required for strategy evaluation."""
        return 30
