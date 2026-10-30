"""Trade intent contract for per-trade strategies."""

from typing import Any, Dict, Literal, Optional

from pydantic import validator

from .base import TradingSignal


class TradeIntent(TradingSignal):
    bucket_id: str
    strategy_id: str
    symbol: str
    order_type: Literal["market", "stop", "limit"] = "market"
    entry_expires_bars: Optional[int] = None
    atr_at_signal: float
    target_is_structural: bool = False
    invalidation_rules: Dict[str, Any] = {}
    reasoning_metadata: Dict[str, Any] = {}

    @validator("stop_loss")
    def stop_loss_required(cls, value):
        if value is None:
            raise ValueError("stop_loss is required for TradeIntent")
        return value
