"""Exit checks evaluated at a candle's close."""

from typing import Callable, Optional, Tuple

from ..models import ActivePosition, ExitPolicy, OrderSide
from .exit_reasons import STOP_LOSS, TAKE_PROFIT, TIME_EXPIRED, TRAILING_STOP


def should_exit_at_close(
    position: ActivePosition,
    current_price: float,
    check_trailing_stop: Callable[[ActivePosition, float], bool],
) -> Tuple[bool, Optional[str]]:
    """Apply the strategy's default stop, target, trailing, and time rules."""
    if position.side == OrderSide.BUY:
        if current_price <= float(position.stop_loss):
            return (True, STOP_LOSS)
    elif current_price >= float(position.stop_loss):
        return (True, STOP_LOSS)

    if position.side == OrderSide.BUY:
        if current_price >= float(position.take_profit):
            return (True, TAKE_PROFIT)
    elif current_price <= float(position.take_profit):
        return (True, TAKE_PROFIT)

    if position.exit_policy == ExitPolicy.TRAILING_STOP and check_trailing_stop(
        position, current_price
    ):
        return (True, TRAILING_STOP)

    if (
        position.max_hold_candles
        and position.candles_since_entry >= position.max_hold_candles
        and position.exit_policy == ExitPolicy.TIME_BASED
    ):
        return (True, TIME_EXPIRED)

    return (False, None)
