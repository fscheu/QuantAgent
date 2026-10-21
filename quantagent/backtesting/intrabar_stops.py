"""Pure intrabar stop evaluation shared by backtest and paper trading."""

from typing import Any, Optional, Tuple

from quantagent.strategy.exit_reasons import STOP_LOSS, TAKE_PROFIT


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Extract a named field from a dict, pandas Series, or object."""
    for candidate in (key, key.capitalize(), key.upper(), key.lower()):
        if isinstance(obj, dict) and candidate in obj:
            return obj[candidate]
        if hasattr(obj, candidate):
            return getattr(obj, candidate)
        try:
            if candidate in obj:
                return obj[candidate]
        except (KeyError, TypeError, IndexError):
            pass
    return default


def check_intrabar_stops(
    position: Any,
    candle: Any,
) -> Optional[Tuple[str, float]]:
    """Evaluate whether an active position hits stop loss or take profit intrabar.

    Pure function: (position, candle) -> (reason, exit_price) or None.
    """
    raw_side = _get_field(position, "side")
    if raw_side is None:
        return None
    if hasattr(raw_side, "value"):
        raw_side = raw_side.value
    side_str = str(raw_side).lower()

    is_long = side_str in ("buy", "long")
    is_short = side_str in ("sell", "short")
    if not (is_long or is_short):
        return None

    raw_sl = _get_field(position, "stop_loss")
    raw_tp = _get_field(position, "take_profit")
    sl = float(raw_sl) if raw_sl is not None and float(raw_sl) > 0 else None
    tp = float(raw_tp) if raw_tp is not None and float(raw_tp) > 0 else None

    open_val = _get_field(candle, "open")
    high_val = _get_field(candle, "high")
    low_val = _get_field(candle, "low")
    if open_val is None or high_val is None or low_val is None:
        return None

    open_price = float(open_val)
    high_price = float(high_val)
    low_price = float(low_val)

    if is_long:
        # 1. Opening gap
        if sl is not None and open_price <= sl:
            return (STOP_LOSS, open_price)
        if tp is not None and open_price >= tp:
            return (TAKE_PROFIT, open_price)

        # 2 & 3. Intrabar hit (if both, STOP_LOSS wins)
        hit_sl = sl is not None and low_price <= sl
        hit_tp = tp is not None and high_price >= tp

        if hit_sl:
            return (STOP_LOSS, sl)
        if hit_tp:
            return (TAKE_PROFIT, tp)

        return None

    else:  # is_short
        # 1. Opening gap
        if sl is not None and open_price >= sl:
            return (STOP_LOSS, open_price)
        if tp is not None and open_price <= tp:
            return (TAKE_PROFIT, open_price)

        # 2 & 3. Intrabar hit (if both, STOP_LOSS wins)
        hit_sl = sl is not None and high_price >= sl
        hit_tp = tp is not None and low_price <= tp

        if hit_sl:
            return (STOP_LOSS, sl)
        if hit_tp:
            return (TAKE_PROFIT, tp)

        return None
