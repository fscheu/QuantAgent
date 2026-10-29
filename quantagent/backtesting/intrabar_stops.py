"""Pure intrabar stop evaluation shared by backtest and paper trading."""

from typing import Any, Optional, Tuple

from quantagent.strategy.exit_reasons import STOP_LOSS, TAKE_PROFIT
from quantagent.strategy.close_exit import should_exit_at_close


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


def evaluate_position_exit(
    position: Any,
    candle: Any,
) -> Optional[Tuple[str, float]]:
    """Return the first intrabar or close exit as (reason, execution price)."""
    intrabar_exit = check_intrabar_stops(position, candle)
    if intrabar_exit is not None:
        return intrabar_exit
    close = _get_field(candle, "close")
    if close is None:
        return None
    def trailing_stop(pos: Any, price: float) -> bool:
        trailing_pct = _get_field(pos, "trailing_stop_pct")
        if not trailing_pct:
            return False
        raw_side = _get_field(pos, "side")
        side = str(getattr(raw_side, "value", raw_side)).lower()
        if side in ("buy", "long"):
            high = _get_field(pos, "highest_price_seen")
            high = max(float(high), price) if high is not None else price
            return price < high * (1 - float(trailing_pct))
        low = _get_field(pos, "lowest_price_seen")
        low = min(float(low), price) if low is not None else price
        return price > low * (1 + float(trailing_pct))

    should_exit, reason = should_exit_at_close(position, float(close), trailing_stop)
    return (reason, float(close)) if should_exit else None


def check_stop_entry(
    entry_price: Optional[float],
    candle: Any,
) -> Optional[float]:
    """Evaluate whether a buy stop order is filled intrabar.

    Pure function: returns fill_price if filled (high >= entry_price), else None.
    Execution price is max(open, entry_price).
    """
    if entry_price is None:
        return None
    open_val = _get_field(candle, "open")
    high_val = _get_field(candle, "high")
    if open_val is None or high_val is None:
        return None

    open_price = float(open_val)
    high_price = float(high_val)
    stop_price = float(entry_price)

    if high_price >= stop_price:
        return max(open_price, stop_price)
    return None
